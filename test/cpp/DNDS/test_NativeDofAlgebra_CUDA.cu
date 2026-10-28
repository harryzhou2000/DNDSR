#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/CUDAExecution.hpp"
#include <cuda_runtime.h>

namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;

TEST_CASE_TEMPLATE("Native CUDA algebra retains float/double fields, joins streams and excludes padding/ghosts", T, float, double)
{
    R::EagerEngine engine;
    auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto consumer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    for (size_t width : {size_t(5), size_t(6), size_t(7), size_t(114)})
        for (size_t rows : {size_t(0), size_t(257)})
        {
            CAPTURE(width);
            CAPTURE(rows);
            A::DofSpace space(rows, 3);
            A::DofField<T> host(space, width, width + 3);
            {
                auto owned = host.Owned().Storage().Write(), ghost = host.Ghost().Storage().Write();
                std::fill_n(owned.template View<T>().data, host.Owned().Layout().Elements(), T(-919));
                std::fill_n(ghost.template View<T>().data, host.Ghost().Layout().Elements(), T(-727));
            }
            auto x = std::make_unique<A::DofField<T>>(space, host.Owned().To(R::Device::CUDA(0)), host.Ghost().To(R::Device::CUDA(0)));
            A::DofField<T> y(space, host.Owned().To(R::Device::CUDA(0)), host.Ghost().To(R::Device::CUDA(0)));
            A::Fill(engine, producer, *x, T(3));
            A::Scale(engine, producer, *x, T(2));
            consumer.WaitFor(producer.Record());
            A::Copy(engine, consumer, y, *x);
            A::Axpy(engine, consumer, y, T(-0.5), *x);
            A::LinearCombination(engine, consumer, y, T(2), y, T(1), *x);
            x.reset();
            auto dot = A::DotLocal(engine, consumer, y, y);
            auto norm = A::NormLocal(engine, consumer, y);
            const auto sum = A::ComponentSumLocal(engine, consumer, y).Get();
            CHECK(dot.Get() == doctest::Approx(double(rows * width) * 144));
            CHECK(norm.Get() == doctest::Approx(12 * std::sqrt(double(rows * width))));
            for (double value : sum)
                CHECK(value == double(rows) * 12);
            engine.Synchronize();
            const auto owned = y.Owned().To(R::Device::Host()), ghost = y.Ghost().To(R::Device::Host());
            const auto raw = owned.Storage().Read(), ghosts = ghost.Storage().Read();
            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t component = 0; component < width; ++component)
                    CHECK(raw.template View<T>().data[row * (width + 3) + component] == T(12));
                for (size_t component = width; component < width + 3; ++component)
                    CHECK(raw.template View<T>().data[row * (width + 3) + component] == T(-919));
            }
            for (size_t item = 0; item < ghost.Layout().Elements(); ++item)
                CHECK(ghosts.template View<T>().data[item] == T(-727));
            {
                auto write = host.Owned().Write();
                for (size_t row = 0; row < rows; ++row)
                    for (size_t component = 0; component < width; ++component)
                        write.At(row, component) = T(-3.25 + double(row % 17) / 2 - double(component) / 4);
            }
            A::DofField<T> varied(space, host.Owned().To(R::Device::CUDA(0)), host.Ghost().To(R::Device::CUDA(0)));
            const auto minima = A::ComponentMinLocal(engine, consumer, varied).Get();
            const auto maxima = A::ComponentMaxLocal(engine, consumer, varied).Get();
            const auto sums = A::ComponentSumLocal(engine, consumer, varied).Get();
            double expectedSquares = 0;
            for (size_t component = 0; component < width; ++component)
            {
                const double base = -3.25 - double(component) / 4;
                const size_t full = rows / 17, tail = rows % 17;
                const double rowSum = double(full * 136) + double(tail) * double(tail ? tail - 1 : 0) / 2;
                CHECK(sums[component] == double(rows) * base + rowSum / 2);
                CHECK(minima[component] == (rows ? base : std::numeric_limits<double>::infinity()));
                CHECK(maxima[component] == (rows ? base + double(std::min<size_t>(rows - 1, 16)) / 2 : -std::numeric_limits<double>::infinity()));
                for (size_t row = 0; row < rows; ++row)
                {
                    const double value = base + double(row % 17) / 2;
                    expectedSquares += value * value;
                }
            }
            CHECK(A::DotLocal(engine, consumer, varied, varied).Get() == doctest::Approx(expectedSquares));
            CHECK_THROWS(A::Scale(y, T(2))); // Immediate overload is CPU-only.
            auto cpu = engine.CreateCPUStream();
            CHECK_THROWS(A::Fill(engine, cpu, y, T(1)));
        }
    A::DofSpace fixedSpace(13, 2);
    A::DofField<T, 6> fixed(fixedSpace, 6, 9, R::Device::CUDA(0));
    A::Fill(engine, producer, fixed, T(-2));
    A::Scale(engine, producer, fixed, T(3));
    consumer.WaitFor(producer.Record());
    A::Axpy(engine, consumer, fixed, T(-0.5), fixed);
    CHECK(A::DotLocal(engine, consumer, fixed, fixed).Get() == 13 * 6 * 9);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
}
