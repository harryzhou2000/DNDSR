#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Discretization/CUDA/FiniteVolumeReconstructionKernels.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace C = DNDS::Cauchy;
namespace K = C::CUDA;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using DNDS::real;

namespace
{
    template <class T, class F>
    R::Array<T> Table(I rows, I width, F fill)
    {
        R::Array<T> result(R::RowLayout::Dense(rows, width));
        auto write = result.Write();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (I j = 0; j < width; ++j)
                write.View()(iRow, j) = fill(iRow, j);
        return result;
    }
}

TEST_CASE("CUDA analytic mean characteristics match CPU inverse and canonical periodic face frames")
{
    const C::PerfectGas gas;
    const C::EquationSystem equation(gas, false);
    const std::array<C::FlowState, 4> states{
        gas.Conservative({1., .4, .2, -.1, 1.}),
        gas.Conservative({2., -.3, .1, .2, 1.3}),
        gas.Conservative({1., 1e6, 0, 0, 1.}),
        gas.Conservative({.8, .2, -.4, .3, .9})};
    const std::array<C::Vector3, 4> normals{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {.6, .8, 0}}};
    auto samples = Table<C::FiniteVolumeDetail::SampleRecord>(4, 1, [](I iFace, I)
                                                              {
        C::FiniteVolumeDetail::SampleRecord result{};
        result.cell = iFace == 1 ? 1 : iFace == 3 ? 2 : 0;
        result.other = iFace == 0 ? 1 : iFace == 1 ? 3 : DNDS::UnInitIndex;
        result.side = iFace == 1;
        result.point = result.face = iFace;
        result.changesFrame = iFace == 1;
        result.rotation = result.changesFrame ? std::array<real, 9>{0, 1, 0, -1, 0, 0, 0, 0, 1}
                                             : std::array<real, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1};
        return result; });
    auto entries = Table<I>(5, 1, [](I iFace, I)
                            { return iFace == 4 ? DNDS::UnInitIndex : iFace; });
    auto quadrature = Table<real>(4, 7, [&](I iFace, I j)
                                  { return j >= 3 && j < 6 ? normals[iFace][j - 3] : 0.; });
    std::array<C::FlowCharacteristics, 5> expected{};
    {
        const auto read = samples.Read();
        for (I iFace = 0; iFace < 4; ++iFace)
        {
            const auto &sample = read.View()(iFace, 0);
            const auto left = states[sample.side ? sample.other : sample.cell];
            auto right = sample.other == DNDS::UnInitIndex ? left : states[sample.side ? sample.cell : sample.other];
            if (sample.changesFrame)
            {
                const Eigen::Map<const Eigen::Matrix3d> rotation(sample.rotation.data());
                Eigen::Vector3d momentum{right[1], right[2], right[3]};
                momentum = (rotation.transpose() * momentum).eval();
                for (int d = 0; d < 3; ++d)
                    right[1 + d] = momentum[d];
            }
            expected[iFace] = equation.PrepareCharacteristics(left, right, normals[iFace]);
            const auto analytic = C::TryPerfectGasCharacteristics(gas, left, right, normals[iFace]);
            REQUIRE(bool(analytic));
            CHECK(analytic.value.componentwiseFallback == expected[iFace].componentwiseFallback);
            for (int j = 0; j < 25; ++j)
            {
                CHECK(analytic.value.forward[j] == doctest::Approx(expected[iFace].forward[j]).epsilon(2e-10).scale(1));
                CHECK(analytic.value.inverse[j] == doctest::Approx(expected[iFace].inverse[j]).epsilon(2e-10).scale(1));
            }
        }
    }
    CHECK(expected[3].componentwiseFallback);
    for (int j = 0; j < 25; ++j)
        expected[4].forward[j] = expected[4].inverse[j] = (j % 6 == 0);
    const auto device = R::Device::CUDA(0);
    auto ds = samples.To(device);
    auto de = entries.To(device);
    auto dq = quadrature.To(device);
    for (const I width : {I(5), I(6), I(7)})
    {
        auto means = Table<real>(3, width, [&](I iCell, I j)
                                 { return j < 5 ? states[iCell][j] : 1e8; });
        auto ghosts = Table<real>(1, width, [&](I, I j)
                                  { return j < 5 ? states[3][j] : -1e8; });
        auto dm = means.To(device), dg = ghosts.To(device);
        R::Array<real> forward(R::RowLayout::Dense(5, 25), device), inverse(forward.Layout(), device);
        R::Array<DNDS::NumericalStatus> status(R::RowLayout::Dense(5, 1), device);
        R::Array<int> fallback(R::RowLayout::Dense(5, 1), device);
        R::EagerEngine engine;
        const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        const auto submit = [&]
        {
            R::SubmitArrays(engine, stream, [gas](R::ExecutionContext c, auto s, auto e, auto q, auto u, auto g, auto l, auto r, auto errors, auto flags)
                            { K::PrepareFaceCharacteristicMatrices(c, {s, e, q}, {u, g}, gas, l, r, errors, flags); }, R::ReadAs<R::FixedRows<1>>(ds), R::ReadAs<R::FixedRows<1>>(de), R::ReadAs<R::FixedRows<7>>(dq), R::ReadAs<R::TableRows>(dm), R::ReadAs<R::TableRows>(dg), R::WriteAs<R::TableRows>(forward), R::WriteAs<R::TableRows>(inverse), R::WriteAs<R::FixedRows<1>>(status), R::WriteAs<R::FixedRows<1>>(fallback));
        };
        submit();
        engine.Synchronize();
        {
            const auto l = forward.To(R::Device::Host()), r = inverse.To(R::Device::Host());
            const auto s = status.To(R::Device::Host());
            const auto f = fallback.To(R::Device::Host());
            const auto left = l.Read(), right = r.Read();
            const auto errors = s.Read();
            const auto flags = f.Read();
            for (I iFace = 0; iFace < 5; ++iFace)
            {
                REQUIRE(bool(errors.View()(iFace, 0)));
                CHECK(flags.View()(iFace, 0) == int(expected[iFace].componentwiseFallback));
                for (I j = 0; j < 25; ++j)
                {
                    CHECK(left.View()(iFace, j) == doctest::Approx(expected[iFace].forward[j]).epsilon(2e-10).scale(1));
                    CHECK(right.View()(iFace, j) == doctest::Approx(expected[iFace].inverse[j]).epsilon(2e-10).scale(1));
                }
            }
        }
        // Invalid physics is not relabeled as a successful component fallback.
        if (width == 5)
        {
            means.Write().At(0, 0) = -1;
            dm.CopyFrom(means, stream.Context()).Wait();
            submit();
            engine.Synchronize();
            const auto host = status.To(R::Device::Host());
            const auto read = host.Read();
            CHECK_FALSE(bool(read.View()(0, 0)));
            CHECK(bool(read.View()(4, 0)));
        }
    }
}
