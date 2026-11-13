#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <numeric>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using DNDS::real;
using Field = A::DofField<real>;

TEST_CASE("Queued CUDA diagonal SVD and scalar inverses match native CPU actions")
{
    R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    for (I rows : {I(0), I(131)})
        for (int width : {5, 6, 7})
            for (auto mode : {C::DiagonalStorage::Entries, C::DiagonalStorage::Block})
            {
                CAPTURE(rows);
                CAPTURE(width);
                CAPTURE(int(mode));
                const A::DofSpace space(rows);
                std::vector<std::vector<I>> columns(rows);
                std::vector<I> order(rows);
                std::iota(order.begin(), order.end(), I(0));
                C::LinearProblemContext cpu(space, width, columns, order, mode);
                C::LinearProblemContext gpu(space, width, columns, order, mode, C::OffDiagonalStorage::Assembled, context.device);
                {
                    auto write = cpu.Diagonal().Write();
                    const auto diagonal = write.View<R::TableRows>();
                    for (I iCell = 0; iCell < rows; ++iCell)
                        if (mode == C::DiagonalStorage::Entries)
                            for (int j = 0; j < width; ++j)
                                diagonal(iCell, j) = 2 + .125 * (iCell % 3 + j);
                        else
                            for (int j = 0; j < width; ++j)
                                for (int i = 0; i < width; ++i)
                                {
                                    // Exact powers of two make the singular
                                    // row-scaled block all ones (rank one).
                                    const real scale = real(1 << i);
                                    diagonal(iCell, i + width * j) = iCell % 3 == 0 ? scale
                                                                                    : scale * (i == j ? 3 + .125 * i : .01 * (1 + 2 * i - j));
                                }
                }
                gpu.Diagonal().CopyFrom(cpu.Diagonal(), context).Wait();
                gpu.PrepareDiagonalInverse(context);
                cpu.InvertDiagonal();
                gpu.SubmitInvertDiagonal(engine, stream);
                engine.Synchronize();
                gpu.ObserveDiagonalInverse();
                CHECK(gpu.LastInverseFallbacks() == 0);
                CHECK(gpu.LastInverseSVDCells() == (mode == C::DiagonalStorage::Block ? rows : 0));
                Field rhs(space, width), expected(space, width);
                Field deviceRHS(space, width, 0, context.device), actual(space, width, 0, context.device);
                // Probe every matrix column, including filtered modes. This
                // checks the pseudoinverse, not just one favorable RHS vector.
                for (int column = 0; column < width; ++column)
                {
                    {
                        auto write = rhs.Owned().Write();
                        for (I iCell = 0; iCell < rows; ++iCell)
                            for (int j = 0; j < width; ++j)
                                write.View()(iCell, j) = j == column ? 1 : 0;
                    }
                    deviceRHS.Owned().CopyFrom(rhs.Owned(), context).Wait();
                    cpu.JacobiZero(rhs, expected);
                    gpu.SubmitJacobiZero(engine, stream, deviceRHS, actual);
                    engine.Synchronize();
                    const auto host = actual.Owned().To(R::Device::Host());
                    const auto a = host.Read(), b = expected.Owned().Read();
                    for (I iCell = 0; iCell < rows; ++iCell)
                        for (int j = 0; j < width; ++j)
                            CHECK(a.View()(iCell, j) == doctest::Approx(b.View()(iCell, j)).epsilon(2e-11).scale(1));
                }
            }
}

TEST_CASE("CUDA inverse observation rejects invalid diagonal before Jacobi publication")
{
    const A::DofSpace space(1);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    C::LinearProblemContext problem(space, 5, {{}}, {0}, C::DiagonalStorage::Block,
                                    C::OffDiagonalStorage::Assembled, context.device);
    R::Array<real> invalid(R::RowLayout::Dense(1, 25));
    {
        auto write = invalid.Write();
        for (int j = 0; j < 25; ++j)
            write.View()(0, j) = 0;
    }
    problem.Diagonal().CopyFrom(invalid, context).Wait();
    problem.PrepareDiagonalInverse(context);
    problem.SubmitInvertDiagonal(engine, stream);
    engine.Synchronize();
    CHECK_THROWS(problem.ObserveDiagonalInverse());
    Field rhs(space, 5, 0, context.device), result(space, 5, 0, context.device);
    CHECK_THROWS(problem.SubmitJacobiZero(engine, stream, rhs, result));
}
