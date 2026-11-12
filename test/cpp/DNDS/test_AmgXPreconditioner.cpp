#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <cmath>

namespace R = DNDS::Runtime;
using I = DNDS::index;

TEST_CASE("Single-device AmgX applies a fixed linear local preconditioner and refreshes values")
{
    REQUIRE(DNDS::Algebra::AmgXPreconditioner::Available());
    constexpr I n = 31, nnz = 3 * n - 2;
    R::Array<int> offsets(R::RowLayout::Dense(n + 1, 1)), columns(R::RowLayout::Dense(nnz, 1));
    R::Array<double> hostValues(R::RowLayout::Dense(nnz, 1)), hostRhs(R::RowLayout::Dense(n, 1));
    {
        auto o = offsets.Write(), c = columns.Write();
        auto a = hostValues.Write(), r = hostRhs.Write();
        I entry = 0;
        for (I iRow = 0; iRow < n; ++iRow)
        {
            o.View()(iRow, 0) = int(entry);
            for (I j = std::max<I>(0, iRow - 1); j <= std::min<I>(n - 1, iRow + 1); ++j, ++entry)
            {
                c.View()(entry, 0) = int(j);
                a.View()(entry, 0) = iRow == j ? 4 : -1;
            }
            r.View()(iRow, 0) = 1 + 0.1 * iRow;
        }
        o.View()(n, 0) = int(entry);
    }
    const auto device = R::Device::CUDA(0);
    DNDS::Algebra::AmgXPreconditioner provider(device);
    {
        auto o = offsets.Read(), c = columns.Read();
        provider.SetPattern(o.View(), c.View());
    }
    auto values = hostValues.To(device), rhs = hostRhs.To(device);
    R::Array<double> solution(R::RowLayout::Dense(n, 1), device);
    auto apply = [&](bool update)
    {
        R::EagerEngine engine;
        auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        if (update)
            R::SubmitArrays(engine, stream, [&provider](auto context, auto a)
                            { provider.UpdateValuesBorrowed(context, a); }, R::Read(values));
        R::SubmitArrays(engine, stream, [&provider](auto context, auto r, auto z)
                        { provider.ApplyBorrowed(context, r, z); }, R::Read(rhs), R::Write(solution));
        engine.Close();
        return solution.To(R::Device::Host());
    };
    const auto first = apply(true);
    const auto firstRead = first.Read(), rRead = hostRhs.Read();
    double initial = 0, final = 0;
    for (I iRow = 0; iRow < n; ++iRow)
    {
        const auto v = firstRead.View();
        const double residual = rRead.View()(iRow, 0) - 4 * v(iRow, 0) +
                                (iRow ? v(iRow - 1, 0) : 0) + (iRow + 1 < n ? v(iRow + 1, 0) : 0);
        initial += std::pow(rRead.View()(iRow, 0), 2);
        final += residual * residual;
    }
    CHECK(final < 0.1 * initial);
    // Repeated zero-start application must not retain the previous correction.
    const auto again = apply(false);
    const auto againRead = again.Read();
    for (I iRow = 0; iRow < n; ++iRow)
        CHECK(againRead.View()(iRow, 0) == doctest::Approx(firstRead.View()(iRow, 0)).epsilon(1e-12));
    {
        auto a = hostValues.Write();
        for (I iRow = 0; iRow < nnz; ++iRow)
            a.View()(iRow, 0) *= 2;
    }
    values.CopyFrom(hostValues, R::ExecutionContext::CUDA(0)).Wait();
    const auto scaled = apply(true);
    const auto scaledRead = scaled.Read();
    for (I iRow = 0; iRow < n; ++iRow)
        CHECK(2 * scaledRead.View()(iRow, 0) == doctest::Approx(firstRead.View()(iRow, 0)).epsilon(1e-10));
    CHECK_THROWS(provider.ApplyBorrowed(R::ExecutionContext::Host(), {}, {}));
}
