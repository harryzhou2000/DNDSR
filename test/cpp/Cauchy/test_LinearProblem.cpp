#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include <Eigen/LU>
#include <optional>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using Field = A::DofField<DNDS::real>;
using Matrix = Eigen::Matrix<double, 5, 5>;
using Vector = Eigen::Matrix<double, 5, 1>;

namespace
{
    Vector Input(int rank, DNDS::index row)
    {
        Vector result;
        for (int j = 0; j < 5; ++j)
            result[j] = .2 + .07 * rank + .03 * row + .04 * j;
        return result;
    }
    Vector RightHandSide(int rank, DNDS::index row)
    {
        Vector result;
        for (int j = 0; j < 5; ++j)
            result[j] = .4 + .1 * rank - .06 * row + .05 * j;
        return result;
    }
    Matrix Diagonal(DNDS::index row, C::DiagonalStorage mode)
    {
        Matrix result = Matrix::Zero();
        for (int j = 0; j < 5; ++j)
        {
            result(j, j) = 3 + .1 * row + .03 * j;
            if (mode == C::DiagonalStorage::Block)
                result(j, (j + 1) % 5) = .02;
        }
        return result;
    }
    Matrix OffDiagonal(DNDS::index row, DNDS::index column)
    {
        Matrix result = Matrix::Identity() * (-.1 - .001 * row);
        result(1, 4) = .003 * (column + 1);
        return result;
    }
    void FillMatrix(C::LinearProblemContext &problem)
    {
        auto diagonal = problem.Diagonal().Write();
        auto off = problem.OffDiagonal().Write();
        const auto d = diagonal.View<R::TableRows>();
        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
        {
            const auto block = Diagonal(iCell, problem.DiagonalMode());
            if (problem.DiagonalMode() == C::DiagonalStorage::Block)
                for (int j = 0; j < 25; ++j)
                    d(iCell, j) = block.data()[j];
            else
                for (int j = 0; j < 5; ++j)
                    d(iCell, j) = block(j, j);
            for (DNDS::index j = 0; j < DNDS::index(problem.Columns()[iCell].size()); ++j)
                off.View()(iCell, j) = OffDiagonal(iCell, problem.Columns()[iCell][j]);
        }
    }
}
TEST_CASE("Native linear action overlaps direct MPI with core and preserves frozen local SGS semantics")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    const int previous = (rank + size - 1) % size, next = (rank + 1) % size;
    std::vector<std::vector<DNDS::index>> send(size), receive(size);
    send[next] = {2, 3};
    receive[previous] = {0, 1};
    const auto route = R::RowTransferPlan::Known(communication, 4, 2, send, receive);
    const A::DofSpace space(communication, 4, 2, "mock coupled flow5");
    C::EquationSchema schema{"mock-flow5", "mock-energy", "identity", {}};
    C::EquationBlockSchema flow;
    flow.name = "flow";
    for (int j = 0; j < 5; ++j)
        flow.components.push_back({"q" + std::to_string(j), "", "flow"});
    schema.blocks.push_back(std::move(flow));
    const std::vector<std::vector<DNDS::index>> columns{{1}, {0, 2}, {1, 3, 4}, {2, 5}};
    for (auto mode : {C::DiagonalStorage::Entries, C::DiagonalStorage::Block})
        for (int layout = 0; layout < 3; ++layout)
        {
            C::LinearProblemContext problem(space, 5, columns, {1, 0}, mode);
            const auto format = layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                         : R::RowLayout::TiledColMajor(0, 5, 32);
            Field input(space, format), full(space, format), split(space, format), b(space, format), forward(space, format), result(space, format);
            {
                auto write = input.Owned().Write(), rhs = b.Owned().Write();
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                    for (int j = 0; j < 5; ++j)
                    {
                        write.View<R::UniformTableRows>()(iCell, j) = Input(rank, iCell)[j];
                        rhs.View<R::UniformTableRows>()(iCell, j) = RightHandSide(rank, iCell)[j];
                    }
            }
            FillMatrix(problem);
            CHECK_THROWS(problem.ApplySGS(b, input, forward, result));
            problem.InvertDiagonal();
            const auto inputVersion = input.Owned().Version(), rhsVersion = b.Owned().Version();
            R::PayloadPlan payload(route, input.Owned().Layout(), input.Ghost().Layout(), R::ElementSchema::Arithmetic<double>());
            const auto binding = payload.Bind(input.Owned(), input.Ghost());
            {
                auto transfer = binding.Begin();
                problem.Apply(input, split, C::LinearRegion::Core);
                // Ghost completion is still unobserved, even when a small
                // message has already arrived. All/dependent must reject it.
                CHECK_THROWS(problem.Apply(input, full));
                transfer.Wait();
                problem.Apply(input, split, C::LinearRegion::HaloDependent);
            }
            problem.Apply(input, full);
            Field logicalResult(space, format);
            auto logicalInput = C::EquationDofs::Coupled(schema, input), logicalOutput = C::EquationDofs::Coupled(schema, logicalResult);
            {
                auto transfer = binding.Begin();
                problem.Apply(logicalInput, logicalOutput, C::LinearRegion::Core);
                transfer.Wait();
                problem.Apply(logicalInput, logicalOutput, C::LinearRegion::HaloDependent);
            }
            const auto a = full.Owned().Read(), c = split.Owned().Read(), logicalRead = logicalResult.Owned().Read();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
            {
                Vector expected = Diagonal(iCell, mode) * Input(rank, iCell);
                for (auto column : columns[iCell])
                    expected += OffDiagonal(iCell, column) * (column < 4 ? Input(rank, column) : Input(previous, column - 2));
                for (int j = 0; j < 5; ++j)
                {
                    CHECK(a.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-13));
                    CHECK(a.View<R::UniformTableRows>()(iCell, j) == c.View<R::UniformTableRows>()(iCell, j));
                    CHECK(logicalRead.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-13));
                }
            }
            problem.ApplySGS(b, input, forward, result);
            std::array<Vector, 4> lower, upper;
            const auto &order = problem.Order();
            std::array<DNDS::index, 4> position{};
            for (DNDS::index j = 0; j < 4; ++j)
                position[order[j]] = j;
            for (auto iCell : order)
            {
                Vector value = RightHandSide(rank, iCell);
                for (auto column : columns[iCell])
                {
                    const Vector other = column >= 4 ? Input(previous, column - 2) : position[column] < position[iCell] ? lower[column]
                                                                                                                        : Input(rank, column);
                    value -= OffDiagonal(iCell, column) * other;
                }
                lower[iCell] = Diagonal(iCell, mode).fullPivLu().solve(value);
            }
            for (DNDS::index pos = 4; pos > 0; --pos)
            {
                const auto iCell = order[pos - 1];
                Vector value = RightHandSide(rank, iCell);
                for (auto column : columns[iCell])
                {
                    const Vector other = column >= 4 ? Input(previous, column - 2) : position[column] < position[iCell] ? lower[column]
                                                                                                                        : upper[column];
                    value -= OffDiagonal(iCell, column) * other;
                }
                upper[iCell] = Diagonal(iCell, mode).fullPivLu().solve(value);
            }
            {
                const auto actual = result.Owned().Read();
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                    for (int j = 0; j < 5; ++j)
                        CHECK(actual.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(upper[iCell][j]).epsilon(1e-13));
            }
            for (bool middleExchange : {false, true})
            {
                Field stagedForward(space, format), stagedOutput(space, format);
                R::PayloadPlan forwardPayload(route, stagedForward.Owned().Layout(), stagedForward.Ghost().Layout(), R::ElementSchema::Arithmetic<double>());
                const auto forwardBinding = forwardPayload.Bind(stagedForward.Owned(), stagedForward.Ghost());
                auto transfer = binding.Begin();
                auto iteration = problem.BeginSGS(b, input, stagedForward, stagedOutput, middleExchange);
                CHECK_THROWS(iteration.Lower(C::LinearRegion::HaloDependent));
                iteration.Lower(C::LinearRegion::Core);
                CHECK_THROWS(stagedForward.Owned().Read());
                CHECK_THROWS(iteration.Upper());
                CHECK_THROWS(iteration.Finish());
                transfer.Wait();
                iteration.Lower(C::LinearRegion::HaloDependent);
                {
                    auto read = stagedForward.Owned().Read();
                    for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                        for (int j = 0; j < 5; ++j)
                            CHECK(read.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(lower[iCell][j]).epsilon(1e-13));
                }
                std::vector<double> globalLower;
                if (middleExchange)
                {
                    CHECK_THROWS(iteration.Upper(C::LinearRegion::HaloDependent));
                    auto exchange = forwardBinding.Begin();
                    CHECK_THROWS(iteration.Upper(C::LinearRegion::HaloDependent));
                    exchange.Wait(); // Deliberately exposed inter-scan join.
                    std::array<double, 20> localLower;
                    for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                        for (int j = 0; j < 5; ++j)
                            localLower[iCell * 5 + j] = lower[iCell][j];
                    globalLower.resize(size * 20);
                    MPI_Allgather(localLower.data(), 20, MPI_DOUBLE, globalLower.data(), 20, MPI_DOUBLE, MPI_COMM_WORLD);
                }
                CHECK_THROWS(iteration.Upper(C::LinearRegion::Core));
                iteration.Upper(C::LinearRegion::HaloDependent);
                CHECK_THROWS(stagedOutput.Owned().Read());
                iteration.Upper(C::LinearRegion::Core);
                CHECK_THROWS(stagedOutput.Owned().Read());
                iteration.Finish();
                CHECK_THROWS(iteration.Finish());
                auto actual = stagedOutput.Owned().Read(), complete = result.Owned().Read();
                std::array<Vector, 4> expectedUpper = upper;
                if (middleExchange)
                    for (DNDS::index pos = 4; pos > 0; --pos)
                    {
                        const auto iCell = order[pos - 1];
                        Vector value = RightHandSide(rank, iCell);
                        for (auto column : columns[iCell])
                        {
                            Vector other;
                            if (column >= 4)
                                for (int j = 0; j < 5; ++j)
                                    other[j] = globalLower[previous * 20 + (column - 2) * 5 + j];
                            else
                                other = position[column] < position[iCell] ? lower[column] : expectedUpper[column];
                            value -= OffDiagonal(iCell, column) * other;
                        }
                        expectedUpper[iCell] = Diagonal(iCell, mode).fullPivLu().solve(value);
                    }
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                    for (int j = 0; j < 5; ++j)
                    {
                        CHECK(actual.View<R::UniformTableRows>()(iCell, j) == doctest::Approx(expectedUpper[iCell][j]).epsilon(1e-13));
                        if (!middleExchange)
                            CHECK(actual.View<R::UniformTableRows>()(iCell, j) == complete.View<R::UniformTableRows>()(iCell, j));
                    }
            }
            Field abandonedForward(space, format), abandonedOutput(space, format);
            {
                auto unused = problem.BeginSGS(b, input, abandonedForward, abandonedOutput);
            }
            CHECK(abandonedForward.Owned().IsValid());
            CHECK(abandonedOutput.Owned().IsValid());
            {
                auto abandoned = problem.BeginSGS(b, input, abandonedForward, abandonedOutput);
                abandoned.Lower(C::LinearRegion::Core);
            }
            CHECK_FALSE(abandonedForward.Owned().IsValid());
            CHECK_FALSE(abandonedOutput.Owned().IsValid());
            CHECK_THROWS(abandonedOutput.Owned().Read());
            CHECK(input.Owned().Version() == inputVersion);
            CHECK(b.Owned().Version() == rhsVersion);
            CHECK_THROWS(problem.Apply(input, input));
            CHECK_THROWS(problem.ApplySGS(b, input, forward, input));
            CHECK_THROWS(problem.ApplySGS(b, input, result, result));
            {
                auto active = problem.Diagonal().Write();
                CHECK_THROWS(problem.ApplySGS(b, input, forward, result));
            }
            CHECK_THROWS(problem.ApplySGS(b, input, forward, result));
            problem.InvertDiagonal();
            problem.ApplySGS(b, input, forward, result);
        }
}
TEST_CASE("Linear matrix construction rejects invalid topology and supports empty owners")
{
    const A::DofSpace space(2, 1);
    C::LinearProblemContext bound(space, 5, {{1}, {0}}, {}, C::DiagonalStorage::Block);
    const auto columns = std::make_shared<const std::vector<std::vector<DNDS::index>>>(std::vector<std::vector<DNDS::index>>{{1}, {0}});
    CHECK_NOTHROW(bound.ValidateColumns(columns));
    CHECK_NOTHROW(bound.ValidateColumns(columns));
    CHECK_THROWS(bound.ValidateColumns(std::make_shared<const std::vector<std::vector<DNDS::index>>>(std::vector<std::vector<DNDS::index>>{{2}, {0}})));
    CHECK_THROWS(C::LinearProblemContext(space, 5, {{1}, {2}}, {1}, C::DiagonalStorage::Block));
    CHECK_THROWS(C::LinearProblemContext(space, 5, {{1, 1}, {0}}, {}, C::DiagonalStorage::Block));
    CHECK_THROWS(C::LinearProblemContext(space, 5, {{0}, {0}}, {}, C::DiagonalStorage::Block));
    CHECK_THROWS(C::LinearProblemContext(space, 5, {{1}, {3}}, {}, C::DiagonalStorage::Block));
    CHECK_THROWS(C::LinearProblemContext(space, 5, {{1}, {0}}, {0, 0}, C::DiagonalStorage::Block));
    const A::DofSpace empty(0, 0);
    C::LinearProblemContext problem(empty, 5, {}, {}, C::DiagonalStorage::Block);
    Field rhs(empty, 5), input(empty, 5), forward(empty, 5), output(empty, 5);
    problem.InvertDiagonal();
    problem.Apply(input, output);
    problem.ApplySGS(rhs, input, forward, output);
    problem.Diagonal() = R::Array<double>(R::RowLayout::Dense(0, 25));
    CHECK_THROWS(problem.Apply(input, output));
}
TEST_CASE("Cauchy native ILU copies only local coefficients and validates ordering fill epochs and local parts")
{
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    const A::DofSpace space(communication, 4, 2);
    const std::vector<std::vector<DNDS::index>> columns{{1}, {0, 2}, {1, 3, 4}, {2, 5}};
    for (auto mode : {C::DiagonalStorage::Entries, C::DiagonalStorage::Block})
        for (bool coreFirst : {false, true})
            for (bool parts : {false, true})
                for (int fill : {0, 1, -1})
                {
                    INFO("diagonal mode=" << int(mode) << ", coreFirst=" << coreFirst << ", parts=" << parts << ", fill=" << fill);
                    C::LinearProblemContext problem(space, 5, columns, {1}, mode);
                    FillMatrix(problem);
                    Field rhs(space, R::RowLayout::ColMajor(0, 5, {8, 1})), forward(space, R::RowLayout::TiledColMajor(0, 5, 32)), result(space, 5);
                    Eigen::VectorXd b(20);
                    {
                        auto write = rhs.Owned().Write();
                        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                            for (int j = 0; j < 5; ++j)
                                write.View<R::UniformTableRows>()(iCell, j) = b[iCell * 5 + j] = RightHandSide(0, iCell)[j];
                    }
                    CHECK_FALSE(problem.ILUFactorsCurrent());
                    CHECK_THROWS(problem.ApplyILU(rhs, forward, result));
                    CHECK_THROWS(problem.PrepareILUStructure(fill, coreFirst, {0, 5}));
                    CHECK_THROWS(problem.PrepareILUStructure(6, coreFirst));
                    problem.PrepareILUStructure(fill, coreFirst, parts ? std::vector<DNDS::index>{0, 2, 4} : std::vector<DNDS::index>{});
                    CHECK_FALSE(problem.ILUFactorsCurrent());
                    if (fill < 0 || !coreFirst)
                        CHECK(problem.ILUForwardCore().empty());
                    else
                        CHECK(problem.ILUForwardCore() == std::vector<DNDS::index>{1});
                    CHECK(problem.ILUBackwardCore().empty());
                    problem.FactorizeILU();
                    CHECK(problem.ILUFactorsCurrent());
                    {
                        auto activeGhost = rhs.Ghost().Write();
                        problem.ApplyILU(rhs, forward, result); // local solve reads no ghost
                    }
                    Eigen::MatrixXd local = Eigen::MatrixXd::Zero(20, 20);
                    Eigen::VectorXd x(20);
                    {
                        const auto read = result.Owned().Read();
                        for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                        {
                            local.block<5, 5>(5 * iCell, 5 * iCell) = Diagonal(iCell, mode);
                            for (auto column : columns[iCell])
                                if (column < 4 && (!parts || iCell / 2 == column / 2))
                                    local.block<5, 5>(5 * iCell, 5 * column) = OffDiagonal(iCell, column);
                            for (int j = 0; j < 5; ++j)
                                x[iCell * 5 + j] = read.View()(iCell, j);
                        }
                    }
                    CHECK((local * x - b).norm() < .01 * b.norm());
                    if (fill != 0 || parts || !coreFirst)
                        CHECK((x - local.fullPivLu().solve(b)).norm() < 1e-12);
                    CHECK_THROWS(problem.ApplyILU(rhs, rhs, result));
                    CHECK_THROWS(problem.ApplyILU(rhs, forward, forward));
                    {
                        auto active = problem.OffDiagonal().Write();
                        CHECK_FALSE(problem.ILUFactorsCurrent());
                        CHECK_THROWS(problem.ApplyILU(rhs, forward, result));
                    }
                    CHECK_FALSE(problem.ILUFactorsCurrent());
                    CHECK_THROWS(problem.ApplyILU(rhs, forward, result));
                    problem.FactorizeILU();
                    CHECK(problem.ILUFactorsCurrent());
                }
    const A::DofSpace empty(communication, 0, 0);
    C::LinearProblemContext zero(empty, 5, {}, {}, C::DiagonalStorage::Block);
    Field rhs(empty, 5), forward(empty, 5), output(empty, 5);
    zero.PrepareILUStructure();
    zero.FactorizeILU();
    CHECK_NOTHROW(zero.ApplyILU(rhs, forward, output));
}
TEST_CASE("RCM factor ordering preserves local parts and solves the same dense system")
{
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    const A::DofSpace space(communication, 4, 0);
    // A scrambled path 0--2--1--3 has one edge spanning each part.
    const std::vector<std::vector<DNDS::index>> columns{{2}, {2, 3}, {0, 1}, {1}};
    for (bool parts : {false, true})
        for (bool coreFirst : {false, true})
        {
            C::LinearProblemContext problem(space, 5, columns, {0, 1, 2, 3}, C::DiagonalStorage::Block);
            FillMatrix(problem);
            Field rhs(space, 5), forward(space, 5), result(space, 5);
            Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(20, 20);
            Eigen::VectorXd b(20);
            {
                auto write = rhs.Owned().Write();
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                {
                    matrix.block<5, 5>(5 * iCell, 5 * iCell) = Diagonal(iCell, C::DiagonalStorage::Block);
                    for (auto other : columns[iCell])
                        if (!parts || iCell / 2 == other / 2)
                            matrix.block<5, 5>(5 * iCell, 5 * other) = OffDiagonal(iCell, other);
                    for (int j = 0; j < 5; ++j)
                        write.View()(iCell, j) = b[5 * iCell + j] = RightHandSide(0, iCell)[j];
                }
            }
            problem.PrepareILUStructure(-1, coreFirst, parts ? std::vector<DNDS::index>{0, 2, 4} : std::vector<DNDS::index>{}, "rcm");
            problem.FactorizeILU();
            problem.ApplyILU(rhs, forward, result);
            const auto expected = matrix.fullPivLu().solve(b).eval();
            const auto read = result.Owned().Read();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(read.View()(iCell, j) == doctest::Approx(expected[5 * iCell + j]).epsilon(1e-12));
            // For incomplete factors all RHS rows are ready. The returned
            // order exposes the factor scan, independently of dense equality.
            problem.PrepareILUStructure(0, coreFirst, {}, "rcm");
            // Peripheral-root search from0 reaches3. CM from3 visits
            // 3,1,2,0; reversing gives0,2,1,3, distinct from natural0,1,2,3.
            CHECK(problem.ILUForwardCore() == std::vector<DNDS::index>{0, 2, 1, 3});
        }
    const A::DofSpace empty(communication, 0, 0);
    C::LinearProblemContext zero(empty, 5, {}, {}, C::DiagonalStorage::Block);
    CHECK_NOTHROW(zero.PrepareILUStructure(0, true, {}, "rcm"));
    CHECK_THROWS(zero.PrepareILUStructure(0, true, {}, "unknown"));
}
TEST_CASE("Linear diagonal preparation handles general component counts and reports guarded inverse fallback")
{
    const A::DofSpace space(1);
    C::LinearProblemContext problem(space, 2, {{}}, {0}, C::DiagonalStorage::Block);
    Field rhs(space, 2), input(space, 2), forward(space, 2), output(space, 2);
    {
        auto values = problem.Diagonal().Write(), b = rhs.Owned().Write(), old = input.Owned().Write();
        const std::array<double, 4> block{2, .1, .4, 3};
        for (int j = 0; j < 4; ++j)
            values.View()(0, j) = block[j];
        b.View()(0, 0) = 1;
        b.View()(0, 1) = 2;
        old.View()(0, 0) = 7;
        old.View()(0, 1) = 8;
    }
    problem.InvertDiagonal();
    CHECK(problem.LastInverseFallbacks() == 0);
    problem.Apply(input, output);
    {
        const auto read = output.Owned().Read();
        CHECK(read.View()(0, 0) == doctest::Approx(17.2));
        CHECK(read.View()(0, 1) == doctest::Approx(24.7));
    }
    problem.ApplySGS(rhs, input, forward, output);
    {
        const auto read = output.Owned().Read();
        CHECK(2 * read.View()(0, 0) + .4 * read.View()(0, 1) == doctest::Approx(1));
        CHECK(.1 * read.View()(0, 0) + 3 * read.View()(0, 1) == doctest::Approx(2));
    }
    {
        auto write = problem.Diagonal().Write();
        for (int j = 0; j < 4; ++j)
            write.View()(0, j) = 1;
        write.View()(0, 3) += 1e-9;
    }
    problem.InvertDiagonal();
    CHECK(problem.LastInverseFallbacks() == 1);
    {
        auto write = problem.Diagonal().Write();
        write.View()(0, 0) = 0;
    }
    CHECK_THROWS(problem.InvertDiagonal());
    CHECK_THROWS(problem.ApplySGS(rhs, input, forward, output));
}
TEST_CASE("Staged SGS retains numeric allocations and row metadata after original owners expire")
{
    const A::DofSpace space(1);
    Field output(space, 2);
    std::optional<C::LinearProblemContext::SGSIteration> iteration;
    {
        auto problem = std::make_unique<C::LinearProblemContext>(space, 2, std::vector<std::vector<DNDS::index>>{{}}, std::vector<DNDS::index>{0}, C::DiagonalStorage::Entries);
        Field rhs(space, 2), input(space, 2), forward(space, 2);
        {
            auto diagonal = problem->Diagonal().Write(), b = rhs.Owned().Write();
            diagonal.View()(0, 0) = diagonal.View()(0, 1) = 2;
            b.View()(0, 0) = 3;
            b.View()(0, 1) = -1;
        }
        problem->InvertDiagonal();
        iteration.emplace(problem->BeginSGS(rhs, input, forward, output));
    }
    iteration->Lower();
    iteration->Upper();
    iteration->Finish();
    const auto read = output.Owned().Read();
    CHECK(read.View()(0, 0) == 1.5);
    CHECK(read.View()(0, 1) == -.5);
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    MPI_Finalize();
    return result;
}
