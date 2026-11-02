#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include <Eigen/LU>
#include <optional>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using Field = A::DofField<DNDS::real>;
using Matrix = Eigen::Matrix<double, 5, 5>;

namespace
{
    const std::vector<std::vector<DNDS::index>> columns{{1}, {0, 2}, {1, 3, 4}, {2, 5}};
    Matrix Diagonal(DNDS::index iCell)
    {
        Matrix value = Matrix::Identity() * (3 + .1 * iCell);
        for (int j = 0; j < 5; ++j)
            value(j, (j + 1) % 5) = .02;
        return value;
    }
    Matrix Off(DNDS::index iCell, DNDS::index other)
    {
        Matrix value = Matrix::Identity() * (-.1 - .001 * iCell);
        value(1, 4) = .003 * (other + 1);
        return value;
    }
    double RHS(int rank, DNDS::index iCell, int j) { return .4 + .1 * rank - .06 * iCell + .05 * j; }
    struct Fixture
    {
        DNDS::MPIInfo mpi{MPI_COMM_WORLD};
        int rank = mpi.rank, size = mpi.size;
        std::shared_ptr<R::CommunicationContext> communication;
        A::DofSpace space;
        R::RowTransferPlan route;
        std::shared_ptr<C::LinearProblemContext> problem;
        Field rhs;
        static R::RowTransferPlan Route(const std::shared_ptr<R::CommunicationContext> &communication, int rank, int size)
        {
            std::vector<std::vector<DNDS::index>> send(size), receive(size);
            send[(rank + 1) % size] = {2, 3};
            receive[(rank + size - 1) % size] = {0, 1};
            return R::RowTransferPlan::Known(communication, 4, 2, send, receive);
        }
        Fixture(int layout = 0)
            : communication(std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD)),
              space(communication, 4, 2, "coupled linear solver oracle"), route(Route(communication, rank, size)),
              problem(std::make_shared<C::LinearProblemContext>(space, 5, columns, std::vector<DNDS::index>{1, 0}, C::DiagonalStorage::Block)),
              rhs(space, layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                  : R::RowLayout::TiledColMajor(0, 5, 32))
        {
            auto diagonal = problem->Diagonal().Write();
            auto off = problem->OffDiagonal().Write();
            auto b = rhs.Owned().Write();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
            {
                const auto block = Diagonal(iCell);
                for (int j = 0; j < 25; ++j)
                    diagonal.View()(iCell, j) = block.data()[j];
                for (DNDS::index k = 0; k < DNDS::index(columns[iCell].size()); ++k)
                    off.View()(iCell, k) = Off(iCell, columns[iCell][k]);
                for (int j = 0; j < 5; ++j)
                    b.View<R::UniformTableRows>()(iCell, j) = RHS(rank, iCell, j);
            }
        }
        R::TransferBinding Bind(Field &field) const
        {
            return R::PayloadPlan(route, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<double>()).Bind(field.Owned(), field.Ghost());
        }
        Eigen::VectorXd Oracle() const
        {
            // Independent, tiny global dense solve. Ghost 4/5 denote the
            // preceding rank's owned rows 2/3, even for the self-ring case.
            Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(20 * size, 20 * size);
            Eigen::VectorXd b(20 * size);
            for (int owner = 0; owner < size; ++owner)
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                {
                    const DNDS::index row = 20 * owner + 5 * iCell;
                    matrix.block<5, 5>(row, row) += Diagonal(iCell);
                    for (auto other : columns[iCell])
                    {
                        const int source = other < 4 ? owner : (owner + size - 1) % size;
                        const DNDS::index local = other < 4 ? other : other - 2;
                        matrix.block<5, 5>(row, 20 * source + 5 * local) += Off(iCell, other);
                    }
                    for (int j = 0; j < 5; ++j)
                        b[row + j] = RHS(owner, iCell, j);
                }
            return matrix.fullPivLu().solve(b);
        }
    };
}
TEST_CASE("Cauchy coupled linear solvers match a global dense oracle with real persistent halos")
{
    for (int scenario = 0; scenario < 7; ++scenario)
    {
        Fixture fixture(scenario % 3);
        C::LinearSolveSettings settings;
        settings.method = scenario < 3 ? "sgs" : "gmres";
        settings.preconditioner = scenario >= 5 ? "ilu" : "sgs";
        settings.exchangeBetweenSweeps = scenario == 2 || scenario == 4;
        settings.sgsSweeps = 2;
        settings.subspace = 8;
        settings.maxIterations = 80;
        settings.relativeTolerance = 1e-11;
        settings.absoluteTolerance = 1e-13;
        const C::LinearExecution execution{scenario % 2 == 0, scenario % 3 != 1};
        C::LinearSolver solver(fixture.problem, fixture.rhs, [&](Field &field)
                               { return fixture.Bind(field); }, settings, execution);
        const auto version = fixture.rhs.Owned().Version();
        const auto report = solver.Solve(fixture.rhs);
        CHECK(report.converged);
        CHECK(report.usable);
        CHECK(report.iterations > 0);
        CHECK(report.trueResidual <= std::max(settings.absoluteTolerance, settings.relativeTolerance * report.initialTrueResidual));
        CHECK(fixture.rhs.Owned().Version() == version);
        const auto expected = fixture.Oracle();
        std::array<double, 20> first{};
        {
            const auto read = solver.Solution().Owned().Read();
            const auto values = read.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    first[5 * iCell + j] = values(iCell, j);
                    CHECK(values(iCell, j) == doctest::Approx(expected[fixture.rank * 20 + 5 * iCell + j]).epsilon(1e-10));
                }
        }
        // A new fixed preconditioner application starts from zero; no stale
        // previous solve may change either the correction or iteration count.
        const auto repeated = solver.Solve(fixture.rhs);
        CHECK(repeated.converged);
        CHECK(repeated.iterations == report.iterations);
        {
            const auto read = solver.Solution().Owned().Read();
            const auto values = read.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(values(iCell, j) == first[5 * iCell + j]);
        }
        const auto warm = solver.Solve(fixture.rhs, &solver.Solution());
        CHECK(warm.converged);
        CHECK(warm.iterations == 0);
        const auto solutionVersion = solver.Solution().Owned().Version();
        CHECK_THROWS(solver.Solve(solver.Solution()));
        CHECK(solver.Solution().Owned().Version() == solutionVersion);
        CHECK(solver.Solve(fixture.rhs).converged); // Preflight rejection is reusable.
    }
}
TEST_CASE("Cauchy linear controls and rank-local access failures reject collectively before transfers")
{
    Fixture fixture;
    C::LinearSolveSettings settings;
    auto bind = [&](Field &field)
    { return fixture.Bind(field); };
    settings.method = "invalid";
    CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    settings.method = "gmres";
    settings.preconditioner = "ilu";
    settings.exchangeBetweenSweeps = true;
    CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    if (fixture.size > 1)
    {
        settings.exchangeBetweenSweeps = false;
        settings.subspace = 4 + fixture.rank;
        CHECK_THROWS(C::LinearSolver(fixture.problem, fixture.rhs, bind, settings));
    }
    settings.subspace = 8;
    settings.exchangeBetweenSweeps = false;
    settings.relativeTolerance = 1e-10;
    settings.maxIterations = 40;
    C::LinearSolver solver(fixture.problem, fixture.rhs, bind, settings);
    std::optional<R::Array<double>::WriteLease> writer;
    if (fixture.rank == 0)
        writer.emplace(fixture.rhs.Owned().Write());
    CHECK_THROWS(solver.Solve(fixture.rhs));
    writer.reset();
    if (fixture.rank == 0)
        writer.emplace(fixture.problem->Diagonal().Write());
    CHECK_THROWS(solver.Solve(fixture.rhs));
    writer.reset();
    CHECK(solver.Solve(fixture.rhs).converged);
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
