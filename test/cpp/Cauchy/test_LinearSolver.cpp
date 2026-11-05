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
    struct EvaluatedOff final : C::OffDiagonalEvaluator
    {
        std::shared_ptr<const A::DofSpace::Partition> partition;
        std::vector<std::vector<DNDS::index>> declaredColumns = columns;
        DNDS::index components = 5;
        explicit EvaluatedOff(const A::DofSpace &space) : partition(space.Pin()) {}
        DNDS::index Components() const override { return components; }
        auto RowSpaceIdentity() const -> std::shared_ptr<const A::DofSpace::Partition> override { return partition; }
        const std::vector<std::vector<DNDS::index>> &Columns() const override { return declaredColumns; }
        void EvaluateBlock(DNDS::index iRow, DNDS::index iBlock, Eigen::Ref<Eigen::MatrixXd> output) const override
        {
            output = Off(iRow, columns[iRow][iBlock]);
        }
    };
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
        Fixture(int layout = 0, bool evaluated = false)
            : communication(std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD)),
              space(communication, 4, 2, "coupled linear solver oracle"), route(Route(communication, rank, size)),
              problem(std::make_shared<C::LinearProblemContext>(space, 5, columns, std::vector<DNDS::index>{1, 0}, C::DiagonalStorage::Block,
                                                                evaluated ? C::OffDiagonalStorage::SemiMatrixFree : C::OffDiagonalStorage::Assembled)),
              rhs(space, layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                  : R::RowLayout::TiledColMajor(0, 5, 32))
        {
            {
                auto diagonal = problem->Diagonal().Write();
                std::optional<A::BlockMatrixStorage::WriteAccess> off;
                if (!evaluated)
                    off.emplace(problem->OffDiagonal().Write());
                auto b = rhs.Owned().Write();
                for (DNDS::index iCell = 0; iCell < 4; ++iCell)
                {
                    const auto block = Diagonal(iCell);
                    for (int j = 0; j < 25; ++j)
                        diagonal.View()(iCell, j) = block.data()[j];
                    for (DNDS::index k = 0; k < DNDS::index(columns[iCell].size()); ++k)
                        if (off)
                            off->View()(iCell, k) = Off(iCell, columns[iCell][k]);
                    for (int j = 0; j < 5; ++j)
                        b.View<R::UniformTableRows>()(iCell, j) = RHS(rank, iCell, j);
                }
            }
            if (evaluated)
                problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(space));
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
    for (int selection = 0; selection < 16; ++selection)
    {
        const int scenario = selection % 8;
        Fixture fixture(scenario % 3, selection >= 8);
        C::LinearSolveSettings settings;
        settings.method = scenario < 3 ? "sgs" : scenario == 7 ? "ilu"
                                                               : "gmres";
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
TEST_CASE("Evaluated block publication validates row identity, column ordinals and active read scopes")
{
    Fixture fixture(1, true);
    const auto generation = fixture.problem->OffDiagonalGeneration();
    CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(nullptr));
    {
        A::DofSpace other(4, 2);
        CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(other)));
    }
    for (int error = 0; error < 2; ++error)
    {
        auto invalid = std::make_shared<EvaluatedOff>(fixture.space);
        if (error == 0)
            invalid->components = 4;
        else
            std::swap(invalid->declaredColumns[2][0], invalid->declaredColumns[2][1]);
        CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(invalid));
    }
    {
        auto held = fixture.problem->ReadOffDiagonal();
        CHECK_THROWS(fixture.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(fixture.space)));
    }
    CHECK(fixture.problem->OffDiagonalGeneration() == generation);
    fixture.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(fixture.space));
    CHECK(fixture.problem->OffDiagonalGeneration() == generation + 1);
    {
        Fixture assembled;
        CHECK_THROWS(assembled.problem->PublishOffDiagonalEvaluator(std::make_shared<EvaluatedOff>(assembled.space)));
    }
    if (fixture.size > 1)
    {
        Fixture mixed(0, fixture.rank % 2);
        CHECK_THROWS(C::LinearSolver(mixed.problem, mixed.rhs, [&](Field &field)
                                     { return mixed.Bind(field); }));
    }
    std::optional<C::LinearProblemContext::OffDiagonalReadAccess> retained;
    {
        Fixture temporary(2, true);
        retained.emplace(temporary.problem->ReadOffDiagonal());
        temporary.problem.reset();
    }
    // Numeric access retains no communicator owners. Its frozen row descriptor
    // and producer survive the complete original context/fixture lifetime.
    retained->WithBlocks([&](const auto &view)
                         {
    const auto first = view(0, 0), second = view(1, 0);
    CHECK(first.isApprox(Off(0, 1), 1e-15));
    CHECK(second.isApprox(Off(1, 0), 1e-15));
    for (DNDS::index iCell = 0; iCell < 4; ++iCell)
        for (DNDS::index j = 0; j < DNDS::index(columns[iCell].size()); ++j)
            CHECK(view(iCell, j).isApprox(Off(iCell, columns[iCell][j]), 1e-15)); });
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
