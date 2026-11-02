#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Solver/FieldGMRES.hpp"
#include <Eigen/LU>

namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
namespace L = DNDS::Linear;
using Field = A::DofField<DNDS::real>;

static std::vector<double> Products(const Field &x, const std::vector<const Field *> &probes)
{
    const auto source = x.Owned().Read();
    const auto values = source.View<R::UniformTableRows>();
    std::vector<R::Array<double>::ReadLease> retained;
    std::vector<R::ArrayView<const double, R::AccessBoundsChecks, R::UniformTableRows>> inputs;
    for (const auto *probe : probes)
    {
        retained.push_back(probe->Owned().Read());
        inputs.push_back(retained.back().View<R::UniformTableRows>());
    }
    std::vector<double> local(probes.size(), 0), global(probes.size(), 0);
    for (DNDS::index iRow = 0; iRow < values.RowCount(); ++iRow)
        for (DNDS::index j = 0; j < values.RowSize(iRow); ++j)
            for (DNDS::index k = 0; k < DNDS::index(probes.size()); ++k)
                local[k] += values(iRow, j) * inputs[k](iRow, j);
    MPI_Allreduce(local.data(), global.data(), int(probes.size()), MPI_DOUBLE, MPI_SUM, x.Space().Communicator());
    return global;
}
static void ApplyDense(Field &input, Field &output, const Eigen::Matrix4d &matrix)
{
    const auto read = input.Owned().Read();
    auto write = output.Owned().Write();
    const auto x = read.View<R::UniformTableRows>();
    const auto y = write.View<R::UniformTableRows>();
    for (DNDS::index iRow = 0; iRow < y.RowCount(); ++iRow)
        for (DNDS::index j = 0; j < 2; ++j)
        {
            double value = 0;
            for (DNDS::index other = 0; other < x.RowCount(); ++other)
                for (DNDS::index k = 0; k < 2; ++k)
                    value += matrix(iRow * 2 + j, other * 2 + k) * x(other, k);
            y(iRow, j) = value;
        }
}
TEST_CASE("Field GMRES handles first-vector closure true residuals and extreme fixed preconditioner scaling")
{
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    const A::DofSpace space(communication, 2);
    Field rhs(space, 2);
    {
        auto write = rhs.Owned().Write();
        for (DNDS::index iRow = 0; iRow < 2; ++iRow)
            for (int j = 0; j < 2; ++j)
                write.View()(iRow, j) = 1 + iRow + .2 * j;
    }
    for (double scale : {1., 1e-12, 1e12})
    {
        L::FieldGMRES solver(rhs, 3);
        const auto pc = [scale](const Field &input, Field &output)
        { A::Copy(output, input); A::Scale(output, scale); };
        const auto op = [](Field &input, Field &output)
        { ApplyDense(input, output, Eigen::Matrix4d::Identity()); };
        const auto report = solver.Solve(rhs, op, pc, Products, {8, 1e-12, 1e-12});
        CHECK(report.converged);
        CHECK(report.usable);
        CHECK(report.iterations == 1);
        CHECK(report.trueResidual < 1e-11);
        {
            const auto exact = rhs.Owned().Read(), actual = solver.Solution().Owned().Read();
            for (DNDS::index iRow = 0; iRow < 2; ++iRow)
                for (int j = 0; j < 2; ++j)
                    CHECK(actual.View()(iRow, j) == doctest::Approx(exact.View()(iRow, j)).epsilon(1e-13));
        }
        const auto warm = solver.Solve(rhs, op, pc, Products, {8, 1e-12, 1e-12}, &solver.Solution());
        CHECK(warm.converged);
        CHECK(warm.iterations == 0);
        CHECK(std::isnan(warm.preconditionedResidual)); // no PC evaluation was needed
        const auto version = solver.Solution().Owned().Version();
        CHECK_THROWS(solver.Solve(solver.Solution(), op, pc, Products));
        CHECK(solver.Solution().Owned().Version() == version);
    }
}
TEST_CASE("Field GMRES restarts nonnormal systems across layouts with empty local ranks")
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    const A::DofSpace space(communication, rank == 0 ? 2 : 0);
    Eigen::Matrix4d matrix;
    matrix << 4, 1, .2, 0, -1, 3, 1, .1, .3, .1, 2, .4, .2, 0, -.3, 2.5;
    const Eigen::Vector4d b{1, 2, -.5, .3}, exact = matrix.fullPivLu().solve(b);
    for (const auto &format : {R::RowLayout::Dense(space.OwnedRows(), 2, 5), R::RowLayout::ColMajor(space.OwnedRows(), 2, {8, 1}), R::RowLayout::TiledColMajor(space.OwnedRows(), 2, 32)})
    {
        Field rhs(space, format), guess(space, 2);
        {
            auto write = rhs.Owned().Write(), seed = guess.Owned().Write();
            for (DNDS::index iRow = 0; iRow < DNDS::index(space.OwnedRows()); ++iRow)
                for (int j = 0; j < 2; ++j)
                {
                    write.View<R::UniformTableRows>()(iRow, j) = b[2 * iRow + j];
                    seed.View()(iRow, j) = .05;
                }
        }
        L::FieldGMRES solver(rhs, 2);
        const auto pc = [](const Field &input, Field &output)
        { A::Copy(output, input); };
        const auto op = [&](Field &input, Field &output)
        { ApplyDense(input, output, matrix); };
        const auto report = solver.Solve(rhs, op, pc, Products, {120, 1e-12, 1e-13}, &guess);
        CHECK(report.converged);
        CHECK(report.usable);
        CHECK(report.restarts > 0);
        CHECK(report.trueResidual < 3e-12);
        const auto read = solver.Solution().Owned().Read();
        for (DNDS::index iRow = 0; iRow < DNDS::index(space.OwnedRows()); ++iRow)
            for (int j = 0; j < 2; ++j)
                CHECK(read.View<R::UniformTableRows>()(iRow, j) == doctest::Approx(exact[2 * iRow + j]).epsilon(2e-11));
    }
}
TEST_CASE("Field GMRES reports algebraic failure without false convergence")
{
    auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    const A::DofSpace space(communication, 2);
    Field rhs(space, 2);
    A::Fill(rhs, 1.);
    L::FieldGMRES solver(rhs, 3);
    const auto identity = [](Field &input, Field &output)
    { A::Copy(output, input); };
    const auto pc = [](const Field &input, Field &output)
    { A::Copy(output, input); };
    const auto zeroPC = [](const Field &, Field &output)
    { A::Fill(output, 0.); };
    auto report = solver.Solve(rhs, identity, zeroPC, Products);
    CHECK(report.stop == L::LinearStop::PreconditionerBreakdown);
    CHECK_FALSE(report.converged);
    CHECK_FALSE(report.usable);
    const auto zero = [](Field &, Field &output)
    { A::Fill(output, 0.); };
    report = solver.Solve(rhs, zero, pc, Products);
    CHECK(report.stop == L::LinearStop::ArnoldiBreakdown);
    CHECK_FALSE(report.usable);
    const auto nonfinite = [](Field &, Field &output)
    { A::Fill(output, std::numeric_limits<double>::quiet_NaN()); };
    report = solver.Solve(rhs, nonfinite, pc, Products);
    CHECK(report.stop == L::LinearStop::NonFinite);
    CHECK_FALSE(report.usable);
    report = solver.Solve(rhs, identity, pc, Products, {0, 1e-12, 1e-12});
    CHECK(report.stop == L::LinearStop::IterationLimit);
    CHECK(report.usable);
    CHECK_FALSE(report.converged);
    A::Fill(rhs, 0.);
    report = solver.Solve(rhs, identity, pc, Products);
    CHECK(report.converged);
    CHECK(report.iterations == 0);
    CHECK(report.trueResidual == 0);
    Field badGuess(space, 2);
    A::Fill(badGuess, std::numeric_limits<double>::quiet_NaN());
    report = solver.Solve(rhs, zero, pc, Products, {}, &badGuess);
    CHECK(report.stop == L::LinearStop::NonFinite);
    CHECK_FALSE(report.usable);
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
