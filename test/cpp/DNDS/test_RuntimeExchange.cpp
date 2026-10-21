#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/Exchange.hpp"
#include <optional>

using namespace DNDS::Runtime;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int result = context.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Exchange routes arbitrary rows and transpose sums, retaining buffers")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto roots = Buffer::Allocate(2 * sizeof(double));
    {
        auto write = roots.Write();
        write.View<double>()[0] = 100 * rank + 1;
        write.View<double>()[1] = 100 * rank + 2;
    }
    auto leaves = Buffer::Allocate(3 * sizeof(double));
    ExchangePlan plan(MPI_COMM_WORLD, 2, {{(rank + 1) % size, 1}, {rank, 0}, {(rank + 1) % size, 1}});
    auto op = plan.BeginCopy(roots, leaves, sizeof(double));
    CHECK_THROWS(roots.Write());
    CHECK_THROWS(leaves.Read());
    auto moved = std::move(op);
    moved.Wait();
    {
        auto read = leaves.Read();
        CHECK(read.View<double>()[0] == 100 * ((rank + 1) % size) + 2);
        CHECK(read.View<double>()[1] == 100 * rank + 1);
        CHECK(read.View<double>()[2] == read.View<double>()[0]);
    }
    {
        auto write = leaves.Write();
        for (size_t i = 0; i < 3; ++i)
            write.View<double>()[i] = 1;
    }
    auto sums = Buffer::Allocate(2 * sizeof(double));
    plan.BeginAdd(leaves, sums, 1).Wait();
    CHECK(sums.Read().View<double>()[0] == 1);
    CHECK(sums.Read().View<double>()[1] == 2);
    {
        auto retained = plan.BeginCopy(roots, leaves, sizeof(double));
        roots = {};
        retained.Wait();
    }
}

TEST_CASE("Exchange collective validation and ranks with no owner rows")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<RemoteRow> invalid;
    if (rank == 0)
        invalid.push_back({size, 0});
    CHECK_THROWS(ExchangePlan(MPI_COMM_WORLD, 0, invalid));
    invalid = {{0, 0}};
    CHECK_THROWS(ExchangePlan(MPI_COMM_WORLD, 0, invalid));
    auto roots = Buffer::Allocate(rank == 0 ? sizeof(double) : 0);
    if (rank == 0)
        roots.Write().View<double>()[0] = 19;
    auto leaves = Buffer::Allocate(sizeof(double));
    VersionTracker versions;
    ExchangePlan plan(MPI_COMM_WORLD, rank == 0 ? 1 : 0, {{0, 0}}, {versions.Stamp(Change::Layout)});
    {
        std::optional<ReadAccess> read;
        if (rank == 0)
            read.emplace(leaves.Read());
        CHECK_THROWS(plan.BeginCopy(roots, leaves, sizeof(double)));
    }
    if (size > 1)
    {
        CHECK_THROWS(plan.BeginCopy(roots, leaves, rank == 0 ? 1 : sizeof(double)));
        if (rank == 0)
            CHECK_THROWS(plan.BeginAdd(leaves, roots, 1));
        else
            CHECK_THROWS(plan.BeginCopy(roots, leaves, sizeof(double)));
    }
    plan.BeginCopy(roots, leaves, sizeof(double)).Wait();
    CHECK(leaves.Read().View<double>()[0] == 19);
    {
        auto change = versions.BeginMutation(Change::Layout);
        change.Commit();
    }
    CHECK_THROWS(plan.BeginCopy(roots, leaves, sizeof(double)));
    ExchangePlan empty(MPI_COMM_WORLD, 0, {});
    auto a = Buffer::Allocate(0), b = Buffer::Allocate(0);
    empty.BeginCopy(a, b, sizeof(double)).Wait();
    CHECK(a.IsIdle());
    CHECK(b.IsIdle());
}
