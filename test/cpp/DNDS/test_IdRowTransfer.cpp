#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/Communication/IdRowTransfer.hpp"

namespace R = DNDS::Runtime;
using I = DNDS::index;
namespace
{
    constexpr I baseId = 1000000000000LL;
    R::Array<I> Ids(const std::vector<I> &values)
    {
        R::Array<I> result(R::RowLayout::Dense(I(values.size()), 1));
        auto write = result.Write();
        for (I iRow = 0; iRow < I(values.size()); ++iRow)
            write.View()(iRow, 0) = values[iRow];
        return result;
    }
}
TEST_CASE("Exact persistent-ID lookup transfers native rows after arbitrary rank distribution and local permutation")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto context = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    for (int distribution = 0; distribution < 2; ++distribution)
    {
        std::vector<I> oldIds, newIds;
        for (I iRow = 0; iRow < 3; ++iRow)
            oldIds.push_back(baseId + 7 * (3 * mpi.rank + iRow));
        // Small independent test enumeration. Production lookup only processes
        // its local IDs and directory messages, without global ID-sized arrays.
        for (I global = 3 * mpi.size; global > 0; --global)
            if ((distribution == 0 ? (int((global - 1) * 5 + 1) % mpi.size) : 0) == mpi.rank)
                newIds.push_back(baseId + 7 * (global - 1));
        const auto sourceIds = Ids(oldIds), targetIds = Ids(newIds);
        const auto sourceGeneration = sourceIds.Version(), targetGeneration = targetIds.Version();
        auto route = R::BuildRowTransferById(context, sourceIds, targetIds, true);
        CHECK(route.SourceRows() == 3);
        CHECK(route.DestinationRows() == I(newIds.size()));
        CHECK_FALSE(route.HasCollisions());
        CHECK_FALSE(route.Reverse().HasCollisions());
        R::Array<double> source(R::RowLayout::TiledColMajor(3, 3, 8));
        R::Array<double> destination(R::RowLayout::ColMajor(I(newIds.size()), 3, {8, 1}));
        {
            auto write = source.Write();
            const auto values = write.View<R::UniformTableRows>();
            for (I iRow = 0; iRow < 3; ++iRow)
                for (int j = 0; j < 3; ++j)
                    values(iRow, j) = double(oldIds[iRow]) + .25 * j;
        }
        R::PayloadPlan(route, source.Layout(), destination.Layout(), R::ElementSchema::Arithmetic<double>())
            .Bind(source, destination, R::TransferTransport::Direct, {R::TransferMode::Transient, context, 0})
            .Begin()
            .Wait();
        const auto read = destination.Read();
        const auto values = read.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < I(newIds.size()); ++iRow)
            for (int j = 0; j < 3; ++j)
                CHECK(values(iRow, j) == double(newIds[iRow]) + .25 * j);
        CHECK(sourceIds.Version() == sourceGeneration);
        CHECK(targetIds.Version() == targetGeneration);
    }
}
TEST_CASE("ID row transfer rejects duplicate sources missing targets and nonbijective coverage")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto context = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    auto source = Ids({baseId + mpi.rank}), target = Ids({baseId + mpi.rank});
    auto missing = Ids({baseId + mpi.size + mpi.rank});
    CHECK_THROWS(R::BuildRowTransferById(context, source, missing));
    auto duplicated = Ids({baseId + mpi.rank, baseId + mpi.rank});
    CHECK_THROWS(R::BuildRowTransferById(context, duplicated, target));
    auto gather = R::BuildRowTransferById(context, source, duplicated);
    CHECK_FALSE(gather.HasCollisions());
    CHECK(gather.Reverse().HasCollisions());
    CHECK_THROWS(R::BuildRowTransferById(context, source, duplicated, true));
    if (mpi.size > 1)
    {
        auto crossRankDuplicate = Ids({baseId});
        CHECK_THROWS(R::BuildRowTransferById(context, crossRankDuplicate, target));
        CHECK_THROWS(R::BuildRowTransferById(context, source, target, mpi.rank == 0));
    }
    auto empty = Ids({});
    CHECK(R::BuildRowTransferById(context, empty, empty, true).DestinationRows() == 0);
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
