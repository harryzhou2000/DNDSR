#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Access/EntityWorkPartition.hpp"
#include "NativeMeshOverlapFixture.hpp"

using namespace DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Node-star work sets match an independent Cartesian ownership oracle")
{
    for (bool emptyPeer : {false, true})
    {
        auto fixture = OverlapTest::BuildNativeFixture(mpi, emptyPeer);
        auto &mesh = *fixture.mesh;
        auto partition = EntityWorkPartition::CellNodeCell(mesh);
        auto traversed = EntityWorkPartition::Build(mesh, EntityKind::Cell, {{Adj::Cell2Node, Adj::Node2Cell}});
        CHECK(partition.Core() == traversed.Core());
        CHECK(partition.HaloDependent() == traversed.HaloDependent());
        const int active = emptyPeer && mpi.size > 1 ? mpi.size - 1 : mpi.size;
        const auto rows = mpi.rank < active ? DNDS::EvenSplitRange(mpi.rank, active, I(10)) : std::pair<I, I>{10, 10};
        std::vector<I> expectedCore, expectedDependent;
        const auto nCell = partition.Outputs().NumOwned();
        for (I iCell = 0; iCell < nCell; ++iCell)
        {
            const I original = mesh.OriginalEntityId(EntityKind::Cell, iCell), y = original / 10;
            // Node-star neighbors occupy the immediately adjacent grid rows.
            // At a physical boundary the absent row contributes no dependency.
            const bool core = (y == 0 || y - 1 >= rows.first) && (y == 9 || y + 1 < rows.second);
            (core ? expectedCore : expectedDependent).push_back(iCell);
        }
        CHECK(partition.Core() == expectedCore);
        CHECK(partition.HaloDependent() == expectedDependent);
        CHECK(partition.Core().size() + partition.HaloDependent().size() == size_t(nCell));

        auto nodeInput = EntityWorkPartition::Build(mesh, EntityKind::Cell, {{Adj::Cell2Node}});
        const auto nodeRows = mpi.rank < active ? DNDS::EvenSplitRange(mpi.rank, active, I(121)) : std::pair<I, I>{121, 121};
        expectedCore.clear();
        for (I iCell = 0; iCell < nCell; ++iCell)
        {
            const I original = mesh.OriginalEntityId(EntityKind::Cell, iCell), x = original % 10, y = original / 10;
            const I first = 11 * y + x, last = first + 12;
            if (first >= nodeRows.first && last < nodeRows.second)
                expectedCore.push_back(iCell);
        }
        CHECK(nodeInput.Core() == expectedCore);
        CHECK_THROWS(EntityWorkPartition::Build(mesh, EntityKind::Cell, {{Adj::Node2Cell}}));
        CHECK_THROWS(EntityWorkPartition::Build(mesh, EntityKind::Cell, {{}}));
        CHECK_NOTHROW(partition.Validate());
        // Pure geometry generations do not change this topological partition.
        {
            auto edit = mesh.versions.BeginMutation(DNDS::Runtime::Change::Geometry);
            edit.Commit();
        }
        CHECK_NOTHROW(partition.Validate());
        {
            auto edit = mesh.versions.BeginMutation(DNDS::Runtime::Change::Layout);
            edit.Commit();
        }
        CHECK_THROWS(partition.Validate());
    }
}

TEST_CASE("Work classification rejects missing deeper closure without fetching rows")
{
    auto fixture = OverlapTest::BuildNativeFixture(mpi);
    std::vector<AdjacencyDescriptor> path(20, Adj::Cell2Cell);
    if (mpi.size > 1)
        CHECK_THROWS(EntityWorkPartition::Build(*fixture.mesh, EntityKind::Cell, {path}));
    else
        CHECK_NOTHROW(EntityWorkPartition::Build(*fixture.mesh, EntityKind::Cell, {path}));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
