#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeMeshOverlapFixture.hpp"
#include "NativePeriodicFixtures.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Access/EntityField.hpp"
#include "Geom/IO/NativeMeshOutput.hpp"
#include "Geom/IO/NativeMeshSerialOutput.hpp"
#include "Geom/Refinement/NativeMeshExtraction.hpp"
#include "Geom/Transition/LocalCellOrdering.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "NativeReorderChecks.hpp"
#include <atomic>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
static std::atomic<bool> counting{false};
static std::atomic<int> reductions{0};
extern "C" int MPI_Allreduce(const void *send, void *receive, int count, MPI_Datatype type, MPI_Op operation, MPI_Comm comm)
{
    if (counting)
        ++reductions;
    return PMPI_Allreduce(send, receive, count, type, operation, comm);
}
namespace
{
    struct Count
    {
        Count()
        {
            reductions = 0;
            counting = true;
        }
        ~Count() { counting = false; }
    };
    std::vector<I> Connectivity(G::UnstructuredMesh &mesh)
    {
        std::vector<I> result;
        auto read = mesh.AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto view = read.HostView();
        for (I iCell = 0; iCell < view.NumCell(); ++iCell)
        {
            for (auto iNode : view.CellNodes(iCell))
                result.push_back(mesh.Entities(G::EntityKind::Node).LocalToGlobal(iNode));
            for (auto iFace : view.CellFaces(iCell))
                result.push_back(mesh.Entities(G::EntityKind::Face).LocalToGlobal(iFace));
        }
        return result;
    }
}

TEST_CASE("Cold mesh preparation shares an explicit policy without changing connectivity or halos")
{
    std::vector<I> expected;
    std::vector<I> expectedOutput;
    std::vector<double> expectedCoordinates;
    std::vector<I> expectedOrdering;
    int enabledSetup = 0;
    int enabledOutput = 0;
    for (bool enabled : {true, false})
    {
        auto context = std::make_shared<R::CommunicationContext>(mpi.comm, enabled);
        std::shared_ptr<G::UnstructuredMesh> mesh;
        {
            Count count;
            mesh = G::OverlapTest::BuildNativeFixture(mpi, true, 4, context).mesh;
            if (enabled)
                enabledSetup = reductions;
            else
                CHECK(reductions.load() < enabledSetup); // Required protocol/numerical reductions remain.
        }
        const auto actual = Connectivity(*mesh);
        if (enabled)
            expected = actual;
        else
            CHECK(actual == expected);
        G::LocalCellOrderingSettings ordering;
        ordering.subregions = 2;
        ordering.coreSplit = G::CellCoreSplit::CellNodeCell;
        ordering.aggregateSendRegions = true;
        const auto permutation = G::BuildLocalCellOrdering(*mesh, ordering);
        if (enabled)
            expectedOrdering = permutation.oldToNew;
        else
            CHECK(permutation.oldToNew == expectedOrdering);
        {
            G::AdjacencyRegistry registry(mpi);
            Count count;
            registry.RegisterMesh(*mesh, G::MeshReadRequirements::CellFlux());
            CHECK(reductions.load() == (enabled ? 3 : 1)); // One required policy agreement; two optional stages.
        }
        for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Face, G::EntityKind::Bnd})
            CHECK(mesh->Entities(kind).ChecksCollectivePreflight() == enabled);
        const auto cells = mesh->Entities(G::EntityKind::Cell);
        auto opposite = std::make_shared<R::CommunicationContext>(mpi.comm, !enabled);
        auto rebound = cells.WithCommunication(opposite);
        CHECK(rebound.SameOrder(cells));
        CHECK(cells.ChecksCollectivePreflight() == enabled); // retained descriptor unchanged
        CHECK(rebound.ChecksCollectivePreflight() == !enabled);
        auto sibling = cells.CreateSibling(G::EntityKind::Cell, cells.NumOwned());
        CHECK(sibling.ChecksCollectivePreflight() == enabled);
        int value = 0;
        {
            Count count;
            for (int iteration = 0; iteration < 4; ++iteration)
                cells.LocalStage([&]
                                 { ++value; });
            CHECK(reductions.load() == (enabled ? 4 : 0));
        }
        CHECK(value == 4);
        {
            Count count;
            mesh->CheckCollectiveMutationAvailable(enabled);
            CHECK(reductions.load() == (enabled ? 1 : 0));
        }
        auto route = cells.BuildRowTransferPlan();
        R::Array<double> source(R::RowLayout::Dense(cells.NumOwned(), 2));
        R::Array<double> destination(R::RowLayout::Dense(cells.NumGhost(), 2));
        {
            auto write = source.Write();
            for (I iCell = 0; iCell < cells.NumOwned(); ++iCell)
                for (I j = 0; j < 2; ++j)
                    write.View()(iCell, j) = 2 * cells.LocalToGlobal(iCell) + j;
        }
        R::PayloadPlan payload(route, source.Layout(), destination.Layout(), R::ElementSchema::For<double>());
        auto binding = payload.Bind(source, destination);
        {
            Count count;
            for (int iteration = 0; iteration < 3; ++iteration)
                binding.Begin().Wait();
            CHECK(reductions.load() == (enabled ? 3 : 0));
        }
        auto read = destination.Read();
        for (I iGhost = 0; iGhost < cells.NumGhost(); ++iGhost)
            for (I j = 0; j < 2; ++j)
                CHECK(read.View()(iGhost, j) == 2 * cells.LocalToGlobal(cells.NumOwned() + iGhost) + j);
        G::EntityField<double> field(cells, source);
        auto fieldBinding = field.BuildHaloRoute(cells).Bind(field, cells);
        {
            Count count;
            fieldBinding.Begin(R::ExecutionContext::Host()).Wait();
            CHECK(reductions.load() == (enabled ? 2 : 0));
        }
        auto ghostRead = fieldBinding.GhostValues().Read();
        for (I iGhost = 0; iGhost < cells.NumGhost(); ++iGhost)
            for (I j = 0; j < 2; ++j)
                CHECK(ghostRead.View()(iGhost, j) == 2 * cells.LocalToGlobal(cells.NumOwned() + iGhost) + j);
        MPI_Comm outputComm = MPI_COMM_NULL;
        MPI_Comm_dup(mpi.comm, &outputComm);
        {
            // Output legitimately uses an isolated congruent communicator;
            // retain the mode, never reuse requests from the original comm.
            Count count;
            const DNDS::MPIInfo outputMPI(outputComm);
            auto output = G::NativeMeshOutput::Build(*mesh->CurrentNativeState(), outputMPI);
            auto serial = G::NativeMeshSerialOutput::Build(*mesh->CurrentNativeState(), outputMPI, mpi.size - 1);
            std::vector<I> rows = output.connectivity;
            rows.insert(rows.end(), output.offsets.begin(), output.offsets.end());
            auto indices = serial.cellNodes.Read();
            for (I iCell = 0; iCell < I(serial.cellNodes.RowCount()); ++iCell)
                for (I iNode : indices.View()[iCell])
                    rows.push_back(iNode);
            std::vector<double> points;
            for (const auto *array : {&output.coordinates, &serial.coordinates})
            {
                auto values = array->Read();
                for (I iNode = 0; iNode < I(array->RowCount()); ++iNode)
                    for (I component = 0; component < 3; ++component)
                        points.push_back(values.View()(iNode, component));
            }
            if (enabled)
            {
                enabledOutput = reductions;
                expectedOutput = rows;
                expectedCoordinates = points;
            }
            else
            {
                CHECK(reductions.load() < enabledOutput);
                CHECK(rows == expectedOutput);
                CHECK(points == expectedCoordinates);
            }
        }
        MPI_Comm_free(&outputComm);
    }
}

TEST_CASE("CGNS owned import partition refinement reorder and publication inherit preflight")
{
    std::vector<I> expected;
    for (bool enabled : {true, false})
    {
        CAPTURE(enabled);
        auto context = std::make_shared<R::CommunicationContext>(mpi.comm, enabled);
        const auto path = (NativeReorderChecks::Root() / "data/mesh/UniformSquare_10.cgns").string();
        G::PartitionOptions partition;
        partition.metisSeed = 43;
        auto owned = G::ReadAndDeriveNativeMeshFromCGNSOwned(
            mpi, 2, path, partition, G::FBC_Name_2_ID_Default, {}, 1, 1, mpi.size - 1, {}, context);
        REQUIRE(owned.communication == context);
        for (const auto &[kind, space] : owned.spaces)
        {
            CHECK(space.NumGhost() == 0);
            CHECK(space.ChecksCollectivePreflight() == enabled);
        }
        R::Array<I> companion(R::RowLayout::Dense(owned.spaces.at(G::EntityKind::Cell).NumOwned(), 2));
        {
            auto ids = owned.ids.at(G::EntityKind::Cell).Read();
            auto write = companion.Write();
            for (I iCell = 0; iCell < I(companion.RowCount()); ++iCell)
                for (I j = 0; j < 2; ++j)
                    write.View()(iCell, j) = 31 * ids.View()(iCell, 0) + j;
        }
        auto reordered = G::NativeMeshTransition::ReorderOwnedCells(owned, mpi, 2, 1);
        auto entry = G::RegisterArray(companion, G::EntityKind::Cell, "id-based companion",
                                      R::ElementSchema::For<I>(), R::ExecutionContext::Host(),
                                      G::CompanionDisposition::TransferIfReordered, {}, context);
        entry.fn(reordered.transfers.at(G::EntityKind::Cell), mpi);
        owned = std::move(reordered.owned);
        {
            auto ids = owned.ids.at(G::EntityKind::Cell).Read();
            auto read = companion.Read();
            for (I iCell = 0; iCell < I(companion.RowCount()); ++iCell)
                for (I j = 0; j < 2; ++j)
                    CHECK(read.View()(iCell, j) == 31 * ids.View()(iCell, 0) + j);
        }
        G::PrepareMeshOptions options;
        options.reorderCells = 1;
        options.reorderParts = 2;
        auto state = G::PrepareNativeMeshForSolverOwned(std::move(owned), mpi, options);
        auto mesh = G::UnstructuredMesh::FromNative(mpi, std::move(state));
        REQUIRE(mesh->PreparationCommunication() == context);
        const auto actual = Connectivity(*mesh);
        if (enabled)
            expected = actual;
        else
            CHECK(actual == expected);
        mesh->CurrentNativeState()->Audit({true, true});
        auto extracted = G::NativeMeshExtraction::ExtractBoundary(*mesh->CurrentNativeState(), mpi);
        CHECK(extracted.state->Entities(G::EntityKind::Cell).NumGlobal() == mesh->Entities(G::EntityKind::Bnd).NumGlobal());
        for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
            CHECK(extracted.state->Entities(kind).ChecksCollectivePreflight() == enabled);
    }
}

TEST_CASE("Periodic owned topology and publication preserve disabled preflight")
{
    auto context = std::make_shared<R::CommunicationContext>(mpi.comm, false);
    auto owned = G::NativeMeshBuilder::Build(NativePeriodicFixtures::QuadraticTorus(mpi), {1, true, std::nullopt, context});
    auto topology = G::NativeMeshTopology::Build(std::move(owned), mpi);
    auto spec = G::DefaultNativeMeshGhostSpec(topology, 1, true);
    auto state = G::NativeMeshAssembler::Build(mpi, std::move(topology), spec);
    for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Face, G::EntityKind::Bnd})
        CHECK_FALSE(state->Entities(kind).ChecksCollectivePreflight());
    state->Audit({true, true});
}

TEST_CASE("Conflicting selections of agreed mesh contexts reject before optional preparation stages")
{
    if (mpi.size < 2)
        return;
    auto checked = std::make_shared<R::CommunicationContext>(mpi.comm, true);
    auto unchecked = std::make_shared<R::CommunicationContext>(mpi.comm, false);
    auto input = NativePeriodicFixtures::QuadraticTorus(mpi);
    G::NativeMeshBuildOptions options;
    options.communication = mpi.rank == 0 ? checked : unchecked;
    CHECK_THROWS_WITH(G::NativeMeshBuilder::Build(std::move(input), options), doctest::Contains("preflight policy differs"));
}

int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    DNDS::MPI::Finalize();
    return result;
}
