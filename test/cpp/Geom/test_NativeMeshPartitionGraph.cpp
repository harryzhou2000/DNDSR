#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshPartitionGraph.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "Geom/Unstructured/ReaderPartitionGraph.hpp"
#include "Geom/Refinement/NativeMeshExtraction.hpp"
#include "NativePeriodicFixtures.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    G::NativeMeshInput ReaderPeriodic(const DNDS::MPIInfo &mpi, bool triangles)
    {
        auto input = NativePeriodicFixtures::TwoCells(mpi);
        const size_t count = triangles ? 4 : 2;
        input.cells = input.nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? I(count) : 0);
        const bool active = input.cells.NumOwned() != 0;
        const std::vector<std::vector<I>> cones = triangles ? std::vector<std::vector<I>>{{0, 1, 3}, {0, 3, 2}, {1, 0, 2}, {1, 2, 3}}
                                                            : std::vector<std::vector<I>>{{0, 1, 3, 2}, {1, 0, 2, 3}};
        const std::vector<std::vector<G::NodePeriodicBits>> bits = triangles ? std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {0}, {0}}, {{0}, {0}, {0}}, {{0}, {1}, {1}}, {{0}, {1}, {0}}}
                                                                             : std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {0}, {0}, {0}}, {{0}, {1}, {1}, {0}}};
        input.cellNodes = NativePeriodicFixtures::Rows<I>(active ? cones : std::vector<std::vector<I>>{});
        input.cellNodesPbi = NativePeriodicFixtures::Rows<G::NodePeriodicBits>(active ? bits : std::vector<std::vector<G::NodePeriodicBits>>{});
        input.cellInfo = R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(input.cells.NumOwned()), 1));
        auto info = input.cellInfo.Write();
        for (size_t row = 0; row < size_t(input.cells.NumOwned()); ++row)
            info.View()(row, 0) = G::ElemInfo{triangles ? G::Elem::Tri3 : G::Elem::Quad4, G::INTERNAL_ZONE};
        return input;
    }

    G::NativeMeshInput ReaderHex(const DNDS::MPIInfo &mpi)
    {
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 6 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 2 : 0);
        auto bnd = nodes.CreateSibling(G::EntityKind::Bnd, mpi.rank == mpi.size / 2 ? 4 : 0);
        std::vector<std::vector<I>> cones, boundaries;
        std::vector<std::vector<G::NodePeriodicBits>> bits, boundaryBits;
        if (cells.NumOwned())
        {
            cones = {{0, 0, 1, 1, 3, 3, 4, 4}, {1, 1, 2, 2, 4, 4, 5, 5}};
            bits = {{{0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}}, {{0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}}};
        }
        if (bnd.NumOwned())
        {
            boundaries = {{0, 1, 4, 3}, {0, 3, 4, 1}, {1, 2, 5, 4}, {1, 4, 5, 2}};
            boundaryBits = {{{1}, {1}, {1}, {1}}, {{0}, {0}, {0}, {0}}, {{1}, {1}, {1}, {1}}, {{0}, {0}, {0}, {0}}};
        }
        G::NativeMeshInput input{mpi, 3, cells, nodes, bnd, NativePeriodicFixtures::Rows(cones), NativePeriodicFixtures::Rows(boundaries),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(bnd.NumOwned()), 1))};
        input.periodic = true;
        input.cellNodesPbi = NativePeriodicFixtures::Rows(bits);
        input.boundaryNodesPbi = NativePeriodicFixtures::Rows(boundaryBits);
        {
            auto coordinates = input.coordinates.Write();
            for (size_t row = 0; row < size_t(nodes.NumOwned()); ++row)
            {
                coordinates.View()(row, 0) = 0;
                coordinates.View()(row, 1) = DNDS::real(row % 3);
                coordinates.View()(row, 2) = DNDS::real(row / 3);
            }
            auto info = input.cellInfo.Write();
            for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
                info.View()(row, 0) = {G::Elem::Hex8, G::INTERNAL_ZONE};
            auto boundary = input.boundaryInfo.Write();
            for (size_t row = 0; row < size_t(bnd.NumOwned()); ++row)
                boundary.View()(row, 0) = {G::Elem::Quad4, row % 2 ? G::BC_ID_PERIODIC_1 : G::BC_ID_PERIODIC_1_DONOR};
        }
        return input;
    }

    void CheckReaderGraphAndPartition(const G::UnstructuredMeshState &source, const DNDS::MPIInfo &mpi, bool partition = true)
    {
        const auto graph = G::detail::BuildReaderPartitionGraph(source, mpi);
        const auto full = G::PrepareNativeMeshTopology(source, mpi);
        const auto oracle = G::detail::BuildNativePartitionGraph(*full, mpi, 0);
        CHECK(graph.offsets == oracle.offsets);
        CHECK(graph.adjacency == oracle.adjacency);
        const auto owned = G::NativeMeshAssembler::CaptureOwned(source, mpi);
        const auto primary = G::detail::BuildOwnedPrimaryCellGraph(owned, mpi);
        CHECK(primary.offsets == oracle.offsets);
        CHECK(primary.adjacency == oracle.adjacency);
        CHECK(graph.weights.empty());
        if (!partition)
            return;
        for (const auto *method : {"KWAY", "RB"})
        {
            CAPTURE(std::string(method));
            G::NativePartitionOptions options;
            options.metisType = method;
            const auto fast = G::NativeMeshPreparation::Partition(source, mpi, options);
            const auto reference = G::NativeMeshPreparation::Partition(*full, mpi, options);
            options.deferNewFacesUntilPreparation = true;
            const auto deferred = G::NativeMeshPreparation::Partition(source, mpi, options);
            const auto retainedFaces = G::NativeMeshPreparation::Partition(*full, mpi, options);
            CHECK_FALSE(deferred.state->HasEntities(G::EntityKind::Face));
            CHECK_FALSE(deferred.state->Requirements().faces);
            CHECK(deferred.state->LocalCellNeighbors().empty());
            CHECK(deferred.state->IdHighWater() == source.IdHighWater());
            CHECK(deferred.state->MeshNamespace() == source.MeshNamespace());
            CHECK(retainedFaces.state->Requirements().faces);
            CHECK(retainedFaces.state->IdHighWater() == reference.state->IdHighWater());
            auto localOracle = fast.state->BuildLocalCellNeighbors();
            for (auto &row : localOracle)
            {
                std::sort(row.begin(), row.end());
                row.erase(std::unique(row.begin(), row.end()), row.end());
            }
            CHECK(G::detail::BuildPrimaryLocalCellNeighbors(*deferred.state) == localOracle);
            CHECK(fast.state->MeshNamespace() == source.MeshNamespace());
            for (const auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
            {
                CHECK(fast.sourceSpaces.at(kind).SameOrder(source.Entities(kind)));
                REQUIRE(fast.state->NumOwned(kind) == reference.state->NumOwned(kind));
                REQUIRE(deferred.state->NumOwned(kind) == fast.state->NumOwned(kind));
                CHECK(deferred.sourceSpaces.at(kind).SameOrder(source.Entities(kind)));
                CHECK(fast.state->IdHighWater().at(kind) == source.IdHighWater().at(kind));
                for (I row = 0; row < fast.state->NumOwned(kind); ++row)
                {
                    CHECK(fast.state->OriginalEntityId(kind, row) == reference.state->OriginalEntityId(kind, row));
                    CHECK(deferred.state->OriginalEntityId(kind, row) == fast.state->OriginalEntityId(kind, row));
                }
            }
            CHECK_FALSE(fast.transfers.count(G::EntityKind::Face));
            CHECK(fast.state->NumGlobal(G::EntityKind::Face) == full->NumGlobal(G::EntityKind::Face));
            const auto beforeA = fast.state->BorrowHostView(), beforeB = reference.state->BorrowHostView();
            for (I cell = 0; cell < beforeA.NumCell(); ++cell)
                for (size_t slot = 0; slot < beforeA.CellFaces(cell).size(); ++slot)
                {
                    const auto fa = beforeA.CellFaces(cell)[slot], fb = beforeB.CellFaces(cell)[slot];
                    // Fresh faces may select the opposite periodic reference
                    // side before final cell reorder. Physical boundary zones
                    // and coordinates in this cell's frame must still agree.
                    auto zoneOnCell = [&](const auto &view, I face)
                    {
                        const auto zone = view.GetFaceZone(face);
                        if (!G::FaceIDIsPeriodic(zone))
                            return zone;
                        const bool back = view.CellIsFaceBack(cell, face, DNDS::rowsize(slot));
                        const bool main = G::FaceIDIsPeriodicMain(zone) == back;
                        return G::t_index(-(((-int(zone) - 1) % 3) + 1) - (main ? 0 : 3));
                    };
                    CHECK(zoneOnCell(beforeA, fa) == zoneOnCell(beforeB, fb));
                }
            // PrepareMesh reorders cells and refreshes faces. This exact stage
            // must restore the same final orientation, slots and PBI, not merely
            // an isomorphic graph with new face labels.
            auto finalA = G::NativeMeshTransition::ReorderLocalCells(*deferred.state, mpi).state;
            auto finalB = G::NativeMeshTransition::ReorderLocalCells(*reference.state, mpi).state;
            finalA = G::NativeMeshPreparation::PrepareTopology(*finalA, mpi);
            finalB = G::NativeMeshPreparation::PrepareTopology(*finalB, mpi);
            const auto a = finalA->BorrowHostView(), b = finalB->BorrowHostView();
            for (I cell = 0; cell < a.NumCell(); ++cell)
            {
                REQUIRE(a.CellNodes(cell).size() == b.CellNodes(cell).size());
                CHECK(a.GetCellInfo(cell).type == b.GetCellInfo(cell).type);
                for (size_t slot = 0; slot < a.CellNodes(cell).size(); ++slot)
                {
                    CHECK(finalA->OriginalEntityId(G::EntityKind::Node, a.CellNodes(cell)[slot]) ==
                          finalB->OriginalEntityId(G::EntityKind::Node, b.CellNodes(cell)[slot]));
                    CHECK(a.GetCoordNodeOnCell(cell, DNDS::rowsize(slot)) == b.GetCoordNodeOnCell(cell, DNDS::rowsize(slot)));
                }
                for (size_t slot = 0; slot < a.CellFaces(cell).size(); ++slot)
                {
                    const auto fa = a.CellFaces(cell)[slot], fb = b.CellFaces(cell)[slot];
                    CHECK(a.GetFaceZone(fa) == b.GetFaceZone(fb));
                    CHECK(a.CellIsFaceBack(cell, fa, DNDS::rowsize(slot)) == b.CellIsFaceBack(cell, fb, DNDS::rowsize(slot)));
                    REQUIRE(a.FaceNodes(fa).size() == b.FaceNodes(fb).size());
                    for (size_t node = 0; node < a.FaceNodes(fa).size(); ++node)
                    {
                        CHECK(finalA->OriginalEntityId(G::EntityKind::Node, a.FaceNodes(fa)[node]) ==
                              finalB->OriginalEntityId(G::EntityKind::Node, b.FaceNodes(fb)[node]));
                        if (a.IsPeriodic())
                            CHECK(a.FaceNodePeriodicBits(fa)[node] == b.FaceNodePeriodicBits(fb)[node]);
                    }
                    const auto na = a.CellFaceOther(cell, fa, DNDS::rowsize(slot));
                    const auto nb = b.CellFaceOther(cell, fb, DNDS::rowsize(slot));
                    CHECK((na == DNDS::UnInitIndex ? na : finalA->OriginalEntityId(G::EntityKind::Cell, na)) ==
                          (nb == DNDS::UnInitIndex ? nb : finalB->OriginalEntityId(G::EntityKind::Cell, nb)));
                }
            }
        }
    }

    std::shared_ptr<const G::NativeMeshState> WeightedStrip(const DNDS::MPIInfo &mpi, bool readerOwned)
    {
        const bool active = !readerOwned || mpi.rank == mpi.size - 1;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? 8 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? 3 : 0);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        R::RowBuilder cones(size_t(cells.NumOwned()), sizeof(I));
        const I first = readerOwned ? 0 : 8 * mpi.rank;
        for (size_t cell = 0; cell < size_t(cells.NumOwned()); ++cell)
        {
            cones.ResizeRow(cell, 4);
            const I values[]{first + I(cell), first + I(cell) + 1, first + I(cell) + 5, first + I(cell) + 4};
            std::copy(std::begin(values), std::end(values), cones.Row<I>(cell).data);
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, R::Array<I>::FromSharedStorage(cones.Commit()), R::Array<I>(),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        {
            auto points = input.coordinates.Write();
            const DNDS::real heights[]{1, 2, 4, 3};
            for (size_t row = 0; row < size_t(nodes.NumOwned()); ++row)
            {
                points.View()(row, 0) = 5 * mpi.rank + row % 4;
                points.View()(row, 1) = row < 4 ? 0 : heights[row % 4] * (mpi.rank + 1);
                points.View()(row, 2) = 0;
            }
            auto info = input.cellInfo.Write();
            for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
                info.View()(row, 0) = G::ElemInfo{G::Elem::Quad4, G::t_index(100 * mpi.rank + int(row))};
        }
        return G::ConstructNativeMesh(std::move(input));
    }
}

TEST_CASE("Reader primary unweighted graph equals full facial publication and partitions identical primary rows")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool triangles : {false, true})
    {
        auto source = G::ConstructNativeMesh(ReaderPeriodic(mpi, triangles));
        const auto graph = G::detail::BuildReaderPartitionGraph(*source, mpi);
        if (mpi.rank == 0)
        {
            CHECK(graph.offsets == (triangles ? std::vector<idx_t>{0, 2, 4, 6, 8} : std::vector<idx_t>{0, 1, 2}));
            CHECK(graph.adjacency == (triangles ? std::vector<idx_t>{1, 3, 0, 2, 1, 3, 0, 2} : std::vector<idx_t>{1, 0}));
            if (mpi.size > 1)
                CHECK(source->NumOwned(G::EntityKind::Node) == 0);
        }
        else
            CHECK(graph.offsets == std::vector<idx_t>{0});
        INFO("periodic two-cell fixture");
        CheckReaderGraphAndPartition(*source, mpi);
    }
    INFO("reader-owned hex fixture");
    CheckReaderGraphAndPartition(*G::ConstructNativeMesh(ReaderHex(mpi)), mpi);
    INFO("quadratic periodic torus fixture");
    CheckReaderGraphAndPartition(*G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi)), mpi);
    INFO("weighted strip fixture");
    CheckReaderGraphAndPartition(*WeightedStrip(mpi, true), mpi);
}

TEST_CASE("Reader scratch rejects dimension-one boundary meshes collectively")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = G::ConstructNativeMesh(ReaderPeriodic(mpi, false));
    auto boundary = G::NativeMeshExtraction::ExtractBoundary(*source, mpi).state;
    CHECK(boundary->Snapshot(R::Device::Host()).AcquireRead().HostView().Dimension() == 1);
    CHECK_THROWS(G::detail::BuildReaderPartitionGraph(*boundary, mpi));
    CHECK_THROWS(G::NativeMeshPreparation::Partition(*boundary, mpi));
    CHECK_THROWS(G::NativeMeshTransition::ReorderLocalCells(*boundary, mpi));
    CHECK_NOTHROW(boundary->Validate());
}

TEST_CASE("Face deferral agrees collectively and preserves weighted fallback")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = WeightedStrip(mpi, true);
    G::NativePartitionOptions options;
    if (mpi.size > 1)
    {
        options.deferNewFacesUntilPreparation = mpi.rank == 0;
        CHECK_THROWS(G::NativeMeshPreparation::Partition(*source, mpi, options));
        CHECK_NOTHROW(source->Validate());
    }
    options.deferNewFacesUntilPreparation = true;
    auto deferred = G::NativeMeshPreparation::Partition(*source, mpi, options);
    CHECK_FALSE(deferred.state->HasEntities(G::EntityKind::Face));
    options.edgeWeightMethod = 1;
    auto weighted = G::NativeMeshPreparation::Partition(*source, mpi, options);
    CHECK(weighted.state->Requirements().faces);
    CHECK(weighted.state->HasEntities(G::EntityKind::Face));
    CHECK(weighted.state->MeshNamespace() == source->MeshNamespace());
}

TEST_CASE("Reader graph retains empty disconnected nonmanifold and cracked high-order behavior")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool disconnected : {false, true})
    {
        auto input = ReaderPeriodic(mpi, false);
        input.periodic = false;
        input.cellNodesPbi.reset();
        input.boundaryNodesPbi.reset();
        input.nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, disconnected && mpi.rank == mpi.size - 1 ? 8 : 0);
        input.cells = input.nodes.CreateSibling(G::EntityKind::Cell, disconnected && mpi.rank == 0 ? 2 : 0);
        input.boundaries = input.nodes.CreateSibling(G::EntityKind::Bnd, 0);
        input.cellNodes = NativePeriodicFixtures::Rows<I>(input.cells.NumOwned() ? std::vector<std::vector<I>>{{0, 1, 2, 3}, {4, 5, 6, 7}} : std::vector<std::vector<I>>{});
        input.boundaryNodes = R::Array<I>();
        input.coordinates = R::Array<DNDS::real>(R::RowLayout::Dense(size_t(input.nodes.NumOwned()), 3));
        input.cellInfo = R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(input.cells.NumOwned()), 1));
        input.boundaryInfo = R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1));
        {
            auto info = input.cellInfo.Write();
            for (size_t row = 0; row < size_t(input.cells.NumOwned()); ++row)
                info.View()(row, 0) = {G::Elem::Quad4, G::INTERNAL_ZONE};
        }
        auto source = G::ConstructNativeMesh(std::move(input));
        CheckReaderGraphAndPartition(*source, mpi, !disconnected);
        if (disconnected)
        {
            const auto full = G::PrepareNativeMeshTopology(*source, mpi);
            for (const auto *method : {"KWAY", "RB"})
            {
                G::NativePartitionOptions options;
                options.metisType = method;
                std::optional<G::NativeMeshTransitionResult> fast, reference;
                try
                {
                    fast.emplace(G::NativeMeshPreparation::Partition(*source, mpi, options));
                }
                catch (const std::exception &)
                {
                }
                try
                {
                    reference.emplace(G::NativeMeshPreparation::Partition(*full, mpi, options));
                }
                catch (const std::exception &)
                {
                }
                // METIS contiguity can reject disconnected input. Keep the
                // reference success/failure behavior as well as assignments.
                REQUIRE(bool(fast) == bool(reference));
                if (fast)
                {
                    REQUIRE(fast->state->NumOwned(G::EntityKind::Cell) == reference->state->NumOwned(G::EntityKind::Cell));
                    for (I row = 0; row < fast->state->NumOwned(G::EntityKind::Cell); ++row)
                        CHECK(fast->state->OriginalEntityId(G::EntityKind::Cell, row) == reference->state->OriginalEntityId(G::EntityKind::Cell, row));
                }
            }
        }
    }
    auto cracked = G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi, true));
    CHECK_THROWS(G::detail::BuildReaderPartitionGraph(*cracked, mpi));
    CHECK_THROWS(G::PrepareNativeMeshTopology(*cracked, mpi));
    CHECK_THROWS(G::NativeMeshTransition::ReorderLocalCells(*cracked, mpi));
    CHECK_NOTHROW(cracked->Validate());
    auto input = ReaderPeriodic(mpi, true);
    input.periodic = false;
    input.cellNodesPbi.reset();
    input.boundaryNodesPbi.reset();
    input.nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 5 : 0);
    input.cells = input.nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 3 : 0);
    input.boundaries = input.nodes.CreateSibling(G::EntityKind::Bnd, 0);
    input.cellNodes = NativePeriodicFixtures::Rows<I>(input.cells.NumOwned() ? std::vector<std::vector<I>>{{0, 1, 2}, {1, 0, 3}, {0, 1, 4}} : std::vector<std::vector<I>>{});
    input.boundaryNodes = R::Array<I>();
    input.coordinates = R::Array<DNDS::real>(R::RowLayout::Dense(size_t(input.nodes.NumOwned()), 3));
    input.cellInfo = R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(input.cells.NumOwned()), 1));
    input.boundaryInfo = R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1));
    {
        auto info = input.cellInfo.Write();
        for (size_t row = 0; row < size_t(input.cells.NumOwned()); ++row)
            info.View()(row, 0) = {G::Elem::Tri3, G::INTERNAL_ZONE};
    }
    auto nonmanifold = G::ConstructNativeMesh(std::move(input));
    CHECK_THROWS(G::detail::BuildReaderPartitionGraph(*nonmanifold, mpi));
    CHECK_THROWS(G::PrepareNativeMeshTopology(*nonmanifold, mpi));
    CHECK_THROWS(G::NativeMeshTransition::ReorderLocalCells(*nonmanifold, mpi));
    CHECK_NOTHROW(nonmanifold->Validate());
    auto retry = G::ConstructNativeMesh(ReaderPeriodic(mpi, false));
    CHECK_NOTHROW(G::NativeMeshTransition::ReorderLocalCells(*retry, mpi));
    if (mpi.size > 1)
    {
        auto distributed = WeightedStrip(mpi, false);
        CHECK_THROWS(G::detail::BuildReaderPartitionGraph(*distributed, mpi));
    }
}

TEST_CASE("Native partition uses exact global CSR and normalized nonuniform physical face diameters")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool readerOwned : {false, true})
    {
        auto source = WeightedStrip(mpi, readerOwned);
        auto prepared = G::PrepareNativeMeshTopology(*source, mpi);
        const auto graph = G::detail::BuildNativePartitionGraph(*prepared, mpi, 1);
        const auto unweighted = G::detail::BuildNativePartitionGraph(*prepared, mpi, 0);
        CHECK(unweighted.offsets == graph.offsets);
        CHECK(unweighted.adjacency == graph.adjacency);
        CHECK(unweighted.weights.empty());
        const bool active = !readerOwned || mpi.rank == mpi.size - 1;
        if (active)
        {
            const idx_t first = readerOwned ? 0 : 3 * mpi.rank;
            // Shared edges have lengths 2*(rank+1), 4*(rank+1).
            // The maximum over all ranks is exactly 4*mpi.size.
            const I numerator = readerOwned ? 1 : mpi.rank + 1;
            const I denominator = readerOwned ? 1 : mpi.size;
            const I cap = std::min<I>(INT_MAX, (I(std::numeric_limits<idx_t>::max()) - 1) / (4 * (readerOwned ? 1 : mpi.size)));
            const idx_t half = idx_t((cap - 1) * numerator / (2 * denominator) + 1);
            const idx_t full = idx_t((cap - 1) * numerator / denominator + 1);
            CHECK(graph.offsets == std::vector<idx_t>{0, 1, 3, 4});
            CHECK(graph.adjacency == std::vector<idx_t>{first + 1, first, first + 2, first + 1});
            CHECK(graph.weights == std::vector<idx_t>{half, half, full, full});
        }
        else
        {
            CHECK(graph.offsets == std::vector<idx_t>{0});
            CHECK(graph.adjacency.empty());
            CHECK(graph.weights.empty());
        }
        G::NativePartitionOptions options;
        options.edgeWeightMethod = 1;
        options.metisNcuts = 2;
        if (readerOwned)
            options.metisType = "RB";
        auto result = G::NativeMeshPreparation::Partition(*source, mpi, options);
        CHECK(result.state->MeshNamespace() == source->MeshNamespace());
        CHECK(result.state->NumGlobal(G::EntityKind::Cell) == source->NumGlobal(G::EntityKind::Cell));
        CHECK(result.state->NumGlobal(G::EntityKind::Node) == source->NumGlobal(G::EntityKind::Node));
        for (const auto &[kind, highWater] : source->IdHighWater())
            CHECK(result.state->IdHighWater().at(kind) == highWater);
        CHECK(result.transfers.count(G::EntityKind::Face));
        const auto view = result.state->BorrowHostView();
        for (I cell = 0; cell < view.NumCell(); ++cell)
        {
            const auto original = result.state->OriginalEntityId(G::EntityKind::Cell, cell);
            const I owner = readerOwned ? mpi.size - 1 : original / 3;
            const I sourceCell = original % 3;
            CHECK(view.GetCellInfo(cell).zone == 100 * owner + sourceCell);
            const I first = readerOwned ? 0 : 8 * owner;
            const I expected[]{first + sourceCell, first + sourceCell + 1, first + sourceCell + 5, first + sourceCell + 4};
            for (size_t slot = 0; slot < 4; ++slot)
                CHECK(result.state->OriginalEntityId(G::EntityKind::Node, view.CellNodes(cell)[slot]) == expected[slot]);
        }
        if (readerOwned)
        {
            CHECK(result.state->NumGlobal(G::EntityKind::Face) == 10);
            std::array<I, 50> originalFaces{};
            if (mpi.rank == mpi.size - 1)
            {
                const auto originalView = prepared->BorrowHostView();
                REQUIRE(originalView.NumFace() == 10);
                for (I face = 0; face < 10; ++face)
                {
                    const auto id = prepared->OriginalEntityId(G::EntityKind::Face, face);
                    REQUIRE(id >= 0);
                    REQUIRE(id < 10);
                    auto row = originalFaces.data() + 5 * id;
                    row[0] = id;
                    for (size_t slot = 0; slot < 2; ++slot)
                    {
                        row[1 + slot] = prepared->OriginalEntityId(G::EntityKind::Node, originalView.FaceNodes(face)[slot]);
                        const auto cell = originalView.FaceCells(face)[slot];
                        row[3 + slot] = cell == DNDS::UnInitIndex ? cell : prepared->OriginalEntityId(G::EntityKind::Cell, cell);
                    }
                }
            }
            MPI_Bcast(originalFaces.data(), int(originalFaces.size()), DNDS::DNDS_MPI_INDEX, mpi.size - 1, mpi.comm);
            if (mpi.size > 1 && result.state->NumOwned(G::EntityKind::Cell) == 0)
                CHECK(result.state->NumOwned(G::EntityKind::Face) == 0);
            if (mpi.size > 1 && mpi.rank == mpi.size - 1)
                CHECK(result.state->NumOwned(G::EntityKind::Face) < 10);
            for (I face = 0; face < view.NumFace(); ++face)
            {
                const auto id = result.state->OriginalEntityId(G::EntityKind::Face, face);
                REQUIRE(id >= 0);
                REQUIRE(id < 10);
                const auto original = originalFaces.data() + 5 * id;
                CHECK(original[0] == id);
                const auto support = view.FaceCells(face);
                I minimumCellOwner = mpi.size;
                for (I cell : support)
                    if (cell != DNDS::UnInitIndex)
                    {
                        REQUIRE(cell >= 0);
                        const auto cells = result.state->Entities(G::EntityKind::Cell);
                        minimumCellOwner = std::min<I>(minimumCellOwner, cells.Owner(cells.LocalToGlobal(cell)));
                    }
                CHECK(minimumCellOwner == mpi.rank);
                std::vector<I> nodes;
                for (size_t slot = 0; slot < 2; ++slot)
                {
                    const auto node = result.state->OriginalEntityId(G::EntityKind::Node, view.FaceNodes(face)[slot]);
                    nodes.push_back(node);
                    CHECK(node == original[1 + slot]);
                    const auto cell = support[slot];
                    CHECK((cell == DNDS::UnInitIndex ? cell : result.state->OriginalEntityId(G::EntityKind::Cell, cell)) == original[3 + slot]);
                }
                std::sort(nodes.begin(), nodes.end());
                REQUIRE(nodes.size() == 2);
                const I a = nodes[0], b = nodes[1];
                CHECK(((a < 4 && b == a + 1) || (a >= 4 && b == a + 1) || b == a + 4));
            }
        }
        if (!readerOwned && mpi.size > 1)
        {
            options.metisType = "RB";
            CHECK_THROWS(G::NativeMeshPreparation::Partition(*source, mpi, options));
            options.metisType = "KWAY";
            CHECK_NOTHROW(G::NativeMeshPreparation::Partition(*source, mpi, options));
        }
        CHECK_THROWS(G::detail::BuildNativePartitionGraph(*prepared, mpi, mpi.rank == 0 ? 2 : 1));
        CHECK_NOTHROW(G::detail::BuildNativePartitionGraph(*prepared, mpi, 1));
        CHECK_NOTHROW(source->Validate());
    }
}

TEST_CASE("Several physical faces of one cell pair contribute their maximum diameter")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 13 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 5 : 0);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    R::RowBuilder cones(size_t(cells.NumOwned()), sizeof(I));
    const I inputCones[5][4]{{0, 1, 2, 3}, {1, 0, 4, 2}, {5, 6, 10, 9}, {6, 7, 11, 10}, {7, 8, 12, 11}};
    for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
    {
        cones.ResizeRow(row, 4);
        std::copy(std::begin(inputCones[row]), std::end(inputCones[row]), cones.Row<I>(row).data);
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, R::Array<I>::FromSharedStorage(cones.Commit()), R::Array<I>(),
                             R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
    {
        auto points = input.coordinates.Write();
        const DNDS::real coordinates[13][3]{{0, 0, 0}, {1, 0, 0}, {1, 2, 0}, {0, 2, 0}, {2, -1, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}, {13, 0, 0}, {10, 1, 0}, {11, 2, 0}, {12, 4, 0}, {13, 3, 0}};
        for (size_t row = 0; row < size_t(nodes.NumOwned()); ++row)
            for (size_t j = 0; j < 3; ++j)
                points.View()(row, j) = coordinates[row][j];
        auto info = input.cellInfo.Write();
        for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
            info.View()(row, 0) = G::ElemInfo{G::Elem::Quad4, 0};
    }
    auto source = G::PrepareNativeMeshTopology(*G::ConstructNativeMesh(std::move(input)), mpi);
    auto graph = G::detail::BuildNativePartitionGraph(*source, mpi, 1);
    if (mpi.rank == 0)
    {
        const idx_t cap = idx_t(std::min<I>(INT_MAX, (I(std::numeric_limits<idx_t>::max()) - 1) / 6));
        const idx_t half = idx_t((I(cap) - 1) / 2 + 1);
        CHECK(graph.offsets == std::vector<idx_t>{0, 1, 2, 3, 5, 6});
        CHECK(graph.adjacency == std::vector<idx_t>{1, 0, 3, 2, 4, 3});
        CHECK(graph.weights == std::vector<idx_t>{half, half, half, half, cap, cap});
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
