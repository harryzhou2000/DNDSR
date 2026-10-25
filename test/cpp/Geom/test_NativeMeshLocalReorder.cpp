#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/Mesh_CellPermutation.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "Geom/Mesh/FieldTransfer.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    G::AdjKind OrderedConeKey()
    {
        auto key = G::Adj::Cell2Node;
        key.path = key.Path();
        key.selection = G::RelationSelection::Incidence;
        key.order = G::RelationOrder::Topological;
        key.periodic = G::RelationPeriodicPolicy::FrameMatched;
        key.semanticName = "local-reorder.ordered-cone";
        key.semanticVersion = 1;
        return key;
    }
    R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(I));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            std::copy(rows[row].begin(), rows[row].end(), builder.Row<I>(row).data);
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }
    std::shared_ptr<const G::NativeMeshState> Fixture(const DNDS::MPIInfo &mpi, bool empty)
    {
        const bool active = !empty || mpi.rank == 0;
        const I base = 12 * mpi.rank;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? 12 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? 5 : 0);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, active ? 1 : 0);
        std::vector<std::vector<I>> cones, bnd;
        if (active)
        {
            for (I cell = 0; cell < 4; ++cell)
                cones.push_back({base + cell, base + cell + 1, base + cell + 6, base + cell + 5});
            // On multiple ranks, the triangle touches the previous rank's
            // first quad at a single node, never at an actual face.
            const I previous = empty ? 0 : (mpi.rank + mpi.size - 1) % mpi.size;
            cones.push_back({12 * previous, base + 10, base + 11});
            bnd.push_back({base, base + 1});
        }
        R::Array<DNDS::real> points(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
        R::Array<G::ElemInfo> info(R::RowLayout::Dense(cones.size(), 1)), bndInfo(R::RowLayout::Dense(bnd.size(), 1));
        {
            auto write = points.Write();
            for (size_t node = 0; node < points.Rows(); ++node)
            {
                write.View()(node, 0) = 8 * mpi.rank + double(node < 10 ? node % 5 : node - 10);
                write.View()(node, 1) = node < 10 ? double(node / 5) : -2.0;
                write.View()(node, 2) = 0;
            }
            auto metadata = info.Write(), boundary = bndInfo.Write();
            for (size_t cell = 0; cell < cones.size(); ++cell)
                metadata.View()(cell, 0) = {cell == 4 ? G::Elem::Tri3 : G::Elem::Quad4, G::t_index(100 + cell)};
            for (size_t row = 0; row < bnd.size(); ++row)
                boundary.View()(row, 0) = {G::Elem::Line2, 24};
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(cones), Rows(bnd), points, info, bndInfo};
        auto primary = G::NativeMeshBuilder::Build(std::move(input), {1, true});
        auto topology = G::NativeMeshTopology::Build(*primary, mpi);
        auto owned = G::NativeMeshAssembler::CaptureOwned(*topology, mpi);
        owned.nodeWallDist.emplace(R::RowLayout::Dense(points.Rows(), 3));
        {
            auto write = owned.nodeWallDist->Write();
            for (size_t row = 0; row < points.Rows(); ++row)
                for (size_t component = 0; component < 3; ++component)
                    write.View()(row, component) = 10 + row + component;
        }
        owned.requirements.wallDistance = true;
        // Consistent nonzero node frames exercise companion transfer without
        // requiring this geometric fixture to model a periodic interface.
        owned.periodic = true;
        auto bits = [](const R::Array<I> &cone, uint8_t value)
        {
            R::Array<G::NodePeriodicBits> output(cone.Layout());
            auto write = output.Write();
            for (size_t row = 0; row < output.Rows(); ++row)
                for (size_t slot = 0; slot < write.View().RowSize(row); ++slot)
                    write.View()(row, slot) = G::NodePeriodicBits{value};
            return output;
        };
        owned.cellNodesPbi = bits(*owned.relations[size_t(G::MeshRelation::CellNodes)], 1);
        owned.bndNodesPbi = bits(*owned.relations[size_t(G::MeshRelation::BoundaryNodes)], 1);
        owned.faceNodesPbi = bits(*owned.relations[size_t(G::MeshRelation::FaceNodes)], 1);
        owned.cellFacesPbi = bits(*owned.relations[size_t(G::MeshRelation::CellFaces)], 0);
        owned.namedRelations.emplace(OrderedConeKey(), *owned.relations[size_t(G::MeshRelation::CellNodes)]);
        owned.provenance.emplace();
        owned.provenance->sourceNamespace = topology->MeshNamespace();
        owned.provenance->sourceKinds.emplace(G::EntityKind::Cell, G::EntityKind::Cell);
        std::vector<std::vector<I>> ancestry(cones.size());
        for (size_t row = 0; row < cones.size(); ++row)
            ancestry[row] = {9000000000000LL + 5 * mpi.rank + I(row)};
        owned.provenance->sourceIds.emplace(G::EntityKind::Cell, Rows(ancestry));
        return G::NativeMeshAssembler::Build(mpi, std::move(owned));
    }

    void CheckRows(const G::NativeMeshState &before, const G::NativeMeshState &after,
                   const G::NativeMeshTransitionResult &result, const DNDS::MPIInfo &mpi)
    {
        auto old = before.Snapshot(R::Device::Host()).AcquireRead();
        auto current = after.Snapshot(R::Device::Host()).AcquireRead();
        const auto a = old.HostView(), b = current.HostView();
        const auto oldCells = before.Entities(G::EntityKind::Cell), cells = after.Entities(G::EntityKind::Cell);
        const auto &transfer = result.transfers.at(G::EntityKind::Cell);
        CHECK(result.sourceSpaces.at(G::EntityKind::Cell).SameOrder(oldCells));
        CHECK(after.MeshNamespace() == before.MeshNamespace());
        CHECK(after.IdHighWater() == before.IdHighWater());
        CHECK(after.RelationKeys() == before.RelationKeys());
        for (auto kind : {G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face})
        {
            CHECK(result.sourceSpaces.at(kind).SameOrder(before.Entities(kind)));
            CHECK(result.transfers.at(kind).isLocalOnly);
            for (I row = 0; row < before.NumOwned(kind); ++row)
            {
                CHECK(result.transfers.at(kind).localOld2New[size_t(row)] == row);
                CHECK(after.OriginalEntityId(kind, row) == before.OriginalEntityId(kind, row));
            }
        }
        for (I node = 0; node < b.NumNode(); ++node)
        {
            CHECK(b.Coord(node) == a.Coord(node));
            CHECK(b.NodeWallDist(node) == a.NodeWallDist(node));
        }
        auto oldAncestry = before.Provenance()->sourceIds.at(G::EntityKind::Cell).Read();
        auto ancestry = after.Provenance()->sourceIds.at(G::EntityKind::Cell).Read();
        for (I oldRow = 0; oldRow < a.NumCell(); ++oldRow)
        {
            const I row = transfer.localOld2New[size_t(oldRow)];
            CHECK(after.OriginalEntityId(G::EntityKind::Cell, row) == before.OriginalEntityId(G::EntityKind::Cell, oldRow));
            CHECK(b.GetCellInfo(row).zone == a.GetCellInfo(oldRow).zone);
            CHECK(b.CellNodePeriodicBits(row).size() == a.CellNodePeriodicBits(oldRow).size());
            for (size_t slot = 0; slot < b.CellNodePeriodicBits(row).size(); ++slot)
                CHECK(b.CellNodePeriodicBits(row)[slot] == a.CellNodePeriodicBits(oldRow)[slot]);
            CHECK(ancestry.View()(size_t(row), 0) == oldAncestry.View()(size_t(oldRow), 0));
            std::vector<I> cone;
            for (auto node : b.CellNodes(row))
                cone.push_back(after.OriginalEntityId(G::EntityKind::Node, node));
            std::vector<I> expected;
            for (auto node : a.CellNodes(oldRow))
                expected.push_back(before.OriginalEntityId(G::EntityKind::Node, node));
            CHECK(cone == expected);
            auto named = after.Relation(OrderedConeKey()).GlobalRow(row);
            std::vector<I> globals;
            for (auto node : b.CellNodes(row))
                globals.push_back(after.Entities(G::EntityKind::Node).LocalToGlobal(node));
            CHECK(named == globals);
        }
        // Every companion relation reference to a cell must retain that cell's
        // persistent identity after remote sparse old-to-new remapping.
        auto remapId = [&](I value, const G::EntitySpace &space, const G::NativeMeshState &state)
        {
            if (value == DNDS::UnInitIndex)
                return value;
            if (value >= 0)
                return state.OriginalEntityId(G::EntityKind::Cell, value);
            const I global = -1 - value;
            const auto local = space.GlobalToLocal(global);
            DNDS_check_throw_info(local, "test cell closure missing");
            return state.OriginalEntityId(G::EntityKind::Cell, *local);
        };
        for (I face = 0; face < b.NumFace(); ++face)
        {
            std::vector<I> expected, actual;
            for (auto cell : a.FaceCells(face))
                expected.push_back(remapId(cell, oldCells, before));
            for (auto cell : b.FaceCells(face))
                actual.push_back(remapId(cell, cells, after));
            CHECK(actual == expected);
        }
        for (I node = 0; node < b.NumNode(); ++node)
        {
            std::vector<I> expected, actual;
            for (auto cell : a.NodeCells(node))
                expected.push_back(remapId(cell, oldCells, before));
            for (auto cell : b.NodeCells(node))
                actual.push_back(remapId(cell, cells, after));
            std::sort(expected.begin(), expected.end());
            std::sort(actual.begin(), actual.end());
            CHECK(actual == expected);
        }
        for (I boundary = 0; boundary < b.NumBnd(); ++boundary)
        {
            CHECK(remapId(b.BoundaryCells(boundary)[0], cells, after) == remapId(a.BoundaryCells(boundary)[0], oldCells, before));
            CHECK(b.BoundaryCells(boundary)[1] == DNDS::UnInitIndex);
        }
        R::Array<I> fieldValues(R::RowLayout::Dense(size_t(before.NumOwned(G::EntityKind::Cell)), 1));
        {
            auto write = fieldValues.Write();
            for (size_t row = 0; row < fieldValues.Rows(); ++row)
                write.View()(row, 0) = before.OriginalEntityId(G::EntityKind::Cell, I(row));
        }
        G::EntityField<I> field(oldCells, fieldValues);
        CHECK_FALSE(cells.SameOrder(oldCells));
        CHECK_NOTHROW(field.Read(oldCells));
        CHECK_THROWS(field.Read(cells));
        auto route = G::NativeRowTransfer(transfer, mpi);
        R::Array<I> moved(route.DestinationLayout(fieldValues.Layout()));
        R::PayloadPlan(route, fieldValues.Layout(), moved.Layout(), R::ElementSchema::Arithmetic<I>())
            .Bind(fieldValues, moved, R::TransferTransport::Direct, {R::TransferMode::Transient})
            .Begin()
            .Wait();
        auto movedRead = moved.Read();
        for (size_t row = 0; row < moved.Rows(); ++row)
            CHECK(movedRead.View()(row, 0) == after.OriginalEntityId(G::EntityKind::Cell, I(row)));
        G::EntityField<I> destination(cells, moved);
        auto halo = destination.BuildHaloRoute(cells);
        auto binding = halo.Bind(destination, cells);
        binding.Begin().Wait();
        auto ownedValues = destination.Read(cells);
        auto ghostValues = binding.GhostValues().Read();
        for (I oldRow = 0; oldRow < a.NumCell(); ++oldRow)
        {
            I expected = 0, actual = 0;
            for (auto neighbor : a.CellCells(oldRow))
                expected += before.OriginalEntityId(G::EntityKind::Cell, neighbor);
            const I row = transfer.localOld2New[size_t(oldRow)];
            for (auto neighbor : b.CellCells(row))
                actual += neighbor < cells.NumOwned() ? ownedValues.View()(size_t(neighbor), 0)
                                                      : ghostValues.View()(size_t(neighbor - cells.NumOwned()), 0);
            CHECK(actual == expected);
        }
    }
}

TEST_CASE("Native local reorder shares permutation semantics and preserves current identity payloads")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool empty : {false, true})
        for (const auto &parts : {std::pair<int, int>{1, 1}, {2, 1}, {2, 2}})
        {
            CAPTURE(empty);
            CAPTURE(parts.first);
            CAPTURE(parts.second);
            auto source = Fixture(mpi, empty);
            auto facade = G::UnstructuredMesh::FromNative(mpi, source);
            auto legacy = G::CloneCurrentMeshForSetup(*facade, 1, G::TransitionCachePolicy::Invalidate);
            auto graph = legacy->GetCell2CellFaceVLocal();
            G::detail::CellPermutationResult expected;
            if (legacy->NumCell())
                expected = G::detail::ComputeCellPermutation(graph, legacy->cell2cell, legacy->NumCell(), parts.first, parts.second);
            auto retained = facade->AcquireRead(source->Requirements());
            auto result = G::NativeMeshTransition::ReorderLocalCells(*source, mpi, parts.first, parts.second);
            CHECK(result.transfers.at(G::EntityKind::Cell).localOld2New == expected.cellOld2New);
            CheckRows(*source, *result.state, result, mpi);
            CHECK(retained.HostView().NumCell() == source->NumOwned(G::EntityKind::Cell));
            for (int part = 0; part < result.state->NLocalParts(); ++part)
            {
                bool outer = false;
                for (I row = result.state->LocalPartStart(part); row < result.state->LocalPartEnd(part); ++row)
                {
                    const auto oldRow = expected.cellNew2Old[size_t(row)];
                    bool ghost = false;
                    for (auto cell : retained.HostView().CellCells(oldRow))
                        ghost = ghost || cell >= source->NumOwned(G::EntityKind::Cell);
                    CHECK_FALSE((outer && !ghost));
                    outer = outer || ghost;
                }
            }
            if (!empty && mpi.size > 1 && parts.first == 1)
            {
                // Quad0's only remote contact is a vertex. Private sorting
                // must use full node-neighbor CellCells, not face neighbors.
                CHECK(result.transfers.at(G::EntityKind::Cell).localOld2New[0] >= 3);
            }
            CHECK_THROWS(facade->ReorderLocalCells(parts.first, parts.second));
        }
}

TEST_CASE("Native facade local reorder validates inputs and publishes only after readers release")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = Fixture(mpi, false);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, source);
    const auto oldOrder = mesh->Entities(G::EntityKind::Cell);
    const auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(G::NativeMeshTransition::ReorderLocalCells(*source, mpi, INT_MAX, 2));
    CHECK_THROWS(G::NativeMeshTransition::ReorderLocalCells(*source, mpi, 1, 1, 0));
    if (mpi.size > 1)
        CHECK_THROWS(G::NativeMeshTransition::ReorderLocalCells(*source, mpi, mpi.rank == 0 ? 1 : 2));
    CHECK(stamp.Valid());
    mesh->ReorderLocalCells(2, 2);
    CHECK_FALSE(stamp.Valid());
    CHECK_FALSE(oldOrder.Valid());
    CHECK(mesh->HasNativeState());
    CHECK_FALSE(mesh->cell2node.father);
    CHECK(mesh->NLocalParts() == 2);
    CHECK(mesh->LocalCellNeighbors().size() == 5);
    for (int part = 0; part < mesh->NLocalParts(); ++part)
        for (I row = mesh->LocalPartStart(part); row < mesh->LocalPartEnd(part); ++row)
            for (auto other : mesh->LocalCellNeighbors()[size_t(row)])
                CHECK((other >= mesh->LocalPartStart(part) && other < mesh->LocalPartEnd(part)));
}

TEST_CASE("Shared local partition helper handles disconnected empty and singleton face graphs")
{
    const G::tLocalMatStruct original{{1}, {0}, {3}, {2}, {}};
    const auto partitions = G::PartitionSerialAdj_Metis(original.begin(), original.end(), 2);
    REQUIRE(partitions.size() == original.size());
    for (auto part : partitions)
        CHECK((part >= 0 && part < 2));
    auto graph = original;
    const std::vector<std::vector<I>> fullRows{{1}, {0}, {3}, {2}, {}};
    const auto result = G::detail::ComputeCellPermutation(graph, fullRows, 5, 2, 2);
    CHECK(result.localPartitionStarts.front() == 0);
    CHECK(result.localPartitionStarts.back() == 5);
    std::vector<I> covered = result.cellNew2Old;
    std::sort(covered.begin(), covered.end());
    CHECK(covered == std::vector<I>{0, 1, 2, 3, 4});
    for (I row = 0; row < 5; ++row)
        CHECK(result.cellOld2New[size_t(result.cellNew2Old[size_t(row)])] == row);
    const G::tLocalMatStruct empty;
    CHECK(G::PartitionSerialAdj_Metis(empty.begin(), empty.end(), 2).empty());
    const G::tLocalMatStruct singleton{{}};
    CHECK(G::PartitionSerialAdj_Metis(singleton.begin(), singleton.end(), 2) == std::vector<idx_t>{0});
    G::tLocalMatStruct one{{}};
    const auto singlePermutation = G::detail::ComputeCellPermutation(one, std::vector<std::vector<I>>{{}}, 1, 2, 2);
    CHECK(singlePermutation.cellOld2New == std::vector<I>{0});
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
