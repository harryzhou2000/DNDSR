#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshArchive.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/AdjacencyRegistry.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unistd.h>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
namespace S = DNDS::Serializer;
using I = DNDS::index;

namespace
{
    constexpr const char *Namespace = "c4a67b5d-49e2-43f1-8123-7890abcdef12";
    const std::vector<std::vector<I>> CellNodes{{0, 1, 2}, {1, 3, 2}, {4, 5, 6}};
    const std::vector<std::vector<I>> BoundaryNodes{{0, 1}, {1, 3}, {4, 5}};
    const std::vector<std::vector<I>> CellNeighbors{{1}, {0}, {}};
    const std::vector<std::vector<I>> NodeCells{{0}, {0, 1}, {0, 1}, {1}, {2}, {2}, {2}};
    const std::vector<std::vector<I>> NodeBoundaries{{0}, {0, 1}, {}, {1}, {2}, {2}, {}};
    const DNDS::real Points[7][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {3, 0, 0}, {4, 0, 0}, {3, 1, 0}};
    const std::vector<std::vector<I>> AnyRows{{0, 1}, {0, 1}, {2}};
    const std::vector<std::vector<I>> AllRows{{0}, {1}, {2}};

    DNDS::MPIInfo MPI(MPI_Comm communicator = MPI_COMM_WORLD) { return DNDS::MPIInfo(communicator); }
    I PersistentId(G::EntityKind kind, I global) { return 1000000000000LL + 1000000 * int(kind) + 17 * global; }
    I HighWater(G::EntityKind kind) { return 2000000000000LL + 1000000 * int(kind); }
    G::AdjacencyDescriptor AnyKey()
    {
        return G::AdjacencyDescriptor::Composition(G::Adj::Cell2Node, G::Adj::Cell2Node.Inverted(),
                                                   G::RelationSelection::Any, false, "archive.cell-node-cell.any", 1);
    }
    G::AdjacencyDescriptor AllKey()
    {
        return G::AdjacencyDescriptor::Composition(G::Adj::Cell2Node, G::Adj::Cell2Node.Inverted(),
                                                   G::RelationSelection::All, false, "archive.cell-node-cell.all", 1);
    }
    R::Array<I> Rows(const G::EntitySpace &space, const std::vector<std::vector<I>> &globalRows)
    {
        R::RowBuilder builder(size_t(space.NumOwned()), sizeof(I));
        for (I row = 0; row < space.NumOwned(); ++row)
        {
            const auto &values = globalRows.at(size_t(space.LocalToGlobal(row)));
            builder.ResizeRow(size_t(row), values.size());
            for (size_t slot = 0; slot < values.size(); ++slot)
                builder.Row<I>(size_t(row))[slot] = values[slot];
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }
    std::shared_ptr<const G::NativeMeshState> Fixture(const DNDS::MPIInfo &mpi, bool emptyRanks)
    {
        // Node, cell and boundary owners differ. The rank-zero-only variant
        // also exercises fully empty participants independently of MPI size.
        const bool colocated = emptyRanks || mpi.size == 1;
        const I nodeCount = mpi.rank == (colocated ? 0 : 1) ? 7 : 0;
        const I cellCount = colocated ? (mpi.rank == 0 ? 3 : 0) : mpi.rank == 0          ? 1
                                                              : mpi.rank == mpi.size - 1 ? 2
                                                                                         : 0;
        const I boundaryCount = mpi.rank == 0 ? 3 : 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodeCount);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, cellCount);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, boundaryCount);
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(cells, CellNodes), Rows(boundaries, BoundaryNodes), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodeCount), 3, 4)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cellCount), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(boundaryCount), 1)), {}};
        {
            auto coordinates = input.coordinates.Write();
            for (I row = 0; row < nodeCount; ++row)
                for (size_t component = 0; component < 3; ++component)
                    coordinates.View()(size_t(row), component) = Points[nodes.LocalToGlobal(row)][component];
            auto cellInfo = input.cellInfo.Write();
            for (I row = 0; row < cellCount; ++row)
                cellInfo.View()(size_t(row), 0) = G::ElemInfo{G::Elem::Tri3, G::t_index(100 + cells.LocalToGlobal(row))};
            auto boundaryInfo = input.boundaryInfo.Write();
            for (I row = 0; row < boundaryCount; ++row)
                boundaryInfo.View()(size_t(row), 0) = G::ElemInfo{G::Elem::Line2, G::t_index(20 + boundaries.LocalToGlobal(row))};
        }
        for (const auto *space : {&cells, &nodes, &boundaries})
        {
            R::Array<I> values(R::RowLayout::Dense(size_t(space->NumOwned()), 1));
            auto write = values.Write();
            for (I row = 0; row < space->NumOwned(); ++row)
                write.View()(size_t(row), 0) = PersistentId(space->Kind(), space->LocalToGlobal(row));
            input.ids.emplace(space->Kind(), std::move(values));
        }
        G::NativeMeshBuildOptions options;
        options.completeNodeStar = true;
        auto base = G::ConstructNativeMesh(std::move(input), options);
        auto owned = G::NativeMeshAssembler::CaptureOwned(*base, mpi);
        owned.identity.meshNamespace = Namespace;
        for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
            owned.identity.nextIds[kind] = HighWater(kind);
        owned.namedRelations.emplace(AnyKey(), Rows(owned.spaces.at(G::EntityKind::Cell), AnyRows));
        owned.namedRelations.emplace(AllKey(), Rows(owned.spaces.at(G::EntityKind::Cell), AllRows));
        return G::AssembleNativeMesh(mpi, std::move(owned), 2);
    }
    std::string FileName(const DNDS::MPIInfo &mpi, const std::string &label, const std::string &suffix)
    {
        int pid = getpid();
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() / ("dndsr-native-archive-" + std::to_string(pid) + "-" + label + suffix)).string();
    }
    std::vector<I> Decode(const G::EntitySpace &space, G::MeshRowSpan<I> entries)
    {
        std::vector<I> result;
        for (auto value : entries)
            result.push_back(value == DNDS::UnInitIndex ? value : value >= 0 ? space.LocalToGlobal(value)
                                                                             : -1 - value);
        return result;
    }
    void CheckState(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi,
                    const std::vector<G::AdjacencyDescriptor> &expectedKeys)
    {
        CHECK(state->MeshNamespace() == Namespace);
        CHECK(state->RelationKeys() == expectedKeys);
        const auto cells = state->Entities(G::EntityKind::Cell);
        const auto nodes = state->Entities(G::EntityKind::Node);
        const auto boundaries = state->Entities(G::EntityKind::Bnd);
        auto facade = G::UnstructuredMesh::FromNative(mpi, state);
        CHECK(facade->HasNativeState());
        auto scope = facade->AcquireRead(state->Requirements());
        const auto view = scope.HostView();
        for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
        {
            CHECK(state->NextEntityId(kind) == HighWater(kind));
            const auto space = state->Entities(kind);
            for (I row = 0; row < space.NumLocal(); ++row)
                CHECK(state->OriginalEntityId(kind, row) == PersistentId(kind, space.LocalToGlobal(row)));
        }
        for (I row = 0; row < nodes.NumLocal(); ++row)
        {
            const I global = nodes.LocalToGlobal(row);
            for (size_t component = 0; component < 3; ++component)
                CHECK(view.Coord(row)[component] == Points[global][component]);
            CHECK(Decode(cells, view.NodeCells(row)) == NodeCells.at(size_t(global)));
            CHECK(Decode(boundaries, view.NodeBoundaries(row)) == NodeBoundaries.at(size_t(global)));
        }
        for (I row = 0; row < cells.NumLocal(); ++row)
        {
            const I global = cells.LocalToGlobal(row);
            CHECK(Decode(nodes, view.CellNodes(row)) == CellNodes.at(size_t(global)));
            CHECK(Decode(cells, view.CellCells(row)) == CellNeighbors.at(size_t(global)));
            CHECK(view.GetCellInfo(row).getElemType() == G::Elem::Tri3);
            CHECK(view.GetCellInfo(row).zone == 100 + global);
        }
        for (I row = 0; row < boundaries.NumLocal(); ++row)
        {
            const I global = boundaries.LocalToGlobal(row);
            CHECK(Decode(nodes, view.BoundaryNodes(row)) == BoundaryNodes.at(size_t(global)));
            CHECK(Decode(cells, view.BoundaryCells(row)) == std::vector<I>{global, DNDS::UnInitIndex});
            CHECK(view.GetBndInfo(row).getElemType() == G::Elem::Line2);
            CHECK(view.GetBndInfo(row).zone == 20 + global);
        }
        for (const auto &[key, expected] : std::vector<std::pair<G::AdjacencyDescriptor, const std::vector<std::vector<I>> *>>{
                 {AnyKey(), &AnyRows}, {AllKey(), &AllRows}})
        {
            auto relation = state->Relation(key);
            for (I row = 0; row < relation.RowCount(); ++row)
                CHECK(relation.GlobalRow(row) == expected->at(size_t(cells.LocalToGlobal(row))));
        }
        // Reconstructed distributions must drive a real owner-to-ghost field
        // exchange and node-cone stencil, independently of the archive codec.
        R::Array<DNDS::real> values(R::RowLayout::Dense(size_t(nodes.NumOwned()), 1));
        {
            auto write = values.Write();
            for (I row = 0; row < nodes.NumOwned(); ++row)
            {
                const I global = nodes.LocalToGlobal(row);
                write.View()(size_t(row), 0) = 5 + Points[global][0] + 2 * Points[global][1];
            }
        }
        G::EntityField<DNDS::real> field(nodes, values);
        auto route = field.BuildHaloRoute(nodes);
        auto binding = route.Bind(field, nodes);
        binding.Begin().Wait();
        auto owned = field.Read(nodes);
        auto ghost = binding.GhostValues().Read();
        const DNDS::real expectedStencils[]{18, 21, 27};
        for (I row = 0; row < cells.NumOwned(); ++row)
        {
            DNDS::real sum = 0;
            for (auto node : view.CellNodes(row))
                sum += node < nodes.NumOwned() ? owned.View()(size_t(node), 0) : ghost.View()(size_t(node - nodes.NumOwned()), 0);
            CHECK(sum == expectedStencils[cells.LocalToGlobal(row)]);
        }
    }

    const std::vector<std::vector<I>> TetFaces{{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
    const std::vector<std::vector<I>> TetEdges{{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
    const DNDS::real TetPoints[4][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    template <class T, class F>
    R::Array<T> Table(const G::EntitySpace &space, size_t width, F &&value)
    {
        R::Array<T> result(R::RowLayout::Dense(size_t(space.NumOwned()), width));
        auto write = result.Write();
        for (I row = 0; row < space.NumOwned(); ++row)
            for (size_t component = 0; component < width; ++component)
                write.View()(size_t(row), component) = value(space.LocalToGlobal(row), component);
        return result;
    }
    std::shared_ptr<const G::NativeMeshState> Tetrahedron(const DNDS::MPIInfo &mpi)
    {
        G::NativeOwnedMeshData data;
        data.dimension = 3;
        data.periodic = true;
        data.requirements = {true, true, true, true, true, true};
        data.identity.meshNamespace = Namespace;
        // One independent tetrahedron per writer rank, but every node/face/
        // edge/boundary is owned by the next rank. All five families therefore
        // need real companion exchanges while element slot numbers stay fixed.
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 4);
        data.spaces.emplace(G::EntityKind::Node, nodes);
        data.spaces.emplace(G::EntityKind::Cell, nodes.CreateSibling(G::EntityKind::Cell, 1));
        data.spaces.emplace(G::EntityKind::Bnd, nodes.CreateSibling(G::EntityKind::Bnd, 4));
        data.spaces.emplace(G::EntityKind::Face, nodes.CreateSibling(G::EntityKind::Face, 4));
        data.spaces.emplace(G::EntityKind::Edge, nodes.CreateSibling(G::EntityKind::Edge, 6));
        const auto &cells = data.spaces.at(G::EntityKind::Cell);
        const auto &boundaries = data.spaces.at(G::EntityKind::Bnd);
        const auto &faces = data.spaces.at(G::EntityKind::Face);
        const auto &edges = data.spaces.at(G::EntityKind::Edge);
        auto ownerCell = [&](I group)
        { return (group + mpi.size - 1) % mpi.size; };
        auto nodeRows = [&](const G::EntitySpace &space, const auto &patterns)
        {
            R::RowBuilder builder(size_t(space.NumOwned()), sizeof(I));
            for (I row = 0; row < space.NumOwned(); ++row)
            {
                const I global = space.LocalToGlobal(row), group = global / I(patterns.size());
                const auto &pattern = patterns[size_t(global % I(patterns.size()))];
                builder.ResizeRow(size_t(row), pattern.size());
                for (size_t slot = 0; slot < pattern.size(); ++slot)
                    builder.Row<I>(size_t(row))[slot] = 4 * group + pattern[slot];
            }
            return R::Array<I>::FromSharedStorage(builder.Commit());
        };
        auto set = [&](G::MeshRelation slot, R::Array<I> rows)
        { data.relations[size_t(slot)] = std::move(rows); };
        set(G::MeshRelation::CellNodes, Table<I>(cells, 4, [&](I cell, size_t slot)
                                                 { return 4 * ((cell + 1) % mpi.size) + I(slot); }));
        set(G::MeshRelation::CellCells, Rows(cells, std::vector<std::vector<I>>(size_t(mpi.size))));
        set(G::MeshRelation::BoundaryNodes, nodeRows(boundaries, TetFaces));
        set(G::MeshRelation::BoundaryCells, Table<I>(boundaries, 2, [&](I bnd, size_t slot)
                                                     { return slot ? DNDS::UnInitIndex : ownerCell(bnd / 4); }));
        set(G::MeshRelation::CellFaces, Table<I>(cells, 4, [&](I cell, size_t slot)
                                                 { return 4 * ((cell + 1) % mpi.size) + I(slot); }));
        set(G::MeshRelation::FaceNodes, nodeRows(faces, TetFaces));
        set(G::MeshRelation::FaceCells, Table<I>(faces, 2, [&](I face, size_t slot)
                                                 { return slot ? DNDS::UnInitIndex : ownerCell(face / 4); }));
        set(G::MeshRelation::FaceBoundaries, Table<I>(faces, 1, [](I face, size_t)
                                                      { return face; }));
        set(G::MeshRelation::BoundaryFaces, Table<I>(boundaries, 1, [](I bnd, size_t)
                                                     { return bnd; }));
        set(G::MeshRelation::CellEdges, Table<I>(cells, 6, [&](I cell, size_t slot)
                                                 { return 6 * ((cell + 1) % mpi.size) + I(slot); }));
        set(G::MeshRelation::EdgeNodes, nodeRows(edges, TetEdges));
        set(G::MeshRelation::EdgeCells, Table<I>(edges, 1, [&](I edge, size_t)
                                                 { return ownerCell(edge / 6); }));
        set(G::MeshRelation::CellFaceCells, Table<I>(cells, 4, [](I, size_t)
                                                     { return DNDS::UnInitIndex; }));
        set(G::MeshRelation::NodeCells, Table<I>(nodes, 1, [&](I node, size_t)
                                                 { return ownerCell(node / 4); }));
        {
            R::RowBuilder builder(4, sizeof(I));
            for (I row = 0; row < 4; ++row)
            {
                const I global = nodes.LocalToGlobal(row), group = global / 4, vertex = global % 4;
                std::vector<I> support;
                for (size_t face = 0; face < TetFaces.size(); ++face)
                    if (std::find(TetFaces[face].begin(), TetFaces[face].end(), vertex) != TetFaces[face].end())
                        support.push_back(4 * group + I(face));
                builder.ResizeRow(size_t(row), support.size());
                for (size_t slot = 0; slot < support.size(); ++slot)
                    builder.Row<I>(size_t(row))[slot] = support[slot];
            }
            set(G::MeshRelation::NodeBoundaries, R::Array<I>::FromSharedStorage(builder.Commit()));
        }
        data.coords = Table<DNDS::real>(nodes, 3, [](I node, size_t component)
                                        { return TetPoints[node % 4][component] + (component == 0 ? 10 * (node / 4) : 0); });
        data.nodeWallDist = Table<DNDS::real>(nodes, 3, [](I node, size_t component)
                                              { return DNDS::real(100 + 3 * node + I(component)); });
        data.cellInfo = Table<G::ElemInfo>(cells, 1, [](I cell, size_t)
                                           { return G::ElemInfo{G::Elem::Tet4, G::t_index(100 + cell)}; });
        data.bndInfo = Table<G::ElemInfo>(boundaries, 1, [](I bnd, size_t)
                                          { return G::ElemInfo{G::Elem::Tri3, G::t_index(20 + bnd)}; });
        data.faceInfo = Table<G::ElemInfo>(faces, 1, [](I face, size_t)
                                           { return G::ElemInfo{G::Elem::Tri3, G::t_index(20 + face)}; });
        data.edgeInfo = Table<G::ElemInfo>(edges, 1, [](I edge, size_t)
                                           { return G::ElemInfo{G::Elem::Line2, G::t_index(40 + edge)}; });
        // Use one consistent nonzero node frame on every cone; incidence-frame
        // companions are zero because child and parent use the same frame.
        data.cellNodesPbi = Table<G::NodePeriodicBits>(cells, 4, [](I, size_t)
                                                       { return G::NodePeriodicBits{1}; });
        data.bndNodesPbi = Table<G::NodePeriodicBits>(boundaries, 3, [](I, size_t)
                                                      { return G::NodePeriodicBits{1}; });
        data.faceNodesPbi = Table<G::NodePeriodicBits>(faces, 3, [](I, size_t)
                                                       { return G::NodePeriodicBits{1}; });
        data.edgeNodesPbi = Table<G::NodePeriodicBits>(edges, 2, [](I, size_t)
                                                       { return G::NodePeriodicBits{1}; });
        data.cellFacesPbi = Table<G::NodePeriodicBits>(cells, 4, [](I, size_t)
                                                       { return G::NodePeriodicBits{0}; });
        data.cellEdgesPbi = Table<G::NodePeriodicBits>(cells, 6, [](I, size_t)
                                                       { return G::NodePeriodicBits{0}; });
        data.periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        data.periodicInfo.translation[1].map() = G::tPoint{0.25, 0.5, 0.75};
        data.periodicInfo.rotationCenter[1].map() = G::tPoint{1, 2, 3};
        for (const auto &[kind, space] : data.spaces)
        {
            data.ids.emplace(kind, Table<I>(space, 1, [&](I global, size_t)
                                            { return PersistentId(kind, global); }));
            data.identity.nextIds[kind] = HighWater(kind);
        }
        return G::AssembleNativeMesh(mpi, std::move(data), 1);
    }
    void CheckTetrahedron(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi, I totalCells)
    {
        CHECK(state->MeshNamespace() == Namespace);
        const auto cells = state->Entities(G::EntityKind::Cell), nodes = state->Entities(G::EntityKind::Node);
        const auto faces = state->Entities(G::EntityKind::Face), edges = state->Entities(G::EntityKind::Edge);
        const auto boundaries = state->Entities(G::EntityKind::Bnd);
        auto read = state->Snapshot(R::Device::Host()).AcquireRead();
        const auto view = read.HostView();
        CHECK(view.Dimension() == 3);
        CHECK(view.IsPeriodic());
        CHECK(view.HasNodeWallDist());
        for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
        {
            const auto space = state->Entities(kind);
            CHECK(state->NextEntityId(kind) == HighWater(kind));
            for (I row = 0; row < space.NumLocal(); ++row)
                CHECK(state->OriginalEntityId(kind, row) == PersistentId(kind, space.LocalToGlobal(row)));
        }
        CHECK(view.Periodic().rotation[1].map()(0, 1) == -1);
        CHECK(view.Periodic().rotation[1].map()(1, 0) == 1);
        CHECK(view.Periodic().translation[1].map()[2] == 0.75);
        CHECK(view.Periodic().rotationCenter[1].map()[1] == 2);
        for (I row = 0; row < nodes.NumLocal(); ++row)
        {
            const I global = nodes.LocalToGlobal(row);
            for (size_t component = 0; component < 3; ++component)
            {
                CHECK(view.Coord(row)[component] == TetPoints[global % 4][component] + (component == 0 ? 10 * (global / 4) : 0));
                CHECK(view.NodeWallDist(row)[component] == 100 + 3 * global + I(component));
            }
            CHECK(Decode(cells, view.NodeCells(row)) == std::vector<I>{(global / 4 + totalCells - 1) % totalCells});
            std::vector<I> expectedBoundaries;
            for (size_t face = 0; face < TetFaces.size(); ++face)
                if (std::find(TetFaces[face].begin(), TetFaces[face].end(), global % 4) != TetFaces[face].end())
                    expectedBoundaries.push_back(4 * (global / 4) + I(face));
            CHECK(Decode(boundaries, view.NodeBoundaries(row)) == expectedBoundaries);
        }
        auto checkNodes = [&](const G::EntitySpace &space, bool boundary)
        {
            for (I row = 0; row < space.NumLocal(); ++row)
            {
                const I global = space.LocalToGlobal(row), group = global / 4;
                auto expected = TetFaces[size_t(global % 4)];
                for (auto &value : expected)
                    value += 4 * group;
                CHECK(Decode(nodes, boundary ? view.BoundaryNodes(row) : view.FaceNodes(row)) == expected);
                const auto bits = boundary ? view.BoundaryNodePeriodicBits(row) : view.FaceNodePeriodicBits(row);
                for (auto bit : bits)
                    CHECK(uint8_t(bit) == 1);
                CHECK(Decode(cells, boundary ? view.BoundaryCells(row) : view.FaceCells(row)) == std::vector<I>{(group + totalCells - 1) % totalCells, DNDS::UnInitIndex});
            }
        };
        checkNodes(boundaries, true);
        checkNodes(faces, false);
        for (I row = 0; row < faces.NumLocal(); ++row)
        {
            const I global = faces.LocalToGlobal(row);
            CHECK(Decode(boundaries, view.FaceBoundaries(row)) == std::vector<I>{global});
            CHECK(view.GetFaceInfo(row).getElemType() == G::Elem::Tri3);
            CHECK(view.GetFaceInfo(row).zone == 20 + global);
        }
        for (I row = 0; row < boundaries.NumLocal(); ++row)
        {
            CHECK(Decode(faces, view.BoundaryFaces(row)) == std::vector<I>{boundaries.LocalToGlobal(row)});
            CHECK(view.GetBndInfo(row).getElemType() == G::Elem::Tri3);
            CHECK(view.GetBndInfo(row).zone == 20 + boundaries.LocalToGlobal(row));
        }
        for (I row = 0; row < edges.NumLocal(); ++row)
        {
            const I global = edges.LocalToGlobal(row), group = global / 6;
            auto expected = TetEdges[size_t(global % 6)];
            for (auto &value : expected)
                value += 4 * group;
            CHECK(Decode(nodes, view.EdgeNodes(row)) == expected);
            CHECK(Decode(cells, view.EdgeCells(row)) == std::vector<I>{(group + totalCells - 1) % totalCells});
            CHECK(view.GetEdgeInfo(row).getElemType() == G::Elem::Line2);
            CHECK(view.GetEdgeInfo(row).zone == 40 + global);
            for (auto bit : view.EdgeNodePeriodicBits(row))
                CHECK(uint8_t(bit) == 1);
        }
        for (I row = 0; row < cells.NumLocal(); ++row)
        {
            const I global = cells.LocalToGlobal(row), group = (global + 1) % totalCells;
            CHECK(view.GetCellInfo(row).getElemType() == G::Elem::Tet4);
            CHECK(view.GetCellInfo(row).zone == 100 + global);
            CHECK(Decode(nodes, view.CellNodes(row)) == std::vector<I>{4 * group, 4 * group + 1, 4 * group + 2, 4 * group + 3});
            CHECK(Decode(faces, view.CellFaces(row)) == std::vector<I>{4 * group, 4 * group + 1, 4 * group + 2, 4 * group + 3});
            CHECK(Decode(edges, view.CellEdges(row)) == std::vector<I>{6 * group, 6 * group + 1, 6 * group + 2, 6 * group + 3, 6 * group + 4, 6 * group + 5});
            CHECK(view.CellCells(row).empty());
            for (auto other : view.CellFaceCells(row))
                CHECK(other == DNDS::UnInitIndex);
            for (auto bit : view.CellNodePeriodicBits(row))
                CHECK(uint8_t(bit) == 1);
            for (auto bit : view.CellFacePeriodicBits(row))
                CHECK(uint8_t(bit) == 0);
            for (auto bit : view.CellEdgePeriodicBits(row))
                CHECK(uint8_t(bit) == 0);
            G::MeshQueryCoords physical;
            view.GetCoordsOnCell(row, physical);
            for (size_t slot = 0; slot < 4; ++slot)
            {
                const DNDS::real x = TetPoints[slot][0] + 10 * group, y = TetPoints[slot][1], z = TetPoints[slot][2];
                CHECK(physical(0, slot) == 3.25 - y);
                CHECK(physical(1, slot) == x + 1.5);
                CHECK(physical(2, slot) == z + 0.75);
                const I node = 4 * group + I(slot);
                const auto wall = view.GetCoordWallDistOnCell(row, DNDS::rowsize(slot));
                CHECK(wall[0] == -(101 + 3 * node));
                CHECK(wall[1] == 100 + 3 * node);
                CHECK(wall[2] == 102 + 3 * node);
            }
        }
        auto facade = G::UnstructuredMesh::FromNative(mpi, state);
        CHECK(facade->HasNativeState());
    }
}

TEST_CASE("Native archive preserves a full custom ghost recipe across saved and even partitions")
{
    const auto mpi = MPI();
    auto owned = G::NativeMeshAssembler::CaptureOwned(*Fixture(mpi, false), mpi);
    REQUIRE(owned.ghostSpec);
    auto recipe = *owned.ghostSpec;
    recipe.chains.push_back({G::EntityKind::Cell, {AnyKey()}, G::EntityKind::Cell});
    auto source = G::NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    const auto file = FileName(mpi, "custom-policy", ".h5");
    auto serializer = std::make_shared<S::SerializerH5>(mpi);
    serializer->SetCollectiveRW(true, true);
    serializer->OpenFile(file, false);
    G::WriteNativeMeshArchive(*source, mpi, serializer, "mesh");
    serializer->CloseFile();
    for (auto partition : {S::ArrayGlobalOffset_Unknown, S::ArrayGlobalOffset_EvenSplit})
    {
        serializer->OpenFile(file, true);
        auto decoded = G::ReadNativeMeshArchiveOwned(mpi, serializer, "mesh", partition);
        serializer->CloseFile();
        CHECK_FALSE(decoded.retainedRead);
        for (const auto &[kind, space] : decoded.spaces)
            CHECK(space.NumGhost() == 0);
        REQUIRE(decoded.ghostSpec);
        CHECK(decoded.ghostSpec->chains.size() == recipe.chains.size());
        CHECK_NOTHROW(decoded.coords->Write());
        auto restored = G::AssembleNativeMesh(mpi, std::move(decoded), 2);
        REQUIRE(restored->GhostSpecification());
        const auto &actual = restored->GhostSpecification()->chains;
        REQUIRE(actual.size() == recipe.chains.size());
        for (size_t i = 0; i < actual.size(); ++i)
        {
            CHECK(actual[i].anchor == recipe.chains[i].anchor);
            CHECK(actual[i].target == recipe.chains[i].target);
            CHECK(actual[i].hops == recipe.chains[i].hops);
        }
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Native H5 archive preserves exact topology identity and stencil across saved and even partitions")
{
    const auto mpi = MPI();
    for (bool emptyRanks : {false, true})
    {
        const auto source = Fixture(mpi, emptyRanks);
        const auto keys = source->RelationKeys();
        const auto file = FileName(mpi, emptyRanks ? "empty" : "independent", ".h5");
        auto serializer = std::make_shared<S::SerializerH5>(mpi);
        serializer->SetCollectiveRW(true, true);
        serializer->OpenFile(file, false);
        G::WriteNativeMeshArchive(*source, mpi, serializer, "mesh");
        serializer->CloseFile();
        for (auto partition : {S::ArrayGlobalOffset_Unknown, S::ArrayGlobalOffset_EvenSplit})
        {
            serializer->OpenFile(file, true);
            auto restored = G::ReadNativeMeshArchive(mpi, serializer, "mesh", partition, 2);
            serializer->CloseFile();
            CheckState(restored, mpi, keys);
            for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
            {
                const auto range = DNDS::EvenSplitRange(mpi.rank, mpi.size, source->NumGlobal(kind));
                const I expected = partition == S::ArrayGlobalOffset_Unknown ? source->NumOwned(kind) : range.second - range.first;
                CHECK(restored->NumOwned(kind) == expected);
            }
        }
        serializer.reset();
        if (mpi.rank == 0)
        {
            const auto single = MPI(MPI_COMM_SELF);
            auto reader = std::make_shared<S::SerializerH5>(single);
            reader->OpenFile(file, true);
            auto restored = G::ReadNativeMeshArchive(single, reader, "mesh", S::ArrayGlobalOffset_EvenSplit, 2);
            CheckState(restored, single, keys);
            CHECK(restored->NumOwned(G::EntityKind::Cell) == 3);
            CHECK(restored->NumOwned(G::EntityKind::Node) == 7);
            CHECK(restored->NumGhost(G::EntityKind::Node) == 0);
            reader->CloseFile();
            if (mpi.size > 1)
            {
                reader->OpenFile(file, true);
                CHECK_THROWS(G::ReadNativeMeshArchive(single, reader, "mesh"));
                reader->CloseFile();
            }
        }
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
            std::filesystem::remove(file);
    }
}

TEST_CASE("Native per-rank JSON archive preserves matching topology and rejects incompatible collective read modes")
{
    const auto mpi = MPI();
    const auto source = Fixture(mpi, false);
    const auto file = FileName(mpi, "json", ".r" + std::to_string(mpi.rank) + ".json");
    auto serializer = std::make_shared<S::SerializerJSON>();
    serializer->OpenFile(file, false);
    G::WriteNativeMeshArchive(*source, mpi, serializer, "mesh");
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto restored = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    CheckState(restored, mpi, source->RelationKeys());
    serializer->OpenFile(file, true);
    CHECK_THROWS(G::ReadNativeMeshArchive(mpi, serializer, "mesh", S::ArrayGlobalOffset_EvenSplit));
    serializer->CloseFile();
    if (mpi.size > 1)
    {
        serializer->OpenFile(file, true);
        CHECK_THROWS(G::ReadNativeMeshArchive(mpi, serializer, "mesh", mpi.rank == 0 ? S::ArrayGlobalOffset_EvenSplit : S::ArrayGlobalOffset_Unknown));
        serializer->CloseFile();
    }
    nlohmann::json pristine;
    {
        std::ifstream input(file);
        input >> pristine;
    }
    // Version-one metadata predates saved coverage and must still use a
    // validated explicit default, without requiring a new-format field.
    {
        auto older = pristine;
        auto metadata = nlohmann::json::parse(older.at("mesh").at("metadata_json").get<std::string>());
        metadata.at("version") = 1;
        metadata.erase("ghost_spec");
        older.at("mesh").at("metadata_json") = metadata.dump();
        {
            std::ofstream output(file);
            output << older;
        }
        serializer->OpenFile(file, true);
        auto compatible = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
        serializer->CloseFile();
        CheckState(compatible, mpi, source->RelationKeys());
    }
    for (bool schema : {false, true})
    {
        auto corrupted = pristine;
        if (mpi.rank == 0)
        {
            if (schema)
                corrupted.at("mesh").at("ids_0").at("element_schema") = "wrong.native.ids.schema";
            else
            {
                auto metadata = nlohmann::json::parse(corrupted.at("mesh").at("metadata_json").get<std::string>());
                metadata.at("version") = 999;
                corrupted.at("mesh").at("metadata_json") = metadata.dump();
            }
        }
        {
            std::ofstream output(file);
            output << corrupted;
        }
        serializer->OpenFile(file, true);
        CHECK_THROWS(G::ReadNativeMeshArchive(mpi, serializer, "mesh"));
        serializer->CloseFile();
        {
            std::ofstream output(file);
            output << pristine;
        }
        serializer->OpenFile(file, true);
        auto retry = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
        serializer->CloseFile();
        CheckState(retry, mpi, source->RelationKeys());
    }
    // Identical malformed metadata on all ranks passes collective agreement;
    // parsing must still reject lossy numeric conversions and invalid identity.
    for (const std::string fault : {"version_float", "family_kind_float", "next_id_float", "fixed_width_float",
                                    "named_width_float", "companion_width_float", "writer_count", "namespace", "next_id_low",
                                    "writer_rows_length", "writer_rows_sum", "writer_rows_ownership",
                                    "ghost_policy_missing", "ghost_policy_type", "ghost_chain_empty", "ghost_chain_float"})
    {
        if (fault == "writer_rows_ownership" && mpi.size == 1)
            continue;
        INFO(fault);
        auto corrupted = pristine;
        auto metadata = nlohmann::json::parse(corrupted.at("mesh").at("metadata_json").get<std::string>());
        if (fault == "version_float")
            metadata.at("version") = 1.0;
        if (fault == "ghost_policy_missing")
            metadata.erase("ghost_spec");
        if (fault == "ghost_policy_type")
            metadata.at("ghost_spec") = nlohmann::json::object();
        if (fault == "ghost_chain_empty" || fault == "ghost_chain_float")
            metadata.at("ghost_spec") = nlohmann::json::array({{{"anchor", fault == "ghost_chain_float" ? nlohmann::json(0.5) : nlohmann::json(int(G::EntityKind::Cell))},
                                                                {"target", int(G::EntityKind::Node)},
                                                                {"hops", nlohmann::json::array()}}});
        if (fault == "family_kind_float")
            metadata.at("families").at(0).at("kind") = 0.5;
        if (fault == "next_id_float")
            metadata.at("families").at(0).at("next_id") = double(HighWater(G::EntityKind::Cell)) + 0.5;
        if (fault == "fixed_width_float")
            metadata.at("fixed").at(size_t(G::MeshRelation::BoundaryCells)) = 2.5;
        if (fault == "named_width_float")
            metadata.at("named").at(0).at("width") = -1.5;
        if (fault == "companion_width_float")
            metadata.at("companions").at("coordinates").at("width") = 3.5;
        if (fault == "writer_count")
            metadata.at("writer_ranks") = 0;
        if (fault == "namespace")
            metadata.at("namespace") = "invalid-mesh-namespace";
        if (fault == "next_id_low")
            metadata.at("families").at(0).at("next_id") = 1;
        if (fault == "writer_rows_length")
            metadata.at("families").at(0).at("writer_rows").erase(0);
        if (fault == "writer_rows_sum")
        {
            auto &rows = metadata.at("families").at(0).at("writer_rows");
            rows.at(0) = rows.at(0).get<I>() + 1;
        }
        if (fault == "writer_rows_ownership")
        {
            auto &rows = metadata.at("families").at(0).at("writer_rows");
            rows.at(0) = rows.at(0).get<I>() - 1;
            rows.at(1) = rows.at(1).get<I>() + 1;
        }
        corrupted.at("mesh").at("metadata_json") = metadata.dump();
        {
            std::ofstream output(file);
            output << corrupted;
        }
        serializer->OpenFile(file, true);
        CHECK_THROWS(G::ReadNativeMeshArchive(mpi, serializer, "mesh"));
        serializer->CloseFile();
    }
    {
        std::ofstream output(file);
        output << pristine;
    }
    serializer->OpenFile(file, true);
    auto finalRetry = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    CheckState(finalRetry, mpi, source->RelationKeys());
    std::filesystem::remove(file);
}

TEST_CASE("Native archive preserves all five entity families periodic frames and wall vectors without legacy storage")
{
    const auto mpi = MPI();
    const auto source = Tetrahedron(mpi);
    CheckTetrahedron(source, mpi, mpi.size);
    const auto file = FileName(mpi, "tetrahedron", ".h5");
    auto serializer = std::make_shared<S::SerializerH5>(mpi);
    serializer->SetCollectiveRW(true, true);
    serializer->OpenFile(file, false);
    G::WriteNativeMeshArchive(*source, mpi, serializer, "mesh");
    serializer->CloseFile();
    for (auto partition : {S::ArrayGlobalOffset_Unknown, S::ArrayGlobalOffset_EvenSplit})
    {
        serializer->OpenFile(file, true);
        const auto restored = G::ReadNativeMeshArchive(mpi, serializer, "mesh", partition);
        serializer->CloseFile();
        CheckTetrahedron(restored, mpi, mpi.size);
        if (partition == S::ArrayGlobalOffset_Unknown)
        {
            // Route every family through a different owner after restart,
            // then return it to the original rank/order. Exact slot, ID,
            // frame and wall-vector checks below cover the resulting archive.
            G::MeshTransitionHandle handle(G::UnstructuredMesh::FromNative(mpi, restored));
            R::Array<I> tags(R::RowLayout::Dense(1, 1));
            tags.Write().View()(0, 0) = PersistentId(G::EntityKind::Cell, mpi.rank);
            G::EntityField<I> taggedCells(handle.Get()->Entities(G::EntityKind::Cell), tags);
            auto redistribute = [&](int direction)
            {
                G::ReorderInput input;
                for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
                {
                    const auto space = handle.Get()->Entities(kind);
                    const int target = (mpi.rank + direction + mpi.size) % mpi.size;
                    input.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(space.NumOwned()), target)});
                }
                auto transition = G::PreparedMeshTransition::Prepare(handle, input, {taggedCells.Register("archive.cell.tag")});
                transition.Publish();
                const I original = direction > 0 ? (mpi.rank + mpi.size - 1) % mpi.size : mpi.rank;
                CHECK(handle.Get()->OriginalEntityId(G::EntityKind::Cell, 0) == PersistentId(G::EntityKind::Cell, original));
                CHECK(taggedCells.Read(handle.Get()->Entities(G::EntityKind::Cell)).View()(0, 0) == PersistentId(G::EntityKind::Cell, original));
                auto scope = handle.Get()->AcquireRead(handle.Get()->CurrentNativeRequirements());
                const auto view = scope.HostView();
                for (size_t slot = 0; slot < 4; ++slot)
                    CHECK(handle.Get()->OriginalEntityId(G::EntityKind::Node, view.CellNodes(0)[slot]) ==
                          PersistentId(G::EntityKind::Node, 4 * ((original + 1) % mpi.size) + I(slot)));
            };
            redistribute(1);
            CHECK(handle.Get()->HasNativeState());
            redistribute(-1);
            const auto movedFile = file + ".redistributed.h5";
            auto output = std::make_shared<S::SerializerH5>(mpi);
            output->SetCollectiveRW(true, true);
            output->OpenFile(movedFile, false);
            handle.Get()->WriteNativeArchive(output, "mesh");
            output->CloseFile();
            output->OpenFile(movedFile, true);
            auto roundtrip = G::ReadNativeMeshArchive(mpi, output, "mesh");
            output->CloseFile();
            CheckTetrahedron(roundtrip, mpi, mpi.size);
            output.reset();
            MPI_Barrier(mpi.comm);
            if (mpi.rank == 0)
                std::filesystem::remove(movedFile);
        }
    }
    serializer.reset();
    if (mpi.rank == 0)
    {
        const auto single = MPI(MPI_COMM_SELF);
        auto reader = std::make_shared<S::SerializerH5>(single);
        reader->OpenFile(file, true);
        const auto restored = G::ReadNativeMeshArchive(single, reader, "mesh", S::ArrayGlobalOffset_EvenSplit);
        reader->CloseFile();
        CheckTetrahedron(restored, single, mpi.size);
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
