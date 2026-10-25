#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/RelationAccess.hpp"
#include "Geom/Mesh/NativeMeshClosure.hpp"
#include "Geom/Mesh/NativeMeshArchive.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <filesystem>
#include <unistd.h>
#include "DNDS/Runtime/RowBuilder.hpp"
#include <map>
#include <set>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    DNDS::MPIInfo MPI()
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        return mpi;
    }
    R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(I));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            for (size_t slot = 0; slot < rows[row].size(); ++slot)
                builder.Row<I>(row)[slot] = rows[row][slot];
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }
    std::shared_ptr<const G::NativeMeshState> Fixture(const DNDS::MPIInfo &mpi, int dimension, bool duplicateBoundary = false, bool nonmanifold = false)
    {
        const I nodeCount = mpi.rank == 0 ? (dimension == 3 || nonmanifold ? 5 : 4) : 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodeCount);
        std::vector<std::vector<I>> cellRows, boundaryRows;
        if (nonmanifold)
        {
            if (mpi.rank == 0)
                cellRows = {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}};
        }
        else
        {
            if (mpi.rank == 0)
                cellRows.push_back(dimension == 3 ? std::vector<I>{0, 1, 2, 3} : std::vector<I>{0, 1, 2});
            if (mpi.rank == (mpi.size > 1 ? 1 : 0))
                cellRows.push_back(dimension == 3 ? std::vector<I>{0, 2, 1, 4} : std::vector<I>{1, 3, 2});
            if (dimension == 2 && mpi.rank == mpi.size - 1)
            {
                boundaryRows = {{0, 1}, {1, 3}, {3, 2}, {2, 0}};
                if (duplicateBoundary)
                    boundaryRows.push_back({1, 0});
            }
        }
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(cellRows.size()));
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, I(boundaryRows.size()));
        G::NativeMeshInput input{mpi, dimension, cells, nodes, boundaries, Rows(cellRows), Rows(boundaryRows), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodeCount), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(cellRows.size(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaryRows.size(), 1)), {}};
        {
            auto coords = input.coordinates.Write();
            const DNDS::real points[5][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, -1, 0}};
            for (size_t row = 0; row < size_t(nodeCount); ++row)
                for (size_t component = 0; component < 3; ++component)
                    coords.View()(row, component) = points[row][component];
            if (dimension == 3 && nodeCount)
            {
                coords.View()(3, 0) = 0;
                coords.View()(3, 1) = 0;
                coords.View()(3, 2) = 1;
                coords.View()(4, 0) = 0;
                coords.View()(4, 1) = 0;
                coords.View()(4, 2) = -1;
            }
            auto info = input.cellInfo.Write();
            for (size_t row = 0; row < cellRows.size(); ++row)
                info.View()(row, 0) = G::ElemInfo{dimension == 3 ? G::Elem::Tet4 : G::Elem::Tri3, G::INTERNAL_ZONE};
            auto boundary = input.boundaryInfo.Write();
            for (size_t row = 0; row < boundaryRows.size(); ++row)
                boundary.View()(row, 0) = G::ElemInfo{G::Elem::Line2, G::t_index(2 + row)};
        }
        G::NativeMeshBuildOptions options;
        options.completeNodeStar = true;
        return G::NativeMeshBuilder::Build(std::move(input), options);
    }
    std::shared_ptr<const G::NativeMeshState> Truncated(const DNDS::MPIInfo &mpi)
    {
        auto original = Fixture(mpi, 2);
        auto owned = G::NativeMeshAssembler::CaptureOwned(*original, mpi);
        const auto cells = owned.spaces.at(G::EntityKind::Cell);
        auto nodes = owned.spaces.at(G::EntityKind::Node);
        auto globalNodes = owned.relations[size_t(G::MeshRelation::CellNodes)]->Read();
        std::set<I> remote;
        for (I cell = 0; cell < cells.NumOwned(); ++cell)
            for (auto global : globalNodes.View()[size_t(cell)])
                if (nodes.Owner(global) != mpi.rank)
                    remote.insert(global);
        nodes = nodes.WithGhosts({remote.begin(), remote.end()});
        G::NativeMeshData data;
        data.dimension = 2;
        data.identity = owned.identity;
        data.coords = R::ArrayPair<DNDS::real>{*owned.coords, R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumGhost()), 3))};
        {
            auto write = data.coords->ghost.Write();
            const DNDS::real points[4][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
            for (I row = 0; row < nodes.NumGhost(); ++row)
                for (size_t component = 0; component < 3; ++component)
                    write.View()(size_t(row), component) = points[nodes.LocalToGlobal(nodes.NumOwned() + row)][component];
        }
        data.cellInfo = R::ArrayPair<G::ElemInfo>{*owned.cellInfo, R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        data.bndInfo = R::ArrayPair<G::ElemInfo>{R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        std::vector<std::vector<I>> local(size_t(cells.NumOwned())), empty(size_t(cells.NumOwned()));
        for (I cell = 0; cell < cells.NumOwned(); ++cell)
            for (auto global : globalNodes.View()[size_t(cell)])
                local[size_t(cell)].push_back(*nodes.GlobalToLocal(global));
        data.relations[size_t(G::MeshRelation::CellNodes)] = R::ArrayPair<I>{Rows(local), Rows({})};
        data.relations[size_t(G::MeshRelation::CellCells)] = R::ArrayPair<I>{Rows(empty), Rows({})};
        data.relations[size_t(G::MeshRelation::BoundaryNodes)] = R::ArrayPair<I>{Rows({}), Rows({})};
        data.relations[size_t(G::MeshRelation::BoundaryCells)] = R::ArrayPair<I>{R::Array<I>(R::RowLayout::Dense(0, 2)), R::Array<I>(R::RowLayout::Dense(0, 2))};
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        std::map<G::EntityKind, G::EntitySpace> spaces{{G::EntityKind::Cell, cells}, {G::EntityKind::Node, nodes}, {G::EntityKind::Bnd, boundaries}};
        std::map<G::EntityKind, R::Array<I>> ids;
        for (const auto &[kind, space] : spaces)
        {
            R::Array<I> values(R::RowLayout::Dense(size_t(space.NumLocal()), 1));
            {
                auto write = values.Write();
                for (I row = 0; row < space.NumLocal(); ++row)
                    write.View()(size_t(row), 0) = space.LocalToGlobal(row);
            }
            ids.emplace(kind, std::move(values));
        }
        return G::NativeMeshState::FromNative(mpi, std::move(data), std::move(spaces), std::move(ids), G::MeshReadRequirements::Primary());
    }

    using Oracle = std::map<std::pair<int, I>, std::vector<I>>;
    Oracle PersistentTopology(const G::NativeMeshState &state, const DNDS::MPIInfo &mpi)
    {
        auto read = state.Snapshot(R::Device::Host()).AcquireRead();
        const auto view = read.HostView();
        std::vector<I> packed;
        for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Face, G::EntityKind::Edge})
            for (I row = 0; row < state.NumOwned(kind); ++row)
            {
                std::vector<I> values;
                auto append = [&](G::MeshRowSpan<I> entries, G::EntityKind target)
                {
                    values.push_back(I(entries.size()));
                    for (auto local : entries)
                    {
                        DNDS_check_throw_info(local == DNDS::UnInitIndex || local >= 0, "test oracle requires complete owned topology");
                        values.push_back(local == DNDS::UnInitIndex ? local : state.OriginalEntityId(target, local));
                    }
                };
                if (kind == G::EntityKind::Node)
                    for (int component = 0; component < 3; ++component)
                        values.push_back(I(view.Coord(row)[component]));
                else if (kind == G::EntityKind::Cell)
                {
                    values = {view.GetCellInfo(row).type, view.GetCellInfo(row).zone};
                    append(view.CellNodes(row), G::EntityKind::Node);
                    append(view.CellFaces(row), G::EntityKind::Face);
                    append(view.CellFaceCells(row), G::EntityKind::Cell);
                }
                else if (kind == G::EntityKind::Face)
                {
                    values = {view.GetFaceInfo(row).type, view.GetFaceInfo(row).zone};
                    append(view.FaceNodes(row), G::EntityKind::Node);
                    append(view.FaceCells(row), G::EntityKind::Cell);
                }
                else
                {
                    values = {view.GetEdgeInfo(row).type, view.GetEdgeInfo(row).zone};
                    append(view.EdgeNodes(row), G::EntityKind::Node);
                    append(view.EdgeCells(row), G::EntityKind::Cell);
                }
                packed.insert(packed.end(), {int(kind), state.OriginalEntityId(kind, row), I(values.size())});
                packed.insert(packed.end(), values.begin(), values.end());
            }
        // This fixed tiny fixture oracle gathers records only in the test. The
        // production algorithms never gather global topology or allocate by IDs.
        const int local = int(packed.size());
        std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size));
        MPI_Allgather(&local, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
        int total = 0;
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            offsets[size_t(peer)] = total;
            total += counts[size_t(peer)];
        }
        std::vector<I> global(size_t(total), 0);
        MPI_Allgatherv(packed.data(), local, DNDS::DNDS_MPI_INDEX, global.data(), counts.data(), offsets.data(), DNDS::DNDS_MPI_INDEX, mpi.comm);
        Oracle result;
        for (size_t position = 0; position < global.size();)
        {
            const auto kind = int(global[position++]);
            const auto id = global[position++];
            const auto length = size_t(global[position++]);
            auto inserted = result.emplace(std::make_pair(kind, id), std::vector<I>(global.begin() + position, global.begin() + position + length));
            CHECK(inserted.second);
            position += length;
        }
        return result;
    }

    std::set<I> NodeKey(const G::EntitySpace &nodes, G::MeshRowSpan<I> row)
    {
        std::set<I> result;
        for (auto local : row)
            result.insert(nodes.LocalToGlobal(local));
        return result;
    }
}

TEST_CASE("Native face completion matches independent topology with remote boundary ownership and empty ranks")
{
    const auto mpi = MPI();
    auto primary = Fixture(mpi, 2);
    const auto originalNamespace = primary->MeshNamespace();
    auto complete = G::NativeMeshTopology::Build(*primary, mpi);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 5);
    CHECK(complete->MeshNamespace() == originalNamespace);
    CHECK(complete->NextEntityId(G::EntityKind::Face) == 5);
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    const auto nodes = complete->Entities(G::EntityKind::Node), cells = complete->Entities(G::EntityKind::Cell);
    std::map<std::set<I>, std::vector<I>> expected{{{0, 1}, {0}}, {{0, 2}, {0}}, {{1, 2}, {0, 1}}, {{1, 3}, {1}}, {{2, 3}, {1}}};
    std::map<std::set<I>, I> boundaries{{{0, 1}, 0}, {{1, 3}, 1}, {{2, 3}, 2}, {{0, 2}, 3}};
    const std::vector<std::vector<std::vector<I>>> orderedFaces{{{0, 1}, {1, 2}, {2, 0}}, {{1, 3}, {3, 2}, {2, 1}}};
    auto support = complete->Relation(G::Adj::Cell2Face.Inverted());
    CHECK(support.Kind() != G::Adj::Face2Cell);
    for (I face = 0; face < complete->NumOwned(G::EntityKind::Face); ++face)
    {
        const auto key = NodeKey(nodes, view.FaceNodes(face));
        REQUIRE(expected.count(key));
        CHECK(support.GlobalRow(face) == expected.at(key));
        std::vector<I> actual;
        for (auto local : view.FaceCells(face))
            if (local != DNDS::UnInitIndex)
                actual.push_back(cells.LocalToGlobal(local));
        std::sort(actual.begin(), actual.end());
        CHECK(actual == expected.at(key));
        const auto firstParent = cells.LocalToGlobal(view.FaceCells(face)[0]);
        std::vector<I> ordered;
        for (auto local : view.FaceNodes(face))
            ordered.push_back(nodes.LocalToGlobal(local));
        bool oriented = false;
        for (const auto &candidate : orderedFaces[size_t(firstParent)])
            if (std::set<I>(candidate.begin(), candidate.end()) == key)
                oriented = ordered == candidate;
        CHECK(oriented);
        if (boundaries.count(key))
        {
            CHECK(view.GetFaceZone(face) == 2 + boundaries.at(key));
            CHECK(complete->Entities(G::EntityKind::Bnd).LocalToGlobal(view.FaceBoundaries(face)[0]) == boundaries.at(key));
            CHECK(view.FaceCells(face)[1] == DNDS::UnInitIndex);
        }
        else
        {
            CHECK(view.GetFaceZone(face) == G::BC_ID_INTERNAL);
            CHECK(view.FaceBoundaries(face)[0] == DNDS::UnInitIndex);
        }
    }
    for (I cell = 0; cell < cells.NumOwned(); ++cell)
    {
        const I global = cells.LocalToGlobal(cell);
        for (size_t slot = 0; slot < view.CellFaces(cell).size(); ++slot)
        {
            const auto face = view.CellFaces(cell)[slot];
            const auto key = NodeKey(nodes, view.FaceNodes(face));
            CHECK(key == std::set<I>(orderedFaces[size_t(global)][slot].begin(), orderedFaces[size_t(global)][slot].end()));
            const auto neighbor = view.CellFaceCells(cell)[slot];
            if (key == std::set<I>{1, 2})
                CHECK(cells.LocalToGlobal(neighbor) == 1 - global);
            else
                CHECK(neighbor == DNDS::UnInitIndex);
        }
    }
    CHECK_THROWS(G::NativeMeshTopology::Build(*complete, mpi));
    CHECK_NOTHROW(primary->Validate());
}

TEST_CASE("Native tetrahedron face and edge completion preserves exact incident cell sets")
{
    const auto mpi = MPI();
    auto primary = Fixture(mpi, 3);
    auto complete = G::NativeMeshTopology::Build(*primary, mpi, true, true);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 7);
    CHECK(complete->NumGlobal(G::EntityKind::Edge) == 9);
    CHECK(complete->MeshNamespace() == primary->MeshNamespace());
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    const auto nodes = complete->Entities(G::EntityKind::Node), cells = complete->Entities(G::EntityKind::Cell);
    std::map<std::set<I>, std::vector<I>> expected{{{0, 1}, {0, 1}}, {{0, 2}, {0, 1}}, {{1, 2}, {0, 1}}, {{0, 3}, {0}}, {{1, 3}, {0}}, {{2, 3}, {0}}, {{0, 4}, {1}}, {{1, 4}, {1}}, {{2, 4}, {1}}};
    auto support = complete->Relation(G::Adj::Cell2Edge.Inverted());
    for (I edge = 0; edge < complete->NumOwned(G::EntityKind::Edge); ++edge)
    {
        const auto key = NodeKey(nodes, view.EdgeNodes(edge));
        REQUIRE(expected.count(key));
        CHECK(support.GlobalRow(edge) == expected.at(key));
        std::vector<I> actual;
        for (auto cell : view.EdgeCells(edge))
            actual.push_back(cells.LocalToGlobal(cell));
        CHECK(actual == expected.at(key));
    }
    for (I face = 0; face < complete->NumOwned(G::EntityKind::Face); ++face)
        CHECK(view.GetFaceZone(face) == (view.FaceCells(face)[1] == DNDS::UnInitIndex ? G::BC_ID_NULL : G::BC_ID_INTERNAL));
    auto edgesOnly = G::NativeMeshTopology::Build(*primary, mpi, false, true);
    CHECK_FALSE(edgesOnly->HasEntities(G::EntityKind::Face));
    CHECK(edgesOnly->NumGlobal(G::EntityKind::Edge) == 9);
}

TEST_CASE("Native topology rejects nonmanifold faces and duplicate boundary assignments collectively")
{
    const auto mpi = MPI();
    SUBCASE("three cells share a face")
    {
        auto source = Fixture(mpi, 2, false, true);
        CHECK_THROWS(G::NativeMeshTopology::Build(*source, mpi));
        CHECK_NOTHROW(source->Validate());
    }
    SUBCASE("two independently supplied boundaries name one face")
    {
        auto source = Fixture(mpi, 2, true);
        CHECK_THROWS(G::NativeMeshTopology::Build(*source, mpi));
        CHECK_NOTHROW(source->Validate());
    }
}

TEST_CASE("Native topology repairs missing parent ghosts from authoritative owned node incidences")
{
    const auto mpi = MPI();
    auto source = Truncated(mpi);
    CHECK(source->NumGhost(G::EntityKind::Cell) == 0);
    auto closure = G::BuildNativeCellClosure(*source, mpi);
    auto parents = closure.cellNodes.Read();
    if (mpi.size > 1 && parents.NumOwned())
        CHECK(parents.Source().NumGhost() == 1);
    auto metadata = closure.cellInfo.ReadCombined();
    CHECK(metadata.View().Rows() == size_t(parents.Rows()));
    for (size_t row = 0; row < metadata.View().Rows(); ++row)
        CHECK(metadata.View()(row, 0).type == G::Elem::Tri3);
    auto complete = G::NativeMeshTopology::Build(*source, mpi);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 5);
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    int shared = 0;
    auto inverse = complete->Relation(G::Adj::Cell2Face.Inverted());
    for (I face = 0; face < complete->NumOwned(G::EntityKind::Face); ++face)
        if (NodeKey(complete->Entities(G::EntityKind::Node), read.HostView().FaceNodes(face)) == std::set<I>{1, 2})
        {
            ++shared;
            CHECK(inverse.GlobalRow(face) == std::vector<I>{0, 1});
        }
    MPI_Allreduce(MPI_IN_PLACE, &shared, 1, MPI_INT, MPI_SUM, mpi.comm);
    CHECK(shared == 1);
    CHECK(source->NumGhost(G::EntityKind::Cell) == 0);
}

TEST_CASE("Three tetrahedra retain a shared edge with three incident parents")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 5 : 0);
    const std::vector<std::vector<I>> all{{0, 1, 2, 3}, {0, 1, 3, 4}, {0, 1, 4, 2}};
    std::vector<std::vector<I>> local;
    for (size_t cell = 0; cell < all.size(); ++cell)
        if (int(cell) % mpi.size == mpi.rank)
            local.push_back(all[cell]);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(local.size())), boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries, Rows(local), Rows({}), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(local.size(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), {}};
    {
        auto coords = input.coordinates.Write();
        const DNDS::real points[5][3]{{0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {-0.5, 0.8660254037844386, 0}, {-0.5, -0.8660254037844386, 0}};
        for (I row = 0; row < nodes.NumOwned(); ++row)
            for (int c = 0; c < 3; ++c)
                coords.View()(size_t(row), size_t(c)) = points[row][c];
        auto info = input.cellInfo.Write();
        for (size_t row = 0; row < local.size(); ++row)
            info.View()(row, 0) = G::ElemInfo{G::Elem::Tet4, G::INTERNAL_ZONE};
    }
    auto source = G::NativeMeshBuilder::Build(std::move(input));
    auto complete = G::NativeMeshTopology::Build(*source, mpi, true, true);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 9);
    CHECK(complete->NumGlobal(G::EntityKind::Edge) == 10);
    auto read = complete->Snapshot(R::Device::Host()).AcquireRead();
    auto inverse = complete->Relation(G::Adj::Cell2Edge.Inverted());
    int shared = 0;
    for (I edge = 0; edge < complete->NumOwned(G::EntityKind::Edge); ++edge)
        if (NodeKey(complete->Entities(G::EntityKind::Node), read.HostView().EdgeNodes(edge)) == std::set<I>{0, 1})
        {
            ++shared;
            CHECK(inverse.GlobalRow(edge) == std::vector<I>{0, 1, 2});
            CHECK(read.HostView().EdgeCells(edge).size() == 3);
        }
    MPI_Allreduce(MPI_IN_PLACE, &shared, 1, MPI_INT, MPI_SUM, mpi.comm);
    CHECK(shared == 1);
}

TEST_CASE("Native topology rejects shared quadratic edge vertices with inconsistent midpoint identity")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 10 : 0);
    std::vector<std::vector<I>> rows;
    if (mpi.rank == 0)
        rows.push_back({0, 1, 2, 4, 5, 6});
    if (mpi.rank == (mpi.size > 1 ? 1 : 0))
        rows.push_back({1, 3, 2, 7, 8, 9});
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(rows.size())), boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(rows), Rows({}), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(rows.size(), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), {}};
    {
        auto coords = input.coordinates.Write();
        const DNDS::real points[10][3]{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0.5, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0}, {1, 0.5, 0}, {0.5, 1, 0}, {0.5, 0.5, 0}};
        for (I row = 0; row < nodes.NumOwned(); ++row)
            for (int component = 0; component < 3; ++component)
                coords.View()(size_t(row), size_t(component)) = points[row][component];
        auto metadata = input.cellInfo.Write();
        for (size_t row = 0; row < rows.size(); ++row)
            metadata.View()(row, 0) = G::ElemInfo{G::Elem::Tri6, G::INTERNAL_ZONE};
    }
    auto primary = G::NativeMeshBuilder::Build(input);
    // The shared vertices {1,2} refer to midpoint5 in cell0 and midpoint9 in
    // cell1. Distinct coordinates or a second generated face cannot repair the
    // contradictory topology; every participant must reject it.
    CHECK_THROWS(G::NativeMeshTopology::Build(*primary, mpi));
    CHECK_NOTHROW(primary->Validate());
    input.cellNodes = input.cellNodes.Clone();
    {
        auto write = input.cellNodes.Write();
        if (mpi.rank == (mpi.size > 1 ? 1 : 0))
            write.View()(mpi.size > 1 ? 0 : 1, 5) = 5;
    }
    auto corrected = G::NativeMeshBuilder::Build(input);
    auto complete = G::NativeMeshTopology::Build(*corrected, mpi);
    CHECK(complete->NumGlobal(G::EntityKind::Face) == 5);
}

TEST_CASE("Generated topology keeps persistent IDs and oriented slots through archive and independent redistribution")
{
    const auto mpi = MPI();
    auto primary = Fixture(mpi, 3);
    auto completed = G::NativeMeshTopology::Build(*primary, mpi, true, true);
    const auto expected = PersistentTopology(*completed, mpi);
    const auto name = completed->MeshNamespace();
    const auto marks = completed->IdHighWater();
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("dndsr-topology-" + std::to_string(pid) + "-r" + std::to_string(mpi.rank) + ".json")).string();
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->OpenFile(file, false);
    G::WriteNativeMeshArchive(*completed, mpi, serializer, "mesh");
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto current = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    CHECK(PersistentTopology(*current, mpi) == expected);
    for (int step = 1; step <= 2; ++step)
    {
        G::ReorderInput input;
        for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
            input.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(current->NumOwned(kind)), (mpi.rank + step + int(kind)) % mpi.size)});
        current = G::NativeMeshTransition::Build(*current, mpi, input).state;
        CHECK(current->MeshNamespace() == name);
        CHECK(current->IdHighWater() == marks);
        CHECK(PersistentTopology(*current, mpi) == expected);
    }
    std::filesystem::remove(file);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
