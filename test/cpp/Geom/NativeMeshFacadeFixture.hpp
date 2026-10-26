#pragma once
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/NativeMeshState.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"

namespace NativeFacadeFixture
{
    namespace G = DNDS::Geom;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;
    template <class T>
    R::ArrayPair<T> Rows(const std::vector<std::vector<T>> &rows, size_t width = 0)
    {
        R::Array<T> owned;
        if (width)
        {
            owned = R::Array<T>(R::RowLayout::Dense(rows.size(), width));
            auto write = owned.Write();
            for (size_t row = 0; row < rows.size(); ++row)
                for (size_t slot = 0; slot < rows[row].size(); ++slot)
                    write.View()(row, slot) = rows[row][slot];
        }
        else
        {
            R::RowBuilder builder(rows.size(), sizeof(T));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                std::copy(rows[row].begin(), rows[row].end(), builder.Row<T>(row).data);
            }
            owned = R::Array<T>::FromSharedStorage(builder.Commit());
        }
        return {std::move(owned), R::Array<T>(R::RowLayout::Dense(0, width))};
    }
    inline G::MeshReadRequirements Requirements()
    {
        auto result = G::MeshReadRequirements::CellFlux();
        result.inverse = result.edges = result.cellFaceNeighbors = result.wallDistance = true;
        return result;
    }
    struct Input
    {
        G::NativeMeshData data;
        std::map<G::EntityKind, G::EntitySpace> spaces;
        std::map<G::EntityKind, R::Array<I>> ids;
        std::shared_ptr<G::UnstructuredMesh> Publish(const DNDS::MPIInfo &mpi,
                                                     G::MeshReadRequirements requirements = Requirements())
        {
            return G::UnstructuredMesh::FromNative(mpi, G::NativeMeshState::FromNative(mpi, data, spaces, ids, requirements));
        }
    };
    inline Input Make(const DNDS::MPIInfo &mpi, bool periodic = false, bool empty = false)
    {
        Input input;
        auto &data = input.data;
        const I nodes = empty ? 0 : periodic ? 4
                                             : 3,
                cells = empty ? 0 : 1;
        const I faces = empty ? 0 : 3, boundaries = empty ? 0 : periodic ? 2
                                                                         : 3;
        data.dimension = 2;
        data.periodic = periodic;
        const auto nodeSpace = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodes);
        input.spaces.emplace(G::EntityKind::Node, nodeSpace);
        for (auto [kind, count] : {std::pair{G::EntityKind::Cell, cells}, {G::EntityKind::Face, faces}, {G::EntityKind::Bnd, boundaries}, {G::EntityKind::Edge, faces}})
            input.spaces.emplace(kind, nodeSpace.CreateSibling(kind, count));
        for (const auto &[kind, space] : input.spaces)
        {
            R::Array<I> ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
            auto write = ids.Write();
            for (I row = 0; row < space.NumOwned(); ++row)
                write.View()(size_t(row), 0) = 10000 + 17 * space.LocalToGlobal(row) + I(kind);
            write.Wait();
            input.ids.emplace(kind, std::move(ids));
        }
        std::vector<std::vector<double>> points, walls;
        const std::vector<std::vector<double>> physical{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        for (I node = 0; node < nodes; ++node)
        {
            points.push_back(physical[size_t(node)]);
            walls.push_back({1 + double(node), 2, 0});
        }
        data.coords = Rows(points, 3);
        data.nodeWallDist = Rows(walls, 3);
        std::vector<std::vector<I>> cn, fn, bn, cf, cc, nc, nb;
        if (!empty)
        {
            cn = periodic ? std::vector<std::vector<I>>{{0, 1, 2, 3}} : std::vector<std::vector<I>>{{0, 1, 2}};
            fn = periodic ? std::vector<std::vector<I>>{{0, 3}, {0, 1}, {3, 2}} : std::vector<std::vector<I>>{{0, 1}, {1, 2}, {2, 0}};
            bn = periodic ? std::vector<std::vector<I>>{{0, 1}, {3, 2}} : fn;
            cf = periodic ? std::vector<std::vector<I>>{{0, 1, 0, 2}} : std::vector<std::vector<I>>{{0, 1, 2}};
            cc = periodic ? std::vector<std::vector<I>>{{0}} : std::vector<std::vector<I>>{{}};
            nc.resize(size_t(nodes), {0});
            nb = periodic ? std::vector<std::vector<I>>{{0}, {0}, {1}, {1}} : std::vector<std::vector<I>>{{0, 2}, {0, 1}, {1, 2}};
        }
        auto set = [&](G::MeshRelation relation, const auto &rows, size_t width = 0)
        { data.relations[size_t(relation)] = Rows(rows, width); };
        set(G::MeshRelation::CellNodes, cn);
        set(G::MeshRelation::FaceNodes, fn);
        set(G::MeshRelation::BoundaryNodes, bn);
        set(G::MeshRelation::CellFaces, cf);
        set(G::MeshRelation::CellCells, cc);
        set(G::MeshRelation::NodeCells, nc);
        set(G::MeshRelation::NodeBoundaries, nb);
        auto fc = std::vector<std::vector<I>>(size_t(faces), {0, DNDS::UnInitIndex});
        if (periodic && !empty)
            fc[0] = {0, 0};
        set(G::MeshRelation::FaceCells, fc, 2);
        set(G::MeshRelation::BoundaryCells, std::vector<std::vector<I>>(size_t(boundaries), {0, DNDS::UnInitIndex}), 2);
        const auto fb = empty ? std::vector<std::vector<I>>{} : periodic ? std::vector<std::vector<I>>{{DNDS::UnInitIndex}, {0}, {1}}
                                                                         : std::vector<std::vector<I>>{{0}, {1}, {2}};
        set(G::MeshRelation::FaceBoundaries, fb, 1);
        set(G::MeshRelation::BoundaryFaces, empty ? std::vector<std::vector<I>>{} : periodic ? std::vector<std::vector<I>>{{1}, {2}}
                                                                                             : fb,
            1);
        set(G::MeshRelation::CellEdges, cf);
        set(G::MeshRelation::EdgeNodes, fn);
        set(G::MeshRelation::EdgeCells, std::vector<std::vector<I>>(size_t(faces), {0}));
        set(G::MeshRelation::CellFaceCells, empty ? std::vector<std::vector<I>>{} : periodic ? std::vector<std::vector<I>>{{0, DNDS::UnInitIndex, 0, DNDS::UnInitIndex}}
                                                                                             : std::vector<std::vector<I>>{{DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex}});
        auto info = [](I count, G::Elem::ElemType type, G::t_index zone)
        { return Rows(std::vector<std::vector<G::ElemInfo>>(size_t(count), {G::ElemInfo{type, zone}}), 1); };
        data.cellInfo = info(cells, periodic ? G::Elem::Quad4 : G::Elem::Tri3, 11);
        data.bndInfo = info(boundaries, G::Elem::Line2, 7);
        data.faceInfo = info(faces, G::Elem::Line2, 7);
        data.edgeInfo = info(faces, G::Elem::Line2, 9);
        if (periodic)
        {
            auto bits = [](const auto &rows)
            {
                std::vector<std::vector<G::NodePeriodicBits>> result;
                for (const auto &row : rows)
                    result.emplace_back(row.size(), G::NodePeriodicBits{});
                return result;
            };
            auto cb = bits(cn), faceBits = bits(fn), slots = bits(cf);
            if (!empty)
            {
                cb[0][2] = G::nodePB1;
                faceBits[0][1] = G::nodePB1;
                slots[0][2] = G::nodePB1;
                data.faceInfo->owned.Write().View()(0, 0).zone = G::BC_ID_PERIODIC_1;
            }
            data.cellNodesPbi = Rows(cb);
            data.bndNodesPbi = Rows(bits(bn));
            data.faceNodesPbi = Rows(faceBits);
            data.cellFacesPbi = Rows(slots);
            data.cellEdgesPbi = Rows(slots);
            data.edgeNodesPbi = Rows(faceBits);
            data.periodicInfo.translation[1].map() = G::tPoint{3, 0, 0};
            data.periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        }
        return input;
    }
}
