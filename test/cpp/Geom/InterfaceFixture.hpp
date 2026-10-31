#pragma once
#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace InterfaceFixture
{
    namespace G = DNDS::Geom;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;
    inline R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
    {
        std::vector<size_t> lengths;
        for (const auto &row : rows)
            lengths.push_back(row.size());
        R::Array<I> result(R::RowLayout::Ragged(lengths));
        auto write = result.Write();
        for (I i = 0; i < I(rows.size()); ++i)
            std::copy(rows[i].begin(), rows[i].end(), write.View()[i].begin());
        return result;
    }
    inline std::shared_ptr<G::UnstructuredMesh> SingleCell(const DNDS::MPIInfo &mpi, bool hex, bool overlappingCopy = false)
    {
        std::vector<G::tPoint> points = hex
                                            ? std::vector<G::tPoint>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}
                                            : std::vector<G::tPoint>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        G::Elem::Element element{hex ? G::Elem::Hex8 : G::Elem::Tet4};
        const I nodesPerCell = I(points.size()), copies = overlappingCopy ? 2 : 1;
        if (overlappingCopy)
        {
            const auto extra = points;
            points.insert(points.end(), extra.begin(), extra.end());
        }
        const bool active = mpi.rank == 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? I(points.size()) : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? copies : 0);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, active ? copies * element.GetNumFaces() : 0);
        std::vector<I> cone(size_t(nodesPerCell), 0);
        std::iota(cone.begin(), cone.end(), I(0));
        std::vector<std::vector<I>> faces, cellCones;
        if (active)
            for (I copy = 0; copy < copies; ++copy)
            {
                std::iota(cone.begin(), cone.end(), copy * nodesPerCell);
                cellCones.push_back(cone);
                for (int i = 0; i < element.GetNumFaces(); ++i)
                {
                    const auto face = element.ObtainFace(i);
                    faces.emplace_back(face.GetNumNodes());
                    element.ExtractFaceNodes(i, cone, faces.back());
                }
            }
        G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries,
                                 Rows(cellCones), Rows(faces),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(nodes.NumOwned(), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.NumOwned(), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaries.NumOwned(), 1))};
        {
            auto coordinate = input.coordinates.Write();
            for (I i = 0; i < nodes.NumOwned(); ++i)
                for (int d = 0; d < 3; ++d)
                    coordinate.View()(i, d) = points[i][d];
            auto cellInfo = input.cellInfo.Write(), bndInfo = input.boundaryInfo.Write();
            for (I i = 0; i < cells.NumOwned(); ++i)
                cellInfo.View()(i, 0) = G::ElemInfo{element.type, G::INTERNAL_ZONE};
            for (I i = 0; i < boundaries.NumOwned(); ++i)
                bndInfo.View()(i, 0) = G::ElemInfo{element.ObtainFace(i % element.GetNumFaces()).type, 7};
        }
        auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
        mesh->PrepareNativeTopology();
        return mesh;
    }

    /// One disconnected warped Hex8 per rank. Boundary ownership is shifted
    /// one rank relative to its volume cell, deliberately separating geometric
    /// search ownership from polynomial-data ownership. All IDs are global;
    /// the ordinary owned construction/ghost assembly supplies remote links.
    inline std::shared_ptr<G::UnstructuredMesh> WarpedHexes(const DNDS::MPIInfo &mpi)
    {
        constexpr double warp = 0.25;
        const std::vector<G::tPoint> unit{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        G::Elem::Element element{G::Elem::Hex8};
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 8);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, 1);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 6);
        std::vector<I> ownedCone(8), boundaryCone(8);
        std::iota(ownedCone.begin(), ownedCone.end(), I(8 * mpi.rank));
        const int volumeOwner = (mpi.rank + 1) % mpi.size;
        std::iota(boundaryCone.begin(), boundaryCone.end(), I(8 * volumeOwner));
        std::vector<std::vector<I>> faces(6, std::vector<I>(4));
        for (int iFace = 0; iFace < 6; ++iFace)
            element.ExtractFaceNodes(iFace, boundaryCone, faces[iFace]);
        G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries, Rows({ownedCone}), Rows(faces),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(8, 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(1, 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(6, 1))};
        {
            auto coordinate = input.coordinates.Write();
            for (I iNode = 0; iNode < 8; ++iNode)
            {
                const auto p = unit[iNode];
                coordinate.View()(iNode, 0) = 2 * mpi.rank + p[0];
                coordinate.View()(iNode, 1) = p[1];
                coordinate.View()(iNode, 2) = p[2] + warp * p[0] * p[1];
            }
            auto cellInfo = input.cellInfo.Write();
            auto bndInfo = input.boundaryInfo.Write();
            cellInfo.View()(0, 0) = G::ElemInfo{G::Elem::Hex8, G::INTERNAL_ZONE};
            for (I iBnd = 0; iBnd < 6; ++iBnd)
                bndInfo.View()(iBnd, 0) = G::ElemInfo{G::Elem::Quad4, 7};
        }
        auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
        mesh->PrepareNativeTopology();
        return mesh;
    }
}
