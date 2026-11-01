#pragma once
#include "../Geom/NativePeriodicFixtures.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace CFVTest
{
    namespace G = DNDS::Geom;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;
    inline std::shared_ptr<G::UnstructuredMesh> Sector(DNDS::MPIInfo mpi)
    {
        using NativePeriodicFixtures::Rows;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 4 : 0);
        const bool first = mpi.rank == 0, second = mpi.rank == (mpi.size == 1 ? 0 : 1);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(first) + I(second));
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, mpi.rank == mpi.size / 2 ? 6 : 0);
        std::vector<std::vector<I>> cones, bnd;
        std::vector<std::vector<G::NodePeriodicBits>> cellBits, bndBits;
        if (first)
        {
            cones.push_back({0, 1, 3, 2});
            cellBits.push_back({{0}, {0}, {0}, {0}});
        }
        if (second)
        {
            cones.push_back({2, 3, 1, 0});
            cellBits.push_back({{0}, {0}, {1}, {1}});
        }
        if (boundaries.NumOwned())
        {
            bnd = {{0, 1}, {1, 0}, {1, 3}, {3, 1}, {2, 0}, {0, 2}};
            bndBits = {{{0}, {0}}, {{1}, {1}}, {{0}, {0}}, {{0}, {1}}, {{0}, {0}}, {{1}, {0}}};
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(cones), Rows(bnd),
                                 R::Array<double>(R::RowLayout::Dense(nodes.NumOwned(), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.NumOwned(), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaries.NumOwned(), 1))};
        input.periodic = true;
        input.cellNodesPbi = Rows(cellBits), input.boundaryNodesPbi = Rows(bndBits);
        input.periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
        input.periodicInfo.translation[1].map().setZero();
        {
            const double h = std::sqrt(0.5);
            const G::tPoint points[]{{1, 0, 0}, {2, 0, 0}, {h, h, 0}, {2 * h, 2 * h, 0}};
            auto coordinates = input.coordinates.Write();
            for (I iNode = 0; iNode < nodes.NumOwned(); ++iNode)
                for (int d = 0; d < 3; ++d)
                    coordinates.View()(iNode, d) = points[iNode][d];
            auto c = input.cellInfo.Write(), b = input.boundaryInfo.Write();
            for (I iCell = 0; iCell < cells.NumOwned(); ++iCell)
                c.View()(iCell, 0) = {G::Elem::Quad4, G::INTERNAL_ZONE};
            const int zones[]{G::BC_ID_PERIODIC_1, G::BC_ID_PERIODIC_1_DONOR, 12, 12, 13, 13};
            for (I iBnd = 0; iBnd < boundaries.NumOwned(); ++iBnd)
                b.View()(iBnd, 0) = {G::Elem::Line2, zones[iBnd]};
        }
        auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
        mesh->PrepareNativeTopology();
        return mesh;
    }

}
