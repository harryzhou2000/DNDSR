#pragma once

#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"

namespace NativePeriodicFixtures
{
    namespace G = DNDS::Geom;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;
    template <class T>
    R::Array<T> Rows(const std::vector<std::vector<T>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(T));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            std::copy(rows[row].begin(), rows[row].end(), builder.Row<T>(row).data);
        }
        return R::Array<T>::FromSharedStorage(builder.Commit());
    }
    inline G::NativeMeshInput TwoCells(const DNDS::MPIInfo &mpi)
    {
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 4 : 0);
        const bool first = mpi.rank == 0, second = mpi.rank == (mpi.size == 1 ? 0 : 1);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(first) + I(second));
        auto bnd = nodes.CreateSibling(G::EntityKind::Bnd, mpi.rank == mpi.size / 2 ? 6 : 0);
        std::vector<std::vector<I>> cones, boundaryCones;
        std::vector<std::vector<G::NodePeriodicBits>> cellBits, boundaryBits;
        if (first)
        {
            cones.push_back({0, 1, 3, 2});
            cellBits.push_back({{0}, {0}, {0}, {0}});
        }
        if (second)
        {
            cones.push_back({1, 0, 2, 3});
            cellBits.push_back({{0}, {1}, {1}, {0}});
        }
        if (bnd.NumOwned())
        {
            boundaryCones = {{2, 0}, {0, 2}, {0, 1}, {1, 0}, {3, 2}, {2, 3}};
            boundaryBits = {{{0}, {0}}, {{1}, {1}}, {{0}, {0}}, {{0}, {1}}, {{0}, {0}}, {{1}, {0}}};
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, bnd, Rows(cones), Rows(boundaryCones),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(bnd.NumOwned()), 1))};
        input.periodic = true;
        input.cellNodesPbi = Rows(cellBits);
        input.boundaryNodesPbi = Rows(boundaryBits);
        {
            auto write = input.coordinates.Write();
            for (I node = 0; node < nodes.NumOwned(); ++node)
            {
                write.View()(size_t(node), 0) = .5 * DNDS::real(node % 2);
                write.View()(size_t(node), 1) = DNDS::real(node / 2);
                write.View()(size_t(node), 2) = 0;
            }
            auto info = input.cellInfo.Write();
            for (size_t cell = 0; cell < cones.size(); ++cell)
                info.View()(cell, 0) = G::ElemInfo{G::Elem::Quad4, G::INTERNAL_ZONE};
            auto boundary = input.boundaryInfo.Write();
            const G::t_index zones[]{G::BC_ID_PERIODIC_1, G::BC_ID_PERIODIC_1_DONOR, 12, 12, 13, 13};
            for (size_t row = 0; row < boundaryCones.size(); ++row)
                boundary.View()(row, 0) = G::ElemInfo{G::Elem::Line2, zones[row]};
        }
        return input;
    }
    inline G::NativeMeshInput QuadraticTorus(const DNDS::MPIInfo &mpi, bool corrupt = false)
    {
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? (corrupt ? 5 : 4) : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 1 : 0);
        auto bnd = nodes.CreateSibling(G::EntityKind::Bnd, !corrupt && mpi.rank == mpi.size / 2 ? 4 : 0);
        std::vector<std::vector<I>> cones, boundaries;
        std::vector<std::vector<G::NodePeriodicBits>> cellBits, boundaryBits;
        if (cells.NumOwned())
        {
            cones = {{0, 0, 0, 0, 1, 2, 1, corrupt ? 4 : 2, 3}};
            cellBits = {{{0}, {1}, {3}, {2}, {0}, {1}, {2}, {0}, {0}}};
        }
        if (bnd.NumOwned())
        {
            boundaries = {{0, 0, 1}, {0, 0, 2}, {0, 0, 1}, {0, 0, 2}};
            boundaryBits = {{{0}, {1}, {0}}, {{1}, {3}, {1}}, {{3}, {2}, {2}}, {{2}, {0}, {0}}};
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, bnd, Rows(cones), Rows(boundaries),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(bnd.NumOwned()), 1))};
        input.periodic = true;
        input.cellNodesPbi = Rows(cellBits);
        input.boundaryNodesPbi = Rows(boundaryBits);
        {
            auto write = input.coordinates.Write();
            const DNDS::real points[5][3]{{0, 0, 0}, {.5, 0, 0}, {0, .5, 0}, {.5, .5, 0}, {0, .5, 0}};
            for (I node = 0; node < nodes.NumOwned(); ++node)
                for (size_t component = 0; component < 3; ++component)
                    write.View()(size_t(node), component) = points[node][component];
            if (cells.NumOwned())
            {
                auto info = input.cellInfo.Write();
                info.View()(0, 0) = G::ElemInfo{G::Elem::Quad9, G::INTERNAL_ZONE};
            }
            if (bnd.NumOwned())
            {
                auto info = input.boundaryInfo.Write();
                const G::t_index zones[]{G::BC_ID_PERIODIC_2, G::BC_ID_PERIODIC_1_DONOR, G::BC_ID_PERIODIC_2_DONOR, G::BC_ID_PERIODIC_1};
                for (size_t row = 0; row < 4; ++row)
                    info.View()(row, 0) = G::ElemInfo{G::Elem::Line3, zones[row]};
            }
        }
        return input;
    }
}
