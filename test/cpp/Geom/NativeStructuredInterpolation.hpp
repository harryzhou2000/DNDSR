#pragma once

#include "NativeRelationSynthetic.hpp"
#include <optional>

namespace NativeStructuredInterpolation
{
    namespace G = DNDS::Geom;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;

    struct Hex
    {
        G::MeshAdjacency cone;
        R::Array<G::NodePeriodicBits> ownedBits, ghostBits;
        I nCellLocal;
        std::vector<G::Elem::ElemType> types;
    };

    inline Hex BuildHex(const DNDS::MPIInfo &mpi, I n, uint8_t periodicAxes, bool readerOwned = false)
    {
        const I active = readerOwned ? 1 : mpi.size;
        const I nx = active * n, ny = n, nz = n;
        const I nodeY = (periodicAxes & 2) ? n : n + 1;
        const I nodeZ = (periodicAxes & 4) ? n : n + 1;
        const bool owns = !readerOwned || mpi.rank == 0;
        const I slices = owns ? n + (!(periodicAxes & 1) && mpi.rank == active - 1) : 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, slices * nodeY * nodeZ);
        auto source = nodes.CreateSibling(G::EntityKind::Cell, owns ? n * n * n : 0);
        std::vector<std::vector<I>> rows;
        std::vector<std::vector<G::NodePeriodicBits>> bits;
        const I dx[]{0, 1, 1, 0, 0, 1, 1, 0};
        const I dy[]{0, 0, 1, 1, 0, 0, 1, 1};
        const I dz[]{0, 0, 0, 0, 1, 1, 1, 1};
        for (I row = 0; row < source.NumOwned(); ++row)
        {
            const I x = mpi.rank * n + row / (n * n), y = (row / n) % n, z = row % n;
            rows.emplace_back(8);
            bits.emplace_back(8);
            for (int slot = 0; slot < 8; ++slot)
            {
                const I px = x + dx[slot], py = y + dy[slot], pz = z + dz[slot];
                const I cx = (periodicAxes & 1) ? px % nx : px;
                const I cy = (periodicAxes & 2) ? py % ny : py;
                const I cz = (periodicAxes & 4) ? pz % nz : pz;
                rows.back()[size_t(slot)] = (cx * nodeY + cy) * nodeZ + cz;
                bits.back()[size_t(slot)] = G::NodePeriodicBits{uint8_t(((periodicAxes & 1) && px == nx ? 1 : 0) |
                                                                        ((periodicAxes & 2) && py == ny ? 2 : 0) |
                                                                        ((periodicAxes & 4) && pz == nz ? 4 : 0))};
            }
        }
        std::vector<I> ghosts;
        if (active > 1)
            for (const int step : {-1, 1})
            {
                I neighbor = mpi.rank + step;
                if (periodicAxes & 1)
                    neighbor = (neighbor + active) % active;
                if (neighbor < 0 || neighbor >= active)
                    continue;
                const I x = step < 0 ? n - 1 : 0;
                for (I y = 0; y < n; ++y)
                    for (I z = 0; z < n; ++z)
                        ghosts.push_back(neighbor * n * n * n + (x * n + y) * n + z);
            }
        std::sort(ghosts.begin(), ghosts.end());
        ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
        source = source.WithGhosts(std::move(ghosts));
        G::MeshAdjacency original(G::Adj::Cell2Node, source, nodes, NativePeriodicFixtures::Rows(rows));
        G::EntityGhostMapping route(source);
        auto fetched = original.WithGhostRows(route);
        std::set<I> remoteNodes;
        {
            auto read = fetched.Read();
            for (I row = 0; row < read.RowCount(); ++row)
                for (const auto global : read.GlobalRow(row))
                    if (!nodes.GlobalToLocal(global))
                        remoteNodes.insert(global);
        }
        nodes = nodes.WithGhosts({remoteNodes.begin(), remoteNodes.end()});
        G::MeshAdjacency withNodes(G::Adj::Cell2Node, source, nodes, original.Owned());
        auto cone = withNodes.WithGhostRows(route);
        auto ownedBits = NativePeriodicFixtures::Rows(bits);
        auto ghostBits = route.TransferRows(source, ownedBits, R::ElementSchema(G::NodePeriodicBits::CommType(), sizeof(G::NodePeriodicBits), "Geom.NodePeriodicBits", R::ElementSchema::TypeIdentity<G::NodePeriodicBits>()));
        return {std::move(cone), std::move(ownedBits), std::move(ghostBits), source.NumOwned(),
                std::vector<G::Elem::ElemType>(size_t(source.NumLocal()), G::Elem::Hex8)};
    }
}
