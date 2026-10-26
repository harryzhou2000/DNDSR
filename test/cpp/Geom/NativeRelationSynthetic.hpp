#pragma once

#include "Geom/Mesh/NativeConnectivity.hpp"
#include "NativePeriodicFixtures.hpp"
#include <set>

namespace NativeRelationSynthetic
{
    namespace G = DNDS::Geom;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;

    // Keep the historical rank0/rank1 ownership exactly, including all empty
    // ranks beyond rank1; synthetic expectations are independent fixed tables.
    inline G::NativeRelation FourQuadCones(const DNDS::MPIInfo &mpi)
    {
        const I count = mpi.size == 1 ? 9 : mpi.rank == 0 ? 5
                                        : mpi.rank == 1   ? 4
                                                          : 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, count);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.size == 1 ? 4 : mpi.rank < 2 ? 2
                                                                                               : 0);
        const I input[4][4]{{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
        std::vector<std::vector<I>> rows;
        std::set<I> remote;
        for (I row = 0; row < cells.NumOwned(); ++row)
        {
            const I global = cells.LocalToGlobal(row);
            rows.emplace_back(input[global], input[global] + 4);
            for (const I node : rows.back())
                if (!nodes.GlobalToLocal(node))
                    remote.insert(node);
        }
        nodes = nodes.WithGhosts({remote.begin(), remote.end()});
        return {G::Adj::Cell2Node, cells, nodes, NativePeriodicFixtures::Rows(rows)};
    }

    inline G::NativeRelation AllNeighbors(const G::NativeRelation &cone)
    {
        std::vector<std::vector<I>> rows(size_t(cone.Source().NumOwned()));
        for (I row = 0; row < cone.Source().NumOwned(); ++row)
            for (I neighbor = 0; neighbor < 4; ++neighbor)
                if (neighbor != cone.Source().LocalToGlobal(row))
                    rows[size_t(row)].push_back(neighbor);
        return {G::Adj::Cell2Cell, cone.Source(), cone.Source(), NativePeriodicFixtures::Rows(rows)};
    }

    inline std::vector<std::set<I>> GatherTiny(const G::RelationReadAccess &read, const DNDS::MPIInfo &mpi)
    {
        // A bounded 9x9 synthetic oracle, not global mesh indexing/scratch.
        int values[81]{};
        DNDS_check_throw_info(read.Source().NumGlobal() <= 9 && read.Target().NumGlobal() <= 9,
                              "tiny fixture oracle exceeded its fixed domain");
        for (I row = 0; row < read.NumOwned(); ++row)
            for (const I global : read.GlobalRow(row))
            {
                REQUIRE(global >= 0);
                REQUIRE(global < 9);
                values[9 * read.Source().LocalToGlobal(row) + global] = 1;
            }
        MPI_Allreduce(MPI_IN_PLACE, values, 81, MPI_INT, MPI_MAX, mpi.comm);
        std::vector<std::set<I>> result(size_t(read.Source().NumGlobal()));
        for (size_t row = 0; row < result.size(); ++row)
            for (I target = 0; target < read.Target().NumGlobal(); ++target)
                if (values[9 * row + size_t(target)])
                    result[row].insert(target);
        return result;
    }
}
