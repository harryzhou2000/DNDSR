#pragma once

#include "PublishedAdjacency.hpp"

namespace DNDS::Geom
{
    /// Rank-local classification of owned output rows by explicit adjacency
    /// paths to input rows. Construction never fetches remote connectivity.
    /// Every path's terminal input must exist in its canonical mapping; missing
    /// intermediate adjacency rows are errors. Physical boundary sentinels are
    /// skipped. Completeness of supplied published adjacency rows is a producer
    /// contract, not inferred from row length. This does not certify kernel I/O,
    /// BC readiness or whether an owned row is an active MPI send source.
    class EntityWorkPartition
    {
        struct Data
        {
            EntitySpace outputs;
            Runtime::VersionStamp stamp;
            std::vector<index> core, dependent;
            Data(EntitySpace space, Runtime::VersionStamp version)
                : outputs(std::move(space)), stamp(std::move(version)) {}
        };
        std::shared_ptr<const Data> _data;
        explicit EntityWorkPartition(std::shared_ptr<const Data> data) : _data(std::move(data)) {}

    public:
        /// All paths start at outputKind. Empty dependency lists mean every
        /// output is core. An individual path must contain at least one hop.
        /// Generation checks occur at acquisition/Validate, not in a hot loop.
        static EntityWorkPartition Build(UnstructuredMesh &mesh, EntityKind outputKind,
                                         const std::vector<std::vector<AdjacencyDescriptor>> &paths)
        {
            auto lease = mesh.versions.Read();
            auto result = std::make_shared<Data>(mesh.Entities(outputKind),
                                                 mesh.versions.Stamp(Runtime::Change::Topology | Runtime::Change::Distribution |
                                                                     Runtime::Change::Layout | Runtime::Change::Interface));
            using Read = std::shared_ptr<MeshAdjacencyRead>;
            std::vector<std::vector<Read>> dependencies;
            for (const auto &path : paths)
            {
                DNDS_check_throw_info(!path.empty(), "work partition dependency path is empty");
                auto source = result->outputs;
                std::vector<Read> reads;
                for (const auto &descriptor : path)
                {
                    auto read = std::make_shared<MeshAdjacencyRead>(ReadPublishedAdjacency(mesh, descriptor));
                    DNDS_check_throw_info(read->Source().SameOrder(source), "work partition dependency source order differs");
                    source = read->Target();
                    reads.push_back(std::move(read));
                }
                dependencies.push_back(std::move(reads));
            }
            result->core.reserve(size_t(result->outputs.NumOwned()));
            result->dependent.reserve(size_t(result->outputs.NumOwned() / 10));
            std::vector<index> frontier, next;
            for (index iRow = 0; iRow < result->outputs.NumOwned(); ++iRow)
            {
                bool core = true;
                for (const auto &path : dependencies)
                {
                    frontier.assign(1, iRow);
                    for (index iHop = 0; iHop < index(path.size()); ++iHop)
                    {
                        const auto &read = *path[iHop];
                        next.clear();
                        for (auto source : frontier)
                        {
                            DNDS_check_throw_info(source >= 0 && source < read.RowCount(),
                                                  "work partition is missing an intermediate adjacency row");
                            for (index slot = 0; slot < read.RowSize(source); ++slot)
                            {
                                const auto entry = read.ReadEntry(source, size_t(slot));
                                if (entry.absent)
                                    continue;
                                DNDS_check_throw_info(!entry.unavailable, "work partition requires complete input ghost mapping");
                                const auto local = read.Target().GlobalToLocal(entry.global);
                                DNDS_check_throw_info(local, "work partition input is absent from target mapping");
                                next.push_back(*local);
                                if (iHop + 1 == path.size() && *local >= read.Target().NumOwned())
                                    core = false;
                            }
                        }
                        std::sort(next.begin(), next.end());
                        next.erase(std::unique(next.begin(), next.end()), next.end());
                        frontier.swap(next);
                    }
                }
                (core ? result->core : result->dependent).push_back(iRow);
            }
            return EntityWorkPartition(std::move(result));
        }

        /// Standard conservative cell split matching the node-star halo policy.
        static EntityWorkPartition CellNodeCell(UnstructuredMesh &mesh)
        {
            return Build(mesh, EntityKind::Cell, {{Adj::Cell2Cell}});
        }
        void Validate() const
        {
            DNDS_check_throw_info(_data, "work partition was moved");
            _data->stamp.Validate();
            _data->outputs.Validate();
        }
        const EntitySpace &Outputs() const { return _data->outputs; }
        /// Borrow lists only after Validate, and retain normal mesh/input leases
        /// through execution. A cached partition does not block mesh mutation.
        const std::vector<index> &Core() const { return _data->core; }
        const std::vector<index> &HaloDependent() const { return _data->dependent; }
    };
}
