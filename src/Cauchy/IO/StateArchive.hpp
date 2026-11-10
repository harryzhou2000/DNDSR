#pragma once

#include "Cauchy/State/EquationDofs.hpp"
#include <nlohmann/json.hpp>
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"

namespace DNDS::Cauchy
{
    /// Logical blocks identify ranges in stored physical fields. Each field is
    /// serialized directly, retaining its native table layout/padding. Ghosts
    /// and derived caches are not scientific checkpoint state.
    nlohmann::ordered_json DescribeEquationStorage(const EquationDofs &state);
    void WriteEquationArrays(const EquationDofs &state, const Serializer::SerializerBaseSSP &serializer,
                             const std::string &prefix, const MPIInfo &mpi);

    /// Reader-owned shard, independent of accepted solver allocations. Restore
    /// uses a caller-provided persistent-ID route for redistribution.
    struct StateArchive
    {
        nlohmann::ordered_json storage;
        std::vector<Runtime::Array<real>> fields;
        static StateArchive Read(const Serializer::SerializerBaseSSP &serializer,
                                 const std::string &prefix, nlohmann::ordered_json storage, const MPIInfo &mpi,
                                 Serializer::ArrayGlobalOffset shard);
        bool Compatible(const EquationDofs &destination) const;
        /// Collective. Destination must be unpublished scratch: failure may
        /// alter it. Matching grouping transfers directly into target layouts.
        /// Changed grouping remaps temporary fields then scatters logical blocks.
        void Restore(EquationDofs &destination, const Runtime::RowTransferPlan &route,
                     const std::shared_ptr<Runtime::CommunicationContext> &context) const;
    };
}
