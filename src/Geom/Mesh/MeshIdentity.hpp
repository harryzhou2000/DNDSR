#pragma once

#include "MeshRelationTypes.hpp"
#include "DNDS/MPI.hpp"
#include <limits>
#include <map>
#include <random>

namespace DNDS::Geom
{
    /// Durable identity is independent of process-local EntitySpace tokens.
    /// IDs are scoped by this UUID and entity kind. nextIds is a non-reuse
    /// high-water mark and must survive deletion, reorder and restart.
    struct MeshIdentity
    {
        std::string meshNamespace;
        std::map<EntityKind, index> nextIds;
    };

    /// Collective setup over at most five scalar maxima. No storage depends on
    /// global entity counts or numeric ID magnitudes. Empty namespaces generate
    /// one random UUID on rank zero; explicit UUIDs must be identical on peers.
    inline MeshIdentity PrepareMeshIdentity(const MPIInfo &mpi, MeshIdentity requested,
                                            const std::map<EntityKind, index> &localOwnedMaxima)
    {
        auto agree = [&](bool valid, const char *message)
        {
            int value = valid;
            MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(value, message);
        };
        auto validNamespace = [](const std::string &value)
        {
            if (value.empty())
                return true;
            if (value.size() != 36)
                return false;
            for (size_t i = 0; i < value.size(); ++i)
            {
                const bool dash = i == 8 || i == 13 || i == 18 || i == 23;
                if (dash ? value[i] != '-' : !((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f')))
                    return false;
            }
            return value != "00000000-0000-0000-0000-000000000000";
        };
        bool valid = validNamespace(requested.meshNamespace) && !localOwnedMaxima.empty();
        int kinds = 0;
        for (const auto &[kind, maximum] : localOwnedMaxima)
        {
            const bool kindValid = kind >= EntityKind::Cell && kind < EntityKind::NUM_KINDS;
            valid = valid && kindValid && maximum >= -1;
            if (kindValid)
                kinds |= 1 << int(kind);
        }
        for (const auto &[kind, next] : requested.nextIds)
            valid = valid && localOwnedMaxima.count(kind) && next >= 0;
        agree(valid, "invalid mesh namespace or persistent-ID high-water metadata");
        int kindRange[]{kinds, -kinds};
        MPI_Allreduce(MPI_IN_PLACE, kindRange, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(kindRange[0] == -kindRange[1], "mesh identity entity families differ between ranks");

        // Compare supplied metadata before filling defaults. Missing high-water
        // marks and explicit values must mean the same thing on every rank.
        std::string signature = requested.meshNamespace;
        for (const auto &[kind, next] : requested.nextIds)
            signature += ":" + std::to_string(int(kind)) + ":" + std::to_string(next);
        int length = mpi.rank == 0 ? int(signature.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string canonical(size_t(length), ' ');
        if (mpi.rank == 0)
            canonical = signature;
        MPI_Bcast(canonical.data(), length, MPI_CHAR, 0, mpi.comm);
        agree(signature == canonical, "mesh namespace or persistent-ID high-water metadata differs between ranks");

        if (requested.meshNamespace.empty())
        {
            int generated = 1;
            if (mpi.rank == 0)
                try
                {
                    std::random_device random;
                    unsigned char bytes[16];
                    for (auto &byte : bytes)
                        byte = static_cast<unsigned char>(random());
                    bytes[6] = (bytes[6] & 0x0fU) | 0x40U;
                    bytes[8] = (bytes[8] & 0x3fU) | 0x80U;
                    const char digits[] = "0123456789abcdef";
                    for (size_t i = 0; i < 16; ++i)
                    {
                        if (i == 4 || i == 6 || i == 8 || i == 10)
                            requested.meshNamespace.push_back('-');
                        requested.meshNamespace.push_back(digits[bytes[i] >> 4]);
                        requested.meshNamespace.push_back(digits[bytes[i] & 15]);
                    }
                }
                catch (...)
                {
                    generated = 0;
                }
            agree(generated, "mesh namespace random generation failed");
            if (mpi.rank != 0)
                requested.meshNamespace.resize(36);
            MPI_Bcast(requested.meshNamespace.data(), 36, MPI_CHAR, 0, mpi.comm);
        }
        for (const auto &[kind, maximum] : localOwnedMaxima)
        {
            index globalMaximum = maximum;
            MPI_Allreduce(MPI_IN_PLACE, &globalMaximum, 1, DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(globalMaximum < std::numeric_limits<index>::max(),
                                  "persistent ID leaves no representable next-ID high-water mark");
            const index required = globalMaximum + 1;
            auto supplied = requested.nextIds.find(kind);
            DNDS_check_throw_info(supplied == requested.nextIds.end() || supplied->second >= required,
                                  "persistent-ID high-water mark would reuse an existing ID");
            requested.nextIds[kind] = supplied == requested.nextIds.end() ? required : supplied->second;
        }
        return requested;
    }
}
