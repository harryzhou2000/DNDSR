#pragma once

#include "MeshRelationTypes.hpp"
#include "DNDS/Runtime/Array.hpp"
#include <map>

namespace DNDS::Geom
{
    /// Durable ancestry, independent of source objects and current numbering.
    /// Each destination family has OWNED rows of source persistent IDs and an
    /// explicit source family. CSR rows allow zero or multiple source entities.
    /// These IDs are payload: redistribution moves rows, never renumbers IDs.
    /// Ghost ancestry is queried at its owner when needed, not replicated by
    /// default. This record is not a field interpolation or a live dependency.
    struct NativeMeshProvenance
    {
        std::string sourceNamespace;
        std::map<EntityKind, EntityKind> sourceKinds;
        std::map<EntityKind, Runtime::Array<index>> sourceIds;

        /// Rank-independent schema; logical row counts/values are rank-local.
        std::string Signature() const
        {
            DNDS_check_throw_info(sourceNamespace.size() == 36 && sourceNamespace != "00000000-0000-0000-0000-000000000000",
                                  "invalid provenance source namespace");
            for (size_t i = 0; i < sourceNamespace.size(); ++i)
            {
                const bool dash = i == 8 || i == 13 || i == 18 || i == 23;
                const char c = sourceNamespace[i];
                DNDS_check_throw_info(dash ? c == '-' : ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')),
                                      "invalid provenance source namespace");
            }
            DNDS_check_throw_info(!sourceIds.empty() && sourceKinds.size() == sourceIds.size(), "incomplete provenance family mapping");
            std::string result = sourceNamespace;
            for (const auto &[kind, ids] : sourceIds)
            {
                auto source = sourceKinds.find(kind);
                DNDS_check_throw_info(kind >= EntityKind::Cell && kind < EntityKind::NUM_KINDS && source != sourceKinds.end() &&
                                          source->second >= EntityKind::Cell && source->second < EntityKind::NUM_KINDS,
                                      "invalid provenance entity kind");
                result += ":" + std::to_string(int(kind)) + ":" + std::to_string(int(source->second));
            }
            return result;
        }
    };
}
