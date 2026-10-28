#pragma once

#include "Geom/Access/EntitySpace.hpp"

namespace DNDS::Geom
{
    /// Transitional field-setup adapter. It borrows only maps, leaving field
    /// allocation, TransAttach, createMPITypes and request creation with callers.
    /// The returned legacy transformer retains its documented lifetime contract;
    /// this bridge cannot stamp arbitrary subsequent direct transformer calls.
    /// cloneMaps additionally isolates compatibility callers that can edit map
    /// objects; published native maps are always cloned regardless of this flag.
    class LegacyFieldBridge
    {
    public:
        template <class TTransformer>
        static void BorrowIndexing(TTransformer &transformer, const EntitySpace &space, bool cloneMaps = false)
        {
            space.Validate();
            DNDS_check_throw_info(transformer.father && transformer.son, "attach field arrays before borrowing entity indexing");
            DNDS_check_throw_info(transformer.father->Size() == space.NumOwned(), "field owned rows differ from entity space");
            // Legacy createMPITypes sizes an initially empty ghost allocation.
            DNDS_check_throw_info(transformer.son->Size() == 0 || transformer.son->Size() == space.NumGhost(),
                                  "field ghost rows differ from entity space");
            EntitySpace::CheckCommunicator(transformer.mpi.comm, space._mpi.comm);
            EntitySpace::CheckCommunicator(transformer.father->getMPI().comm, space._mpi.comm);
            EntitySpace::CheckCommunicator(transformer.son->getMPI().comm, space._mpi.comm);
            // Publish only after all recoverable local checks have succeeded.
            auto global = cloneMaps || space._publishedMaps ? std::make_shared<GlobalOffsetsMapping>(*space._global) : space._global;
            auto ghost = cloneMaps || space._publishedMaps ? std::make_shared<OffsetAscendIndexMapping>(*space._ghostMap) : space._ghostMap;
            transformer.pLGlobalMapping = global;
            transformer.pLGhostMapping = ghost;
            transformer.father->pLGlobalMapping = std::move(global);
        }
    };
}
