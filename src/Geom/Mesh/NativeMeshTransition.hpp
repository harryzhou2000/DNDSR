#pragma once
#include "NativeMeshState.hpp"
#include "ReorderPlan.hpp"

namespace DNDS::Geom
{
    struct NativeMeshTransitionResult
    {
        std::shared_ptr<const NativeMeshState> state;
        std::unordered_map<EntityKind, PermutationTransfer> transfers;
    };

    /// Detached host-native transition. Persistent IDs and ordered companions
    /// move as payload; topology indices use sparse owner-directed remapping.
    class NativeMeshTransition
    {
    public:
        static NativeMeshTransitionResult Build(const NativeMeshState &source, const MPIInfo &mpi,
                                                const ReorderInput &input, int ghostLayers = 1);
    };
}
