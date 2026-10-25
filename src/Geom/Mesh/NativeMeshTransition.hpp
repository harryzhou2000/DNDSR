#pragma once
#include "NativeMeshState.hpp"
#include "MeshReorderTypes.hpp"
#include "DNDS/PermutationTransfer.hpp"
#include <unordered_map>

namespace DNDS::Geom
{
    struct NativeMeshTransitionResult
    {
        std::shared_ptr<const NativeMeshState> state;
        std::unordered_map<EntityKind, PermutationTransfer> transfers;
        /// Exact input associations for validating external fields before
        /// transfer. Numeric identity maps do not imply matching entity stamps.
        std::map<EntityKind, EntitySpace> sourceSpaces;
    };

    /// Detached host-native transition. Persistent IDs and ordered companions
    /// move as payload; topology indices use sparse owner-directed remapping.
    class NativeMeshTransition
    {
        static NativeMeshTransitionResult BuildImpl(const NativeMeshState &source, const MPIInfo &mpi,
                                                    const ReorderInput &input, int ghostLayers,
                                                    const std::vector<index> *cellOld2New);

    public:
        static NativeMeshTransitionResult Build(const NativeMeshState &source, const MPIInfo &mpi,
                                                const ReorderInput &input, int ghostLayers = 1);
        /// Detached setup-only rank-local cell permutation. Other family owned
        /// order stays unchanged. All fixed/named references, companions, IDs
        /// and ancestry follow the same native transfer machinery. Existing
        /// fields/routes remain associated with source; transfers names every
        /// family (identity maps included) for explicit external field transfer.
        static NativeMeshTransitionResult ReorderLocalCells(const NativeMeshState &source, const MPIInfo &mpi,
                                                            int nParts = 1, int nPartsInner = 1, int ghostLayers = 1);
    };
}
