#pragma once

#include "Geom/Unstructured/OwnedMeshData.hpp"

namespace DNDS::Geom
{
    /// Owned host input in explicit rank-offset global numbering. The three
    /// spaces must be sibling families without ghosts. Cone slots follow the
    /// element topology; optional IDs are owned, one-column persistent values.
    /// Coordinates (three components) and ElemInfo (one component) accept dense,
    /// padded or uniform CSR rows; publication normalizes only their metadata.
    struct NativeMeshInput
    {
        MPIInfo mpi;
        int dimension;
        EntitySpace cells, nodes, boundaries;
        Runtime::Array<index> cellNodes, boundaryNodes;
        Runtime::Array<real> coordinates;
        Runtime::Array<ElemInfo> cellInfo, boundaryInfo;
        std::map<EntityKind, Runtime::Array<index>> ids;
        bool periodic = false;
        /// Empty namespace creates a new logical mesh; restart supplies the
        /// saved namespace and high-water marks unchanged.
        MeshIdentity identity;
        std::optional<NativeMeshProvenance> provenance;
        /// Already canonical node IDs with ordered physical image labels.
        /// Periodic cones allow repeated IDs in distinct frames. All ranks
        /// supply both companions (including empty rows) and identical frames.
        std::optional<Runtime::Array<NodePeriodicBits>> cellNodesPbi, boundaryNodesPbi;
        Periodicity periodicInfo;
    };

    struct NativeMeshBuildOptions
    {
        /// Finalization frontend's default cell-ring depth, at least one.
        /// The owned builder validates agreement but creates no compute ghosts.
        int cellLayers = 1;
        /// Finalization frontend's owned-node coverage policy. Owned topology
        /// always retains complete GLOBAL inverse rows; this controls which
        /// remote targets become locally dereferenceable at final assembly.
        bool completeNodeStar = false;
        /// Unset means full audit for fresh input, or inherit the source policy
        /// for refinement/extraction. Set explicitly to override that policy.
        std::optional<MeshAuditOptions> audit;
    };

    /// Collective owned primary construction over sparse native relations.
    /// Returns owned GLOBAL-index topology; no final ghost mesh is assembled
    /// and no UnstructuredMeshState is published. Temporary support rows are
    /// fetched only to perform inversion, composition and physical matching.
    /// All ranks, including empty ones, invoke Build in the same order.
    /// Canonical periodic inputs provide ordered image companions explicitly;
    /// external boundaries match one physical face, periodic boundaries two
    /// side occurrences (possibly in the same cell).
    class NativeMeshBuilder
    {
    public:
        static NativeOwnedMeshData Build(
            NativeMeshInput input, NativeMeshBuildOptions options = {});
    };

    /// Explicit completed primary-mesh operation for callers requesting a read
    /// state immediately. Runs owned construction, selects the frontend's
    /// default recipe, then invokes the final assembler exactly once. Pipelines
    /// that still need partition/reorder/face construction use Build's owned
    /// result instead and finalize after those steps.
    std::shared_ptr<const UnstructuredMeshState> ConstructNativeMesh(
        NativeMeshInput input, NativeMeshBuildOptions options = {});
}
