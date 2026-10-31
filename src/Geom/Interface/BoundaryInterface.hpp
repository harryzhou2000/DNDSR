#pragma once

#include "Geom/Spatial/BoundingBoxTree.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::Geom
{
    struct RigidPose
    {
        tGPoint rotation = tGPoint::Identity();
        tPoint translation = tPoint::Zero();
        void Validate() const;
        tPoint Apply(const tPoint &point) const { return rotation * point + translation; }
        tPoint Inverse(const tPoint &point) const { return rotation.transpose() * (point - translation); }
    };
    /// Explicit stencil preparation uses the ordinary DSL and ghost mapping;
    /// it does not mutate or assume the mesh's default compute halo.
    class BoundaryStencilPlan
    {
        std::shared_ptr<const AdjacencyGhostPublication> _publication;

    public:
        static BoundaryStencilPlan Prepare(UnstructuredMesh &mesh, const GhostSpec &spec,
                                           MeshReadRequirements requirements);
        MeshAdjacencyRead Read(const AdjacencyDescriptor &descriptor) const;
    };
    struct BoundaryElementRecord
    {
        index boundary, face, cell, cellFaceSlot, originalId;
        Elem::ElemType type;
    };
    namespace Detail
    {
        struct BoundaryPatchData;
        struct InterfacePointMapData;
    }
    /// Selected owned boundary elements, their adjacent-volume records, and a
    /// geometry placement. Preparation reads the volume mesh directly; only
    /// selected surface coordinates enter the index. No full-volume copy.
    class BoundaryPatch
    {
        std::shared_ptr<const Detail::BoundaryPatchData> _data;
        explicit BoundaryPatch(std::shared_ptr<const Detail::BoundaryPatchData> data) : _data(std::move(data)) {}
        friend class InterfacePointMap;
        friend class DistributedInterfacePointMap;

    public:
        static BoundaryPatch Create(ssp<UnstructuredMesh> mesh, const std::vector<index> &boundaries, RigidPose pose = {});
        BoundaryPatch WithPose(RigidPose pose) const;
        void Validate() const;
        index Size() const;
        BoundaryElementRecord Element(index iElement) const;
        const EntitySpace &Cells() const;
        RigidPose Pose() const;
    };
    struct InterfaceSearchSettings
    {
        real maxGap = 1e-10, referenceTolerance = 1e-10, projectionTolerance = 1e-12;
        int maxIterations = 20;
        /// 1..10 conservative boxes per rank, clustered in the longest world
        /// direction. This changes candidate routing only, never arbitration.
        int coarseBoxesPerRank = 1;
        void Validate() const;
    };
    struct InterfacePointMatch
    {
        index queryId, donorElement, donorBoundary, donorCell, donorOriginalId;
        tPoint reference, projectedPoint, donorPoint;
        real distance;
    };
    /// Immutable single-rank point correspondence. Query order is preserved;
    /// ties use persistent boundary ID, independent of patch ordering. A new
    /// pose requires a new map. Match performs no numerical interpolation.
    class InterfacePointMap
    {
        std::shared_ptr<const Detail::InterfacePointMapData> _data;
        explicit InterfacePointMap(std::shared_ptr<const Detail::InterfacePointMapData> data) : _data(std::move(data)) {}

    public:
        static InterfacePointMap Match(BoundaryPatch donor, const Runtime::Array<real> &points,
                                       const Runtime::Array<index> &queryIds, InterfaceSearchSettings settings = {});
        void Validate() const;
        index Size() const;
        InterfacePointMatch Get(index iQuery) const;
        const BoundaryPatch &Donor() const;
    };

    enum class InterfaceRoutingMode
    {
        CoarseBoxes,
        AllGatherBoundary
    };
    struct DistributedInterfaceMatch
    {
        InterfacePointMatch geometry;
        int geometryRank = 0, donorRank = 0;
        index donorOwnedCell = 0;
    };
    namespace Detail
    {
        struct DistributedInterfaceMapData;
    }

    /// Collective distributed boundary point matching. Coarse boxes only route
    /// candidates; exact proposals return to the query origin for arbitration.
    /// AllGatherBoundary replicates selected surface geometry as an independent
    /// routing baseline. Mesh coordinates always remain in their reference frame.
    /// Query IDs are unique per origin rank. Every rank calls with the same mode
    /// and settings, including empty query/patch ranks. Retain MPI until disposal.
    class DistributedInterfacePointMap
    {
        std::shared_ptr<const Detail::DistributedInterfaceMapData> _data;
        explicit DistributedInterfacePointMap(std::shared_ptr<const Detail::DistributedInterfaceMapData> data)
            : _data(std::move(data)) {}

    public:
        static DistributedInterfacePointMap Match(BoundaryPatch donor, const Runtime::Array<real> &points,
                                                  const Runtime::Array<index> &queryIds, InterfaceRoutingMode mode = InterfaceRoutingMode::CoarseBoxes,
                                                  InterfaceSearchSettings settings = {});
        void Validate() const;
        index Size() const;
        DistributedInterfaceMatch Get(index iQuery) const;
        /// Explicitly reject a retained map when the active immutable pose or
        /// patch selection has been replaced. Version validity alone describes
        /// the old patch, which can still be internally valid.
        void ValidateFor(const BoundaryPatch &active) const;
        const BoundaryPatch &Donor() const;
    };
}
