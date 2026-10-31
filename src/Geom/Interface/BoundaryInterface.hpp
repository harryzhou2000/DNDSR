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
}
