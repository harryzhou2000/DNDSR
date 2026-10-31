#pragma once

#include "ElementGeometry.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include <memory>

namespace DNDS::Geom
{
    struct SurfaceElementRecord
    {
        index id;
        Elem::ElemType type;
    };
    struct SurfaceSearchSettings : PointProjectionSettings
    {
        int coarseBoxesPerRank = 1;
        void Validate() const;
    };
    enum class SurfaceRoutingMode
    {
        CoarseBoxes,
        AllGatherSurface
    };
    namespace Detail
    {
        struct SurfacePatchData;
        struct SurfacePointMapData;
        struct DistributedSurfacePointMapData;
    }
    /// Immutable geometry snapshot, independent of UnstructuredMesh, cell
    /// numbering, equations, or sliding. Element IDs are unique locally; the
    /// distributed caller supplies unique IDs across ranks. Coordinates are
    /// [element,3*node] Runtime rows; a retained read lease freezes the shared
    /// allocation/metadata against writes through other Runtime handles for the
    /// snapshot's lifetime.
    /// Foreign raw-pointer/tensor writes are outside lease enforcement and must
    /// also be excluded by the caller while this snapshot is retained.
    /// Rigid placement creates a new snapshot sharing reference geometry and
    /// the immutable reference BVH; queries are transformed into that frame.
    class SurfacePatch
    {
        std::shared_ptr<const Detail::SurfacePatchData> _data;
        explicit SurfacePatch(std::shared_ptr<const Detail::SurfacePatchData> data) : _data(std::move(data)) {}
        friend class SurfacePointMap;
        friend class DistributedSurfacePointMap;

    public:
        static SurfacePatch Create(std::vector<SurfaceElementRecord> elements, Runtime::Array<real> coordinates, RigidPose pose = {},
                                   BoundingBoxBackend backend = BoundingBoxBackend::CGAL);
        SurfacePatch WithPose(RigidPose pose) const;
        /// Explicitly rebuild only the candidate index; reference geometry and
        /// its read lease stay shared. Existing maps retain their old snapshot.
        /// Backend is local policy and need not agree across MPI ranks; search
        /// settings/collective order still must agree. No global default state.
        SurfacePatch WithBackend(BoundingBoxBackend backend) const;
        BoundingBoxBackend Backend() const;
        index Size() const;
        SurfaceElementRecord Element(index iElement) const;
        /// Borrowed view; retain this SurfacePatch through use.
        ElementGeometryView Geometry(index iElement) const;
        RigidPose Pose() const;
        bool Same(const SurfacePatch &other) const { return _data == other._data; }
    };
    struct SurfacePointMatch
    {
        /// element indexes the selected surface rows on geometryRank, not the
        /// receiving rank. elementId is the caller's persistent global ID.
        index queryId, element, elementId;
        int geometryRank;
        /// reference: element parameter coordinates; projectedPoint/normal:
        /// world frame; surfacePoint: physical donor reference frame (before
        /// rigid placement), ready for a mesh-bound polynomial evaluator.
        tPoint reference, projectedPoint, surfacePoint, normal;
        real distance;
    };
    struct SurfaceMatchProposal
    {
        SurfacePointMatch match;
        bool interior;
    };
    /// Pure query-owner dispute resolution shared by local/distributed search.
    /// Empty input returns no match; two distinct near-nearest interiors throw.
    std::optional<SurfacePointMatch> ResolveSurfaceMatches(const std::vector<SurfaceMatchProposal> &proposals, real distanceTolerance);
    /// Local query order is preserved. Competing near-equal-distance proposals
    /// resolve equivalent seams by persistent element ID; multiple accepted
    /// interiors are rejected. This is donor arbitration, not a reduction of
    /// sampled field values. Projection is normal projection + containment,
    /// not clamped closest-point search.
    class SurfacePointMap
    {
        std::shared_ptr<const Detail::SurfacePointMapData> _data;
        explicit SurfacePointMap(std::shared_ptr<const Detail::SurfacePointMapData> data) : _data(std::move(data)) {}

    public:
        static SurfacePointMap Match(SurfacePatch surface, const Runtime::Array<real> &points,
                                     const Runtime::Array<index> &queryIds, SurfaceSearchSettings settings = {});
        index Size() const;
        SurfacePointMatch Get(index iQuery) const;
        void ValidateFor(const SurfacePatch &active) const;
    };
    /// General collective point-to-surface locator: coarse boxes -> sparse
    /// query fan-out -> geometric proposals -> query-owner arbitration.
    /// AllGatherSurface is a replicated-surface routing baseline using the same
    /// exact projection/arbitration kernels. All ranks call in the same order
    /// with matching settings, including empty owners/queries. Caller retains
    /// mpi.comm through routes/operations. No volume mesh or solution payload
    /// belongs here. A new pose/geometry requires an explicitly rebuilt map.
    class DistributedSurfacePointMap
    {
        std::shared_ptr<const Detail::DistributedSurfacePointMapData> _data;
        explicit DistributedSurfacePointMap(std::shared_ptr<const Detail::DistributedSurfacePointMapData> data) : _data(std::move(data)) {}

    public:
        static DistributedSurfacePointMap Match(MPIInfo mpi, SurfacePatch surface, const Runtime::Array<real> &points,
                                                const Runtime::Array<index> &queryIds,
                                                SurfaceRoutingMode mode = SurfaceRoutingMode::CoarseBoxes,
                                                SurfaceSearchSettings settings = {});
        index Size() const;
        SurfacePointMatch Get(index iQuery) const;
        void ValidateFor(const SurfacePatch &active) const;
        /// Collective reusable route from matched geometry-owner element rows
        /// into original query order. Bind any compatible element-associated
        /// Runtime payload/schema; mesh adapters use it for volume-cell links.
        /// It does not imply a DOF or a cell halo mapping.
        Runtime::RowTransferPlan BuildElementDataRoute() const;
    };
}
