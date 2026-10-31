#pragma once

#include "BoundingBoxTree.hpp"
#include "Geom/Elements.hpp"

namespace DNDS::Geom
{
    /// Placement is independent of geometry storage and mesh generations.
    struct RigidPose
    {
        tGPoint rotation = tGPoint::Identity();
        tPoint translation = tPoint::Zero();
        void Validate() const;
        tPoint Apply(const tPoint &point) const { return rotation * point + translation; }
        tPoint Inverse(const tPoint &point) const { return rotation.transpose() * (point - translation); }
    };

    struct PointProjectionSettings
    {
        real maxGap = 1e-10, referenceTolerance = 1e-10, projectionTolerance = 1e-12;
        int maxIterations = 20;
        void Validate() const;
    };
    enum class PointProjectionStatus
    {
        Success,
        Degenerate,
        NotConverged,
        OutsideReference,
        GapExceeded
    };
    struct PointProjection
    {
        PointProjectionStatus status = PointProjectionStatus::NotConverged;
        tPoint reference = tPoint::Zero(), point = tPoint::Zero(), normal = tPoint::Zero();
        real distance = 0, stationarity = 0;
        int iterations = 0;
        bool Accepted() const { return status == PointProjectionStatus::Success; }
    };

    /// Borrowed column-major coordinates [3,node], without mesh, MPI, or storage
    /// ownership. Retain the input buffer/read lease through every operation.
    /// Initial surface family: Line2, Tri3, bilinear Quad4 (including warped
    /// quads). Bounds use the convex hull of these shape functions and include
    /// the reference-tolerance envelope. High-order nodal bounds need a separate
    /// conservative construction and are deliberately not guessed here.
    class ElementGeometryView
    {
        Elem::ElemType _type;
        const real *_coordinates;
        index _nodes;

    public:
        ElementGeometryView(Elem::ElemType type, const real *coordinates, index nodes);
        Elem::ElemType Type() const { return _type; }
        Eigen::Map<const tSmallCoords> Coordinates() const { return {_coordinates, 3, _nodes}; }
        bool ContainsReference(const tPoint &reference, real tolerance = 0) const;
        tPoint Point(const tPoint &reference) const;
        tGPoint Jacobian(const tPoint &reference) const;
        SpatialBox Bounds(RigidPose pose = {}, real referenceTolerance = 0) const;
        static real ReferencePadding(real diameter, real referenceTolerance);
        void Validate() const;
        /// Unconstrained normal projection followed by containment/gap checks.
        /// It does not clamp outside projections to an edge/vertex or certify
        /// the global closest point of a folded/nonconvex surface. normal is a
        /// unit oriented normal for Tri3/Quad4 and zero for a 3D line, whose
        /// normal is not unique. Nonfinite caller inputs throw; geometric
        /// rejection returns an explicit status rather than one opaque false.
        PointProjection Project(const tPoint &query, PointProjectionSettings settings = {}) const;
    };
}
