#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "Geom/Spatial/ElementGeometry.hpp"
#include "DNDS/Config/ConfigParam.hpp"

namespace DNDS::Cauchy
{
    /// Prescribed constant angular/translation velocity. Reference coordinates
    /// stay unchanged; absolute fluid vectors use the moving mesh's axes.
    struct RigidMotionSettings
    {
        Vector3 angularVelocity{}, translationVelocity{}, center{};
        void Validate(int dimension) const;
        DNDS_DECLARE_CONFIG(RigidMotionSettings)
        {
            DNDS_FIELD(angularVelocity, "Constant angular velocity in mesh/world initial axes; 2D requires the z axis");
            DNDS_FIELD(translationVelocity, "Constant ground-frame velocity of the reference center");
            DNDS_FIELD(center, "Fixed reference rotation center at time zero");
        }
    };

    /// Immutable stage-time kinematics, prepared once outside point loops.
    /// Stored momentum is absolute velocity expressed in these moving axes;
    /// energy is absolute total energy. No relative kinetic-energy correction.
    struct RigidFrame
    {
        Geom::RigidPose pose;
        Vector3 angularVelocity{}, centerVelocity{}, center{};
        bool moving = false;
        Vector3 GridVelocity(const Vector3 &referencePoint) const;
        Vector3 WorldVector(const Vector3 &local) const;
        Vector3 LocalVector(const Vector3 &world) const;
        Vector3 WorldPoint(const Vector3 &referencePoint) const;
        FlowState WorldState(FlowState local) const;
        FlowState LocalState(FlowState world) const;
        /// Component-axis source only: -Omega cross momentum; no energy work.
        FlowState RotationSource(const FlowState &state) const;
        /// Column-major negative source derivative used by the implicit
        /// residual mass*U - scale*RHS. Volume/scaling are caller policy.
        std::array<real, 25> NegativeSourceJacobian() const;
    };

    /// Solver motion policy; Geom consumes the resulting pose without any
    /// knowledge of physical time, fluid states or moving-frame source terms.
    class RigidMotion
    {
        RigidMotionSettings _settings;

    public:
        explicit RigidMotion(RigidMotionSettings settings, int dimension);
        const RigidMotionSettings &Settings() const { return _settings; }
        RigidFrame At(real time) const;
    };
}
