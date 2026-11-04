#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include <set>

namespace DNDS::Cauchy
{
    enum class BoundaryKind
    {
        Interior,
        Extrapolate,
        Farfield,
        SlipWall,
        NoSlipWall,
        Analytic,
        DoubleMachBottom,
        Interface
    };
    /// Resolved numerical BC, independent of mesh storage and time scheduling.
    /// Construction validates names; a kernel receives plain parameters.
    struct BoundaryCondition
    {
        BoundaryKind kind = BoundaryKind::Interior;
        FlowState farfield{};
        Vector3 velocity{};
        real temperature = 1;
        bool isothermal = false;
        bool wallMovesWithMesh = true;
        FlowProfileSettings profile;
        FlowState prescribedPrimitive{};
        /// Generic metadata resolves primitive inputs without inventing a
        /// perfect-gas conservative state. Historical PG methods reject it.
        bool hasPerfectGasState = true;
        /// Resolved absolute wall velocity in the mesh-bound Cartesian frame.
        /// Shared kinematics for equation-specific adiabatic wall producers.
        Vector3 WallVelocityAt(const Vector3 &point, const RigidFrame *frame = nullptr) const;
        /// Physical face value for reconstruction/gradient input. Wall normal
        /// is outward from the interior parent. Extrapolation leaves U unchanged.
        FlowState FaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
        /// Riemann exterior state. Wall velocity is prescribed; this initial
        /// rigid-frame implementation does not imply deforming ALE support.
        FlowState Exterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
        /// Column-major derivative of Exterior with respect to its conservative
        /// interior state, with branch, normal, prescribed data and time frozen.
        /// The derivative is one-sided at characteristic/DMR branch switches;
        /// callers linearize a selected physical state, not the switch itself.
        /// This model-owned BC derivative is separate from Riemann dissipation.
        std::array<real, 25> ExteriorJacobian(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
    };
    class BoundarySystem
    {
        std::map<int, BoundaryCondition> _conditions;
        BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas *gas);

    public:
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas &gas);
        /// Resolve metadata only; equation-specific operators prepare values.
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings);
        const BoundaryCondition &At(int zone) const;
        void ValidateUsed(const std::set<int> &zones) const;
    };
}
