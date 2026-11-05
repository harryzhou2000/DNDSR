#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Physics/Chemistry/ChemicalSource.hpp"
#include <set>

namespace DNDS::Cauchy
{
    class EquationSystem;
    class EquationPointEvaluator;
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
        /// Resolved Ns-1 composition for prescribed states. Empty for a
        /// species-free profile; never resolve names inside numerical loops.
        std::vector<real> prescribedFractions;
        std::array<real, 2> prescribedRANS{};
        /// Validate the equation-state producer at operator setup. The legacy
        /// perfect-gas methods retain their existing larger profile set.
        void ValidateEquationProducer() const;
        /// EOS-aware face/exterior state. rhoY is preallocated Ns-1 point
        /// scratch and is replaced only when the BC changes composition or
        /// density. No exchange, Array access, worker acquisition or allocation.
        /// Prescribed velocities use ground axes; results use mesh axes.
        FlowState FaceValue(EquationPointEvaluator &evaluator, const FlowState &inside,
                            Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                            const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
        FlowState Exterior(EquationPointEvaluator &evaluator, const FlowState &inside,
                           Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                           const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
        /// Resolved absolute wall velocity in the mesh-bound Cartesian frame.
        /// Shared kinematics for equation-specific adiabatic wall producers.
        Vector3 WallVelocityAt(const Vector3 &point, const RigidFrame *frame = nullptr) const;
        /// Physical wall trace for diagnostics/prescribed wall data. Wall normal
        /// is outward from the interior parent. Extrapolation leaves U unchanged.
        FlowState FaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
        /// Euler-compatible Riemann/VR exterior state. Isothermal no-slip
        /// reflects velocity and sets rho=p/(R*T_wall), preserving composition.
        /// Direct conservative gradients instead consume (inside+exterior)/2.
        /// Wall velocity is prescribed; this initial
        /// rigid-frame implementation does not imply deforming ALE support.
        FlowState Exterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
        /// Column-major derivative of Exterior with respect to its conservative
        /// interior state, with branch, normal, prescribed data and time frozen.
        /// The derivative is one-sided at characteristic/DMR branch switches;
        /// callers linearize a selected physical state, not the switch itself.
        /// This model-owned BC derivative is separate from Riemann dissipation.
        std::array<real, 25> ExteriorJacobian(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
        /// Euler's approximate linMode=1 protocol: homogeneous wall reflection
        /// before thermal/scalar correction; zero for non-wall boundaries.
        /// Distinct from the exact physical derivative above. No branch
        /// derivative or wall-temperature derivative enters LU-SGS/Roe blocks.
        std::array<real, 25> ApproximateExteriorJacobian(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0, const RigidFrame *frame = nullptr) const;
    };
    class BoundarySystem
    {
        std::map<int, BoundaryCondition> _conditions;
        BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas *gas);

    public:
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas &gas);
        /// Resolve equation-specific composition once; retains no worker.
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const EquationSystem &equation);
        /// Resolve metadata only; equation-specific operators prepare values.
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings);
        const BoundaryCondition &At(int zone) const;
        void ValidateUsed(const std::set<int> &zones) const;
    };
}
