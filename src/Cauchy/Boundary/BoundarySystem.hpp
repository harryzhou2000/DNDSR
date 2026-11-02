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
        DoubleMachBottom
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
        FlowProfileSettings profile;
        FlowState prescribedPrimitive{};
        /// Physical face value for reconstruction/gradient input. Wall normal
        /// is outward from the interior parent. Extrapolation leaves U unchanged.
        FlowState FaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0) const;
        /// Riemann exterior state. Wall velocity is prescribed; this initial
        /// fixed-grid implementation does not imply ALE geometric conservation.
        FlowState Exterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0) const;
        /// Column-major derivative of Exterior with respect to its conservative
        /// interior state, with branch, normal, prescribed data and time frozen.
        /// The derivative is one-sided at characteristic/DMR branch switches;
        /// callers linearize a selected physical state, not the switch itself.
        /// This model-owned BC derivative is separate from Riemann dissipation.
        std::array<real, 25> ExteriorJacobian(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point = {}, real time = 0) const;
    };
    class BoundarySystem
    {
        std::map<int, BoundaryCondition> _conditions;

    public:
        explicit BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas &gas);
        const BoundaryCondition &At(int zone) const;
        void ValidateUsed(const std::set<int> &zones) const;
    };
}
