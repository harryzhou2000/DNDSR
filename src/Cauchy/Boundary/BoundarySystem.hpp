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
        NoSlipWall
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
        /// Physical face value for reconstruction/gradient input. Wall normal
        /// is outward from the interior parent. Extrapolation leaves U unchanged.
        FlowState FaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal) const;
        /// Riemann exterior state. Wall velocity is prescribed; this initial
        /// fixed-grid implementation does not imply ALE geometric conservation.
        FlowState Exterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal) const;
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
