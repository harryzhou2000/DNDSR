#pragma once
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    template <class View>
    FlowState Load(const View &input, index iCell)
    {
        FlowState state;
        const auto row = input[iCell];
        for (int j = 0; j < 5; ++j)
            state[j] = row[j];
        return state;
    }
    inline FlowState RotateState(FlowState state, const Geom::tGPoint &rotation)
    {
        const Geom::tPoint momentum{state[1], state[2], state[3]};
        const Geom::tPoint rotated = rotation * momentum;
        for (int d = 0; d < 3; ++d)
            state[1 + d] = rotated[d];
        return state;
    }
    /// Physical displacement from a cell average's barycenter, in that cell's
    /// own axes. The neighbor center already carries the periodic coordinate
    /// image; only vectors need the inverse frame map. No Taylor basis here.
    template <class Sample>
    inline Geom::tPoint DirectFVDisplacement(const CFV::FiniteVolumeHostView &metric, const Sample &sample, bool other)
    {
        if (!other)
            return sample.ownPoint - metric.GetCellBary(sample.cell);
        const Geom::tPoint delta = sample.ownPoint - sample.otherPoint;
        return (sample.changesFrame ? Geom::tPoint(sample.rotation.transpose() * delta) : delta) +
               metric.GetCellCent(sample.other) - metric.GetCellBary(sample.other);
    }
    template <class Sample>
    inline NumericalResult<FlowState> BoundaryFaceValue(const Sample &sample, const PerfectGas &gas, const FlowState &inside,
                                                        real time, const RigidFrame &frame, RHSScalarView incoming) noexcept
    {
        if (sample.boundary.kind == BoundaryKind::Interface)
            return Load(incoming, sample.boundaryPoint);
        return sample.boundary.TryFaceValue(gas, inside, {sample.normal[0], sample.normal[1], sample.normal[2]},
                                            {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time,
                                            OptionalRef<const RigidFrame>(frame));
    }
    template <class Sample>
    inline NumericalResult<FlowState> BoundaryExterior(const Sample &sample, const PerfectGas &gas, const FlowState &inside,
                                                       real time, const RigidFrame &frame, RHSScalarView incoming) noexcept
    {
        if (sample.boundary.kind == BoundaryKind::Interface)
            return Load(incoming, sample.boundaryPoint);
        return sample.boundary.TryExterior(gas, inside, {sample.normal[0], sample.normal[1], sample.normal[2]},
                                           {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time,
                                           OptionalRef<const RigidFrame>(frame));
    }
    /// Physical trace for the viscous gradient. An interface supplies the
    /// neighboring state; the face target is its conservative midpoint.
    template <class Sample>
    inline NumericalResult<FlowState> BoundaryGradientValue(const Sample &sample, const PerfectGas &gas, const FlowState &inside,
                                                            real time, const RigidFrame &frame, RHSScalarView incoming) noexcept
    {
        auto result = BoundaryFaceValue(sample, gas, inside, time, frame, incoming);
        if (!result)
            return result.status;
        if (sample.boundary.kind == BoundaryKind::Interface)
            for (int j = 0; j < 5; ++j)
                result.value[j] = .5 * (inside[j] + result.value[j]);
        return result;
    }
}
