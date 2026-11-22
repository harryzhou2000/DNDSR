#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/Positivity.hpp"
#include "CFV/Math/GradientEvaluation.hpp"
#include "CFV/Math/LimitedVR.hpp"

namespace DNDS::Cauchy
{
    template <class Means, class Gradients>
    DNDS_DEVICE_CALLABLE FlowState EvaluateLVRFaceTrace(
        const FiniteVolumeStencilView &stencil, Means means, Gradients gradients,
        index cell, const Geom::tPoint &point, int dimension, index flowFirst) noexcept
    {
        FlowState value{};
        const Geom::tPoint displacement = point - stencil.metric.GetCellBary(cell);
        for (int j = 0; j < 5; ++j)
        {
            const index component = flowFirst + j;
            value[j] = means(cell, component);
            for (int d = 0; d < dimension; ++d)
                value[j] += gradients(cell, component * dimension + d) * displacement[d];
        }
        return value;
    }

    template <class Means, class Gradients, class Output>
    DNDS_DEVICE_CALLABLE NumericalStatus EvaluateLVRFaceSignals(
        const FiniteVolumeStencilView &stencil, Means means, Gradients gradients, Gradients raw, Output signals,
        index slot, int dimension, index flowFirst, const PerfectGas &gas,
        const ThermalBounds &bounds, const CFV::LimitedVRKernelParameters &settings) noexcept
    {
        const index face = stencil.stencilFaces(slot, 0);
        const auto parents = stencil.topology.FaceCells(face);
        if (parents[1] == UnInitIndex)
        {
            for (int j = 0; j < signals.Width(); ++j)
                signals(slot, j) = 0;
            return {};
        }
        const auto center = stencil.metric.GetFaceCent(face);
        const auto left = EvaluateLVRFaceTrace(stencil, means, gradients, parents[0], center, dimension, flowFirst);
        auto right = EvaluateLVRFaceTrace(stencil, means, gradients, parents[1],
                                          stencil.metric.periodic.FacePoint(face, 1, center), dimension, flowFirst);
        if (stencil.metric.periodic.Rotates(face))
        {
            const auto rotation = stencil.metric.periodic.OtherRotation(face, 0);
            const auto old = right;
            for (int d = 0; d < 3; ++d)
            {
                right[1 + d] = 0;
                for (int k = 0; k < 3; ++k)
                    right[1 + d] += rotation(d, k) * old[1 + k];
            }
        }
        if (!(ThermallyAdmissible(left, bounds) && ThermallyAdmissible(right, bounds)))
            return NumericalStatus{NumericalError::InvalidState, 1}.At(parents[0], face);
        const auto l = gas.Primitive(left), r = gas.Primitive(right);
        const auto normal = stencil.metric.GetFaceMeanNorm(face);
        real unLeft = 0, unRight = 0;
        for (int d = 0; d < 3; ++d)
        {
            unLeft += l.velocity[d] * normal[d];
            unRight += r.velocity[d] * normal[d];
        }
        const auto sensor = CFV::Math::EvaluateLimitedVRFaceSensor(
            l.density, r.density, l.pressure, r.pressure, unLeft, unRight, l.soundSpeed, r.soundSpeed, settings);
        const auto support = CFV::Math::EvaluateLimitedVRFaceSensor(
                                 means(parents[0], flowFirst), means(parents[1], flowFirst), 1, 1, 0, 0, 1, 1, settings)
                                 .contact;
        signals(slot, 0) = sensor.shock;
        signals(slot, 1) = settings.contactUseCellMeanJump ? support : sensor.contact;
        signals(slot, 2) = settings.contactExtremumUseCellMeanSupport ? support : signals(slot, 1);
        if (signals.Width() == 4)
        {
            real weak = 0;
            if (settings.weakAlphaMax > 0)
            {
                const auto evaluate = [&](index cell, const Geom::tPoint &point)
                {
                    FlowState mean{};
                    for (int j = 0; j < 5; ++j)
                        mean[j] = means(cell, flowFirst + j);
                    const auto value = gas.Primitive(mean);
                    const Geom::tPoint displacement = point - stencil.metric.GetCellBary(cell);
                    real speedSquared = 0, dp = 0;
                    for (int d = 0; d < 3; ++d)
                        speedSquared += value.velocity[d] * value.velocity[d];
                    for (int d = 0; d < dimension; ++d)
                    {
                        const auto slope = [&](index column)
                        { return raw(cell, (flowFirst + column) * dimension + d); };
                        real increment = slope(4) + .5 * speedSquared * slope(0);
                        for (int j = 0; j < 3; ++j)
                            increment -= value.velocity[j] * slope(1 + j);
                        dp += (gas.gamma - 1) * increment * displacement[d];
                    }
                    return std::pair{value, dp};
                };
                const auto [meanLeft, dpLeft] = evaluate(parents[0], center);
                const auto [meanRight, dpRight] = evaluate(parents[1], stencil.metric.periodic.FacePoint(face, 1, center));
                real compression = 0;
                const bool rotating = stencil.metric.periodic.Rotates(face);
                for (int d = 0; d < 3; ++d)
                {
                    real rightVelocity = meanRight.velocity[d];
                    if (rotating)
                    {
                        rightVelocity = 0;
                        const auto rotation = stencil.metric.periodic.OtherRotation(face, 0);
                        for (int j = 0; j < 3; ++j)
                            rightVelocity += rotation(d, j) * meanRight.velocity[j];
                    }
                    compression += (meanLeft.velocity[d] - rightVelocity) * normal[d];
                }
                const real impedance = .5 * (meanLeft.density * meanLeft.soundSpeed + meanRight.density * meanRight.soundSpeed);
                weak = CFV::Math::LimitedVRWeakShock(meanLeft.pressure, meanRight.pressure, dpLeft, dpRight,
                                                     compression, impedance, settings);
            }
            signals(slot, 3) = weak;
        }
        return {};
    }
}
