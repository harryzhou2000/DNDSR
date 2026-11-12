#pragma once
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Motion/RigidMotion.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::CPU
{
    struct SourceGradientView
    {
        RHSScalarView values;
        int componentStride = 1, direction = 0;
        real operator()(index iPoint, index component) const
        {
            return values(iPoint, component * componentStride + direction);
        }
    };
    struct SourceKernelRequest
    {
        CFV::FiniteVolumeHostView metric;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> weights, lengths;
        SpanView<const EquationBlockReadView> blocks;
        SpanView<const EquationBlockLocation> locations;
        SpanView<const index> logicalFirst;
        SpanView<const RHSScalarView> samples;
        SpanView<const std::array<SourceGradientView, 3>> gradients;
        WallDistanceView walls;
        EquationBlockReadView ransMean;
        EquationBlockArrayView<real> speciesOutput, ransOutput;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> matrix;
        std::optional<EquationFVDetail::MeanLinearizationOutput> linearization;
        index cells = 0, components = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int dimension = 0, independentSpecies = 0;
        bool chemical = false, fullBlock = false;
        real scale = 1;
    };
    NumericalStatus AddMeanSourceJacobian(const SourceKernelRequest &, const EquationSystem &, PointEvaluator &) noexcept;
    NumericalStatus AddQuadratureSourceJacobian(const SourceKernelRequest &, const EquationSystem &, PointEvaluator &) noexcept;
    NumericalStatus AddVolumeSources(const SourceKernelRequest &, const EquationSystem &, PointEvaluator &) noexcept;
    void ReplicateSourceValues(Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets,
                               RHSScalarView input, RHSOutputView output, int stride = 1, int direction = 0);
    void AddEquationRotationSource(CFV::FiniteVolumeHostView metric, const RigidFrame &frame,
                                   EquationBlockReadView flow, EquationBlockArrayView<real> output,
                                   std::optional<EquationBlockArrayView<real>> lowOutput);
}
