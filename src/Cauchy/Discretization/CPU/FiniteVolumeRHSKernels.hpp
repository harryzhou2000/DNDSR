#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Discretization/FiniteVolumeKernelPolicy.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Cauchy/State/EquationViews.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"

namespace DNDS::Cauchy::CPU
{
    /// Geometry, flags and explicit operands of the fixed-size flow5 leaf.
    /// No FiniteVolume/Workspace owner or access acquisition crosses this boundary.
    struct FlowRHSRequest
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        std::optional<CFV::ReconstructionBasisEvaluationView> basis;
        SpanView<const BoundaryCondition> conditions;
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        ThermalBounds thermalBounds;
        RigidFrame frame;
        index ownedCells = 0, allCells = 0, forceSlots = 0;
        int dimension = 0, modes = 0;
        bool direct = false, linear = false, firstOrder = false;
        bool polynomialViscosity = false, hasLocalInterfaceRows = false, useStencil = false;
        real time = 0;
    };
    NumericalResult<RHSKernelDiagnostics> EvaluateFlowFlux(const FlowRHSRequest &, const EquationSystem &,
                                                           const Riemann::RiemannFlux &, const RHSReadViews &,
                                                           const RHSWriteViews &, CFV::ReconstructionRegion) noexcept;
    NumericalStatus AddFlowRotationSource(const FlowRHSRequest &, const RHSReadViews &,
                                          const RHSWriteViews &, CFV::ReconstructionRegion) noexcept;

    /// Layout-independent equation leaf input. Per-equation descriptor tables
    /// are small immutable metadata; mesh-sized fields are explicit Array views.
    struct EquationRHSRequest
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        std::optional<CFV::ReconstructionBasisEvaluationView> basis;
        SpanView<const BoundaryCondition> conditions;
        SpanView<const EquationBlockReadView> meanBlocks, recBlocks, viscousRecBlocks;
        SpanView<const EquationBlockLocation> directLocations;
        SpanView<const RHSFieldView> directViews, directViscousViews;
        SpanView<const index> logicalFirst, vectorStarts;
        EquationBlockReadView ransMean;
        std::optional<RHSScalarView> preparedMean;
        RHSFieldView waveJump, pressureMinimum, primitiveGradientValues;
        RHSScalarView incoming;
        RHSOutputView faceValues, lowFaceValues;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> wallFlux;
        WallDistanceView wall;
        EquationThermalView thermal;
        RigidFrame frame;
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        index ownedCells = 0, width = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int dimension = 0, modes = 0, species = 0;
        bool direct = false, linear = false, firstOrder = false, viscous = false;
        bool primitiveGradient = false, hasWall = false, positivity = false, faceFallback = false;
        bool useStencil = false, preparedWaveJump = false, preparedPressureMinimum = false, produceLow = false;
        real time = 0;
    };
    /// Shape dispatch and all numeric instantiations stay in the CPU TU.
    /// Returns local fallback count; the facade publishes diagnostics afterward.
    NumericalResult<index> EvaluateEquationFlux(const EquationRHSRequest &, const EquationSystem &,
                                                const BoundaryPointOperator &, const Riemann::RiemannFlux &,
                                                EquationPointEvaluator &, EquationLayoutView<const real>,
                                                EquationLayoutView<const real>, EquationLayoutView<real>,
                                                EquationLayoutView<real>, CFV::ReconstructionRegion) noexcept;
}
