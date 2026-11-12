#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::CPU
{
    /// Explicit numeric operands for one auxiliary phase. All field views are
    /// pinned by AuxiliaryPreparation, which also owns the worker lease.
    struct AuxiliaryKernelRequest
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        SpanView<const EquationBlockReadView> blocks;
        SpanView<const RHSFieldView> gradients;
        SpanView<const EquationBlockLocation> locations;
        SpanView<const index> vectorStarts;
        SpanView<const BoundaryCondition> conditions;
        WallDistanceView wall;
        bool hasWall = false;
        std::optional<MeanThermodynamicView> derived;
        std::array<bool, 7> enabled{};
        std::array<RHSOutputView, 7> values;
        RHSOutputView scratch;
        bool useScratch = false;
        RHSScalarView incoming;
        RigidFrame frame;
        index components = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int dimension = 0, independentSpecies = 0;
        real time = 0;
    };
    NumericalStatus PrepareAuxiliaryMeans(const AuxiliaryKernelRequest &request,
                                          const EquationSystem &equation, EquationPointEvaluator &worker,
                                          index first, index end) noexcept;
    NumericalStatus PrepareAuxiliaryFaces(const AuxiliaryKernelRequest &request,
                                          const EquationSystem &equation, const BoundaryPointOperator &boundary,
                                          EquationPointEvaluator &worker,
                                          FiniteVolumeStencilView::Indices faces) noexcept;
    void ReduceAuxiliaryCells(const AuxiliaryKernelRequest &request,
                              FiniteVolumeStencilView::Indices cells);
}
