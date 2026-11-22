#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    void PrepareLVRScalarsAndTarget(CFV::Detail::GradientCellGeometry cells, RHSScalarView means,
                                    RHSScalarView gradients, RHSOutputView scalars, RHSOutputView target,
                                    const PerfectGas &gas, int dimension, index modes, index flowFirst)
    {
        OMP::ForRows(0, means.RowCount(), [&](index cell)
                     {
            FlowState flow{};
            for (int component = 0; component < 5; ++component)
                flow[component] = means(cell, flowFirst + component);
            scalars(cell, 0) = gas.Pressure(flow);
            scalars(cell, 1) = flow[0];
            if (modes)
                CFV::Math::ConvertGradientRow(cells, gradients, target, cell, 0, dimension, modes, true); });
    }
    NumericalStatus BuildLVRFaceSensors(FiniteVolumeStencilView stencil, RHSFieldView means,
                                        RHSFieldView gradients, RHSFieldView raw, int dimension, index flowFirst,
                                        const PerfectGas &gas, const ThermalBounds &bounds,
                                        const CFV::LimitedVRKernelParameters &settings,
                                        RHSOutputView sensors, Runtime::RowSelection selection) noexcept
    {
        return OMP::ForRowsStatus(0, selection.RowCount(), [&](index row) -> NumericalStatus
                                  { return EvaluateLVRFaceSignals(stencil, means, gradients, raw, sensors, selection.RowAt(row),
                                                                  dimension, flowFirst, gas, bounds, settings); });
    }
    void GatherLVRBranches(FiniteVolumeStencilView stencil, RHSScalarView sensors, RHSOutputView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base)
    {
        DNDS_assert(base.RowCount() == branches.RowCount());
        OMP::ForRows(0, branches.RowCount(), [&](index iCell)
                     {
            std::array<real, 4> values{};
            for (const index iFace : stencil.topology.CellFaces(iCell))
            {
                const index iAux = stencil.routing(iFace, 0).auxiliaryFace;
                for (int j = 0; j < branches.Width(); ++j)
                    values[j] = std::max(values[j], sensors(iAux, j));
            }
            for (int j = 0; j < branches.Width(); ++j)
                branches(iCell, j) = values[j];
            for (int j = 0; j < 2; ++j)
                base(iCell, j) = values[j]; });
    }
    void MergeLVRBase(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> b,
                      RHSOutputView s, int column)
    {
        OMP::ForRows(0, s.RowCount(), [&](index iCell)
                     { s(iCell, column) = std::max(s(iCell, column), b(iCell, column)); });
    }
    void PublishLVRPenalty(RHSScalarView g,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> h,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> a, CFV::LimitedVRTemporalUpdate update)
    {
        OMP::ForRows(0, g.RowCount(), [&](index iCell)
                     { a(iCell, 0) = CFV::Math::PublishLimitedVRCell(g, h, iCell, update); });
    }
}
