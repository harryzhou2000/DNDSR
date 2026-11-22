#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        using BaseInput = R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<2>>;
        using BaseOutput = R::ArrayView<real, R::AccessBoundsChecks, R::FixedRows<2>>;
        using ScalarOutput = R::ArrayView<real, R::AccessBoundsChecks, R::FixedRows<1>>;
        using StatusInput = R::ArrayView<const NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>>;
        __global__ void PrepareLVRScalarsAndTargetKernel(CFV::Detail::GradientCellGeometry cells,
                                                         LVRInput means, LVRInput gradients,
                                                         LVROutput scalars, LVROutput target, PerfectGas gas,
                                                         int dimension, index modes, index flowFirst, LVRStatus status)
        {
            for (index cell = index(blockIdx.x) * blockDim.x + threadIdx.x; cell < means.RowCount(); cell += index(gridDim.x) * blockDim.x)
            {
                FlowState flow{};
                for (int component = 0; component < 5; ++component)
                    flow[component] = means(cell, flowFirst + component);
                const auto point = TryPerfectGasPoint(gas, flow, false);
                status(cell, 0) = point ? NumericalStatus{} : point.status.At(cell);
                if (point)
                {
                    scalars(cell, 0) = point.value.thermo.pressure;
                    scalars(cell, 1) = flow[0];
                    if (modes)
                        CFV::Math::ConvertGradientRow(cells, gradients, target, cell, 0, dimension, modes, true);
                }
            }
        }
        __global__ void BuildLVRFaceSensorsKernel(LVRStencil descriptor, LVRMeans means, LVRMeans gradients, LVRMeans raw,
                                                  int dimension, index flowFirst, PerfectGas gas, ThermalBounds bounds,
                                                  CFV::LimitedVRKernelParameters settings, LVROutput output, LVRStatus status,
                                                  R::RowSelection selection)
        {
            const auto &stencil = descriptor(0, 0);
            for (index row = index(blockIdx.x) * blockDim.x + threadIdx.x; row < selection.RowCount(); row += index(gridDim.x) * blockDim.x)
            {
                const index slot = selection.RowAt(row);
                status(slot, 0) = EvaluateLVRFaceSignals(stencil, means, gradients, raw, output, slot,
                                                         dimension, flowFirst, gas, bounds, settings);
            }
        }
        __global__ void GatherLVRBranchesKernel(LVRStencil descriptor, LVRInput sensors, StatusInput faceStatus,
                                                LVROutput branches, BaseOutput base, LVRStatus status)
        {
            const auto &stencil = descriptor(0, 0);
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < branches.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                std::array<real, 4> values{};
                NumericalStatus result;
                for (const index iFace : stencil.topology.CellFaces(iCell))
                {
                    const index iAux = stencil.routing(iFace, 0).auxiliaryFace;
                    if (!faceStatus(iAux, 0))
                    {
                        result = faceStatus(iAux, 0).At(iCell, iFace);
                        break;
                    }
                    for (int j = 0; j < branches.Width(); ++j)
                        values[j] = std::max(values[j], sensors(iAux, j));
                }
                status(iCell, 0) = result;
                if (result)
                {
                    for (int j = 0; j < branches.Width(); ++j)
                        branches(iCell, j) = values[j];
                    for (int j = 0; j < 2; ++j)
                        base(iCell, j) = values[j];
                }
            }
        }
        __global__ void MergeLVRBaseKernel(BaseInput base, LVROutput seeds, int column)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < seeds.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                seeds(iCell, column) = std::max(seeds(iCell, column), base(iCell, column));
        }
        __global__ void PublishLVRPenaltyKernel(LVRInput branches, LVROutput history, ScalarOutput penalty, CFV::LimitedVRTemporalUpdate update)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < branches.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                penalty(iCell, 0) = CFV::Math::PublishLimitedVRCell(branches, history, iCell, update);
            }
        }
        unsigned Blocks(index count) { return unsigned(std::min<index>((count + 127) / 128, 1024)); }
        void CheckLaunch() { DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA physical LVR launch failed"); }
        template <class... Views>
        void CheckDevice(R::ExecutionContext context, const Views &...views)
        {
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA LVR requires a CUDA context");
            R::ValidateViewDevices(context.device, views...);
        }
        template <class View>
        void Table(View view, index rows, index width)
        {
            DNDS_check_throw_info(view.RowCount() == rows && view.Width() == width && view.kind == R::RowLayoutKind::Table,
                                  "CUDA physical LVR table shape differs");
        }
    }
    void PrepareLVRScalarsAndTarget(R::ExecutionContext c, CFV::Detail::GradientCellGeometry cells,
                                    LVRInput means, LVRInput gradients, LVROutput scalars, LVROutput target,
                                    PerfectGas gas, int dimension, index modes, index flowFirst, LVRStatus status)
    {
        CheckDevice(c, means, scalars, status);
        gas.Validate();
        Table(scalars, means.RowCount(), 2);
        DNDS_check_throw_info(flowFirst >= 0 && flowFirst + 5 <= means.Width() && status.RowCount() == means.RowCount(),
                              "invalid CUDA LVR scalar shape");
        if (modes)
        {
            CheckDevice(c, cells, gradients, target);
            Table(gradients, means.RowCount(), means.Width() * dimension);
            Table(target, means.RowCount(), means.Width() * modes);
        }
        if (!means.RowCount())
            return;
        PrepareLVRScalarsAndTargetKernel<<<Blocks(means.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(
            cells, means, gradients, scalars, target, gas, dimension, modes, flowFirst, status);
        CheckLaunch();
    }
    void BuildLVRFaceSensors(R::ExecutionContext c, LVRStencil stencil, LVRMeans means, LVRMeans gradients, LVRMeans raw,
                             int dimension, index flowFirst, PerfectGas gas, ThermalBounds bounds,
                             CFV::LimitedVRKernelParameters settings, LVROutput sensors, LVRStatus status,
                             R::RowSelection selection)
    {
        CheckDevice(c, stencil, means, gradients, raw, sensors, status, selection);
        gas.Validate();
        Table(gradients.owned, means.owned.RowCount(), means.owned.Width() * dimension);
        if (means.ghost.RowCount())
            Table(gradients.ghost, means.ghost.RowCount(), means.owned.Width() * dimension);
        Table(sensors, sensors.RowCount(), sensors.Width());
        DNDS_check_throw_info(stencil.RowCount() == 1 && flowFirst >= 0 && flowFirst + 5 <= means.owned.Width() &&
                                  status.RowCount() == sensors.RowCount(),
                              "invalid CUDA LVR face status/target shape");
        if (!selection.RowCount())
            return;
        BuildLVRFaceSensorsKernel<<<Blocks(selection.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(
            stencil, means, gradients, raw, dimension, flowFirst, gas, bounds, settings, sensors, status, selection);
        CheckLaunch();
    }
    void GatherLVRBranches(R::ExecutionContext c, LVRStencil stencil, LVRInput sensors, StatusInput faceStatus,
                           LVROutput branches, BaseOutput base, LVRStatus status)
    {
        CheckDevice(c, stencil, sensors, faceStatus, branches, base, status);
        Table(sensors, sensors.RowCount(), sensors.Width());
        Table(branches, branches.RowCount(), sensors.Width());
        DNDS_check_throw_info(stencil.RowCount() == 1 && base.RowCount() == branches.RowCount() && status.RowCount() == branches.RowCount() &&
                                  faceStatus.RowCount() == sensors.RowCount(),
                              "invalid CUDA LVR gather shape");
        R::CheckKernelArgumentBudget<LVRStencil, LVRInput, StatusInput, LVROutput, BaseOutput, LVRStatus>();
        if (!branches.RowCount())
            return;
        GatherLVRBranchesKernel<<<Blocks(branches.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(stencil, sensors, faceStatus, branches, base, status);
        CheckLaunch();
    }
    void MergeLVRBase(R::ExecutionContext c, BaseInput base, LVROutput seeds, int column)
    {
        CheckDevice(c, base, seeds);
        Table(seeds, base.RowCount(), seeds.Width());
        DNDS_check_throw_info(column == 0 || column == 1, "invalid LVR base column");
        R::CheckKernelArgumentBudget<BaseInput, LVROutput, int>();
        if (!seeds.RowCount())
            return;
        MergeLVRBaseKernel<<<Blocks(seeds.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(base, seeds, column);
        CheckLaunch();
    }
    void PublishLVRPenalty(R::ExecutionContext c, LVRInput branches, LVROutput history, ScalarOutput penalty, CFV::LimitedVRTemporalUpdate update)
    {
        CheckDevice(c, branches, history, penalty);
        Table(branches, history.RowCount(), history.Width() + 1);
        DNDS_check_throw_info(penalty.RowCount() == branches.RowCount(), "invalid LVR penalty shape");
        R::CheckKernelArgumentBudget<LVRInput, LVROutput, ScalarOutput, CFV::LimitedVRTemporalUpdate>();
        if (!branches.RowCount())
            return;
        PublishLVRPenaltyKernel<<<Blocks(branches.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(c)>>>(branches, history, penalty, update);
        CheckLaunch();
    }
}
