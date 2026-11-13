#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <limits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        using Cells = R::ArrayView<const real, false, R::FixedRows<8>>;
        using InputSummary = R::ArrayView<const StateReductionValue, R::AccessBoundsChecks, R::FixedRows<1>>;
        __device__ StateReductionValue Merge(StateReductionValue a, StateReductionValue b)
        {
            return {fmin(a.value, b.value), a.firstInvalid < 0 ? b.firstInvalid : b.firstInvalid < 0 ? a.firstInvalid
                                                                                                     : (a.firstInvalid < b.firstInvalid ? a.firstInvalid : b.firstInvalid)};
        }
        struct MergeValues
        {
            __device__ StateReductionValue operator()(StateReductionValue a, StateReductionValue b) const { return Merge(a, b); }
        };
        __device__ void BlockResult(StateReductionValue value, StateReductionOutput output)
        {
            using Reduce = cub::BlockReduce<StateReductionValue, Threads, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
            __shared__ typename Reduce::TempStorage temporary;
            const auto reduced = Reduce(temporary).Reduce(value, MergeValues{});
            if (!threadIdx.x)
                output(blockIdx.x, 0) = reduced;
        }
        __global__ void AdmissibilityRows(StateInput means, AdmissibilityView bounds, StateReductionOutput scratch)
        {
            StateReductionValue value{1, -1};
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                if (!bounds.Admissible(means[iCell]))
                    value = Merge(value, {0, iCell});
            BlockResult(value, scratch);
        }
        __global__ void StableStepRows(StateInput sums, Cells cells, real cfl, StateReductionOutput scratch)
        {
            StateReductionValue value{std::numeric_limits<real>::max(), -1};
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < sums.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const real radius = sums(iCell, 0);
                if (!(isfinite(radius) && radius > 0))
                    value = Merge(value, {value.value, iCell});
                else
                    value = Merge(value, {cfl * cells(iCell, 0) / radius, -1});
            }
            BlockResult(value, scratch);
        }
        __global__ void Finish(InputSummary scratch, StateReductionOutput result)
        {
            StateReductionValue value{std::numeric_limits<real>::max(), -1};
            for (index iRow = threadIdx.x; iRow < scratch.RowCount(); iRow += blockDim.x)
                value = Merge(value, scratch(iRow, 0));
            BlockResult(value, result);
        }
        void Check(R::ExecutionContext context, StateInput input, StateReductionOutput scratch, StateReductionOutput result)
        {
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA state reduction needs a CUDA context");
            R::ValidateViewDevices(context.device, input, scratch, result);
            DNDS_check_throw_info(input.kind == R::RowLayoutKind::Table && scratch.RowCount() == StateReductionScratchRows(input.RowCount()) &&
                                      result.RowCount() == 1 && scratch.data != result.data,
                                  "CUDA state reduction workspace shape/alias differs");
        }
        void Complete(R::ExecutionContext context, StateReductionOutput scratch, StateReductionOutput result)
        {
            R::CheckKernelArgumentBudget<InputSummary, StateReductionOutput>();
            Finish<<<1, Threads, 0, static_cast<cudaStream_t>(context.stream)>>>(scratch.ConstView(), result);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA state reduction launch failed");
        }
    }
    index StateReductionScratchRows(index ownedRows)
    {
        DNDS_check_throw_info(ownedRows >= 0, "negative CUDA state reduction row count");
        return std::max<index>(1, std::min<index>(256, ownedRows / Threads + (ownedRows % Threads != 0)));
    }
    void AdmissibleMeans(R::ExecutionContext context, StateInput means, AdmissibilityView bounds,
                         StateReductionOutput scratch, StateReductionOutput result)
    {
        Check(context, means, scratch, result);
        DNDS_check_throw_info(means.Width() >= 5 && means.Width() <= 7 && bounds.thermal.components == means.Width() &&
                                  bounds.thermal.flowFirst == 0 && bounds.thermal.independent == 0 &&
                                  bounds.thermal.speciesEnergies[0].empty() && bounds.thermal.speciesEnergies[1].empty() &&
                                  bounds.rans.components == means.Width() - 5 && (!bounds.rans.components || bounds.ransFirst == 5),
                              "CUDA admissibility supports combined perfect-gas flow5/SA/2EQ only");
        R::CheckKernelArgumentBudget<StateInput, AdmissibilityView, StateReductionOutput>();
        AdmissibilityRows<<<unsigned(scratch.RowCount()), Threads, 0, static_cast<cudaStream_t>(context.stream)>>>(means, bounds, scratch);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA admissibility row launch failed");
        Complete(context, scratch, result);
    }
    void MinimumStableStep(R::ExecutionContext context, StateInput sums, Cells cells, real cfl,
                           StateReductionOutput scratch, StateReductionOutput result)
    {
        Check(context, sums, scratch, result);
        cells.ValidateDevice(context.device);
        DNDS_check_throw_info(sums.Width() == 1 && cells.RowCount() >= sums.RowCount() && std::isfinite(cfl) && cfl > 0,
                              "CUDA stable-step shape/control differs");
        R::CheckKernelArgumentBudget<StateInput, Cells, real, StateReductionOutput>();
        StableStepRows<<<unsigned(scratch.RowCount()), Threads, 0, static_cast<cudaStream_t>(context.stream)>>>(sums, cells, cfl, scratch);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA stable-step row launch failed");
        Complete(context, scratch, result);
    }
}
