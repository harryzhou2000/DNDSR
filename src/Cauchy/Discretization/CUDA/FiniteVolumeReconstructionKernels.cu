#include "Cauchy/Discretization/CUDA/FiniteVolumeReconstructionKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        using Means = R::PairReadView<real, R::AccessBoundsChecks, R::TableRows>;
        using Status = R::ArrayView<NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>>;
        using Flags = R::ArrayView<int, R::AccessBoundsChecks, R::FixedRows<1>>;
        __device__ FlowState Load(Means mean, index iCell)
        {
            FlowState value{};
            for (int j = 0; j < 5; ++j)
                value[j] = mean(iCell, j);
            return value;
        }
        __global__ void Characteristics(CharacteristicGeometry geometry, Means mean, PerfectGas gas,
                                        CharacteristicOutput forward, CharacteristicOutput inverse, Status status, Flags fallback)
        {
            for (index iFace = index(blockIdx.x) * blockDim.x + threadIdx.x; iFace < forward.RowCount(); iFace += index(gridDim.x) * blockDim.x)
            {
                for (int j = 0; j < 25; ++j)
                    forward(iFace, j) = inverse(iFace, j) = (j % 6 == 0);
                status(iFace, 0) = {};
                fallback(iFace, 0) = 0;
                const index entry = geometry.faceEntries(iFace, 0);
                if (entry == UnInitIndex)
                    continue;
                const auto &sample = geometry.samples(entry, 0);
                const index leftCell = sample.side ? sample.other : sample.cell;
                const index rightCell = sample.side ? sample.cell : sample.other;
                const auto left = Load(mean, leftCell);
                auto right = rightCell == UnInitIndex ? left : Load(mean, rightCell);
                if (rightCell != UnInitIndex && sample.changesFrame)
                {
                    const auto original = right;
                    for (int i = 0; i < 3; ++i)
                    {
                        right[1 + i] = 0;
                        for (int j = 0; j < 3; ++j)
                            right[1 + i] += sample.rotation[sample.side ? j + 3 * i : i + 3 * j] * original[1 + j];
                    }
                }
                const Vector3 normal{geometry.quadrature(sample.point, 3), geometry.quadrature(sample.point, 4), geometry.quadrature(sample.point, 5)};
                const auto result = TryPerfectGasCharacteristics(gas, left, right, normal);
                status(iFace, 0) = result.status.At(-1, iFace);
                if (!result)
                    continue;
                fallback(iFace, 0) = result.value.componentwiseFallback;
                for (int j = 0; j < 25; ++j)
                {
                    forward(iFace, j) = result.value.forward[j];
                    inverse(iFace, j) = result.value.inverse[j];
                }
            }
        }
    }
    void PrepareFaceCharacteristicMatrices(R::ExecutionContext context, CharacteristicGeometry geometry, Means mean,
                                           PerfectGas gas, CharacteristicOutput forward, CharacteristicOutput inverse, Status status, Flags fallback)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA characteristics require a CUDA context");
        gas.Validate();
        R::ValidateViewDevices(context.device, geometry, mean, forward, inverse, status, fallback);
        DNDS_check_throw_info(mean.owned.kind == R::RowLayoutKind::Table && mean.owned.Width() >= 5 && mean.owned.Width() <= 7 &&
                                  (!mean.ghost.RowCount() || (mean.ghost.kind == R::RowLayoutKind::Table && mean.ghost.Width() == mean.owned.Width())) &&
                                  forward.kind == R::RowLayoutKind::Table && inverse.kind == R::RowLayoutKind::Table &&
                                  forward.Width() == 25 && inverse.Width() == 25 && forward.RowCount() == inverse.RowCount() &&
                                  geometry.faceEntries.RowCount() == forward.RowCount() && status.RowCount() == forward.RowCount() && fallback.RowCount() == forward.RowCount(),
                              "CUDA characteristic field/geometry shape differs");
        R::CheckKernelArgumentBudget<CharacteristicGeometry, Means, PerfectGas, CharacteristicOutput, CharacteristicOutput, Status, Flags>();
        if (!forward.RowCount())
            return;
        Characteristics<<<unsigned(std::min<index>(256, (forward.RowCount() + 127) / 128)), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
            geometry, mean, gas, forward, inverse, status, fallback);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA characteristic producer launch failed");
    }
}
