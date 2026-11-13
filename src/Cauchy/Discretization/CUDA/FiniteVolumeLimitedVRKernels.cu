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
        using Faces = R::ArrayView<const LimitedVRFace, R::AccessBoundsChecks, R::FixedRows<1>>;
        using StatusInput = R::ArrayView<const NumericalStatus, R::AccessBoundsChecks, R::FixedRows<1>>;
        DNDS_DEVICE_CALLABLE FlowState Load(LVRInput input, index row)
        {
            FlowState value;
            for (int j = 0; j < 5; ++j)
                value[j] = input(row, j);
            return value;
        }
        struct MeanValues
        {
            FlowState value;
            DNDS_DEVICE_CALLABLE real operator()(index, int j) const { return value[j]; }
        };
        __global__ void Target(LVRInput means, LVRInput points, LVROutput coefficients, LVRIndices offsets,
                               ThermalBounds bounds, int power, real cutoff, LVRStatus status)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                NumericalStatus result;
                const auto mean = Load(means, iCell);
                const index begin = offsets(iCell, 0), end = offsets(iCell + 1, 0);
                for (index q = begin; q < end; ++q)
                    for (int j = 0; j < 5; ++j)
                        if (!std::isfinite(points(q, j)))
                            result = NumericalStatus{NumericalError::InvalidState, 1}.At(iCell, q);
                if (!ThermallyAdmissible(mean, bounds))
                    result = NumericalStatus{NumericalError::InvalidState, 2}.At(iCell);
                if (result)
                {
                    const real beta = ReconstructionFraction(mean, MeanValues{mean}, points, begin, end, bounds, power, cutoff);
                    if (!(std::isfinite(beta) && beta >= 0 && beta <= 1))
                        result = NumericalStatus{NumericalError::InvalidState, 3}.At(iCell);
                    else if (beta < 1)
                        for (index j = 0; j < coefficients.Width(); ++j)
                            coefficients(iCell, j) *= beta;
                }
                status(iCell, 0) = result;
            }
        }
        __global__ void Scalars(LVRInput means, LVROutput scalars, PerfectGas gas, LVRStatus status)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < means.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                const auto state = Load(means, iCell);
                const auto point = TryPerfectGasPoint(gas, state, false);
                status(iCell, 0) = point ? NumericalStatus{} : point.status.At(iCell);
                if (point)
                {
                    scalars(iCell, 0) = point.value.thermo.pressure;
                    scalars(iCell, 1) = state[0];
                }
            }
        }
        __global__ void Sensors(LVRInput samples, LVRMeans means, Faces faces, PerfectGas gas, ThermalBounds bounds,
                                CFV::LimitedVRKernelParameters settings, LVROutput output, LVRStatus status)
        {
            for (index iFace = index(blockIdx.x) * blockDim.x + threadIdx.x; iFace < faces.RowCount(); iFace += index(gridDim.x) * blockDim.x)
            {
                const auto &face = faces(iFace, 0);
                const auto left = Load(samples, 2 * iFace);
                auto right = Load(samples, 2 * iFace + 1);
                if (face.changesFrame)
                {
                    const auto old = right;
                    for (int d = 0; d < 3; ++d)
                    {
                        right[1 + d] = 0;
                        for (int k = 0; k < 3; ++k)
                            right[1 + d] += face.rotation[d + 3 * k] * old[1 + k];
                    }
                }
                if (!(ThermallyAdmissible(left, bounds) && ThermallyAdmissible(right, bounds)))
                {
                    status(iFace, 0) = NumericalStatus{NumericalError::InvalidState, 1}.At(-1, iFace);
                    continue;
                }
                const auto l = gas.Primitive(left), r = gas.Primitive(right);
                real unLeft = 0, unRight = 0;
                for (int d = 0; d < 3; ++d)
                {
                    unLeft += l.velocity[d] * face.normal[d];
                    unRight += r.velocity[d] * face.normal[d];
                }
                const auto sensor = CFV::Math::EvaluateLimitedVRFaceSensor(l.density, r.density, l.pressure, r.pressure, unLeft, unRight, l.soundSpeed, r.soundSpeed, settings);
                const auto support = CFV::Math::EvaluateLimitedVRFaceSensor(means(face.cell, 0), means(face.other, 0), 1, 1, 0, 0, 1, 1, settings).contact;
                output(iFace, 0) = sensor.shock;
                output(iFace, 1) = sensor.contact;
                output(iFace, 2) = support;
                status(iFace, 0) = {};
            }
        }
        __global__ void Gather(LVRIndices offsets, LVRIndices indices, LVRInput sensors, StatusInput faceStatus,
                               LVROutput branches, BaseOutput base, LVRStatus status)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < branches.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                std::array<real, 3> values{};
                NumericalStatus result;
                for (index entry = offsets(iCell, 0); entry < offsets(iCell + 1, 0); ++entry)
                {
                    const index face = indices(entry, 0);
                    if (!faceStatus(face, 0))
                    {
                        result = faceStatus(face, 0).At(iCell, face);
                        break;
                    }
                    for (int j = 0; j < 3; ++j)
                        values[j] = std::max(values[j], sensors(face, j));
                }
                status(iCell, 0) = result;
                if (result)
                {
                    for (int j = 0; j < 3; ++j)
                        branches(iCell, j) = values[j];
                    for (int j = 0; j < 2; ++j)
                        base(iCell, j) = values[j];
                }
            }
        }
        __global__ void Merge(BaseInput base, LVROutput seeds, int column)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < seeds.RowCount(); iCell += index(gridDim.x) * blockDim.x)
                seeds(iCell, column) = std::max(seeds(iCell, column), base(iCell, column));
        }
        __global__ void Publish(LVRInput branches, BaseOutput history, ScalarOutput penalty)
        {
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < branches.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            {
                for (int j = 0; j < 2; ++j)
                    history(iCell, j) = branches(iCell, j);
                penalty(iCell, 0) = std::max(branches(iCell, 0), branches(iCell, 1));
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
    void LimitLVRTarget(R::ExecutionContext c, LVRInput means, LVRInput points, LVROutput coefficients,
                        LVRIndices offsets, ThermalBounds bounds, int power, real cutoff, LVRStatus status)
    {
        CheckDevice(c, means, points, coefficients, offsets, status);
        Table(means, means.RowCount(), 5);
        Table(points, points.RowCount(), 5);
        Table(coefficients, means.RowCount(), coefficients.Width());
        DNDS_check_throw_info(coefficients.Width() % 5 == 0 && coefficients.Width() > 0 && offsets.RowCount() == means.RowCount() + 1 &&
                                  status.RowCount() == means.RowCount() && power >= 0 && std::isfinite(cutoff) && cutoff >= 0 && cutoff <= 1,
                              "invalid CUDA LVR target controls/shape");
        R::CheckKernelArgumentBudget<LVRInput, LVRInput, LVROutput, LVRIndices, ThermalBounds, int, real, LVRStatus>();
        if (!means.RowCount())
            return;
        Target<<<Blocks(means.RowCount()), 128, 0, static_cast<cudaStream_t>(c.stream)>>>(means, points, coefficients, offsets, bounds, power, cutoff, status);
        CheckLaunch();
    }
    void FillLVRScalars(R::ExecutionContext c, LVRInput means, LVROutput scalars, PerfectGas gas, LVRStatus status)
    {
        CheckDevice(c, means, scalars, status);
        gas.Validate();
        Table(means, means.RowCount(), 5);
        Table(scalars, means.RowCount(), 2);
        DNDS_check_throw_info(status.RowCount() == means.RowCount(), "invalid CUDA LVR scalar status shape");
        R::CheckKernelArgumentBudget<LVRInput, LVROutput, PerfectGas, LVRStatus>();
        if (!means.RowCount())
            return;
        Scalars<<<Blocks(means.RowCount()), 128, 0, static_cast<cudaStream_t>(c.stream)>>>(means, scalars, gas, status);
        CheckLaunch();
    }
    void BuildLVRFaceSensors(R::ExecutionContext c, LVRInput samples, LVRMeans means, Faces faces, PerfectGas gas, ThermalBounds bounds,
                             CFV::LimitedVRKernelParameters settings, LVROutput sensors, LVRStatus status)
    {
        CheckDevice(c, samples, means, faces, sensors, status);
        gas.Validate();
        Table(samples, 2 * faces.RowCount(), 5);
        Table(means.owned, means.owned.RowCount(), 5);
        if (means.ghost.RowCount())
            Table(means.ghost, means.ghost.RowCount(), 5);
        Table(sensors, faces.RowCount(), 3);
        DNDS_check_throw_info(status.RowCount() == faces.RowCount(), "invalid CUDA LVR face status shape");
        R::CheckKernelArgumentBudget<LVRInput, LVRMeans, Faces, PerfectGas, ThermalBounds, CFV::LimitedVRKernelParameters, LVROutput, LVRStatus>();
        if (!faces.RowCount())
            return;
        Sensors<<<Blocks(faces.RowCount()), 128, 0, static_cast<cudaStream_t>(c.stream)>>>(samples, means, faces, gas, bounds, settings, sensors, status);
        CheckLaunch();
    }
    void GatherLVRBranches(R::ExecutionContext c, LVRIndices offsets, LVRIndices faceIndices, LVRInput sensors, StatusInput faceStatus,
                           LVROutput branches, BaseOutput base, LVRStatus status)
    {
        CheckDevice(c, offsets, faceIndices, sensors, faceStatus, branches, base, status);
        Table(sensors, sensors.RowCount(), 3);
        Table(branches, branches.RowCount(), 3);
        DNDS_check_throw_info(offsets.RowCount() == branches.RowCount() + 1 && base.RowCount() == branches.RowCount() && status.RowCount() == branches.RowCount() && faceStatus.RowCount() == sensors.RowCount(), "invalid CUDA LVR gather shape");
        R::CheckKernelArgumentBudget<LVRIndices, LVRIndices, LVRInput, StatusInput, LVROutput, BaseOutput, LVRStatus>();
        if (!branches.RowCount())
            return;
        Gather<<<Blocks(branches.RowCount()), 128, 0, static_cast<cudaStream_t>(c.stream)>>>(offsets, faceIndices, sensors, faceStatus, branches, base, status);
        CheckLaunch();
    }
    void MergeLVRBase(R::ExecutionContext c, BaseInput base, LVROutput seeds, int column)
    {
        CheckDevice(c, base, seeds);
        Table(seeds, base.RowCount(), 3);
        DNDS_check_throw_info(column == 0 || column == 1, "invalid LVR base column");
        R::CheckKernelArgumentBudget<BaseInput, LVROutput, int>();
        if (!seeds.RowCount())
            return;
        Merge<<<Blocks(seeds.RowCount()), 128, 0, static_cast<cudaStream_t>(c.stream)>>>(base, seeds, column);
        CheckLaunch();
    }
    void PublishLVRPenalty(R::ExecutionContext c, LVRInput branches, BaseOutput history, ScalarOutput penalty)
    {
        CheckDevice(c, branches, history, penalty);
        Table(branches, history.RowCount(), 3);
        DNDS_check_throw_info(penalty.RowCount() == branches.RowCount(), "invalid LVR penalty shape");
        R::CheckKernelArgumentBudget<LVRInput, BaseOutput, ScalarOutput>();
        if (!branches.RowCount())
            return;
        Publish<<<Blocks(branches.RowCount()), 128, 0, static_cast<cudaStream_t>(c.stream)>>>(branches, history, penalty);
        CheckLaunch();
    }
}
