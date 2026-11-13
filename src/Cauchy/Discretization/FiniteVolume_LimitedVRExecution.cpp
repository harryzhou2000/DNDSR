#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;

    namespace
    {
        /// CUDA specialization admits only combined AoS. Uniform CPU views
        /// retain general field layouts; conversion here is wrapper-local.
        template <class T>
        auto AoS(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
        {
            return R::ArrayView<T, R::AccessBoundsChecks, R::TableRows>{view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device};
        }
    }

    void FiniteVolume::LimitedVRWorkspace::Build(const Data &data, const CFV::FVField &mean,
                                                 R::Array<real> &previous,
                                                 CFV::FaceCharacteristicTransformRef transform,
                                                 R::EagerEngine &engine,
                                                 const R::ExecutionStream &stream,
                                                 const R::ExecutionStream &communication)
    {
        const auto &geometry = *data.limitedVRGeometry;
        const auto &settings = data.settings.numerics.limitedVR;
        const auto &controls = data.settings.numerics;
        const auto &gas = data.equation.Gas();
        const auto bounds = data.FixedFlowThermalBounds();
        const auto placement = data.metrics.Device();
        const bool cuda = placement != R::Device::Host();
        const bool overlap = data.settings.execution.overlap;
        // Admission belongs before queued kernels and matched transfers. A
        // retained external history reader is a local access error, not a
        // numerical worker failure that should poison an otherwise reusable
        // checked-test stream or let peers enter another transfer generation.
        // Production LocalStage still aborts on failure; only coordinated UTs
        // may catch this rejection. The domain session and caller's exclusive
        // mutation contract exclude new external accesses after admission,
        // just as for the other FV output buffers.
        data.metrics.Communication()->LocalStage([&]
                                                 {
            DNDS_check_throw_info(engine.Owns(stream) && engine.Owns(communication) &&
                                      stream.Context().device == placement && communication.Context().device == placement &&
                                      mean.Device() == placement && previous.GetDevice() == placement,
                                  "LVR stream or field placement differs");
            if (cuda)
                DNDS_check_throw_info(mean.Owned().Layout().Kind() == R::RowLayoutKind::Table &&
                                          mean.Ghost().Layout().Kind() == R::RowLayoutKind::Table,
                                      "CUDA LVR requires combined AoS");
            auto probe = previous.Write({placement, nullptr});
            probe.CancelUnmodified(); });
        OperatorSchedule schedule(engine, stream, communication, overlap);
        if (cuda)
        {
#ifdef DNDS_USE_CUDA
            if (!device)
            {
                const index cells = data.metrics.NumCell(), ghosts = data.metrics.NumCellProc() - cells, faces = geometry.faces.RowCount();
                device.emplace(DeviceStorage{
                    R::Array<real>(R::RowLayout::Dense(faces, 3), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(cells, 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(ghosts, 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(faces, 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(std::max({cells, ghosts, faces}) + 1), 1), placement),
                    R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), placement)});
                device->sensors.SetName("Cauchy/LVR/faceSensors", DNDS_ALLOCATION_SITE());
                device->ownedStatus.SetName("Cauchy/LVR/ownedStatus", DNDS_ALLOCATION_SITE());
                device->ghostStatus.SetName("Cauchy/LVR/ghostStatus", DNDS_ALLOCATION_SITE());
                device->faceStatus.SetName("Cauchy/LVR/faceStatus", DNDS_ALLOCATION_SITE());
                device->partialStatus.SetName("Cauchy/LVR/statusPartials", DNDS_ALLOCATION_SITE());
                device->resultStatus.SetName("Cauchy/LVR/status", DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto status)
                            { CUDA::ClearNumericalStatus(context, status); }, R::WriteAs<R::FixedRows<1>>(device->resultStatus));
#else
            DNDS_check_throw_info(false, "CUDA LVR is not built");
#endif
        }
        const auto accumulateStatus = [&](const R::Array<NumericalStatus> &rows)
        {
#ifdef DNDS_USE_CUDA
            if (cuda)
                R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto partial, auto result)
                                { CUDA::ReduceNumericalStatus(context, rows, result.ConstView(), partial, result); }, R::ReadAs<R::FixedRows<1>>(rows), R::WriteAs<R::FixedRows<1>>(device->partialStatus), R::ReadWriteAs<R::FixedRows<1>>(device->resultStatus));
#endif
        };
        geometry.gradient.ConvertToCoefficients(engine, stream, gradients, *data.reconstruction, targetCoefficients);
        {
            auto transfer = schedule.BeginAfterCompute(targetHalo);
            if (overlap)
                data.reconstruction->SubmitLimiterWBAP_3OrderTo(engine, stream, targetCoefficients, target, 1, transform,
                                                                CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            data.reconstruction->SubmitLimiterWBAP_3OrderTo(
                engine, stream, targetCoefficients, target, 1, transform,
                overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
            data.reconstruction->SubmitCommitReconstruction(engine, stream, targetCoefficients, target);
        }
        // PP POLICY -- this is the pure-flow5 O2 target guard inside LVR,
        // distinct from the later equation-aware high-order publication policy.
        // Reuse the Euler common-beta mathematics on a mean-based O2 target,
        // independently of the final high-order PP policy. No alternate limiter
        // or mean clipping is introduced. All PP samples belong to owned cells.
        data.positivitySamples->Evaluate(engine, stream, mean, targetCoefficients, sampledPP);
        auto &ownedStatus = cuda ? device->ownedStatus : emptyStatus;
        R::SubmitArrays(engine, stream, [bounds, power = controls.positivityBetaPower, cutoff = controls.positivityBetaCutoff](R::ExecutionContext context, auto means, auto points, auto coefficients, auto offsets, auto status)
                        {
                            if (context.device == R::Device::Host())
                                RequireNumericalSuccess(CPU::LimitLVRTarget(means, points, coefficients, offsets, bounds, power, cutoff), "LimitLVRTarget");
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::LimitLVRTarget(context, AoS(means), AoS(points), AoS(coefficients), offsets, bounds, power, cutoff, status);
#endif
                        },
                        R::ReadAs<R::UniformTableRows>(mean.Owned()), R::ReadAs<R::UniformTableRows>(sampledPP), R::ReadWriteAs<R::UniformTableRows>(targetCoefficients.Owned()), R::ReadAs<R::FixedRows<1>>(data.stencil.positivityOffsets), R::WriteAs<R::FixedRows<1>>(ownedStatus));
        accumulateStatus(ownedStatus);
        {
            auto transfer = schedule.BeginAfterCompute(targetHalo);
            // The mean halo is already observed. Derived mean scalars need no
            // extra communication, including their ghost rows.
            const auto fill = [&](const R::Array<real> &input, R::Array<real> &output, R::Array<NumericalStatus> &status)
            {
                R::SubmitArrays(engine, stream, [gas](R::ExecutionContext context, auto input, auto output, auto status)
                                {
                                    if (context.device == R::Device::Host())
                                        CPU::FillLVRScalars(input, output, gas);
#ifdef DNDS_USE_CUDA
                                    else
                                        CUDA::FillLVRScalars(context, AoS(input), AoS(output), gas, status);
#endif
                                },
                                R::ReadAs<R::UniformTableRows>(input), R::WriteAs<R::UniformTableRows>(output), R::WriteAs<R::FixedRows<1>>(status));
                accumulateStatus(status);
            };
            fill(mean.Owned(), scalars.Owned(), ownedStatus);
            fill(mean.Ghost(), scalars.Ghost(), cuda ? device->ghostStatus : emptyStatus);
            schedule.Await(transfer);
        }
        geometry.sampling->Evaluate(engine, stream, mean, targetCoefficients, sampledFaces);
        if (!cuda)
        {
            if (!cpuSensors)
            {
                cpuSensors.emplace(R::RowLayout::Dense(geometry.faces.RowCount(), 3));
                cpuSensors->SetName("Cauchy/LVR/faceSensors", DNDS_ALLOCATION_SITE());
            }
            R::SubmitArrays(engine, stream, [gas, bounds, settings](R::ExecutionContext, auto samples, auto own, auto ghost, auto faces, auto sensors)
                            { RequireNumericalSuccess(CPU::BuildLVRFaceSensors(samples, {own, ghost}, faces, gas, bounds, settings, sensors), "BuildLVRFaceSensors"); }, R::ReadAs<R::UniformTableRows>(sampledFaces), R::ReadAs<R::UniformTableRows>(mean.Owned()), R::ReadAs<R::UniformTableRows>(mean.Ghost()), R::ReadAs<R::FixedRows<1>>(geometry.faces), R::WriteAs<R::UniformTableRows>(*cpuSensors));
            R::SubmitArrays(engine, stream, [](R::ExecutionContext, auto offsets, auto indices, auto sensors, auto branches, auto base)
                            { CPU::GatherLVRBranches(offsets, indices, sensors, branches, base); }, R::ReadAs<R::FixedRows<1>>(geometry.cellFaceOffsets), R::ReadAs<R::FixedRows<1>>(geometry.cellFaces), R::ReadAs<R::UniformTableRows>(*cpuSensors), R::WriteAs<R::UniformTableRows>(branches.Owned()), R::WriteAs<R::FixedRows<2>>(base));
        }
#ifdef DNDS_USE_CUDA
        else
        {
            R::SubmitArrays(engine, stream, [gas, bounds, settings = CFV::LimitedVRKernelParameters::FromSettings(settings)](R::ExecutionContext context, auto samples, auto own, auto ghost, auto faces, auto sensors, auto status)
                            { CUDA::BuildLVRFaceSensors(context, samples, {own, ghost}, faces, gas, bounds, settings, sensors, status); }, R::ReadAs<R::TableRows>(sampledFaces), R::ReadAs<R::TableRows>(mean.Owned()), R::ReadAs<R::TableRows>(mean.Ghost()), R::ReadAs<R::FixedRows<1>>(geometry.faces), R::WriteAs<R::TableRows>(device->sensors), R::WriteAs<R::FixedRows<1>>(device->faceStatus));
            accumulateStatus(device->faceStatus);
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto offsets, auto indices, auto sensors, auto faceStatus, auto branches, auto base, auto status)
                            { CUDA::GatherLVRBranches(context, offsets, indices, sensors, faceStatus, branches, base, status); }, R::ReadAs<R::FixedRows<1>>(geometry.cellFaceOffsets), R::ReadAs<R::FixedRows<1>>(geometry.cellFaces), R::ReadAs<R::TableRows>(device->sensors), R::ReadAs<R::FixedRows<1>>(device->faceStatus), R::WriteAs<R::TableRows>(branches.Owned()), R::WriteAs<R::FixedRows<2>>(base), R::WriteAs<R::FixedRows<1>>(ownedStatus));
            accumulateStatus(ownedStatus);
        }
#endif
        {
            auto transfer = schedule.BeginAfterCompute(branchHalo);
            if (overlap)
                geometry.gates.BuildExtremumSeeds(engine, stream, scalars, branches, previous, seeds, settings, CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            geometry.gates.BuildExtremumSeeds(engine, stream, scalars, branches, previous, seeds, settings, schedule.DependentRegion());
        }
        const auto maximum = [&](CFV::FVField &input, R::TransferBinding &halo, std::array<bool, 3> active, bool excludeBoundary, bool saturated)
        {
            auto transfer = schedule.BeginAfterCompute(halo);
            if (overlap)
                geometry.gates.NeighborMaximum(engine, stream, input, temporary, active, excludeBoundary, saturated,
                                               {settings.alphaMax, settings.contactAlphaMax, 1}, CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            geometry.gates.NeighborMaximum(engine, stream, input, temporary, active, excludeBoundary, saturated,
                                           {settings.alphaMax, settings.contactAlphaMax, 1}, schedule.DependentRegion());
            Algebra::Copy(engine, stream, input, temporary);
        };
        const auto merge = [&](int column)
        {
            R::SubmitArrays(engine, stream, [column](R::ExecutionContext context, auto base, auto seeds)
                            {
                                if (context.device == R::Device::Host())
                                    CPU::MergeLVRBase(base, seeds, column);
#ifdef DNDS_USE_CUDA
                                else
                                    CUDA::MergeLVRBase(context, base, AoS(seeds), column);
#endif
                            },
                            R::ReadAs<R::FixedRows<2>>(base), R::ReadWriteAs<R::UniformTableRows>(seeds.Owned()));
        };
        const int pressureLayers = settings.alphaMax > 0 ? 1 : 0;
        const int contactLayers = settings.variant == "v3" && settings.contactAlphaMax > 0 ? settings.contactLayers : 0;
        if (!pressureLayers)
            merge(0);
        if (!contactLayers)
            merge(1);
        for (int layer = 0; layer < std::max(pressureLayers, contactLayers); ++layer)
        {
            maximum(seeds, seedHalo, {layer < pressureLayers, layer < contactLayers, false}, true, false);
            if (layer + 1 == pressureLayers)
                merge(0);
            if (layer + 1 == contactLayers)
                merge(1);
        }
        Algebra::Copy(engine, stream, branches, seeds);
        for (int layer = 0; layer < settings.neighborLayers; ++layer)
            maximum(branches, branchHalo, {true, false, false}, false, settings.saturatedOnly);
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto gates, auto history, auto penalty)
                        {
                            if (context.device == R::Device::Host())
                                CPU::PublishLVRPenalty(gates, history, penalty);
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::PublishLVRPenalty(context, AoS(gates), history, penalty);
#endif
                        },
                        R::ReadAs<R::UniformTableRows>(branches.Owned()), R::WriteAs<R::FixedRows<2>>(previous), R::WriteAs<R::FixedRows<1>>(alpha));
        // Synchronous compound-operator return. Fixed neighborhood layers use
        // producer-prefix exchanges; no engine drain separates their kernels.
        schedule.Join();
        if (cuda)
        {
            const auto status = device->resultStatus.To(R::Device::Host());
            RequireNumericalSuccess(status.Read().View()(0, 0), "limited VR preparation");
        }
    }
}
