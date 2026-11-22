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

namespace DNDS::Cauchy
{
    namespace R = Runtime;

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

    void FiniteVolume::LimitedVRWorkspace::Build(const Data &data, const EquationDofs &state, const std::vector<CFV::FVField> &gradients,
                                                 const CFV::FVField &rawFlowGradient, bool newStage,
                                                 R::Array<real> &previous,
                                                 std::vector<R::TransferBinding> &gradientHalo,
                                                 R::EagerEngine &engine,
                                                 const R::ExecutionStream &stream,
                                                 const R::ExecutionStream &communication)
    {
        const auto &geometry = *data.limitedVRGeometry;
        const auto &settings = data.settings.numerics.limitedVR;
        const auto flow = state.LocateBlock("flow");
        const auto &mean = state.PhysicalField(flow.physicalField);
        const auto &gradient = gradients[flow.physicalField];
        const bool refreshWeak = settings.weakAlphaMax > 0 && (!settings.weakFreezeWithinImplicitStage || newStage);
        auto seedSettings = settings;
        if (!refreshWeak)
            seedSettings.weakHaloLayers = 0;
        auto parameters = CFV::LimitedVRKernelParameters::FromSettings(seedSettings);
        if (!refreshWeak)
            parameters.weakAlphaMax = 0;
        const auto gateWidth = settings.GateWidth();
        const index modes = data.reconstruction->NRecDOF();
        const auto &gas = data.equation.Gas();
        const auto bounds = data.FixedFlowThermalBounds();
        const auto placement = data.metrics.Device();
        const bool cuda = placement != R::Device::Host();
        const bool overlap = data.settings.execution.overlap;
        const int dimension = data.metrics.Dimension();
        // The immutable shared publication pins topology, metric fields and
        // compact periodic maps through all CPU worker/device submissions.
        const auto stencil = std::make_shared<FiniteVolumeStencilStorage::ReadAccess>(data.stencil.Read({placement, nullptr}));
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
                const index cells = data.metrics.NumCell(), ghosts = data.metrics.NumCellProc() - cells, faces = data.stencil.stencilFaces.RowCount();
                device.emplace(DeviceStorage{
                    R::Array<real>(R::RowLayout::Dense(faces, gateWidth), placement),
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
        auto gradientOperations = schedule.BeginAfterCompute(gradientHalo);
        auto &ownedStatus = cuda ? device->ownedStatus : emptyStatus;
        auto &ghostStatus = cuda ? device->ghostStatus : emptyStatus;
        auto &flowTarget = targetCoefficients[flow.physicalField];
        const auto fill = [&](bool ghost)
        {
            const auto &values = ghost ? mean.Ghost() : mean.Owned();
            auto &out = ghost ? scalars.Ghost() : scalars.Owned();
            auto &status = ghost ? ghostStatus : ownedStatus;
            R::Array<real> empty(R::RowLayout::Dense(0, 1), placement);
            auto &target = ghost ? empty : flowTarget.Owned();
            const auto &input = ghost ? empty : gradient.Owned();
            R::SubmitArrays(engine, stream, [gas, dimension, count = ghost ? index(0) : modes, first = flow.firstComponent](R::ExecutionContext context, auto cells, auto means, auto slopes, auto output, auto coefficients, auto status)
                            {
                                if (context.device == R::Device::Host())
                                    CPU::PrepareLVRScalarsAndTarget(cells, means, slopes, output, coefficients, gas, dimension, count, first);
#ifdef DNDS_USE_CUDA
                                else
                                    CUDA::PrepareLVRScalarsAndTarget(context, cells, AoS(means), AoS(slopes), AoS(output), AoS(coefficients), gas, dimension, count, first, status);
#endif
                            },
                            R::ReadAs<R::FixedRows<8>>(data.metrics.CellMetrics()), R::ReadAs<R::UniformTableRows>(values), R::ReadAs<R::UniformTableRows>(input), R::WriteAs<R::UniformTableRows>(out), R::WriteAs<R::UniformTableRows>(target), R::WriteAs<R::FixedRows<1>>(status));
            accumulateStatus(status);
        };
        fill(false);
        fill(true);
        for (index field = 0; field < state.PhysicalFieldNumber(); ++field)
            if (field != flow.physicalField)
                data.gradient.ConvertToCoefficients(engine, stream, gradients[field], *data.reconstruction, targetCoefficients[field]);

        const index faceCount = data.stencil.stencilFaces.RowCount();
        if (!cuda && !cpuSensors)
        {
            cpuSensors.emplace(R::RowLayout::Dense(faceCount, gateWidth));
            cpuSensors->SetName("Cauchy/LVR/faceSensors", DNDS_ALLOCATION_SITE());
        }
        auto &sensors = cuda ? device->sensors : *cpuSensors;
        auto &faceStatus = cuda ? device->faceStatus : emptyStatus;
        const R::Array<real> empty(R::RowLayout::Dense(0, 1), placement);
        const auto sample = [&](CFV::ReconstructionRegion region)
        {
            const bool core = region == CFV::ReconstructionRegion::Core;
            const bool all = region == CFV::ReconstructionRegion::All;
            const auto selection = all ? R::RowSelectionShape::AllRows(faceCount)
                                       : data.stencil.auxiliaryRegions[size_t(core ? AuxiliaryRegion::Core : AuxiliaryRegion::Dependent)].faces;
            R::SubmitArrays(engine, stream, [stencil, dimension, first = flow.firstComponent, gas, bounds, parameters, selection, all, core](R::ExecutionContext context, auto own, auto ghost, auto gradOwn, auto gradGhost, auto rawOwn, auto rawGhost, auto output, auto status)
                            {
                                const auto view = stencil->View();
                                const R::RowSelection rows{selection, all ? decltype(view.auxiliaryCoreFaces){} : (core ? view.auxiliaryCoreFaces : view.auxiliaryDependentFaces)};
                                if (context.device == R::Device::Host())
                                    RequireNumericalSuccess(CPU::BuildLVRFaceSensors(view, {own, ghost}, {gradOwn, gradGhost}, {rawOwn, rawGhost},
                                                                                     dimension, first, gas, bounds, parameters, output, rows),
                                                            "BuildLVRFaceSensors");
#ifdef DNDS_USE_CUDA
                                else
                                    CUDA::BuildLVRFaceSensors(context, stencil->KernelDescriptor(), {AoS(own), AoS(ghost)},
                                                              {AoS(gradOwn), AoS(gradGhost)}, {AoS(rawOwn), AoS(rawGhost)}, dimension, first, gas, bounds, parameters, AoS(output), status, rows);
#endif
                            },
                            R::ReadAs<R::UniformTableRows>(mean.Owned()), R::ReadAs<R::UniformTableRows>(core ? empty : mean.Ghost()), R::ReadAs<R::UniformTableRows>(gradient.Owned()), R::ReadAs<R::UniformTableRows>(core ? empty : gradient.Ghost()), R::ReadAs<R::UniformTableRows>(refreshWeak ? rawFlowGradient.Owned() : empty), R::ReadAs<R::UniformTableRows>(refreshWeak && !core ? rawFlowGradient.Ghost() : empty), R::ReadWriteAs<R::UniformTableRows>(sensors), R::ReadWriteAs<R::FixedRows<1>>(faceStatus));
        };
        if (overlap)
            sample(CFV::ReconstructionRegion::Core);
        schedule.Await(gradientOperations);
        sample(schedule.DependentRegion());
        accumulateStatus(faceStatus);
        R::SubmitArrays(engine, stream, [stencil](R::ExecutionContext context, auto sensors, auto faceStatus, auto branches, auto base, auto status)
                        {
                            if (context.device == R::Device::Host())
                                CPU::GatherLVRBranches(stencil->View(), sensors, branches, base);
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::GatherLVRBranches(context, stencil->KernelDescriptor(), AoS(sensors), faceStatus,
                                                        AoS(branches), base, status);
#endif
                        },
                        R::ReadAs<R::UniformTableRows>(sensors), R::ReadAs<R::FixedRows<1>>(faceStatus), R::WriteAs<R::UniformTableRows>(branches.Owned()), R::WriteAs<R::FixedRows<2>>(base), R::WriteAs<R::FixedRows<1>>(ownedStatus));
        accumulateStatus(ownedStatus);
        const bool correctPressure = parameters.pressureCorrection && settings.alphaMax > 0;
        const bool correctContact = parameters.contactCorrection && settings.contactAlphaMax > 0;
        const auto buildSeeds = [&](CFV::ReconstructionRegion region)
        {
            geometry.gates.BuildExtremumSeeds(engine, stream, scalars, branches, previous, seeds, seedSettings, region);
        };
        if (correctPressure || correctContact)
        {
            auto transfer = schedule.BeginAfterCompute(branchHalo);
            if (overlap)
                buildSeeds(CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            buildSeeds(schedule.DependentRegion());
        }
        else
            buildSeeds(CFV::ReconstructionRegion::All);
        CFV::FVField *gates = &seeds;
        CFV::FVField *scratch = &temporary;
        auto *gateHalo = &seedHalo;
        auto *scratchHalo = &temporaryHalo;
        const std::array<real, 4> caps{settings.alphaMax, settings.contactAlphaMax, 1, settings.weakAlphaMax};
        const auto maximum = [&](std::array<bool, 4> active, bool excludeBoundary, bool saturated, unsigned mergeMask)
        {
            auto transfer = schedule.BeginAfterCompute(*gateHalo);
            const auto update = [&](CFV::ReconstructionRegion region)
            {
                geometry.gates.NeighborMaximum(engine, stream, *gates, *scratch, active, excludeBoundary, saturated,
                                               caps, region, mergeMask ? &base : nullptr, mergeMask);
            };
            if (overlap)
                update(CFV::ReconstructionRegion::Core);
            schedule.Await(transfer);
            update(schedule.DependentRegion());
            std::swap(gates, scratch);
            std::swap(gateHalo, scratchHalo);
        };
        const auto merge = [&](int column)
        {
            R::SubmitArrays(engine, stream, [column](R::ExecutionContext context, auto base, auto gates)
                            {
                                if (context.device == R::Device::Host())
                                    CPU::MergeLVRBase(base, gates, column);
#ifdef DNDS_USE_CUDA
                                else
                                    CUDA::MergeLVRBase(context, base, AoS(gates), column);
#endif
                            },
                            R::ReadAs<R::FixedRows<2>>(base), R::ReadWriteAs<R::UniformTableRows>(gates->Owned()));
        };
        const int pressureLayers = correctPressure ? 1 : 0;
        const int contactLayers = correctContact ? settings.contactLayers : 0;
        const int weakLayers = refreshWeak ? settings.weakHaloLayers - int(parameters.weakFirstMaximum) : 0;
        if (!pressureLayers)
            merge(0);
        if (!contactLayers)
            merge(1);
        for (int layer = 0; layer < std::max({pressureLayers, contactLayers, weakLayers}); ++layer)
        {
            const unsigned mergeMask = unsigned(layer + 1 == pressureLayers) |
                                       (unsigned(layer + 1 == contactLayers) << 1);
            // Weak propagation includes boundary-adjacent cells; pressure
            // and contact corrections exclude them.
            maximum({layer < pressureLayers, layer < contactLayers, false, layer < weakLayers}, true, false, mergeMask);
        }
        for (int layer = 0; layer < settings.neighborLayers; ++layer)
            maximum({true, false, false, false}, false, settings.saturatedOnly, 0);
        const CFV::LimitedVRTemporalUpdate temporal{settings.alphaUpdateRelaxation, settings.weakAlphaUpdateRelaxation,
                                                    settings.weakAlphaDecayRetention, newStage, refreshWeak};
        R::SubmitArrays(engine, stream, [temporal](R::ExecutionContext context, auto gates, auto history, auto penalty)
                        {
                            if (context.device == R::Device::Host())
                                CPU::PublishLVRPenalty(gates, history, penalty, temporal);
#ifdef DNDS_USE_CUDA
                            else
                                CUDA::PublishLVRPenalty(context, AoS(gates), AoS(history), penalty, temporal);
#endif
                        },
                        R::ReadAs<R::UniformTableRows>(gates->Owned()), R::ReadWriteAs<R::UniformTableRows>(previous), R::WriteAs<R::FixedRows<1>>(alpha));
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
