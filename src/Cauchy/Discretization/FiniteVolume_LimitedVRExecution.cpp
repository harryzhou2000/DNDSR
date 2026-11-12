#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
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
        /// Execute inside the already queued CPU operation. The private LVR
        /// work fields are distinct; only owned rows are copied, while the
        /// next layer's explicit halo operation publishes its remote rows.
        /// Immediate algebra would create and drain a second engine here.
        void CopyOwnedLVRRows(const CFV::FVField &input, CFV::FVField &output)
        {
            input.Validate();
            output.Validate();
            DNDS_check_throw_info(input.Space().Same(output.Space()) &&
                                      input.Width() == output.Width() &&
                                      input.Device() == R::Device::Host() &&
                                      output.Device() == R::Device::Host(),
                                  "LVR row copy requires matching host fields");
            auto read = input.Owned().Read();
            auto write = output.Owned().Write();
            CPU::CopyRows(read.View<R::UniformTableRows>(),
                          write.View<R::UniformTableRows>(), output.Width());
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
            auto probe = previous.Write();
            probe.CancelUnmodified(); });
        OperatorSchedule schedule(engine, stream, communication, overlap);
        schedule.Submit([&]
                        { geometry.gradient.ConvertToCoefficients(gradients, *data.reconstruction, targetCoefficients); });
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
        schedule.Submit([&]
                        {
            data.positivitySamples->Evaluate(mean, targetCoefficients, sampledPP);
            auto meanRead = mean.Owned().Read(), pointsRead = sampledPP.Read();
            const auto u = meanRead.View<R::UniformTableRows>(), points = pointsRead.View<R::UniformTableRows>();
            auto write = targetCoefficients.Owned().Write();
            const auto offsetRead = data.stencil.positivityOffsets.Read();
            RequireNumericalSuccess(CPU::LimitLVRTarget(u, points, write.View<R::UniformTableRows>(),
                offsetRead.View<R::FixedRows<1>>(), data.FixedFlowThermalBounds(),
                controls.positivityBetaPower, controls.positivityBetaCutoff), "LimitLVRTarget"); });
        {
            auto transfer = schedule.BeginAfterCompute(targetHalo);
            // The mean halo is already observed. Derived mean scalars need no
            // extra communication, including their ghost rows.
            schedule.Submit([&]
                            {
                const auto fill = [&](const auto &input, auto &output)
                {
                    auto read = input.Read(); auto write = output.Write();
                    const auto u = read.template View<R::UniformTableRows>();
                    const auto s = write.template View<R::UniformTableRows>();
                    CPU::FillLVRScalars(u, s, gas);
                };
                fill(mean.Owned(), scalars.Owned()); fill(mean.Ghost(), scalars.Ghost()); });
            schedule.Await(transfer);
        }
        schedule.Submit([&]
                        {
            geometry.sampling->Evaluate(mean, targetCoefficients, sampledFaces);
            auto values = sampledFaces.Read(), own = mean.Owned().Read(), ghost = mean.Ghost().Read();
            const auto samples = values.View<R::UniformTableRows>();
            const R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> means{own.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
            auto gates = branches.Owned().Write();
            const auto g = gates.View<R::UniformTableRows>();
            const auto faceRead = geometry.faces.Read();
            auto baseWrite = base.Write();
            RequireNumericalSuccess(CPU::BuildLVRBranches(samples, means, faceRead.View<R::FixedRows<1>>(),
                gas, data.FixedFlowThermalBounds(), settings, g, baseWrite.View<R::FixedRows<2>>()), "BuildLVRBranches"); });
        {
            auto transfer = schedule.BeginAfterCompute(branchHalo);
            if (overlap)
                schedule.Submit([&]
                                { geometry.gates.BuildExtremumSeeds(scalars, branches, previous, seeds, settings, CFV::ReconstructionRegion::Core); });
            schedule.Await(transfer);
            schedule.Submit([&]
                            { geometry.gates.BuildExtremumSeeds(scalars, branches, previous, seeds, settings, overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All); });
        }
        const auto maximum = [&](CFV::FVField &input, R::TransferBinding &halo, std::array<bool, 3> active, bool excludeBoundary, bool saturated)
        {
            auto transfer = schedule.BeginAfterCompute(halo);
            if (overlap)
                schedule.Submit([&, active, excludeBoundary, saturated]
                                { geometry.gates.NeighborMaximum(input, temporary, active, excludeBoundary, saturated, {settings.alphaMax, settings.contactAlphaMax, 1}, CFV::ReconstructionRegion::Core); });
            schedule.Await(transfer);
            schedule.Submit([&, active, excludeBoundary, saturated]
                            {
                geometry.gates.NeighborMaximum(input, temporary, active, excludeBoundary, saturated, {settings.alphaMax, settings.contactAlphaMax, 1},
                                               overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
                CopyOwnedLVRRows(temporary, input); });
        };
        const auto merge = [&](int column)
        {
            schedule.Submit([&, column]
                            {
                auto read = base.Read();
                auto write = seeds.Owned().Write();
                const auto b = read.View<R::FixedRows<2>>();
                const auto s = write.View<R::UniformTableRows>();
                CPU::MergeLVRBase(b, s, column); });
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
        schedule.Submit([&]
                        { CopyOwnedLVRRows(seeds, branches); });
        for (int layer = 0; layer < settings.neighborLayers; ++layer)
            maximum(branches, branchHalo, {true, false, false}, false, settings.saturatedOnly);
        schedule.Submit([&]
                        {
            auto read = branches.Owned().Read(); auto history = previous.Write(), penalty = alpha.Write();
            const auto g = read.View<R::UniformTableRows>(); const auto h = history.View<R::FixedRows<2>>(); const auto a = penalty.View<R::FixedRows<1>>();
            CPU::PublishLVRPenalty(g, h, a); });
        // Synchronous compound-operator return. Fixed neighborhood layers use
        // producer-prefix exchanges; no engine drain separates their kernels.
        schedule.Join();
    }
}
