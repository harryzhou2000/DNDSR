#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;

    void FiniteVolume::LimitedVRWorkspace::Build(const Data &data, const CFV::FVField &mean,
                                                 R::Array<real> &previous,
                                                 CFV::FaceCharacteristicTransformRef transform,
                                                 R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
        const MPIInfo mpi(data.metrics.Communicator());
        const auto &geometry = *data.limitedVRGeometry;
        const auto &settings = data.settings.numerics.limitedVR;
        const auto &controls = data.settings.numerics;
        const auto &gas = data.equation.Gas();
        const bool overlap = data.settings.execution.overlap;
        const auto begin = [&](R::TransferBinding &binding)
        { return engine.ChecksAccess() ? binding.BeginChecked(stream) : binding.BeginScheduled(stream); };
        MPI::LocalStage(mpi, [&]
                        { geometry.gradient.ConvertToCoefficients(gradients, *data.reconstruction, targetCoefficients); });
        {
            auto transfer = begin(targetHalo);
            if (overlap)
                data.reconstruction->SubmitLimiterWBAP_3OrderTo(engine, stream, targetCoefficients, target, 1, transform,
                                                                CFV::ReconstructionRegion::Core);
            transfer.Wait();
            data.reconstruction->SubmitLimiterWBAP_3OrderTo(
                engine, stream, targetCoefficients, target, 1, transform,
                overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
            data.reconstruction->SubmitCommitReconstruction(engine, stream, targetCoefficients, target);
            stream.Record().WaitHost();
        }
        // PP POLICY -- this is the pure-flow5 O2 target guard inside LVR,
        // distinct from the later equation-aware high-order publication policy.
        // Reuse the Euler common-beta mathematics on a mean-based O2 target,
        // independently of the final high-order PP policy. No alternate limiter
        // or mean clipping is introduced. All PP samples belong to owned cells.
        MPI::LocalStage(mpi, [&]
                        {
            data.positivitySamples->Evaluate(mean, targetCoefficients, sampledPP);
            auto meanRead = mean.Owned().Read(), pointsRead = sampledPP.Read();
            const auto u = meanRead.View<R::UniformTableRows>(), points = pointsRead.View<R::UniformTableRows>();
            for (index iPoint = 0; iPoint < points.RowCount(); ++iPoint)
                for (int j = 0; j < 5; ++j)
                    DNDS_check_throw_info(std::isfinite(points(iPoint, j)), "nonfinite LVR O2 target sample");
            auto write = targetCoefficients.Owned().Write();
            const auto coefficients = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
            {
                const auto value = Load(u, iCell);
                DNDS_check_throw_info(ThermallyAdmissible(value, data.FixedFlowThermalBounds()), "LVR O2 PP requires admissible means");
                struct MeanValues
                {
                    FlowState value;
                    real operator()(index, int j) const { return value[j]; }
                };
                const real beta = ReconstructionFraction(value, MeanValues{value}, points, data.positivityOffsets[iCell], data.positivityOffsets[iCell + 1],
                                                         data.FixedFlowThermalBounds(), controls.positivityBetaPower, controls.positivityBetaCutoff);
                DNDS_check_throw_info(std::isfinite(beta) && beta >= 0 && beta <= 1, "invalid LVR O2 positivity fraction");
                if (beta < 1)
                    for (index j = 0; j < coefficients.RowSize(iCell); ++j) coefficients(iCell, j) *= beta;
            } });
        {
            auto transfer = begin(targetHalo);
            // The mean halo is already observed. Derived mean scalars need no
            // extra communication, including their ghost rows.
            MPI::LocalStage(mpi, [&]
                            {
                const auto fill = [&](const auto &input, auto &output)
                {
                    auto read = input.Read(); auto write = output.Write();
                    const auto u = read.template View<R::UniformTableRows>();
                    const auto s = write.template View<R::UniformTableRows>();
                    for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                    {
                        const auto state = Load(u, iCell);
                        s(iCell, 0) = gas.Pressure(state);
                        s(iCell, 1) = state[0];
                    }
                };
                fill(mean.Owned(), scalars.Owned()); fill(mean.Ghost(), scalars.Ghost()); });
            transfer.Wait();
        }
        MPI::LocalStage(mpi, [&]
                        {
            geometry.sampling->Evaluate(mean, targetCoefficients, sampledFaces);
            auto values = sampledFaces.Read(), own = mean.Owned().Read(), ghost = mean.Ghost().Read();
            const auto samples = values.View<R::UniformTableRows>();
            const R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> means{own.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
            auto gates = branches.Owned().Write();
            const auto g = gates.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
                for (int j = 0; j < 3; ++j) g(iCell, j) = 0;
            for (index iFace = 0; iFace < index(geometry.faces.size()); ++iFace)
            {
                const auto &face = geometry.faces[iFace];
                const auto left = Load(samples, 2 * iFace), right = RotateState(Load(samples, 2 * iFace + 1), face.rotation);
                DNDS_check_throw_info(ThermallyAdmissible(left, data.FixedFlowThermalBounds()) && ThermallyAdmissible(right, data.FixedFlowThermalBounds()),
                                      "LVR finalized O2 face center violates the thermal bound");
                const auto l = gas.Primitive(left), r = gas.Primitive(right);
                real unLeft = 0, unRight = 0;
                for (int d = 0; d < 3; ++d) unLeft += l.velocity[d] * face.normal[d], unRight += r.velocity[d] * face.normal[d];
                const auto sensor = CFV::Math::EvaluateLimitedVRFaceSensor(l.density, r.density, l.pressure, r.pressure, unLeft, unRight, l.soundSpeed, r.soundSpeed, settings);
                // Current v3 uses mean-density support independently of the
                // finalized-O2 base contact gate. Pressure/velocity data here
                // are neutral placeholders for this density-only calculation.
                const auto support = CFV::Math::EvaluateLimitedVRFaceSensor(means(face.cell, 0), means(face.other, 0), 1, 1, 0, 0, 1, 1, settings).contact;
                for (const auto iCell : {face.cell, face.other})
                    if (iCell < data.metrics.NumCell())
                    {
                        g(iCell, 0) = std::max(g(iCell, 0), sensor.shock);
                        g(iCell, 1) = std::max(g(iCell, 1), sensor.contact);
                        g(iCell, 2) = std::max(g(iCell, 2), support);
                    }
            }
            auto baseWrite = base.Write(); const auto b = baseWrite.View<R::FixedRows<2>>();
            for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
                for (int j = 0; j < 2; ++j) b(iCell, j) = g(iCell, j); });
        {
            auto transfer = begin(branchHalo);
            if (overlap)
                geometry.gates.BuildExtremumSeeds(scalars, branches, previous, seeds, settings, CFV::ReconstructionRegion::Core);
            transfer.Wait();
            MPI::LocalStage(mpi, [&]
                            { geometry.gates.BuildExtremumSeeds(scalars, branches, previous, seeds, settings, overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All); });
        }
        const auto maximum = [&](CFV::FVField &input, R::TransferBinding &halo, std::array<bool, 3> active, bool excludeBoundary, bool saturated)
        {
            auto transfer = begin(halo);
            if (overlap)
                geometry.gates.NeighborMaximum(input, temporary, active, excludeBoundary, saturated, {settings.alphaMax, settings.contactAlphaMax, 1}, CFV::ReconstructionRegion::Core);
            transfer.Wait();
            MPI::LocalStage(mpi, [&]
                            {
                geometry.gates.NeighborMaximum(input, temporary, active, excludeBoundary, saturated, {settings.alphaMax, settings.contactAlphaMax, 1},
                                               overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
                Algebra::Copy(input, temporary); });
        };
        const auto merge = [&](int column)
        {
            auto read = base.Read();
            auto write = seeds.Owned().Write();
            const auto b = read.View<R::FixedRows<2>>();
            const auto s = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
                s(iCell, column) = std::max(s(iCell, column), b(iCell, column));
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
        Algebra::Copy(branches, seeds);
        for (int layer = 0; layer < settings.neighborLayers; ++layer)
            maximum(branches, branchHalo, {true, false, false}, false, settings.saturatedOnly);
        MPI::LocalStage(mpi, [&]
                        {
            auto read = branches.Owned().Read(); auto history = previous.Write(), penalty = alpha.Write();
            const auto g = read.View<R::UniformTableRows>(); const auto h = history.View<R::FixedRows<2>>(); const auto a = penalty.View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < data.metrics.NumCell(); ++iCell)
            {
                for (int j = 0; j < 2; ++j) h(iCell, j) = g(iCell, j);
                a(iCell, 0) = std::max(g(iCell, 0), g(iCell, 1));
            } });
    }
}
