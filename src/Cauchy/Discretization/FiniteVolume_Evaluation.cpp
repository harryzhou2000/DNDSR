#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;
    namespace
    {
        void PreflightHalo(EquationDofs &fields)
        {
            auto read = fields.ReadOwned();
            std::vector<R::Array<real>::WriteLease> ghosts;
            ghosts.reserve(size_t(fields.PhysicalFieldNumber()));
            try
            {
                for (index iField = 0; iField < fields.PhysicalFieldNumber(); ++iField)
                    ghosts.emplace_back(fields.PhysicalField(iField).Ghost().Write());
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy equation finite-volume preparation failed");
                for (auto &lease : ghosts)
                    lease.CancelUnmodified();
                throw;
            }
            for (auto &lease : ghosts)
                lease.CancelUnmodified();
        }
    }
    void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction,
                                            Workspace &work, AuxiliaryValues &auxiliary,
                                            R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.FinalizedValues());
        const MPIInfo mpi(Metrics().Communicator());
        reconstruction.binding.Check(state);
        const auto local = [&](auto &&action)
        {
            Metrics().Communication()->LocalStage(std::forward<decltype(action)>(action));
        };
        local([&]
              {
            ValidateWorkspace(state, work);
            if (reconstruction.FinalizedCurrent()) return;
            const bool filtered = _data->settings.numerics.limiter == "WBAP" ||
                                  bool(work.limitedVR) || _data->settings.numerics.positivity;
            DNDS_check_throw_info(!filtered,
                                  "correction source sampling requires a current limited/PP reconstruction publication");
            reconstruction.PublishFinalized(); });
        if (work.meanGradients)
        {
            ValidateWorkspace(state, work);
            AuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary.Emplace(*work.incoming);
            PrepareAuxValues(state, auxiliary, *work.meanGradients, engine, stream, time, work.workers, inputs);
            local([&]
                  {
                if (work.SourceSamplesCurrent(state, time)) return;
                auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {true});
                work.PrepareVolumeSource(state, {}, evaluator, time);
                work.PublishSourceState(state, time); });
            return;
        }
        std::optional<AuxiliaryPreparation> preparation;
        local([&]
              {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(stream.Context().device == R::Device::Host(), "equation correction requires a CPU stream");
            PreflightHalo(state);
            AuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary.Emplace(*work.incoming);
            preparation.emplace(BeginAuxValues(state, auxiliary, time, work.workers, inputs)); });
        OperatorSchedule schedule(engine, stream, _data->settings.execution.overlap);
        std::vector<Algebra::DofHalo<>::Operation> transfers;
        transfers.reserve(reconstruction.meanHalo.size());
        ++reconstruction.meanEpoch;
        for (auto &halo : reconstruction.meanHalo)
            transfers.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        if (_data->settings.execution.overlap)
            schedule.Submit([&]
                            { preparation->Apply(CFV::ReconstructionRegion::Core); });
        OperatorSchedule::Wait(transfers);
        const auto rest = schedule.DependentRegion();
        local([&]
              {
            if (_data->settings.CheckHaloFreshness())
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch);
            schedule.Submit([&, rest]
            {
                preparation->Apply(rest);
                preparation->Finish(auxiliary);
                // Source Jacobians use the retained current polynomial. RANS
                // inputs must be current before publishing these samples.
                // No VR sweep or chemical-rate evaluation is hidden here.
                if (work.SourceSamplesCurrent(state, time)) return;
                if (_data->equation.HasRANS())
                {
                    auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                                        {true, false, false, false, false});
                    work.PrepareVolumeSource(state, {}, evaluator, time);
                }
                else work.SampleSourceState(state, time);
                work.PublishSourceState(state, time);
            });
            schedule.Join(); });
    }
    void FiniteVolume::RefreshCorrectionSource(EquationDofs &state,
                                               ReconstructionState &reconstruction,
                                               Workspace &work, real time) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients,
                                                            reconstruction.FinalizedValues());
        Metrics().Communication()->LocalStage([&]
                                              {
            ValidateWorkspace(state, work);
            reconstruction.binding.Check(state);
            if (!reconstruction.FinalizedCurrent())
            {
                const bool filtered = _data->settings.numerics.limiter == "WBAP" ||
                                      bool(work.limitedVR) || _data->settings.numerics.positivity;
                DNDS_check_throw_info(!filtered,
                                      "source refresh requires a current limited/PP reconstruction publication");
                reconstruction.PublishFinalized();
            }
            if (work.SourceSamplesCurrent(state, time)) return;
            if (_data->equation.HasRANS())
            {
                auto evaluator = _data->equation.MakePointEvaluator(
                    work.workers ? work.workers->Acquire()
                                 : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                    {true, false, false, false, false});
                work.PrepareVolumeSource(state, {}, evaluator, time);
            }
            else
                work.SampleSourceState(state, time);
            work.PublishSourceState(state, time); });
    }
    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &work,
                                R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual), reconstruction, work, engine, stream, time);
    }
    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &work,
                                AuxiliaryValues &auxiliary, R::EagerEngine &engine,
                                const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual), reconstruction, work, engine, stream, time,
                     OptionalRef<AuxiliaryValues>(auxiliary));
    }
    void FiniteVolume::PrepareReconstruction(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, R::EagerEngine &engine,
                                             const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, {}, reconstruction, work, engine, stream, time);
    }
    void FiniteVolume::RefreshSamplingInputs(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, R::EagerEngine &engine,
                                             const R::ExecutionStream &stream) const
    {
        ValidateWorkspace(state, work);
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.FinalizedValues());
        Metrics().Communication()->LocalStage([&]
                                              {
            DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                                  "sampling refresh requires the workspace CPU engine");
            PreflightHalo(state);
            PreflightHalo(reconstruction.FinalizedValues()); });
        ++reconstruction.meanEpoch;
        std::vector<Algebra::DofHalo<>::Operation> means;
        means.reserve(reconstruction.meanHalo.size());
        for (auto &halo : reconstruction.meanHalo)
            means.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        for (auto &operation : means)
            operation.Wait();
        if (_data->settings.CheckHaloFreshness())
            Metrics().Communication()->LocalStage([&]
                                                  {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch); });
        for (auto &binding : reconstruction.FinalizedHalo())
        {
            auto operation = engine.ChecksAccess() ? binding.BeginChecked(stream) : binding.BeginScheduled(stream);
            operation.Wait();
        }
    }
    void FiniteVolume::Execute(EquationDofs &state,
                               ReconstructionState &reconstruction,
                               Workspace &work,
                               DomainExecutionContext &context,
                               FiniteVolumeEvaluation evaluation) const
    {
        auto &engine = context.Engine();
        const auto &stream = context.PrimaryStream();
        switch (evaluation.mode)
        {
        case FiniteVolumeEvaluationMode::SpatialResidual:
            DNDS_check_throw_info(evaluation.residual,
                                  "spatial FV evaluation requires a residual output");
            if (evaluation.auxiliary)
                Evaluate(state, *evaluation.residual, reconstruction, work,
                         *evaluation.auxiliary, engine, stream, evaluation.time);
            else
                Evaluate(state, *evaluation.residual, reconstruction, work,
                         engine, stream, evaluation.time);
            return;
        case FiniteVolumeEvaluationMode::ReconstructionOnly:
            DNDS_check_throw_info(!evaluation.residual && !evaluation.auxiliary,
                                  "reconstruction-only FV evaluation has no residual or auxiliary output");
            PrepareReconstruction(state, reconstruction, work, engine, stream,
                                  evaluation.time);
            return;
        case FiniteVolumeEvaluationMode::SamplingInputs:
            DNDS_check_throw_info(!evaluation.residual && !evaluation.auxiliary,
                                  "sampling-input FV evaluation has no residual or auxiliary output");
            RefreshSamplingInputs(state, reconstruction, work, engine, stream);
            return;
        }
        DNDS_check_throw_info(false, "unknown finite-volume evaluation mode");
    }
    void FiniteVolume::EvaluateMean(EquationDofs &state, EquationDofs &residual, MeanWorkspace &work,
                                    R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual), *work.reconstruction, *work.work, engine, stream, time);
    }
    void FiniteVolume::EvaluateImpl(EquationDofs &state,
                                    OptionalRef<EquationDofs> residual,
                                    ReconstructionState &reconstruction, Workspace &work, R::EagerEngine &engine,
                                    const R::ExecutionStream &stream, real time,
                                    OptionalRef<AuxiliaryValues> auxiliary) const
    {
        if (!auxiliary && work.rhsAuxiliary)
            auxiliary.Emplace(*work.rhsAuxiliary);
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.FinalizedValues());
        const Workspace::PhaseTime timing(work, Workspace::TotalPhase);
        work.sourceState.reset();
        work.sourceCoefficients.reset();
        work.rhsState.reset();
        work.rhsCoefficients.reset();
        work.relativeUpdate = !work.meanOnly && _data->settings.numerics.reconstruction == "variational" ? std::numeric_limits<real>::infinity() : 0;
        work.faceFallbacks = {};
        work.diagnostics.Reset();
        work.characteristicFallbacks = 0;
        const MPIInfo mpi(Metrics().Communicator());
        // Repeated entry/CPU-join agreement follows the same checked/eager
        // policy as later local kernels. Unchecked execution still validates
        // locally and propagates failures to the fail-fast frontend. Payload
        // transfer preflight remains mandatory and unchanged.
        const auto local = [&](auto &&action)
        {
            Metrics().Communication()->LocalStage(std::forward<decltype(action)>(action));
        };
        // Observe only this lane's producer prefix. MPI remains on the caller's
        // controller thread, including when the CPU launcher uses worker threads.
        local([&]
              { stream.Record().WaitHost(); });
        std::optional<EquationPointEvaluator> evaluator;
        std::optional<AuxiliaryPreparation> auxiliaryPreparation;
        std::optional<R::Array<real>::ReadLease> incomingRead;
        RHSScalarView incoming;
        // A Cantera auxiliary pass and RHS pass may share a single worker
        // context. Do not acquire the RHS lease until aux has published and
        // returned its worker. Perfect-gas passes need no such worker lease.
        const bool deferredEvaluator = auxiliary && _data->equation.HasMixture();
        const auto createEvaluator = [&]
        {
            const bool viscous = !work.meanOnly && _data->equation.Viscous();
            evaluator.emplace(_data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                                 {viscous || _data->equation.HasRANS(), viscous, viscous, work.chemical, false}));
        };
        local([&]
              {
            DNDS_check_throw_info(work.owner == _data && reconstruction.owner == _data && std::isfinite(time) && stream.Context().device == R::Device::Host(),
                                  "foreign equation workspace/state, invalid time or missing CPU backend");
            ValidateWorkspace(state, work);
            reconstruction.binding.Check(state);
            if (residual)
            {
                auto &output = *residual;
                _data->equation.ValidateStateSchema(output);
                state.RequireCompatible(output);
                EquationDofOperations::RequireDistinct(state, output);
                EquationDofOperations::RequireDistinct(reconstruction.coefficients, output);
                EquationDofOperations::RequireDistinct(reconstruction.FinalizedValues(), output);
                auto write = output.WriteOwned(); write.CancelUnmodified();
            }
            PreflightHalo(state); PreflightHalo(reconstruction.coefficients); PreflightHalo(reconstruction.FinalizedValues());
            DNDS_check_throw_info(!HasInterfaces() || work.incoming,
                                  "FV evaluation needs a ready interface packet");
            if (work.incoming)
            {
                ValidateBoundaryInputs(*work.incoming, time);
                incomingRead.emplace(
                    work.incoming->QuadratureValues().CoupledField().Owned().Read());
                incoming = incomingRead->View<R::UniformTableRows>();
            }
            _data->partition.Validate();
            work.boundaryMomentumCurrent = false;
            if (!deferredEvaluator) createEvaluator(); });
        if (auxiliary && !work.meanGradients)
            local([&]
                  {
                      AuxiliaryInputs inputs;
                      if (work.incoming)
                          inputs.boundary.Emplace(*work.incoming);
                      auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers, inputs)); });
        const bool overlap = _data->settings.execution.overlap;
        const auto sweep = [&](CFV::ReconstructionRegion region)
        {
            if (work.limitedVR)
            {
                auto &field = work.fields.front();
                DNDS_check_throw_info(_data->reconstruction && field.reconstruction &&
                                          field.reconstructionBoundary,
                                      "limited VR schedule lacks polynomial workspace");
                _data->reconstruction->SubmitReconstructionIterLimitedTo(
                    engine, stream, state.CoupledField(), reconstruction.coefficients.CoupledField(),
                    *field.reconstruction, *field.reconstructionBoundary, work.limitedVR->targetCoefficients,
                    work.limitedVR->alpha, region);
                return;
            }
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                auto &field = work.fields[iField];
                DNDS_check_throw_info(_data->reconstruction && field.reconstruction &&
                                          field.reconstructionBoundary,
                                      "VR schedule lacks polynomial workspace");
                _data->reconstruction->SubmitReconstructionIterTo(
                    engine, stream, state.PhysicalField(iField), reconstruction.coefficients.PhysicalField(iField),
                    *field.reconstruction, *field.reconstructionBoundary, region);
            }
        };
        const auto gradient = [&](CFV::ReconstructionRegion region)
        {
            const Workspace::PhaseTime timing(work, Workspace::GradientPhase);
            if (work.limitedVR)
                _data->limitedVRGeometry->gradient.Evaluate(state.CoupledField(), *work.fields.front().gradientBoundary,
                                                            work.limitedVR->gradients, {1}, region);
            else
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    auto &field = work.fields[iField];
                    if (field.gradient)
                        _data->gradient.Evaluate(state.PhysicalField(iField), *field.gradientBoundary, *field.gradient, field.vectorStarts, region);
                }
            work.PreparePrimitiveGradient(state, region);
        };
        // Destruction order is part of the worker lifetime contract: gradient
        // is captured by reference below, so the schedule must drain before
        // this closure and the earlier retained operands leave scope.
        OperatorSchedule schedule(engine, stream, overlap);
        const auto rest = schedule.DependentRegion();
        // Payload operations retain their own inputs; queued callbacks only
        // borrow objects declared before schedule (or capture values explicitly).
        std::vector<Algebra::DofHalo<>::Operation> meanOperations;
        meanOperations.reserve(reconstruction.meanHalo.size());
        ++reconstruction.meanEpoch;
        for (auto &halo : reconstruction.meanHalo)
            meanOperations.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        std::vector<R::TransferOperation> firstCoefficients;
        const auto &numerics = _data->settings.numerics;
        const bool variational = !work.meanOnly && numerics.reconstruction == "variational";
        const bool firstSweep = overlap && variational && !work.limitedVR && numerics.reconstructionIterations > 0 && !deferredEvaluator;
        if (firstSweep)
            firstCoefficients = schedule.Begin(reconstruction.coefficientHalo);
        // Post both transfers before queueing core work. A CPU-visible MPI
        // bridge may otherwise join that work while checking producer readiness.
        if (auxiliaryPreparation && overlap)
            schedule.Submit([&]
                            { auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });
        if (!deferredEvaluator)
            schedule.Submit([&, variational]
                            { work.FillBoundary(state, variational && _data->settings.numerics.reconstructionBoundaryFromPolynomial,
                                                time, *evaluator, incoming); });
        // P0 RANS sources still need physical gradients during a mean-only
        // timestep forecast; disabling reconstruction must not skip this pass.
        const bool directGradientWork = (!work.meanOnly && numerics.reconstruction == "gradient") ||
                                        (_data->equation.HasRANS() && numerics.reconstruction == "constant");
        const bool gradientWork = directGradientWork || bool(work.limitedVR) || bool(work.primitiveGradient);
        if (gradientWork && overlap && !deferredEvaluator)
            schedule.Submit([&]
                            { gradient(CFV::ReconstructionRegion::Core); });
        if (firstSweep)
            sweep(CFV::ReconstructionRegion::Core);
        OperatorSchedule::Wait(meanOperations);
        if (auxiliaryPreparation)
        {
            schedule.Submit([&]
                            {
                auxiliaryPreparation->Apply(rest);
                auxiliaryPreparation->Finish(*auxiliary); });
            local([&]
                  { schedule.Join(); });
        }
        else if (auxiliary)
        {
            // Two-equation transport explicitly needs mean gradients; those
            // are independent of the reconstruction being computed below.
            AuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary.Emplace(*work.incoming);
            PrepareAuxValues(state, *auxiliary, *work.meanGradients, engine, stream, time, work.workers, inputs);
        }
        if (deferredEvaluator)
        {
            local(createEvaluator);
            schedule.Submit([&, variational]
                            { work.FillBoundary(state, variational && numerics.reconstructionBoundaryFromPolynomial,
                                                time, *evaluator, incoming); });
        }
        if (_data->settings.CheckHaloFreshness())
            local([&]
                  {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch); });
        if (gradientWork)
        {
            schedule.Submit([&, rest]
                            { gradient(deferredEvaluator ? CFV::ReconstructionRegion::All : rest); });
            if (directGradientWork)
                schedule.Submit([&]
                                {
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                {
                    auto &field = work.fields[iField];
                    if (numerics.reconstruction == "gradient" && !work.nativeDirectGradient)
                    {
                        DNDS_check_throw_info(_data->reconstruction,
                                              "gradient-to-modal conversion requires a polynomial plan");
                        _data->gradient.ConvertToCoefficients(*field.gradient, *_data->reconstruction,
                                                             reconstruction.coefficients.PhysicalField(iField));
                    }
                    else if (_data->reconstruction)
                        Algebra::Fill(reconstruction.coefficients.PhysicalField(iField), 0.);
                } });
            local([&]
                  { schedule.Join(); });
        }
        std::optional<R::TransferOperation> primitiveGradientOperation;
        if (work.primitiveGradientHalo)
            primitiveGradientOperation.emplace(schedule.Begin(*work.primitiveGradientHalo));
        if (work.limitedVR)
        {
            if (work.flowCharacteristic)
                local([&]
                      {
                const auto flow = state.LocateBlock("flow");
                work.characteristicFallbacks = PrepareFaceCharacteristics(
                    state.PhysicalField(flow.physicalField), *work.flowCharacteristic, flow.firstComponent); });
            DNDS_check_throw_info(reconstruction.limitedVRHistory, "LVR reconstruction state has no history");
            work.limitedVR->Build(*_data, state.CoupledField(), *reconstruction.limitedVRHistory,
                                  work.flowCharacteristic ? CFV::FaceCharacteristicTransformRef(std::cref(*work.flowCharacteristic))
                                                          : std::nullopt,
                                  engine, stream);
        }
        if (variational)
        {
            for (int iteration = 0; iteration < numerics.reconstructionIterations; ++iteration)
            {
                std::vector<R::TransferOperation> operations;
                if (iteration == 0 && firstSweep)
                    operations = std::move(firstCoefficients);
                else
                {
                    local([&]
                          { schedule.Join(); });
                    if (numerics.reconstructionBoundaryFromPolynomial)
                        local([&]
                              { work.FillBoundary(state, true, time, *evaluator, incoming); });
                    operations = schedule.Begin(reconstruction.coefficientHalo);
                    if (overlap)
                        sweep(CFV::ReconstructionRegion::Core);
                }
                OperatorSchedule::Wait(operations);
                sweep(rest);
                local([&]
                      { schedule.Join(); });
                bool converged = false;
                if (numerics.reconstructionTolerance > 0 && (iteration % 4 == 3 || iteration + 1 == numerics.reconstructionIterations))
                {
                    real norms[2]{};
                    local([&]
                          {
                        for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        {
                            const auto aRead = reconstruction.coefficients.PhysicalField(iField).Owned().Read();
                            DNDS_check_throw_info(work.fields[iField].reconstruction,
                                                  "VR convergence check lacks workspace");
                            const auto bRead = work.fields[iField].reconstruction->temporary.Owned().Read();
                            const auto a = aRead.View<R::UniformTableRows>(), b = bRead.View<R::UniformTableRows>();
                            CPU::AccumulateReconstructionUpdateNorm(a, b, {norms, 2});
                        } });
                    MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                        MPI::ContiguousView<real>{norms, 2}, mpi.comm);
                    work.relativeUpdate = std::sqrt(norms[0] / std::max(norms[1], real(1e-300)));
                    converged = work.relativeUpdate <= numerics.reconstructionTolerance;
                }
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    _data->reconstruction->SubmitCommitReconstruction(engine, stream,
                                                                      reconstruction.coefficients.PhysicalField(iField),
                                                                      *work.fields[iField].reconstruction);
                local([&]
                      { schedule.Join(); });
                work.iterations = iteration + 1;
                if (converged)
                    break;
            }
        }
        else if (!directGradientWork && _data->reconstruction)
            local([&]
                  {
                schedule.Join();
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    Algebra::Fill(reconstruction.coefficients.PhysicalField(iField), 0.); });
        const bool useFluxScratch = reconstruction.distinctFinalized;
        if (useFluxScratch)
            local([&]
                  { EquationDofOperations::Copy(reconstruction.MutableFinalized(), reconstruction.coefficients); });
        if (work.nativeDirectGradient && numerics.limiter != "WBAP" &&
            (numerics.limiter == "Barth" || numerics.positivity))
            local([&]
                  { work.PrepareDirectGradientLimiter(state, numerics.limiter == "Barth"); });
        if (numerics.limiter == "WBAP")
        {
            if (work.nativeDirectGradient)
            {
                if (work.gradientFlowCharacteristic)
                    local([&]
                          {
                    const auto flow = state.LocateBlock("flow");
                    DNDS_check_throw_info(flow.physicalField == work.flowCharacteristicField,
                                          "characteristic flow allocation changed after workspace setup");
                    work.characteristicFallbacks = PrepareFaceCharacteristics(
                        state.PhysicalField(flow.physicalField), *work.gradientFlowCharacteristic,
                        flow.firstComponent); });
                auto gradientOperations = schedule.Begin(work.directGradientInputHalo);
                const auto limit = [&](CFV::ReconstructionRegion region)
                {
                    for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        _data->gradient.LimitWBAP3(
                            *work.fields[iField].gradient, work.directGradients[iField],
                            work.fields[iField].vectorStarts,
                            iField == work.flowCharacteristicField && work.gradientFlowCharacteristic
                                ? CFV::GradientFaceCharacteristicTransformRef(
                                      std::cref(*work.gradientFlowCharacteristic))
                                : std::nullopt,
                            region, std::nullopt, 0, numerics.cfv.WBAP_nStd);
                };
                if (overlap)
                    limit(CFV::ReconstructionRegion::Core);
                OperatorSchedule::Wait(gradientOperations);
                limit(rest);
                local([&]
                      { schedule.Join(); });
            }
            else
            {
                if (work.flowCharacteristic && !work.limitedVR)
                    local([&]
                          {
                              const auto flow = state.LocateBlock("flow");
                              DNDS_check_throw_info(flow.physicalField == work.flowCharacteristicField,
                                                    "characteristic flow allocation changed after workspace setup");
                              work.characteristicFallbacks = PrepareFaceCharacteristics(
                                  state.PhysicalField(flow.physicalField), *work.flowCharacteristic, flow.firstComponent); });
                for (int order = numerics.cfv.maxOrder; order >= 1; --order)
                {
                    auto limiterOperations = schedule.Begin(reconstruction.FinalizedHalo());
                    const auto limit = [&, order](CFV::ReconstructionRegion region)
                    {
                        for (index iField = 0; iField < index(work.fields.size()); ++iField)
                            _data->reconstruction->SubmitLimiterWBAP_3OrderTo(
                                engine, stream, reconstruction.MutableFinalized().PhysicalField(iField),
                                *work.fields[iField].reconstruction, order,
                                iField == work.flowCharacteristicField && work.flowCharacteristic
                                    ? CFV::FaceCharacteristicTransformRef(std::cref(*work.flowCharacteristic))
                                    : std::nullopt,
                                region);
                    };
                    if (overlap)
                        limit(CFV::ReconstructionRegion::Core);
                    OperatorSchedule::Wait(limiterOperations);
                    limit(rest);
                    local([&]
                          { schedule.Join(); });
                    for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        _data->reconstruction->SubmitCommitReconstruction(
                            engine, stream, reconstruction.MutableFinalized().PhysicalField(iField),
                            *work.fields[iField].reconstruction);
                    local([&]
                          { schedule.Join(); });
                }
            }
        }
        if (work.nativeDirectGradient && numerics.positivity)
            local([&]
                  {
                      const Workspace::PhaseTime timing(work, Workspace::PPPhase);
                      work.ApplyDirectGradientPositivity(state); });
        if (work.positivity)
            local([&]
                  { const Workspace::PhaseTime timing(work, Workspace::PPPhase);
                    work.positivity->Apply(state, reconstruction.coefficients, reconstruction.MutableFinalized(), false); });
        local([&]
              { reconstruction.PublishFinalized(); });
        auto operations = work.nativeDirectGradient
                              ? schedule.Begin(work.directGradientHalo)
                              : schedule.Begin(reconstruction.FinalizedHalo());
        std::vector<R::TransferOperation> unlimitedViscousOperations;
        if (useFluxScratch && _data->equation.Viscous() && numerics.viscousGradient == "polynomial" && !numerics.viscousLimited)
            unlimitedViscousOperations = schedule.Begin(reconstruction.coefficientHalo);
        auto directViscousOperations = schedule.Begin(work.directViscousGradientHalo);
        // Publish the direct physical gradients after limiter/PP, once per row.
        // The physical-gradient RHS itself never interprets coefficient data.
        schedule.Submit([&]
                        { work.RefreshDirectGradients(state, false, time); });
        if (overlap && residual)
            schedule.Submit([&]
                            { work.ApplyRHS(state, *residual, CFV::ReconstructionRegion::Core, *evaluator, time,
                                            OptionalRef<const AuxiliaryValues>(auxiliary), {}, incoming); });
        OperatorSchedule::Wait(operations);
        OperatorSchedule::Wait(unlimitedViscousOperations);
        OperatorSchedule::Wait(directViscousOperations);
        if (primitiveGradientOperation)
            primitiveGradientOperation->Wait();
        schedule.Submit([&]
                        { work.RefreshDirectGradients(state, true, time); });
        if (residual)
        {
            schedule.Submit([&, rest]
                            { work.ApplyRHS(state, *residual, rest, *evaluator, time,
                                            OptionalRef<const AuxiliaryValues>(auxiliary), {}, incoming); });
            schedule.Submit([&]
                            { work.PrepareVolumeSource(
                                  state, OptionalRef<EquationDofs>(*residual), *evaluator, time,
                                  work.lowOrderResidual
                                      ? OptionalRef<EquationDofs>(*work.lowOrderResidual)
                                      : OptionalRef<EquationDofs>{}); });
        }
        local([&]
              { schedule.Join(); });
        if (residual && (work.chemical || work.ransFirst >= 0) && work.sourceSampling)
            local([&]
                  { work.PublishSourceState(state, time); });
        work.boundaryMomentumCurrent = bool(residual);
        work.rhsState.emplace(state.NumericAlias());
        work.rhsCoefficients.emplace(reconstruction.FinalizedValues());
        work.rhsTime = time;
    }

}
