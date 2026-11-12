#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
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
                                            Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const
    {
        PrepareCorrectionAux(state, reconstruction, work, auxiliary, engine,
                             stream, stream, time);
    }
    void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction,
                                            Workspace &work, AuxiliaryValues &auxiliary,
                                            Runtime::EagerEngine &engine,
                                            const Runtime::ExecutionStream &stream,
                                            const Runtime::ExecutionStream &communication,
                                            real time,
                                            OptionalRef<const QueuedEvaluationDependency> dependency) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction);
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
            if (dependency)
            {
                // The current RANS gradient preparer acquires its boundary
                // scope on the controller. Keep this explicit bridge until
                // that preparer accepts a deferred packet too; no MPI runs in
                // the queued publication callback.
                OperatorSchedule boundarySchedule(engine, stream, communication, false);
                boundarySchedule.ComputeWaitFor(dependency->completion);
                boundarySchedule.Submit([&]
                                        { dependency->publish(); });
                boundarySchedule.Join();
            }
            ValidateWorkspace(state, work);
            AuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary.Emplace(*work.incoming);
            PrepareAuxValues(state, auxiliary, *work.meanGradients, engine,
                             stream, communication, time, work.workers,
                             inputs);
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
            inputs.deferBoundary = bool(dependency);
            if (work.incoming && !dependency)
                inputs.boundary.Emplace(*work.incoming);
            preparation.emplace(BeginAuxValues(state, auxiliary, time, work.workers, inputs)); });
        OperatorSchedule schedule(engine, stream, communication,
                                  _data->settings.execution.overlap);
        std::vector<Algebra::DofHalo<>::Operation> transfers;
        transfers.reserve(reconstruction.meanHalo.size());
        ++reconstruction.meanEpoch;
        for (auto &halo : reconstruction.meanHalo)
            transfers.push_back(halo.BeginReady(reconstruction.meanEpoch, engine, communication,
                                                _data->settings.CheckHaloFreshness()));
        if (_data->settings.execution.overlap)
            schedule.Submit([&]
                            {
                if (dependency)
                    preparation->ApplyInterfaceIndependent();
                else
                    preparation->Apply(CFV::ReconstructionRegion::Core); });
        if (dependency)
        {
            schedule.ComputeWaitFor(dependency->completion);
            schedule.Submit([&]
                            {
                dependency->publish();
                DNDS_check_throw_info(work.incoming,
                                      "correction interface refresh did not bind a packet");
                preparation->BindBoundary(*work.incoming);
                if (_data->settings.execution.overlap)
                    preparation->Apply(CFV::ReconstructionRegion::Core); });
        }
        for (auto &transfer : transfers)
            schedule.AwaitPublished(std::move(transfer));
        transfers.clear();
        const auto rest = schedule.DependentRegion();
        local([&]
              {
            schedule.Submit([&, rest]
            {
                if (_data->settings.CheckHaloFreshness())
                    for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                        reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch);
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
        auto activeReconstruction = work.BindReconstruction(reconstruction);
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
                                Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual), reconstruction,
                     work, engine, stream, stream, time);
    }
    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &work,
                                AuxiliaryValues &auxiliary, Runtime::EagerEngine &engine,
                                const Runtime::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual), reconstruction,
                     work, engine, stream, stream, time,
                     OptionalRef<AuxiliaryValues>(auxiliary));
    }
    void FiniteVolume::PrepareReconstruction(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, Runtime::EagerEngine &engine,
                                             const Runtime::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, {}, reconstruction, work, engine, stream, stream,
                     time);
    }
    void FiniteVolume::RefreshSamplingInputs(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, Runtime::EagerEngine &engine,
                                             const Runtime::ExecutionStream &stream,
                                             const Runtime::ExecutionStream &communication) const
    {
        ValidateWorkspace(state, work);
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction);
        Metrics().Communication()->LocalStage([&]
                                              {
            DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                                  "sampling refresh requires the workspace CPU engine");
            PreflightHalo(state);
            if (reconstruction.coefficients)
                PreflightHalo(reconstruction.FinalizedValues()); });
        ++reconstruction.meanEpoch;
        std::vector<Algebra::DofHalo<>::Operation> means;
        means.reserve(reconstruction.meanHalo.size());
        OperatorSchedule schedule(engine, stream, communication, true);
        for (auto &halo : reconstruction.meanHalo)
            means.push_back(halo.BeginReady(reconstruction.meanEpoch, engine,
                                            communication, _data->settings.CheckHaloFreshness()));
        // Modal coefficients are distinct from means whenever this list is
        // nonempty. Post all independent payloads before the final join;
        // gradient/P0 representations have no modal-coefficient exchange.
        auto coefficients = schedule.Begin(reconstruction.FinalizedHalo());
        for (auto &operation : means)
            schedule.AwaitPublished(std::move(operation));
        means.clear();
        schedule.Await(coefficients);
        // Sampling consumes CPU-visible inputs today. The transfer segment
        // itself uses the same event/publication protocol as RHS preparation.
        schedule.Join();
        if (_data->settings.CheckHaloFreshness())
            Metrics().Communication()->LocalStage([&]
                                                  {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch); });
    }
    void FiniteVolume::Execute(EquationDofs &state,
                               ReconstructionState &reconstruction,
                               Workspace &work,
                               DomainExecutionContext &context,
                               FiniteVolumeEvaluation evaluation) const
    {
        auto &engine = context.Engine();
        const auto &stream = context.ComputeStream();
        const auto &communication = context.CommunicationStream();
        switch (evaluation.mode)
        {
        case FiniteVolumeEvaluationMode::SpatialResidual:
            DNDS_check_throw_info(evaluation.residual,
                                  "spatial FV evaluation requires a residual output");
            {
                auto dependency = context.AcquireEvaluationDependency();
                if (evaluation.auxiliary)
                    EvaluateImpl(state, OptionalRef<EquationDofs>(*evaluation.residual),
                                 reconstruction, work, engine, stream, communication,
                                 evaluation.time,
                                 OptionalRef<AuxiliaryValues>(*evaluation.auxiliary),
                                 dependency ? OptionalRef<const QueuedEvaluationDependency>(*dependency)
                                            : OptionalRef<const QueuedEvaluationDependency>{});
                else
                    EvaluateImpl(state, OptionalRef<EquationDofs>(*evaluation.residual),
                                 reconstruction, work, engine, stream, communication,
                                 evaluation.time, {},
                                 dependency ? OptionalRef<const QueuedEvaluationDependency>(*dependency)
                                            : OptionalRef<const QueuedEvaluationDependency>{});
                return;
            }
        case FiniteVolumeEvaluationMode::ReconstructionOnly:
            DNDS_check_throw_info(!evaluation.residual && !evaluation.auxiliary,
                                  "reconstruction-only FV evaluation has no residual or auxiliary output");
            EvaluateImpl(state, {}, reconstruction, work, engine, stream,
                         communication, evaluation.time);
            return;
        case FiniteVolumeEvaluationMode::SamplingInputs:
            DNDS_check_throw_info(!evaluation.residual && !evaluation.auxiliary,
                                  "sampling-input FV evaluation has no residual or auxiliary output");
            RefreshSamplingInputs(state, reconstruction, work, engine, stream,
                                  communication);
            return;
        }
        DNDS_check_throw_info(false, "unknown finite-volume evaluation mode");
    }
    void FiniteVolume::EvaluateMean(EquationDofs &state, EquationDofs &residual, MeanWorkspace &work,
                                    Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, OptionalRef<EquationDofs>(residual),
                     *work.reconstruction, *work.work, engine, stream, stream,
                     time);
    }
    void FiniteVolume::EvaluateImpl(EquationDofs &state,
                                    OptionalRef<EquationDofs> residual,
                                    ReconstructionState &reconstruction, Workspace &work, Runtime::EagerEngine &engine,
                                    const Runtime::ExecutionStream &stream,
                                    const Runtime::ExecutionStream &communication, real time,
                                    OptionalRef<AuxiliaryValues> auxiliary,
                                    OptionalRef<const QueuedEvaluationDependency> dependency) const
    {
        if (!auxiliary && work.rhsAuxiliary)
            auxiliary.Emplace(*work.rhsAuxiliary);
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction);
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
        DNDS_check_throw_info(engine.Owns(stream) &&
                                  engine.Owns(communication),
                              "FV execution lanes belong to another runtime");
        std::optional<PointEvaluator> evaluator;
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
            DNDS_check_throw_info(work.owner == _data && reconstruction.owner == _data &&
                                      std::isfinite(time) &&
                                      stream.Context().device == R::Device::Host() &&
                                      communication.Context().device == R::Device::Host(),
                                  "foreign equation workspace/state, invalid time or missing CPU backend");
            ValidateWorkspace(state, work);
            reconstruction.binding.Check(state);
            if (residual)
            {
                auto &output = *residual;
                output.RequireSchema(_data->equation.Schema());
                state.RequireCompatible(output);
                EquationDofOperations::RequireDistinct(state, output);
                if (reconstruction.coefficients)
                {
                    EquationDofOperations::RequireDistinct(reconstruction.WarmValues(), output);
                    EquationDofOperations::RequireDistinct(reconstruction.FinalizedValues(), output);
                }
                auto write = output.WriteOwned(); write.CancelUnmodified();
            }
            PreflightHalo(state);
            if (reconstruction.coefficients)
            {
                PreflightHalo(reconstruction.WarmValues());
                PreflightHalo(reconstruction.FinalizedValues());
            }
            DNDS_check_throw_info(!HasInterfaces() || work.incoming || dependency,
                                  "FV evaluation needs a ready interface packet");
            if (work.incoming && !dependency)
            {
                ValidateBoundaryInputs(*work.incoming, time);
                incomingRead.emplace(
                    work.incoming->QuadratureValues().CoupledField().Owned().Read());
                incoming = incomingRead->View<R::UniformTableRows>();
            }
            _data->partition.Validate();
            work.boundaryMomentumCurrent = false;
            if (!deferredEvaluator) createEvaluator(); });
        const bool needsAuxiliaryPreparation = auxiliary && !work.meanGradients;
        if (needsAuxiliaryPreparation)
            local([&]
                  {
                      AuxiliaryInputs inputs;
                      inputs.deferBoundary = bool(dependency);
                      if (work.incoming && !dependency)
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
                    engine, stream, state.CoupledField(), reconstruction.WarmValues().CoupledField(),
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
                    engine, stream, state.PhysicalField(iField), reconstruction.WarmValues().PhysicalField(iField),
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
        OperatorSchedule schedule(engine, stream, communication, overlap);
        const auto rest = schedule.DependentRegion();
        // Payload operations retain their own inputs; queued callbacks only
        // borrow objects declared before schedule (or capture values explicitly).
        std::vector<Algebra::DofHalo<>::Operation> meanOperations;
        meanOperations.reserve(reconstruction.meanHalo.size());
        ++reconstruction.meanEpoch;
        for (auto &halo : reconstruction.meanHalo)
            meanOperations.push_back(halo.BeginReady(
                reconstruction.meanEpoch, engine, communication,
                _data->settings.CheckHaloFreshness()));
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
                            {
                if (dependency)
                    auxiliaryPreparation->ApplyInterfaceIndependent();
                else
                    auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });
        if (!deferredEvaluator && !dependency)
            schedule.Submit([&, variational]
                            { work.FillBoundary(state, variational && _data->settings.numerics.reconstructionBoundaryFromPolynomial,
                                                time, *evaluator, incoming); });
        // P0 RANS sources still need physical gradients during a mean-only
        // timestep forecast; disabling reconstruction must not skip this pass.
        const bool directGradientWork = (!work.meanOnly && numerics.reconstruction == "gradient") ||
                                        (_data->equation.HasRANS() &&
                                         (numerics.reconstruction == "constant" ||
                                          (work.meanOnly && numerics.reconstruction == "gradient")));
        const bool gradientWork = directGradientWork || bool(work.limitedVR) || bool(work.primitiveGradient);
        if (gradientWork && overlap && !deferredEvaluator && !dependency)
            schedule.Submit([&]
                            { gradient(CFV::ReconstructionRegion::Core); });
        if (firstSweep && !dependency)
            sweep(CFV::ReconstructionRegion::Core);
        if (dependency)
        {
            schedule.ComputeWaitFor(dependency->completion);
            schedule.Submit([&]
                            {
                dependency->publish();
                DNDS_check_throw_info(work.incoming,
                                      "interface refresh did not bind a boundary packet");
                ValidateBoundaryInputs(*work.incoming, time);
                incomingRead.emplace(
                    work.incoming->QuadratureValues().CoupledField().Owned().Read());
                incoming = incomingRead->View<R::UniformTableRows>();
                if (needsAuxiliaryPreparation)
                    auxiliaryPreparation->BindBoundary(*work.incoming); });
            if (needsAuxiliaryPreparation && overlap)
                schedule.Submit([&]
                                { auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });
            if (!deferredEvaluator)
                schedule.Submit([&, variational]
                                { work.FillBoundary(state, variational && _data->settings.numerics.reconstructionBoundaryFromPolynomial,
                                                    time, *evaluator, incoming); });
            // The ordinary MPI core may contain cells incident to a physical
            // interface. Until a separately prepared interface-independent
            // partition is supplied, run these boundary-consuming core passes
            // after the interface event; never infer safety from MPI locality.
            if (gradientWork && overlap && !deferredEvaluator)
                schedule.Submit([&]
                                { gradient(CFV::ReconstructionRegion::Core); });
            if (firstSweep)
                sweep(CFV::ReconstructionRegion::Core);
        }
        for (auto &operation : meanOperations)
            schedule.AwaitPublished(std::move(operation));
        meanOperations.clear();
        if (needsAuxiliaryPreparation)
        {
            schedule.Submit([&, rest]
                            {
                auxiliaryPreparation->Apply(rest);
                auxiliaryPreparation->Finish(*auxiliary); });
        }
        else if (auxiliary)
        {
            // Two-equation transport explicitly needs mean gradients; those
            // are independent of the reconstruction being computed below.
            // The current mean-gradient preparation acquires its CPU scopes
            // synchronously, so publish any queued packet before binding them.
            // This is a host-access bridge, not an MPI-core safety assumption.
            local([&]
                  { schedule.Join(); });
            AuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary.Emplace(*work.incoming);
            PrepareAuxValues(state, *auxiliary, *work.meanGradients, engine,
                             stream, communication, time, work.workers,
                             inputs);
        }
        if (deferredEvaluator)
        {
            schedule.Submit([&, variational]
                            {
                // Auxiliary Finish returns its chemistry lease before this
                // next command acquires the RHS worker. The handoff is a lane
                // dependency, not a controller-side reason to drain the engine.
                createEvaluator();
                work.FillBoundary(state, variational && numerics.reconstructionBoundaryFromPolynomial,
                                  time, *evaluator, incoming); });
        }
        if (_data->settings.CheckHaloFreshness())
            schedule.Submit([&]
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
                    if (!work.meanOnly && numerics.reconstruction == "gradient" && !work.nativeDirectGradient)
                    {
                        DNDS_check_throw_info(_data->reconstruction,
                                              "gradient-to-modal conversion requires a polynomial plan");
                        _data->gradient.ConvertToCoefficients(*field.gradient, *_data->reconstruction,
                                                             reconstruction.WarmValues().PhysicalField(iField));
                    }
                    else if (_data->reconstruction)
                    {
                        auto write = reconstruction.WarmValues().PhysicalField(iField).Owned().Write();
                        const auto values = write.View<R::UniformTableRows>();
                        CPU::FillRows(values, values.Width(), 0.);
                    }
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
                                  engine, stream, communication);
        }
        if (variational)
        {
            for (int iteration = 0; iteration < numerics.reconstructionIterations; ++iteration)
            {
                std::vector<R::TransferOperation> operations;
                if (iteration == 0 && firstSweep)
                {
                    operations = std::move(firstCoefficients);
                }
                else
                {
                    if (numerics.reconstructionBoundaryFromPolynomial)
                        schedule.Submit([&]
                                        { work.FillBoundary(state, true, time, *evaluator, incoming); });
                    operations = schedule.BeginAfterCompute(reconstruction.coefficientHalo);
                    if (overlap)
                        sweep(CFV::ReconstructionRegion::Core);
                }
                schedule.Await(operations);
                sweep(rest);
                bool converged = false;
                // Explicit host stopping boundary. Fixed-count sweeps do not
                // observe a residual. Device-side conditional iteration / queue
                // mutation is future work, not hidden inside this segment.
                if (numerics.reconstructionTolerance > 0 &&
                    ((iteration + 1) % numerics.reconstructionResidualCheckInterval == 0 ||
                     iteration + 1 == numerics.reconstructionIterations))
                {
                    local([&]
                          { schedule.Join(); });
                    real norms[2]{};
                    local([&]
                          {
                        for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        {
                            const auto aRead = reconstruction.WarmValues().PhysicalField(iField).Owned().Read();
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
                                                                      reconstruction.WarmValues().PhysicalField(iField),
                                                                      *work.fields[iField].reconstruction);
                // Fixed-count sweeps enqueue commit and the next boundary
                // producer in order. Only requested norm observations above
                // and the portable-MPI readiness bridge need host visibility.
                work.iterations = iteration + 1;
                if (converged)
                    break;
            }
        }
        else if (!directGradientWork && _data->reconstruction)
            local([&]
                  {
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    Algebra::Fill(engine, stream, reconstruction.WarmValues().PhysicalField(iField), 0.); });
        const bool useFluxScratch = reconstruction.distinctFinalized;
        if (useFluxScratch)
            schedule.Submit([&]
                            { EquationDofOperations::Copy(reconstruction.MutableFinalized(), reconstruction.WarmValues()); });
        // These current CPU postprocessors acquire controller-visible scopes
        // and publish transform/version metadata. Keep one explicit boundary for
        // them, not a drain in every fixed-count VR sweep. Their prepared
        // device implementations remain a separate operator migration.
        const bool hostPostprocessing = work.positivity ||
                                        (work.nativeDirectGradient && numerics.positivity) ||
                                        (!work.meanOnly && (numerics.limiter == "WBAP" || numerics.limiter == "Barth"));
        if (hostPostprocessing)
            local([&]
                  { schedule.Join(); });
        if (work.nativeDirectGradient && numerics.limiter != "WBAP" &&
            (numerics.limiter == "Barth" || numerics.positivity))
            local([&]
                  { work.PrepareDirectGradientLimiter(state, numerics.limiter == "Barth"); });
        // Mean-only forecasts do not publish a reconstructed face field, even
        // when the regular operator is configured with gradient/modal WBAP.
        if (!work.meanOnly && numerics.limiter == "WBAP")
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
                    schedule.Submit([limit]
                                    { limit(CFV::ReconstructionRegion::Core); });
                schedule.Await(gradientOperations);
                schedule.Submit([limit, rest]
                                { limit(rest); });
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
                    auto limiterOperations = schedule.BeginAfterCompute(reconstruction.FinalizedHalo());
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
                    schedule.Await(limiterOperations);
                    limit(rest);
                    for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        _data->reconstruction->SubmitCommitReconstruction(
                            engine, stream, reconstruction.MutableFinalized().PhysicalField(iField),
                            *work.fields[iField].reconstruction);
                    // Commit precedes the next order's producer-prefix
                    // exchange; no host observation is needed between orders.
                }
            }
        }
        if (work.nativeDirectGradient && numerics.positivity)
            local([&]
                  {
                      const Workspace::PhaseTime timing(work, Workspace::PPPhase);
                      work.ApplyDirectGradientPositivity(state); });
        if (work.positivity)
            local([&] { // CPU polynomial PP inspects limiter publications and
                // uses rank-local host decisions for reference selection and
                // recovery. This is not a numerical MPI reduction boundary.
                schedule.Join();
                const Workspace::PhaseTime timing(work, Workspace::PPPhase);
                work.positivity->Apply(state, reconstruction.WarmValues(), reconstruction.MutableFinalized(), false);
            });
        schedule.Submit([&]
                        { reconstruction.PublishFinalized(); });
        auto operations = work.nativeDirectGradient
                              ? schedule.BeginAfterCompute(work.directGradientHalo)
                              : schedule.BeginAfterCompute(reconstruction.FinalizedHalo());
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
        schedule.Await(operations);
        schedule.Await(unlimitedViscousOperations);
        schedule.Await(directViscousOperations);
        if (primitiveGradientOperation)
            schedule.Await(*primitiveGradientOperation);
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
        if (reconstruction.coefficients)
            work.rhsCoefficients.emplace(reconstruction.FinalizedValues());
        work.rhsTime = time;
    }

}
