#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"

#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "Cauchy/Implicit/CPU/NonlinearKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"
#include "Cauchy/Core/LocalStage.hpp"
#include "Cauchy/Implicit/LinearScaling.hpp"
#include "Cauchy/Implicit/NonlinearController.hpp"
#include <exception>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void to_json(Json &value, const NonlinearSolveState &state)
    {
        value = Json{{"schema_version", state.schemaVersion},
                     {"attempt", state.attempt},
                     {"iteration", state.iteration},
                     {"stage", state.stage},
                     {"coupled_stage", state.coupledStage},
                     {"converged", state.converged},
                     {"terminal", state.terminal},
                     {"capped", state.capped},
                     {"rejected", state.rejected},
                     {"residual", state.residual},
                     {"initial_residual", state.initialResidual},
                     {"reference_residual", state.referenceResidual},
                     {"convergence_components", state.convergenceComponents},
                     {"residual_target_ratio", state.residualTargetRatio},
                     {"pseudo_cfl", state.pseudoCFL},
                     {"linear_iterations", state.linearIterations},
                     {"multigrid_cycles", state.multigridCycles},
                     {"level_corrections", state.levelCorrections},
                     {"rhs_limited", state.rhsLimited},
                     {"increment_limited", state.incrementLimited},
                     {"minimum_rhs_alpha", state.minimumRHSAlpha},
                     {"minimum_increment_alpha", state.minimumIncrementAlpha},
                     {"reconstruction_iterations", state.reconstructionIterations},
                     {"reconstruction_relative_update", state.reconstructionRelativeUpdate},
                     {"reconstruction_update_measured", state.reconstructionUpdateMeasured}};
    }
    void from_json(const Json &value, NonlinearSolveState &state)
    {
        DNDS_check_throw_info(value.is_object() && value.value("schema_version", 0) == 1,
                              "unsupported nonlinear solve state schema");
        state.schemaVersion = 1;
        state.attempt = value.at("attempt").get<index>();
        state.iteration = value.at("iteration").get<index>();
        state.stage = value.at("stage").get<int>();
        state.coupledStage = value.at("coupled_stage").get<int>();
        state.converged = value.at("converged").get<bool>();
        state.terminal = value.at("terminal").get<bool>();
        state.capped = value.at("capped").get<bool>();
        state.rejected = value.at("rejected").get<bool>();
        state.residual = value.at("residual").get<std::vector<real>>();
        state.initialResidual = value.at("initial_residual").get<std::vector<real>>();
        state.referenceResidual = value.at("reference_residual").get<std::vector<real>>();
        state.convergenceComponents = value.at("convergence_components").get<std::vector<std::string>>();
        state.residualTargetRatio = value.at("residual_target_ratio").get<real>();
        state.pseudoCFL = value.at("pseudo_cfl").get<real>();
        state.linearIterations = value.at("linear_iterations").get<index>();
        state.multigridCycles = value.at("multigrid_cycles").get<index>();
        state.levelCorrections = value.at("level_corrections").get<std::vector<index>>();
        state.rhsLimited = value.at("rhs_limited").get<index>();
        state.incrementLimited = value.at("increment_limited").get<index>();
        state.minimumRHSAlpha = value.at("minimum_rhs_alpha").get<real>();
        state.minimumIncrementAlpha = value.at("minimum_increment_alpha").get<real>();
        state.reconstructionIterations = value.at("reconstruction_iterations").get<int>();
        state.reconstructionRelativeUpdate = value.at("reconstruction_relative_update").get<real>();
        state.reconstructionUpdateMeasured = value.value("reconstruction_update_measured", true);
        DNDS_check_throw_info(state.attempt >= 0 && state.iteration >= 0 && std::isfinite(state.pseudoCFL) &&
                                  std::isfinite(state.residualTargetRatio) && std::isfinite(state.minimumRHSAlpha) &&
                                  std::isfinite(state.minimumIncrementAlpha) && std::isfinite(state.reconstructionRelativeUpdate),
                              "invalid nonlinear solve state values");
    }
    Json NonlinearSolveState::ToJson() const { return Json(*this); }
    NonlinearSolveState NonlinearSolveState::FromJson(const Json &value)
    {
        return value.get<NonlinearSolveState>();
    }
    struct NonlinearSolver::Data
    {
        // Declared first so it is destroyed last, after every stream, solver,
        // workspace and retained operation handle in this bound problem.
        std::optional<DomainExecutionContext> executionOwner;
        R::SessionAccess session;
        FiniteVolume discrete;
        std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction;
        std::shared_ptr<FiniteVolume::Workspace> workspace;
        R::ExecutionStream stream;
        ImplicitSettings settings;
        MPIInfo mpi;
        EquationDofOperations::Binding binding;
        EquationDofs spatial, defect, candidate;
        R::Array<real> inversePseudoTime;
        AuxiliaryValues auxiliary;
        std::optional<EquationDofOperations::Binding> auxiliaryState;
        std::vector<uint64_t> auxiliaryGhostVersions;
        real auxiliaryTime = 0;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        EquationAdmissibilityPolicy admissibility;
        std::shared_ptr<LinearProblemContext> problem;
        std::unique_ptr<DofLinearSolver> linear;
        bool positivity;
        AlphaReduction rhsFPPReduction = AlphaReduction::Cell;
        std::optional<R::Array<real>> cellRHSAlpha;
        real reconstructionTolerance;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        NonlinearSolveState nonlinearState;
        std::unique_ptr<NonlinearController> controller;

        Data(const FiniteVolume &fv, EquationDofs &state,
             std::shared_ptr<FiniteVolume::ReconstructionState> reconstructionState,
             std::shared_ptr<FiniteVolume::Workspace> work,
             R::EagerEngine &executor, R::ExecutionStream lane, const DomainSettings &controls, ImplicitSettings implicit,
             std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
            : discrete(fv), reconstruction(std::move(reconstructionState)), workspace(std::move(work)), stream(std::move(lane)), settings(std::move(implicit)),
              mpi(fv.Metrics().Communicator()), binding(state), spatial(EquationDofOperations::NewVector(state)),
              defect(EquationDofOperations::NewVector(state)), candidate(EquationDofOperations::NewVector(state)),
              inversePseudoTime(R::RowLayout::Dense(fv.Metrics().NumCell(), 1)), workers(std::move(pool)),
              admissibility(fv.PrepareAdmissibility(workers)),
              problem(fv.BuildLinearProblem(state, settings.diagonal == "block" ? DiagonalStorage::Block : DiagonalStorage::Entries,
                                            (settings.jacobianAction == "directEvaluated" || settings.jacobianAction == "semiMatrixFree") ? OffDiagonalStorage::DirectEvaluated : OffDiagonalStorage::Assembled)),
              positivity(controls.numerics.positivity), reconstructionTolerance(settings.requireConvergedReconstruction && controls.numerics.reconstruction == "variational" ? controls.numerics.reconstructionTolerance : 0)
        {
            controller = std::make_unique<NonlinearController>(
                settings, state.Schema(), mpi.rank, "singleStageImplicit",
                UpdatePolicyRequirements{
                    1, NonlinearDetail::ResidualObservation::BeforeUpdate,
                    false});
            AuxiliaryRequest request;
            // This solve always needs face spectra. Prepare each mean's
            // p/a/diffusion bound once, then consume it across incident faces;
            // the buffer is an explicit auxiliary output, not hidden state.
            request.meanThermodynamics = true;
            request.faceSpectralRadius = request.cellSpectralSum = true;
            auxiliary = fv.BuildAuxiliaryValues(fv.CompleteRHSAuxiliaryRequest(request));
            DNDS_check_throw_info(executor.Owns(stream), "nonlinear solver stream belongs to another runtime");
            rhsFPPReduction = settings.positivity.rhsFPPReduction == "cell" ? AlphaReduction::Cell : settings.positivity.rhsFPPReduction == "minimumLimited" ? AlphaReduction::MinimumLimited
                                                                                                                                                             : AlphaReduction::MinimumAll;
            if (positivity && settings.positivity.rhsFPPMode != "none" && rhsFPPReduction != AlphaReduction::Cell)
                cellRHSAlpha.emplace(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
            index first = 0;
            for (const auto &block : state.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                if (block.name == "rans")
                    ransFirst = first;
                first += index(block.components.size());
            }
        }
        template <class F>
        void Stage(F &&run) const
        {
            RunLocalStage(mpi, settings.collectiveChecks,
                          std::forward<F>(run), "equation nonlinear local phase failed");
        }
        void PrepareLinear(EquationDofs &state, bool overlap)
        {
            linear = std::make_unique<DofLinearSolver>(problem, state, [&](auto &field)
                                                       { return discrete.BindHalo(field); }, NonlinearDetail::ResolveLinear(settings), LinearExecution{overlap, settings.collectiveChecks});
        }
        void EvaluateSpatial(EquationDofs &state, real time,
                             R::EagerEngine &fallbackEngine)
        {
            if (executionOwner)
                discrete.Execute(state, *reconstruction, *workspace,
                                 *executionOwner,
                                 FiniteVolumeEvaluation::Spatial(spatial, time,
                                                                 std::ref(auxiliary)));
            else
                discrete.Evaluate(state, spatial, *reconstruction, *workspace,
                                  auxiliary, fallbackEngine, stream, time);
        }
        void PublishAuxiliary(const EquationDofs &state, real time)
        {
            auxiliaryState.emplace(state.NumericAlias());
            auxiliaryGhostVersions.clear();
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                auxiliaryGhostVersions.push_back(state.PhysicalField(iField).Ghost().Version());
            auxiliaryTime = time;
        }
        bool AuxiliaryCurrent(const EquationDofs &state, real time) const
        {
            if (!auxiliaryState || auxiliaryTime != time)
                return false;
            const auto read = state.ReadWithGhost();
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                if (auxiliaryState->versions[iField] != state.PhysicalField(iField).Owned().Version() ||
                    auxiliaryGhostVersions[iField] != state.PhysicalField(iField).Ghost().Version())
                    return false;
            return true;
        }
        void ValidateMeans(const EquationDofs &state) const
        {
            const auto read = state.ReadOwned();
            const EquationViewLayout<const real> layout(state, read);
            const auto bounds = admissibility.View();
            RequireNumericalSuccess(CPU::ValidateImplicitMeans(layout.Descriptor(), bounds), "ValidateImplicitMeans");
        }
    };

    NonlinearSolver::NonlinearSolver(const FiniteVolume &discrete, EquationDofs &state,
                                     std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction,
                                     std::shared_ptr<FiniteVolume::Workspace> workspace, R::EagerEngine &engine, R::ExecutionStream stream,
                                     const DomainSettings &settings, const Algebra::ComponentSelection &residualSelection,
                                     std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
    {
        const MPIInfo mpi(discrete.Metrics().Communicator());
        ImplicitSettings implicit = settings.implicit;
        std::string serialized;
        MPI::LocalStage(mpi, [&]
                        {
            implicit.residualComponents.clear();
            index ordinal = 0;
            for (const auto &block : state.Schema().blocks)
                for (const auto &component : block.components)
                {
                    if (!residualSelection.All() && residualSelection.Includes(ordinal))
                        implicit.residualComponents.push_back(block.name + "." + component.name);
                    ++ordinal;
                }
            DNDS_check_throw_info(residualSelection.All() || residualSelection.Indices().back() < state.ComponentNumber(), "equation residual selection exceeds its dimension");
            implicit.ValidateComponents(state.ComponentNumber());
            settings.numerics.ValidatePositivity();
            DNDS_check_throw_info(reconstruction && workspace && (!implicit.roeDissipation || discrete.HasRoeJacobianKernel(*workspace)),
                                  "equation nonlinear Roe profile is not connected yet");
            DNDS_check_throw_info(!implicit.requireConvergedReconstruction || settings.numerics.reconstruction != "variational" || settings.numerics.reconstructionTolerance > 0,
                                  "strict implicit VR acceptance requires a positive reconstructionTolerance");
            DNDS_check_throw_info(reconstruction && workspace && stream.Context().device == R::Device::Host(), "equation implicit stage needs reconstruction state, CPU workspace and stream");
            stream.Record().WaitHost();
            discrete.Equation().ValidateStateSchema(state);
            discrete.ValidateWorkspace(state, *workspace);
            DNDS_check_throw_info(state.Space().Same(discrete.Metrics().Space()) && discrete.Equation().HasMixture() == bool(workers), "equation nonlinear space/worker backend differs");
            serialized = Json{{"implicit", implicit}, {"numerics", settings.numerics}, {"execution", settings.execution}}.dump();
            DNDS_check_throw_info(serialized.size() <= size_t(INT_MAX), "equation implicit controls exceed MPI string range"); });
        AgreeEquationBinding(state, mpi.comm);
        NonlinearDetail::AgreeControls(mpi, serialized);
        MPI::LocalStage(mpi, [&]
                        { _data = std::make_shared<Data>(discrete, state, std::move(reconstruction), std::move(workspace), engine, std::move(stream), settings, std::move(implicit), std::move(workers)); });
        MPI::LocalStage(mpi, [&]
                        { _data->ValidateMeans(state); });
        _data->PrepareLinear(state, settings.execution.overlap);
    }

    NonlinearSolver::NonlinearSolver(
        const FiniteVolume &discrete, EquationDofs &state,
        std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction,
        std::shared_ptr<FiniteVolume::Workspace> workspace,
        DomainExecutionContext context, const DomainSettings &settings,
        const Algebra::ComponentSelection &residualSelection)
        : NonlinearSolver(discrete, state, std::move(reconstruction),
                          std::move(workspace), context.Engine(),
                          context.PrimaryStream(), settings, residualSelection,
                          context.ChemicalWorkers())
    {
        _data->executionOwner.emplace(std::move(context));
    }

    NonlinearIterationReport NonlinearSolver::Solve(EquationDofs &state, const EquationDofs &source,
                                                    real mass, real rhsScale, real time, int stage, NonlinearSolveControl control)
    {
        return Execute(state, source, mass, rhsScale, time, stage,
                       std::move(control), false, {});
    }
    NonlinearIterationReport NonlinearSolver::SolveWithController(
        const NonlinearController &controller, EquationDofs &state,
        const EquationDofs &source, real mass, real rhsScale, real time,
        int stage, NonlinearSolveControl control)
    {
        return Execute(state, source, mass, rhsScale, time, stage,
                       std::move(control), false, {}, std::cref(controller));
    }
    NonlinearIterationReport NonlinearSolver::CorrectOnce(EquationDofs &state, const EquationDofs &source,
                                                          real mass, real rhsScale, real time, int stage,
                                                          NonlinearSolveControl control,
                                                          OptionalRef<const EquationDofs> cachedRate)
    {
        control.correctionLimit = 1;
        return Execute(state, source, mass, rhsScale, time, stage, std::move(control), true, cachedRate);
    }
    void NonlinearSolver::EvaluateRate(EquationDofs &state, EquationDofs &rate, real time)
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        auto session = _data->session.Acquire(_nestedSession.lock());
        auto &data = *_data;
        auto engine = data.stream.LockEngine();
        data.Stage([&]
                   {
            data.binding.Check(state);
            EquationDofOperations::CheckCompatible(state, rate);
            for (const auto *input : {&state, &data.spatial})
                EquationDofOperations::RequireDistinct(*input, rate);
            auto write = rate.WriteOwned(); write.CancelUnmodified(); });
        data.EvaluateSpatial(state, time, engine);
        data.PublishAuxiliary(state, time);
        data.Stage([&]
                   {
            const auto geometry = data.discrete.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            const auto read = data.spatial.ReadOwned();
            auto write = rate.WriteOwned();
            const auto from = read.ComponentLayout();
            const auto to = write.ComponentLayout();
            const auto input = from.View();
            const auto output = to.View();
            CPU::NormalizeRateByVolume(input, output, metric, input.ComponentNumber()); });
    }
    AuxiliaryValues NonlinearSolver::PreparedAuxValues() const
    {
        DNDS_check_throw_info(_data->auxiliaryState, "equation auxiliary values have not been prepared");
        return _data->auxiliary;
    }
    NonlinearIterationReport NonlinearSolver::Execute(EquationDofs &state, const EquationDofs &source,
                                                      real mass, real rhsScale, real time, int stage,
                                                      NonlinearSolveControl control, bool singleCorrection,
                                                      OptionalRef<const EquationDofs> cachedRate,
                                                      OptionalRef<const NonlinearController> controller)
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        auto session = _data->session.Acquire(_nestedSession.lock());
        auto &data = *_data;
        auto engine = data.stream.LockEngine();
        if (control.profile)
            *control.profile = {};
        auto *profile = control.profile.Get();
        const NonlinearDetail::PhaseTimer total(profile, NonlinearProfile::Total);
        const index nv = data.binding.snapshot.ComponentNumber();
        Algebra::ComponentSelection selection(nv);
        {
            const NonlinearDetail::PhaseTimer entry(profile, NonlinearProfile::Entry);
            std::string serialized;
            data.Stage([&]
                       {
                data.stream.Record().WaitHost();
                data.binding.Check(state);
                if (cachedRate)
                {
                    state.RequireCompatible(*cachedRate);
                    EquationDofOperations::RequireDistinct(data.spatial, *cachedRate);
                }
                data.ValidateMeans(state);
                EquationDofOperations::CheckCompatible(state, source);
                EquationDofOperations::RequireDistinct(state, source);
                selection = Algebra::ComponentSelection(nv, control.updateComponents);
                DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time) &&
                    control.correctionLimit >= -1 && control.correctionLimit <= data.settings.iterations &&
                    (!singleCorrection || !control.resume) &&
                    (control.initialPseudoCFL == 0 || (std::isfinite(control.initialPseudoCFL) && control.initialPseudoCFL >= data.settings.pseudoCFL && control.initialPseudoCFL <= data.settings.maxPseudoCFL)),
                    "invalid equation implicit stage coefficients");
                if (data.settings.collectiveChecks)
                {
                serialized = Json{{"mass", mass}, {"rhsScale", rhsScale}, {"time", time}, {"limit", control.correctionLimit},
                    {"CFL", control.initialPseudoCFL}, {"selection", selection.Indices()},
                    {"singleCorrection", singleCorrection}, {"cachedRate", bool(cachedRate)}}.dump();
                DNDS_check_throw_info(serialized.size() <= size_t(INT_MAX), "equation solve controls exceed MPI string range");
                } });
            if (data.settings.collectiveChecks)
                NonlinearDetail::AgreeControls(data.mpi, serialized);
        }
        const auto activeComponents = R::MetadataBuffer<uint8_t>::Generate(nv, [&](index j)
                                                                           { return uint8_t(selection.Includes(j)); });
        NonlinearIterationReport initial;
        initial.residual.resize(size_t(nv));
        BindControllerState(control);
        auto geometry = data.discrete.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto evaluate = [&]
        {
            if (!cachedRate)
            {
                data.EvaluateSpatial(state, time, engine);
                data.PublishAuxiliary(state, time);
                return;
            }
            // A method/pMG operator may have evaluated this same state through
            // EvaluateEquation/EvaluateRate. Retain its pre-RHS auxiliary
            // publication; preparing it again would erase refined RHS signals.
            // Collective callers must make this cache choice consistently.
            if (!data.AuxiliaryCurrent(state, time))
            {
                data.discrete.PrepareCorrectionAux(state, *data.reconstruction, *data.workspace, data.auxiliary, engine, data.stream, time);
                data.PublishAuxiliary(state, time);
            }
            else
                data.discrete.RefreshCorrectionSource(state, *data.reconstruction,
                                                      *data.workspace, time);
            data.Stage([&]
                       {
                const auto read = cachedRate->ReadOwned(); auto write = data.spatial.WriteOwned();
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto rate = read.Block(iBlock).owned;
                    const auto output = write.Block(iBlock);
                    CPU::ScaleRateByVolume(rate, output, metric, rate.ComponentNumber());
                } });
        };
        const auto measure = [&](auto &norms)
        {
                std::vector<real> local(size_t(nv + 1));
                data.Stage([&]
                {
                    const auto u = state.ReadOwned(), r = data.spatial.ReadOwned(), b = source.ReadOwned();
                    auto write = data.defect.WriteOwned();
                    const EquationViewLayout<const real> ul(state, u), rl(data.spatial, r), bl(source, b);
                    const EquationViewLayout<real> wl(data.defect, write);
                    RequireNumericalSuccess(CPU::AssembleDefect(ul.Descriptor(), rl.Descriptor(), bl.Descriptor(), wl.Descriptor(),
                        metric, mass, rhsScale, activeComponents.HostView(), data.settings.normOrder, !singleCorrection,
                        {local.data(), index(local.size())}), "AssembleDefect");
                });
                if (singleCorrection) return; // Defect is ready; no stopping norm is needed.
                MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                    MPI::ContiguousView<real>{local.data(), size_t(nv + 1)}, data.mpi.comm);
                DNDS_check_throw_info(std::isfinite(local[nv]) && local[nv] > 0, "invalid global equation stage volume");
                for (index j = 0; j < nv; ++j)
                {
                    norms[j] = data.settings.normOrder == 1 ? local[j] : std::sqrt(local[j]);
                    DNDS_check_throw_info(std::isfinite(norms[j]), "nonfinite equation global stage norm");
                } };
        const auto reconstruction = [&]
        { return ReconstructionStats(); };
        const auto advance = [&](auto &report)
        {
                index rhsLimited = 0;
                real rhsAlpha = 1;
                {
                    const NonlinearDetail::PhaseTimer timer(profile, NonlinearProfile::PseudoTimePP);
                    // Aux is prepared before reconstruction/RHS. Never refresh
                    // it here: RHS may consume or refine the prepared signals.
                    data.Stage([&]
                    {
                        const auto read = data.auxiliary.cellSpectralSum->Owned().Read();
                        const auto meanRead = state.ReadOwned(); const auto meanLayout = meanRead.ComponentLayout(); const auto means = meanLayout.View();
                        auto residualWrite = data.defect.WriteOwned(); const auto residualLayout = residualWrite.ComponentLayout(); const auto defect = residualLayout.View();
                        auto write = data.inversePseudoTime.Write();
                        std::optional<R::Array<real>::WriteLease> alphaWrite;
                        if (data.cellRHSAlpha) alphaWrite.emplace(data.cellRHSAlpha->Write());
                        const auto alphas = alphaWrite ? alphaWrite->View<R::UniformTableRows>() : R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>{};
                        const auto spectral = read.View<R::UniformTableRows>();
                        const auto inverse = write.View<R::UniformTableRows>();
                        const auto bounds = data.admissibility.View().thermal;
                        const auto statistics = RequireNumericalSuccess(CPU::PreparePseudoTime({metric, means, defect, spectral, inverse,
                            alphaWrite ? std::optional(alphas) : std::nullopt, bounds, report.pseudoCFL,
                            data.settings.positivity.rhsFPPScale, data.settings.positivity.rhsFPPRelax,
                            data.settings.positivity.compressionPower,
                            data.positivity && data.settings.positivity.rhsFPPMode != "none",
                            data.settings.positivity.rhsFPPMode == "rhs"}), "PreparePseudoTime");
                        rhsLimited = statistics.limited; rhsAlpha = statistics.minimum;
                    });
                    if (data.positivity && data.settings.positivity.rhsFPPMode != "none")
                    {
                        const auto global = AgreeAlphaStatistics(rhsLimited, rhsAlpha, data.rhsFPPReduction, data.mpi.comm);
                        rhsLimited = global.limited; rhsAlpha = global.minimum;
                        if (data.cellRHSAlpha && rhsAlpha < 1) data.Stage([&]
                        {
                            const auto read = data.cellRHSAlpha->Read(); const auto local = read.View<R::UniformTableRows>();
                            auto inverseWrite = data.inversePseudoTime.Write(); const auto inverse = inverseWrite.View<R::UniformTableRows>();
                            auto residualWrite = data.defect.WriteOwned(); const auto layout = residualWrite.ComponentLayout(); const auto defect = layout.View();
                            CPU::ApplyAgreedRHSAlpha(local, inverse, defect, rhsAlpha, data.rhsFPPReduction,
                                data.settings.positivity.rhsFPPMode == "rhs");
                        });
                    }
                }
                {
                    const NonlinearDetail::PhaseTimer timer(profile, NonlinearProfile::Jacobian);
                    data.Stage([&] { data.discrete.PrepareJacobian(state, data.auxiliary, data.inversePseudoTime, *data.problem,
                        {mass, rhsScale, false, selection.Indices()}, time, data.workers,
                        FiniteVolume::JacobianContext{*data.reconstruction, *data.workspace}); });
                }
                {
                    const NonlinearDetail::PhaseTimer timer(profile, NonlinearProfile::LinearSolve);
                    const auto scales = NonlinearDetail::LinearScales(data.settings, state, data.discrete.Metrics());
                    report.lastLinear = data.linear->Solve(data.defect, {}, scales);
                    report.linearIterations += report.lastLinear.iterations;
                    DNDS_check_throw_info(report.lastLinear.usable, "equation implicit linear correction is unusable");
                }
                const NonlinearDetail::PhaseTimer timer(profile, NonlinearProfile::Correction);
                const auto result = ApplyCorrectionImpl(state, data.linear->Solution(), data.settings.relaxation, selection.Indices());
                if (data.positivity)
                {
                    report.rhsLimited += rhsLimited; report.minimumRHSAlpha = std::min(report.minimumRHSAlpha, rhsAlpha);
                    report.incrementLimited += result.first; report.minimumIncrementAlpha = std::min(report.minimumIncrementAlpha, result.second);
                } };
        if (singleCorrection)
        {
            auto report = std::move(initial);
            report.pseudoCFL = control.initialPseudoCFL == 0 ? data.settings.pseudoCFL : control.initialPseudoCFL;
            {
                const NonlinearDetail::PhaseTimer timer(profile, NonlinearProfile::Residual);
                evaluate();
                if (control.profile)
                    ++control.profile->residualEvaluations;
            }
            measure(report.residual);
            std::tie(report.reconstructionIterations, report.reconstructionRelativeUpdate) = reconstruction();
            report.reconstructionUpdateMeasured = std::isfinite(report.reconstructionRelativeUpdate);
            if (!report.reconstructionUpdateMeasured)
                report.reconstructionRelativeUpdate = 0;
            advance(report);
            report.iterations = 1;
            auto &stateSnapshot = data.nonlinearState;
            stateSnapshot.stage = stage;
            stateSnapshot.coupledStage = -1;
            stateSnapshot.iteration = 1;
            stateSnapshot.converged = false;
            stateSnapshot.terminal = true;
            stateSnapshot.capped = false;
            stateSnapshot.rejected = false;
            stateSnapshot.residual = report.residual;
            stateSnapshot.initialResidual = report.initialResidual;
            stateSnapshot.referenceResidual = report.referenceResidual;
            stateSnapshot.residualTargetRatio = report.residualTargetRatio;
            stateSnapshot.pseudoCFL = report.pseudoCFL;
            stateSnapshot.linearIterations = report.linearIterations;
            stateSnapshot.multigridCycles = report.multigridCycles;
            stateSnapshot.levelCorrections = report.levelCorrections;
            stateSnapshot.rhsLimited = report.rhsLimited;
            stateSnapshot.incrementLimited = report.incrementLimited;
            stateSnapshot.minimumRHSAlpha = report.minimumRHSAlpha;
            stateSnapshot.minimumIncrementAlpha = report.minimumIncrementAlpha;
            stateSnapshot.reconstructionIterations = report.reconstructionIterations;
            stateSnapshot.reconstructionRelativeUpdate = report.reconstructionRelativeUpdate;
            stateSnapshot.reconstructionUpdateMeasured = report.reconstructionUpdateMeasured;
            return report;
        }
        const auto &activeController = controller ? *controller
                                                  : *data.controller;
        return activeController.Execute(
            stage, {std::move(control), std::move(initial),
                    data.reconstructionTolerance, evaluate, measure, reconstruction,
                    [advance](int coupledStage, auto &report)
                    {
                        DNDS_check_throw_info(coupledStage == 0,
                                              "single-stage nonlinear policy requested another stage");
                        advance(report);
                    }});
    }
    std::pair<int, real> NonlinearSolver::ReconstructionStats() const
    {
        return {_data->discrete.LastReconstructionIterations(*_data->workspace), _data->discrete.LastReconstructionUpdate(*_data->workspace)};
    }
    const NonlinearSolveState &NonlinearSolver::State() const
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        return _data->nonlinearState;
    }
    void NonlinearSolver::RestoreState(const NonlinearSolveState &state)
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        const auto validated = NonlinearSolveState::FromJson(state.ToJson());
        DNDS_check_throw_info(validated.convergenceComponents == _data->settings.residualComponents,
                              "restored nonlinear convergence components differ from current solver");
        _data->nonlinearState = validated;
    }
    void NonlinearSolver::BindControllerState(NonlinearSolveControl &control)
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        if (!control.resume)
        {
            ++_data->nonlinearState.attempt;
            _data->nonlinearState.convergenceComponents = _data->settings.residualComponents;
        }
        else
            DNDS_check_throw_info(_data->nonlinearState.convergenceComponents == _data->settings.residualComponents,
                                  "restored nonlinear convergence components differ from current controls");
        control.state.Emplace(_data->nonlinearState);
    }
    bool NonlinearSolver::ReconstructionConverged() const
    {
        return _data->reconstructionTolerance <= 0 || ReconstructionStats().second <= _data->reconstructionTolerance;
    }
    real NonlinearSolver::ReconstructionTolerance() const
    {
        return _data->reconstructionTolerance;
    }
    std::pair<index, real> NonlinearSolver::ApplyCorrectionImpl(EquationDofs &state, const EquationDofs &increment, real scale,
                                                                const std::vector<index> &updateComponents)
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        auto &data = *_data;
        index limited = 0;
        real minimum = 1;
        bool frozen = true;
        data.Stage([&]
                   {
            data.binding.Check(state);
            EquationDofOperations::CheckCompatible(state, increment);
            EquationDofOperations::RequireDistinct(state, increment);
            DNDS_check_throw_info(std::isfinite(scale) && scale > 0 && scale <= 1, "invalid equation correction scale");
            const Algebra::ComponentSelection selection(state.ComponentNumber(), updateComponents);
            std::vector<uint8_t> active;
            if (!selection.All())
            {
                active.resize(size_t(state.ComponentNumber()));
                for (index j = 0; j < state.ComponentNumber(); ++j) active[j] = selection.Includes(j);
            }
            const auto u = state.ReadOwned(), d = increment.ReadOwned();
            auto write = data.candidate.WriteOwned();
            const auto ul = u.ComponentLayout(), dl = d.ComponentLayout(); const auto wl = write.ComponentLayout();
            const auto bounds = data.admissibility.View().thermal;
            const auto &rans = data.admissibility.RANS();
            const int independent = data.discrete.Equation().HasMixture() ? data.discrete.Equation().Mixture().SpeciesNumber() - 1 : 0;
            const R::MetadataBuffer<uint8_t> selectionMetadata(active);
            const auto result = RequireNumericalSuccess(CPU::ApplyCorrection({bounds, rans, selectionMetadata.HostView(),
                state.ComponentNumber(), data.flowFirst, data.speciesFirst, data.ransFirst, independent, scale,
                data.settings.positivity.incrementRelax, data.settings.positivity.compressionPower, data.positivity},
                ul.View(), dl.View(), wl.View()), "ApplyCorrection");
            limited = result.limited; minimum = result.minimum; frozen = result.frozen; });
        // Candidate checks precede all state writers. Scalar repair may not
        // silently unfreeze a masked component, even when density changed.
        const auto validateCandidate = [&]
        {
            DNDS_check_throw_info(frozen, "equation scalar repair would change a frozen component");
            data.ValidateMeans(data.candidate);
        };
        if (data.positivity)
        {
            // Carry acceptance with the already-required minimum-alpha
            // reduction. Every rank still validates before any state writer;
            // preserve the originating exception and reject on all peers.
            std::exception_ptr failure;
            real agreement[2]{1, minimum};
            try
            {
                validateCandidate();
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy nonlinear candidate validation failed");
                failure = std::current_exception();
                agreement[0] = 0;
            }
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<real>{agreement, 2}, data.mpi.comm);
            if (failure)
                std::rethrow_exception(failure);
            DNDS_check_throw_info(agreement[0] == 1, "equation correction candidate rejected on another rank");
            minimum = agreement[1];
        }
        else
            MPI::LocalStage(data.mpi, validateCandidate);
        data.Stage([&]
                   { EquationDofOperations::Copy(state, data.candidate); });
        if (data.positivity)
        {
            limited = MPI::AllreduceValue<MPI::Reduction::Sum>(limited, data.mpi.comm);
        }
        return {limited, minimum};
    }
    void NonlinearSolver::EvaluateEquation(EquationDofs &state, EquationDofs &value, real mass, real rhsScale, real time,
                                           OptionalRef<EquationDofs> rate)
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        auto session = _data->session.Acquire(_nestedSession.lock());
        auto &data = *_data;
        auto engine = data.stream.LockEngine();
        data.Stage([&]
                   {
            data.binding.Check(state);
            EquationDofOperations::CheckCompatible(state, value);
            EquationDofOperations::RequireDistinct(state, value);
            DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time), "invalid equation coefficients");
            auto write = value.WriteOwned(); write.CancelUnmodified();
            if (rate)
            {
                EquationDofOperations::CheckCompatible(state, *rate);
                for (const auto *input : {&state, &value, &data.spatial})
                    EquationDofOperations::RequireDistinct(*input, *rate);
                auto rateWrite = rate->WriteOwned(); rateWrite.CancelUnmodified();
            } });
        data.EvaluateSpatial(state, time, engine);
        data.PublishAuxiliary(state, time);
        data.Stage([&]
                   {
            const auto geometry = data.discrete.Metrics().AcquireRead(); const auto metric = geometry.HostView();
            const auto u = state.ReadOwned(), r = data.spatial.ReadOwned(); auto write = value.WriteOwned();
            const auto ul = u.ComponentLayout(), rl = r.ComponentLayout(); const auto wl = write.ComponentLayout();
            const auto mean = ul.View(), residual = rl.View(); const auto output = wl.View();
            std::optional<EquationDofs::WriteScope> rateWrite;
            std::optional<Algebra::DofComponentLayout<real>> rateLayout;
            if (rate) rateWrite.emplace(rate->WriteOwned());
            if (rateWrite) rateLayout.emplace(rateWrite->ComponentLayout());
            const auto spatialRate = rateLayout ? rateLayout->View() : Algebra::DofComponentView<real>{};
            if (rate)
                CPU::AssembleEquationAndRateRows(mean, residual, output, spatialRate, metric,
                                                    mean.ComponentNumber(), mass, rhsScale);
            else
                CPU::AssembleEquationRows(mean, residual, output, metric, mean.ComponentNumber(), mass, rhsScale); });
    }
    R::SessionAccess::Lease NonlinearSolver::ReserveForMultigrid()
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        auto lease = _data->session.Acquire();
        _nestedSession = lease;
        return lease;
    }
    std::pair<index, real> NonlinearSolver::ApplyCorrection(EquationDofs &state, const EquationDofs &increment, real scale, const std::vector<index> &components)
    {
        DNDS_check_throw_info(_data, "nonlinear session was moved");
        auto lease = _data->session.Acquire(_nestedSession.lock());
        return ApplyCorrectionImpl(state, increment, scale, components);
    }

}
