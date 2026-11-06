#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct EquationNonlinearSolver::Data
    {
        FiniteVolume discrete;
        std::shared_ptr<FiniteVolume::EquationWorkspace> workspace;
        R::EagerEngine *engine;
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
        std::shared_ptr<LinearProblemContext> problem;
        std::unique_ptr<EquationLinearSolver> linear;
        bool positivity;
        std::optional<EquationThermalPolicy> thermal;
        std::optional<RANSStateView> rans;
        AlphaReduction rhsFPPReduction = AlphaReduction::Cell;
        std::optional<R::Array<real>> cellRHSAlpha;
        real reconstructionTolerance, densityFloor, temperatureFloor, pressureFloor;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;

        Data(const FiniteVolume &fv, EquationDofs &state, std::shared_ptr<FiniteVolume::EquationWorkspace> work,
             R::EagerEngine &executor, R::ExecutionStream lane, const DomainSettings &controls, ImplicitSettings implicit,
             std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
            : discrete(fv), workspace(std::move(work)), engine(&executor), stream(std::move(lane)), settings(std::move(implicit)),
              mpi(fv.Metrics().Communicator()), binding(state), spatial(EquationDofOperations::NewVector(state)),
              defect(EquationDofOperations::NewVector(state)), candidate(EquationDofOperations::NewVector(state)),
              inversePseudoTime(R::RowLayout::Dense(fv.Metrics().NumCell(), 1)), workers(std::move(pool)),
              problem(fv.BuildLinearProblem(state, settings.diagonal == "block" ? DiagonalStorage::Block : DiagonalStorage::Entries,
                                            settings.jacobianAction == "semiMatrixFree" ? OffDiagonalStorage::SemiMatrixFree : OffDiagonalStorage::Assembled)),
              positivity(controls.numerics.positivity), reconstructionTolerance(settings.requireConvergedReconstruction && controls.numerics.reconstruction == "variational" ? controls.numerics.reconstructionTolerance : 0),
              densityFloor(controls.numerics.densityFloor), temperatureFloor(controls.numerics.temperatureFloor), pressureFloor(controls.numerics.pressureFloor)
        {
            AuxiliaryRequest request;
            // This solve always needs face spectra. Prepare each mean's
            // p/a/diffusion bound once, then consume it across incident faces;
            // the buffer is an explicit auxiliary output, not hidden state.
            request.meanThermodynamics = true;
            request.faceSpectralRadius = request.cellSpectralSum = true;
            auxiliary = fv.BuildAuxiliaryValues(request);
            if (positivity)
                thermal.emplace(fv.Equation(), densityFloor, temperatureFloor, controls.numerics.sensibleEnergyFloor, pressureFloor,
                                workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
            if (fv.Equation().HasRANS())
                rans.emplace(ResolveRANSStatePolicy(fv.Equation(), controls.numerics.ransState));
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
            if (settings.collectiveChecks)
                MPI::LocalStage(mpi, std::forward<F>(run));
            else
                run();
        }
        void PrepareLinear(EquationDofs &state, bool overlap)
        {
            linear = std::make_unique<EquationLinearSolver>(problem, state, [&](auto &field)
                                                            { return discrete.BindHalo(field); }, NonlinearDetail::ResolveLinear(settings), LinearExecution{overlap, settings.collectiveChecks});
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
            const auto &equation = discrete.Equation();
            auto evaluator = equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto read = state.ReadOwned();
            const auto scalars = rans ? read.Block("rans") : EquationBlockReadView{};
            const auto layout = read.ComponentLayout();
            const auto values = layout.View();
            const int independent = equation.HasMixture() ? equation.Mixture().SpeciesNumber() - 1 : 0;
            auto rhoY = evaluator.SpeciesInput();
            const auto bounds = thermal ? thermal->View() : EquationThermalView{};
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                if (thermal)
                    DNDS_check_throw_info(bounds.Admissible(EquationPositivityDetail::PointRow<decltype(values)>{values, iCell}), "implicit equation mean violates its thermal/species bounds");
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    DNDS_check_throw_info(std::isfinite(values(iCell, j)), "nonfinite implicit equation mean");
                if (rans)
                    DNDS_check_throw_info(rans->Admissible(scalars[iCell]), "implicit RANS mean violates its scalar bounds");
                FlowState flow;
                for (int j = 0; j < 5; ++j)
                    flow[j] = values(iCell, flowFirst + j);
                DNDS_check_throw_info(flow[0] >= densityFloor, "implicit equation density is below its floor");
                real sum = 0;
                for (int k = 0; k < independent; ++k)
                {
                    rhoY[k] = values(iCell, speciesFirst + k);
                    DNDS_check_throw_info(rhoY[k] >= 0, "negative implicit equation species mean");
                    sum += rhoY[k];
                }
                DNDS_check_throw_info(sum <= flow[0], "implicit equation species means exceed density");
                if (!equation.HasMixture())
                    DNDS_check_throw_info(ThermallyAdmissible(flow, discrete.ThermalConstraints()), "inadmissible implicit flow mean");
                const auto point = evaluator.Evaluate(flow, {independent ? rhoY.data : nullptr, independent});
                DNDS_check_throw_info(point.thermo.temperature >= temperatureFloor && point.thermo.pressure > 0 && (equation.HasMixture() || point.thermo.pressure >= pressureFloor),
                                      "implicit equation mean violates temperature or pressure floor");
            }
        }
    };

    EquationNonlinearSolver::EquationNonlinearSolver(const FiniteVolume &discrete, EquationDofs &state,
                                                     std::shared_ptr<FiniteVolume::EquationWorkspace> workspace, R::EagerEngine &engine, R::ExecutionStream stream,
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
            DNDS_check_throw_info(!implicit.roeDissipation,
                                  "equation nonlinear Roe profile is not connected yet");
            DNDS_check_throw_info(!implicit.requireConvergedReconstruction || settings.numerics.reconstruction != "variational" || settings.numerics.reconstructionTolerance > 0,
                                  "strict implicit VR acceptance requires a positive reconstructionTolerance");
            DNDS_check_throw_info(workspace && stream.Context().device == R::Device::Host(), "equation implicit stage needs a CPU workspace and stream");
            stream.Record().WaitHost();
            discrete.Equation().ValidateStateSchema(state);
            discrete.ValidateWorkspace(state, *workspace);
            DNDS_check_throw_info(state.Space().Same(discrete.Metrics().Space()) && discrete.Equation().HasMixture() == bool(workers), "equation nonlinear space/worker backend differs");
            serialized = Json{{"implicit", implicit}, {"numerics", settings.numerics}, {"execution", settings.execution}}.dump();
            DNDS_check_throw_info(serialized.size() <= size_t(INT_MAX), "equation implicit controls exceed MPI string range"); });
        AgreeEquationBinding(state, mpi.comm);
        NonlinearDetail::AgreeControls(mpi, serialized);
        MPI::LocalStage(mpi, [&]
                        { _data = std::make_shared<Data>(discrete, state, std::move(workspace), engine, std::move(stream), settings, std::move(implicit), std::move(workers)); });
        MPI::LocalStage(mpi, [&]
                        { _data->ValidateMeans(state); });
        _data->PrepareLinear(state, settings.execution.overlap);
    }

    EquationNonlinearIterationReport EquationNonlinearSolver::Solve(EquationDofs &state, const EquationDofs &source,
                                                                    real mass, real rhsScale, real time, int stage, NonlinearSolveControl control)
    {
        return Execute(state, source, mass, rhsScale, time, stage, std::move(control), false, nullptr);
    }
    EquationNonlinearIterationReport EquationNonlinearSolver::CorrectOnce(EquationDofs &state, const EquationDofs &source,
                                                                          real mass, real rhsScale, real time, int stage, NonlinearSolveControl control, const EquationDofs *cachedRate)
    {
        control.correctionLimit = 1;
        return Execute(state, source, mass, rhsScale, time, stage, std::move(control), true, cachedRate);
    }
    void EquationNonlinearSolver::EvaluateRate(EquationDofs &state, EquationDofs &rate, real time)
    {
        auto &data = *_data;
        data.Stage([&]
                   {
            data.binding.Check(state);
            EquationDofOperations::CheckCompatible(state, rate);
            for (const auto *input : {&state, &data.spatial})
                EquationDofOperations::RequireDistinct(*input, rate);
            auto write = rate.WriteOwned(); write.CancelUnmodified(); });
        data.discrete.Evaluate(state, data.spatial, *data.workspace, data.auxiliary, *data.engine, data.stream, time);
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
            for (index iCell = 0; iCell < input.RowCount(); ++iCell)
                for (index j = 0; j < input.ComponentNumber(); ++j)
                    output(iCell, j) = input(iCell, j) / metric.GetCellVol(iCell); });
    }
    const AuxiliaryValues &EquationNonlinearSolver::PreparedAuxValues() const
    {
        DNDS_check_throw_info(_data->auxiliaryState, "equation auxiliary values have not been prepared");
        return _data->auxiliary;
    }
    EquationNonlinearIterationReport EquationNonlinearSolver::Execute(EquationDofs &state, const EquationDofs &source,
                                                                      real mass, real rhsScale, real time, int stage, NonlinearSolveControl control, bool singleCorrection, const EquationDofs *cachedRate)
    {
        auto &data = *_data;
        if (control.profile)
            *control.profile = {};
        const NonlinearDetail::PhaseTimer total(control.profile, NonlinearProfile::Total);
        const index nv = data.binding.snapshot.ComponentNumber();
        Algebra::ComponentSelection selection(nv);
        {
            const NonlinearDetail::PhaseTimer entry(control.profile, NonlinearProfile::Entry);
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
        EquationNonlinearIterationReport initial;
        initial.residual.resize(size_t(nv));
        auto geometry = data.discrete.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto evaluate = [&]
        {
            if (!cachedRate)
            {
                data.discrete.Evaluate(state, data.spatial, *data.workspace, data.auxiliary, *data.engine, data.stream, time);
                data.PublishAuxiliary(state, time);
                return;
            }
            // A method/pMG operator may have evaluated this same state through
            // EvaluateEquation/EvaluateRate. Retain its pre-RHS auxiliary
            // publication; preparing it again would erase refined RHS signals.
            // Collective callers must make this cache choice consistently.
            if (!data.AuxiliaryCurrent(state, time))
            {
                data.discrete.PrepareCorrectionAux(state, *data.workspace, data.auxiliary, *data.engine, data.stream, time);
                data.PublishAuxiliary(state, time);
            }
            data.Stage([&]
                       {
                const auto read = cachedRate->ReadOwned(); auto write = data.spatial.WriteOwned();
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto rate = read.Block(iBlock).owned;
                    const auto output = write.Block(iBlock);
                    for (index iCell = 0; iCell < rate.RowCount(); ++iCell)
                        for (index j = 0; j < rate.ComponentNumber(); ++j) output(iCell, j) = metric.GetCellVol(iCell) * rate(iCell, j);
                } });
        };
        const auto measure = [&](auto &norms)
        {
                std::vector<real> local(size_t(nv + 1));
                data.Stage([&]
                {
                    const auto u = state.ReadOwned(), r = data.spatial.ReadOwned(), b = source.ReadOwned();
                    auto write = data.defect.WriteOwned();
                    const auto ul = u.ComponentLayout(), rl = r.ComponentLayout(), bl = b.ComponentLayout();
                    const auto wl = write.ComponentLayout();
                    const auto mean = ul.View(), residual = rl.View(), constant = bl.View();
                    const auto output = wl.View();
                    for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                    {
                        const real volume = metric.GetCellVol(iCell); if (!singleCorrection) local[nv] += volume;
                        for (index j = 0; j < nv; ++j)
                        {
                            const real value = rhsScale * residual(iCell, j) + volume * (constant(iCell, j) - mass * mean(iCell, j));
                            DNDS_check_throw_info(std::isfinite(value), "nonfinite equation implicit defect");
                            output(iCell, j) = selection.Includes(j) ? value : 0;
                            if (!singleCorrection) local[j] += data.settings.normOrder == 1 ? std::abs(value) : value * value / volume;
                        }
                    }
                });
                if (singleCorrection) return; // Defect is ready; no stopping norm is needed.
                MPI_Allreduce(MPI_IN_PLACE, local.data(), int(nv + 1), MPI_DOUBLE, MPI_SUM, data.mpi.comm);
                DNDS_check_throw_info(std::isfinite(local[nv]) && local[nv] > 0, "invalid global equation stage volume");
                for (index j = 0; j < nv; ++j)
                {
                    norms[j] = data.settings.normOrder == 1 ? local[j] / local[nv] : std::sqrt(local[j] / local[nv]);
                    DNDS_check_throw_info(std::isfinite(norms[j]), "nonfinite equation global stage norm");
                } };
        const auto reconstruction = [&]
        { return ReconstructionStats(); };
        const auto advance = [&](auto &report)
        {
                index rhsLimited = 0;
                real rhsAlpha = 1;
                {
                    const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::PseudoTimePP);
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
                        const auto bounds = data.thermal ? data.thermal->View() : EquationThermalView{};
                        for (index iCell = 0; iCell < inverse.RowCount(); ++iCell)
                        {
                            const real volume = metric.GetCellVol(iCell);
                            real value = spectral(iCell, 0) / (report.pseudoCFL * volume);
                            DNDS_check_throw_info(std::isfinite(value) && value > 0, "invalid equation pseudo-time spectral sum");
                            if (data.positivity && data.settings.positivity.rhsFPPMode != "none")
                            {
                                const EquationPositivityDetail::PointRow<decltype(means)> mean{means, iCell};
                                struct TrialIncrement
                                {
                                    decltype(defect) values; index row; real factor;
                                    real operator[](index j) const { return values(row, j) * factor; }
                                } delta{defect, iCell, data.settings.positivity.rhsFPPScale / (value * volume)};
                                const real alpha = bounds.IncrementFraction(mean, delta, data.settings.positivity.rhsFPPRelax, 2, data.settings.positivity.compressionPower);
                                DNDS_check_throw_info(std::isfinite(alpha) && alpha >= 0 && alpha <= 1, "invalid equation RHS PP fraction");
                                if (alpha < 1) { ++rhsLimited; rhsAlpha = std::min(rhsAlpha, alpha); }
                                if (alphaWrite) alphas(iCell, 0) = alpha;
                                else if (data.settings.positivity.rhsFPPMode == "rhs") for (index j = 0; j < nv; ++j) defect(iCell, j) *= alpha;
                                else value /= std::max(alpha, smallReal);
                            }
                            inverse(iCell, 0) = value;
                        }
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
                            for (index iCell = 0; iCell < local.RowCount(); ++iCell)
                            {
                                const real alpha = SelectAlpha(local(iCell, 0), rhsAlpha, data.rhsFPPReduction);
                                if (data.settings.positivity.rhsFPPMode == "rhs") for (index j = 0; j < nv; ++j) defect(iCell, j) *= alpha;
                                else inverse(iCell, 0) /= std::max(alpha, smallReal);
                            }
                        });
                    }
                }
                {
                    const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::Jacobian);
                    data.Stage([&] { data.discrete.PrepareJacobian(state, data.auxiliary, data.inversePseudoTime, *data.problem,
                        {mass, rhsScale, false, selection.Indices()}, time, data.workers, data.workspace.get()); });
                }
                {
                    const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::LinearSolve);
                    report.lastLinear = data.linear->Solve(data.defect);
                    report.linearIterations += report.lastLinear.iterations;
                    DNDS_check_throw_info(report.lastLinear.usable, "equation implicit linear correction is unusable");
                }
                const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::Correction);
                const auto result = ApplyCorrection(state, data.linear->Solution(), data.settings.relaxation, selection.Indices());
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
                const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::Residual);
                evaluate();
                if (control.profile)
                    ++control.profile->residualEvaluations;
            }
            measure(report.residual);
            std::tie(report.reconstructionIterations, report.reconstructionRelativeUpdate) = reconstruction();
            advance(report);
            report.iterations = 1;
            return report;
        }
        return NonlinearDetail::Iterate(data.settings, state.Schema(), data.mpi.rank, stage, control, std::move(initial), data.reconstructionTolerance, evaluate, measure, reconstruction, advance);
    }
    std::pair<int, real> EquationNonlinearSolver::ReconstructionStats() const
    {
        return {_data->discrete.LastReconstructionIterations(*_data->workspace), _data->discrete.LastReconstructionUpdate(*_data->workspace)};
    }
    bool EquationNonlinearSolver::ReconstructionConverged() const
    {
        return _data->reconstructionTolerance <= 0 || ReconstructionStats().second <= _data->reconstructionTolerance;
    }
    real EquationNonlinearSolver::ReconstructionTolerance() const
    {
        return _data->reconstructionTolerance;
    }
    std::pair<index, real> EquationNonlinearSolver::ApplyCorrection(EquationDofs &state, const EquationDofs &increment, real scale,
                                                                    const std::vector<index> &updateComponents)
    {
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
            const auto bounds = data.thermal ? data.thermal->View() : EquationThermalView{};
            const int independent = data.discrete.Equation().HasMixture() ? data.discrete.Equation().Mixture().SpeciesNumber() - 1 : 0;
            const index components = state.ComponentNumber();
            const auto apply = [&](auto mean, auto delta, auto candidate)
            {
            for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
            {
                const EquationPositivityDetail::PointRow<decltype(mean)> old{mean, iCell};
                struct SelectedIncrement
                {
                    decltype(delta) values; index row; real scale; const uint8_t *active;
                    real operator[](index j) const { return !active || active[j] ? scale * values(row, j) : 0; }
                } change{delta, iCell, scale, active.empty() ? nullptr : active.data()};
                real fraction = 1;
                if (data.positivity)
                {
                    const real alpha = bounds.IncrementFraction(old, change, data.settings.positivity.incrementRelax, 2, data.settings.positivity.compressionPower);
                    DNDS_check_throw_info(std::isfinite(alpha) && alpha >= 0 && alpha <= 1, "invalid equation increment PP fraction");
                    if (alpha < 1) { ++limited; minimum = std::min(minimum, alpha); }
                    struct LimitedIncrement
                    {
                        const SelectedIncrement &values; real factor;
                        real operator[](index j) const { return factor * values[j]; }
                    } limitedChange{change, alpha};
                    const real fix = bounds.FixFraction(old, limitedChange);
                    DNDS_check_throw_info(std::isfinite(fix) && fix >= 0 && fix <= 1, "invalid equation final increment repair");
                    fraction = alpha * fix;
                }
                for (index j = 0; j < components; ++j)
                {
                    candidate(iCell, j) = selection.Includes(j) ? mean(iCell, j) + fraction * change[j] : mean(iCell, j);
                    // Exponential scalar repair must not hide an infinite or
                    // NaN linear correction behind a finite floor.
                    DNDS_check_throw_info(std::isfinite(candidate(iCell, j)), "nonfinite equation correction candidate");
                }
                if (data.rans)
                {
                    std::array<real, 2> oldRANS{}, changeRANS{}, nextRANS{};
                    for (int j = 0; j < data.rans->components; ++j)
                    {
                        oldRANS[j] = mean(iCell, data.ransFirst + j);
                        changeRANS[j] = fraction * change[data.ransFirst + j];
                    }
                    DNDS_check_throw_info(data.rans->Admissible(oldRANS), "RANS correction requires an admissible old mean");
                    // Preserve Euler's order: whole-vector thermal compression,
                    // then component-wise RANS repair of that scaled increment.
                    data.rans->RepairIncrement(oldRANS, changeRANS);
                    for (int j = 0; j < data.rans->components; ++j) nextRANS[j] = oldRANS[j] + changeRANS[j];
                    data.rans->RepairMean(candidate(iCell, data.flowFirst), nextRANS);
                    for (int j = 0; j < data.rans->components; ++j)
                    {
                        const index component = data.ransFirst + j;
                        candidate(iCell, component) = nextRANS[j];
                        if (!selection.Includes(component)) frozen = frozen && nextRANS[j] == oldRANS[j];
                    }
                }
                if (data.positivity && independent)
                {
                    struct SpeciesRow
                    {
                        decltype(candidate) values; index row, first;
                        real &operator[](index k) const { return values(row, first + k); }
                    } species{candidate, iCell, data.speciesFirst};
                    Physics::Chemistry::RepairSpeciesDensities(candidate(iCell, data.flowFirst), species, independent);
                    for (index j = 0; j < components; ++j)
                        if (!selection.Includes(j)) frozen = frozen && candidate(iCell, j) == mean(iCell, j);
                }
            }
            };
            // Resolve a complete ordered AoS parent once, while all original
            // payload/shape leases remain held. The same scalar PP/repair
            // kernel then indexes direct table rows. Split/permuted or
            // column-major states retain the logical-component path.
            const auto meanTable = Algebra::TryOwnedTable(ul.View()), deltaTable = Algebra::TryOwnedTable(dl.View());
            const auto candidateTable = Algebra::TryOwnedTable(wl.View());
            if (meanTable && deltaTable && candidateTable)
                apply(*meanTable, *deltaTable, *candidateTable);
            else
                apply(ul.View(), dl.View(), wl.View()); });
        // Candidate checks precede all state writers. Scalar repair may not
        // silently unfreeze a masked component, even when density changed.
        MPI::LocalStage(data.mpi, [&]
                        {
            DNDS_check_throw_info(frozen, "equation scalar repair would change a frozen component");
            data.ValidateMeans(data.candidate); });
        data.Stage([&]
                   { EquationDofOperations::Copy(state, data.candidate); });
        if (data.positivity)
        {
            MPI_Allreduce(MPI_IN_PLACE, &limited, 1, MPI_INT64_T, MPI_SUM, data.mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, data.mpi.comm);
        }
        return {limited, minimum};
    }
    void EquationNonlinearSolver::EvaluateEquation(EquationDofs &state, EquationDofs &value, real mass, real rhsScale, real time, EquationDofs *rate)
    {
        auto &data = *_data;
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
        data.discrete.Evaluate(state, data.spatial, *data.workspace, data.auxiliary, *data.engine, data.stream, time);
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
            for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                for (index j = 0; j < mean.ComponentNumber(); ++j)
                {
                    const real rateValue = residual(iCell, j) / metric.GetCellVol(iCell);
                    output(iCell, j) = mass * mean(iCell, j) - rhsScale * residual(iCell, j) / metric.GetCellVol(iCell);
                    if (rate) spatialRate(iCell, j) = rateValue;
                } });
    }
}
