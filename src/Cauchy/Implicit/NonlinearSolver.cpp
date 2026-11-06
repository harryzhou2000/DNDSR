#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using NonlinearPhaseTimer = NonlinearDetail::PhaseTimer;
    struct NonlinearSolver::Data
    {
        FiniteVolume discrete;
        std::shared_ptr<FiniteVolume::Workspace> workspace;
        R::EagerEngine *engine;
        R::ExecutionStream stream;
        ImplicitSettings settings;
        bool positivity;
        real reconstructionTolerance;
        MPIInfo mpi;
        R::ArrayStorage::Shared stateOwned, stateGhost;
        CFV::FVField spatial, defect, candidate;
        R::Array<real> inversePseudoTime;
        AlphaReduction rhsFPPReduction = AlphaReduction::Cell;
        std::optional<R::Array<real>> cellRHSAlpha;
        std::shared_ptr<LinearProblemContext> problem;
        std::unique_ptr<LinearSolver> linear;
        Data(const FiniteVolume &fv, CFV::FVField &state, std::shared_ptr<FiniteVolume::Workspace> work,
             R::EagerEngine &executor, R::ExecutionStream lane, const DomainSettings &controls)
            : discrete(fv), workspace(std::move(work)), engine(&executor), stream(std::move(lane)), settings(controls.implicit),
              positivity(controls.numerics.positivity),
              reconstructionTolerance(controls.implicit.requireConvergedReconstruction && controls.numerics.reconstruction == "variational" ? controls.numerics.reconstructionTolerance : 0),
              mpi(fv.Metrics().Communicator()), stateOwned(state.Owned().SharedStorage()), stateGhost(state.Ghost().SharedStorage()),
              spatial(fv.Metrics().BuildField(state.Owned().Layout())), defect(fv.Metrics().BuildField(state.Owned().Layout())), candidate(fv.Metrics().BuildField(state.Owned().Layout())),
              inversePseudoTime(R::RowLayout::Dense(fv.Metrics().NumCell(), 1)),
              problem(fv.BuildLinearProblem(state, settings.diagonal == "block" ? DiagonalStorage::Block : DiagonalStorage::Entries,
                                            settings.jacobianAction == "semiMatrixFree" ? OffDiagonalStorage::SemiMatrixFree : OffDiagonalStorage::Assembled))
        {
            rhsFPPReduction = settings.positivity.rhsFPPReduction == "cell" ? AlphaReduction::Cell : settings.positivity.rhsFPPReduction == "minimumLimited" ? AlphaReduction::MinimumLimited
                                                                                                                                                             : AlphaReduction::MinimumAll;
            if (positivity && settings.positivity.rhsFPPMode != "none" && rhsFPPReduction != AlphaReduction::Cell)
                cellRHSAlpha.emplace(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
        }
        void PrepareLinear(const CFV::FVField &state, bool overlap)
        {
            linear = std::make_unique<LinearSolver>(problem, state, [&](auto &field)
                                                    { return discrete.BindHalo(field); }, NonlinearDetail::ResolveLinear(settings), LinearExecution{overlap, settings.collectiveChecks});
        }
        template <class F>
        void Stage(F &&run) const
        {
            if (settings.collectiveChecks)
                MPI::LocalStage(mpi, std::forward<F>(run));
            else
                run();
        }
    };
    NonlinearSolver::NonlinearSolver(const FiniteVolume &discrete, CFV::FVField &state, std::shared_ptr<FiniteVolume::Workspace> workspace,
                                     R::EagerEngine &engine, R::ExecutionStream stream, const DomainSettings &settings)
    {
        const MPIInfo mpi(discrete.Metrics().Communicator());
        std::string serialized;
        MPI::LocalStage(mpi, [&]
                        {
            settings.implicit.Validate(); discrete.Equation().Schema().ResolveComponents(settings.implicit.residualComponents); discrete.Metrics().ValidateField(state, 5);
            DNDS_check_throw_info(!settings.implicit.requireConvergedReconstruction || settings.numerics.reconstruction != "variational" || settings.numerics.reconstructionTolerance > 0,
                                  "strict implicit VR acceptance requires a positive reconstructionTolerance");
            DNDS_check_throw_info(workspace && stream.Context().device == R::Device::Host(), "implicit stage needs a CPU workspace and stream");
            AuxiliaryRequest required; required.faceSpectralRadius = required.cellSpectralSum = true;
            discrete.RequireWorkspaceChannels(*workspace, required);
            serialized = Json{{"implicit", settings.implicit}, {"positivity", settings.numerics.positivity}, {"overlap", settings.execution.overlap}}.dump();
            DNDS_check_throw_info(serialized.size() <= size_t(INT_MAX), "implicit controls exceed MPI string range"); });
        NonlinearDetail::AgreeControls(mpi, serialized);
        MPI::LocalStage(mpi, [&]
                        { _data = std::make_shared<Data>(discrete, state, std::move(workspace), engine, std::move(stream), settings); });
        _data->PrepareLinear(state, settings.execution.overlap);
    }
    NonlinearIterationReport NonlinearSolver::Solve(CFV::FVField &state, const CFV::FVField &source,
                                                    real mass, real rhsScale, real time, int stage, NonlinearSolveControl control)
    {
        return Execute(state, source, mass, rhsScale, time, stage, std::move(control), false, nullptr);
    }
    NonlinearIterationReport NonlinearSolver::CorrectOnce(CFV::FVField &state, const CFV::FVField &source,
                                                          real mass, real rhsScale, real time, int stage, NonlinearSolveControl control, const CFV::FVField *cachedRate)
    {
        control.correctionLimit = 1;
        return Execute(state, source, mass, rhsScale, time, stage, std::move(control), true, cachedRate);
    }
    NonlinearIterationReport NonlinearSolver::Execute(CFV::FVField &state, const CFV::FVField &source, real mass, real rhsScale, real time, int stage, NonlinearSolveControl control, bool singleCorrection, const CFV::FVField *cachedRate)
    {
        auto &data = *_data;
        if (control.profile)
            *control.profile = {};
        const NonlinearPhaseTimer totalTimer(control.profile, NonlinearProfile::Total);
        Algebra::ComponentSelection selection(5);
        std::array<bool, 5> active{};
        {
            const NonlinearPhaseTimer entryTimer(control.profile, NonlinearProfile::Entry);
            data.Stage([&]
                       {
            if (cachedRate)
            {
                data.discrete.Metrics().ValidateField(*cachedRate, 5);
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(cachedRate->Owned().Storage(), data.spatial.Owned().Storage()), "cached rate aliases residual scratch");
            }
            selection = Algebra::ComponentSelection(5, control.updateComponents);
            for (int j = 0; j < 5; ++j) active[j] = selection.Includes(j);
            data.discrete.Metrics().ValidateField(state, 5); data.discrete.Metrics().ValidateField(source, 5);
            DNDS_check_throw_info(state.Owned().SharedStorage() == data.stateOwned && state.Ghost().SharedStorage() == data.stateGhost &&
                                      !Algebra::Detail::Overlaps(state.Owned().Storage(), source.Owned().Storage()) &&
                                      !Algebra::Detail::Overlaps(state.Ghost().Storage(), source.Owned().Storage()) &&
                                      std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time) &&
                                      control.correctionLimit >= -1 && control.correctionLimit <= data.settings.iterations &&
                                      (control.initialPseudoCFL == 0 || (std::isfinite(control.initialPseudoCFL) && control.initialPseudoCFL >= data.settings.pseudoCFL && control.initialPseudoCFL <= data.settings.maxPseudoCFL)),
                                  "invalid implicit stage state/source/coefficients"); });
            if (data.settings.collectiveChecks)
            {
                int componentMask = 0;
                for (int j = 0; j < 5; ++j)
                    if (active[j])
                        componentMask |= 1 << j;
                const std::array<real, 6> coefficients{mass, rhsScale, time, control.initialPseudoCFL, real(componentMask),
                                                       real(singleCorrection) + 2 * real(bool(cachedRate))};
                auto minimum = coefficients, maximum = coefficients;
                MPI_Allreduce(MPI_IN_PLACE, minimum.data(), 6, MPI_DOUBLE, MPI_MIN, data.mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, maximum.data(), 6, MPI_DOUBLE, MPI_MAX, data.mpi.comm);
                index minLimit = control.correctionLimit, maxLimit = minLimit;
                MPI_Allreduce(MPI_IN_PLACE, &minLimit, 1, MPI_INT64_T, MPI_MIN, data.mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, &maxLimit, 1, MPI_INT64_T, MPI_MAX, data.mpi.comm);
                DNDS_check_throw_info(minimum == maximum && minLimit == maxLimit, "implicit stage coefficients differ between ranks");
            }
        }
        auto geometry = data.discrete.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto evaluate = [&]
        {
            if (!cachedRate)
            {
                data.discrete.Evaluate(state, data.spatial, *data.workspace, *data.engine, data.stream, time);
                return;
            }
            data.discrete.PrepareCorrectionAux(state, *data.workspace, *data.engine, data.stream, time);
            data.Stage([&]
                       {
                const auto read = cachedRate->Owned().Read(); auto write = data.spatial.Owned().Write();
                const auto rate = read.View<R::UniformTableRows>();
                const auto output = write.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < rate.RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j) output(iCell, j) = metric.GetCellVol(iCell) * rate(iCell, j); });
        };
        const auto measure = [&](auto &norms)
        {
                std::array<real, 6> norm{};
                data.Stage([&]
                           {
                const auto u = state.Owned().Read(), r = data.spatial.Owned().Read(), b = source.Owned().Read();
                auto write = data.defect.Owned().Write();
                const auto mean = u.View<R::UniformTableRows>(), residual = r.View<R::UniformTableRows>(), constant = b.View<R::UniformTableRows>();
                const auto output = write.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                {
                    const real volume = metric.GetCellVol(iCell); if (!singleCorrection) norm[5] += volume;
                    for (int j = 0; j < 5; ++j)
                    {
                        const real value = rhsScale * residual(iCell, j) + volume * (constant(iCell, j) - mass * mean(iCell, j));
                        DNDS_check_throw_info(std::isfinite(value), "nonfinite implicit stage defect");
                        output(iCell, j) = active[j] ? value : 0;
                        if (!singleCorrection) norm[j] += data.settings.normOrder == 1 ? std::abs(value) : value * value / volume;
                    }
                } });
                if (singleCorrection) return; // Defect is ready; no stopping norm is needed.
                MPI_Allreduce(MPI_IN_PLACE, norm.data(), 6, MPI_DOUBLE, MPI_SUM, data.mpi.comm);
                DNDS_check_throw_info(std::isfinite(norm[5]) && norm[5] > 0, "invalid global stage volume");
                for (int j = 0; j < 5; ++j)
                {
                    norms[j] = data.settings.normOrder == 1 ? norm[j] / norm[5] : std::sqrt(norm[j] / norm[5]);
                    DNDS_check_throw_info(std::isfinite(norms[j]), "nonfinite global stage norm");
                } };
        const auto reconstruction = [&]
        { return ReconstructionStats(); };
        const auto advance = [&](auto &report)
        {
            const auto &auxiliary = data.discrete.PreparedAuxValues(*data.workspace);
            index rhsLimited = 0;
            real rhsAlpha = 1;
            {
                const NonlinearPhaseTimer pseudoTimer(control.profile, NonlinearProfile::PseudoTimePP);
                data.Stage([&]
                           {
                const auto signals = auxiliary.cellSpectralSum->Owned().Read(), meanRead = state.Owned().Read();
                auto inverse = data.inversePseudoTime.Write(), residualWrite = data.defect.Owned().Write();
                std::optional<R::Array<real>::WriteLease> alphaWrite;
                if (data.cellRHSAlpha) alphaWrite.emplace(data.cellRHSAlpha->Write());
                const auto alphas = alphaWrite ? alphaWrite->View<R::UniformTableRows>() : R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>{};
                const auto spectral = signals.View<R::UniformTableRows>(), mean = meanRead.View<R::UniformTableRows>();
                const auto invTau = inverse.View<R::UniformTableRows>(), defect = residualWrite.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                {
                    const real volume = metric.GetCellVol(iCell);
                    real value = spectral(iCell, 0) / (report.pseudoCFL * volume);
                    DNDS_check_throw_info(std::isfinite(value) && value > 0, "invalid implicit pseudo-time spectral sum");
                    if (data.positivity && data.settings.positivity.rhsFPPMode != "none")
                    {
                        FlowState u, delta;
                        for (int j = 0; j < 5; ++j) { u[j] = mean(iCell, j); delta[j] = defect(iCell, j) * data.settings.positivity.rhsFPPScale / (value * volume); }
                        const real alpha = FlowIncrementFraction(u, delta, data.discrete.ThermalConstraints(), data.settings.positivity.rhsFPPRelax, 2, data.settings.positivity.compressionPower);
                        DNDS_check_throw_info(std::isfinite(alpha) && alpha >= 0 && alpha <= 1, "invalid implicit RHS PP fraction");
                        if (alpha < 1) { ++rhsLimited; rhsAlpha = std::min(rhsAlpha, alpha); }
                        if (alphaWrite)
                            alphas(iCell, 0) = alpha;
                        else if (data.settings.positivity.rhsFPPMode == "rhs")
                            for (int j = 0; j < 5; ++j) defect(iCell, j) *= alpha;
                        else value /= std::max(alpha, smallReal);
                    }
                    invTau(iCell, 0) = value;
                } });
                if (data.positivity && data.settings.positivity.rhsFPPMode != "none")
                {
                    const auto global = AgreeAlphaStatistics(rhsLimited, rhsAlpha, data.rhsFPPReduction, data.mpi.comm);
                    rhsLimited = global.limited;
                    rhsAlpha = global.minimum;
                    if (data.cellRHSAlpha && rhsAlpha < 1)
                        data.Stage([&]
                                   {
                    auto read = data.cellRHSAlpha->Read(); auto inverse = data.inversePseudoTime.Write(), residual = data.defect.Owned().Write();
                    const auto local = read.View<R::UniformTableRows>();
                    const auto invTau = inverse.View<R::UniformTableRows>(), defect = residual.View<R::UniformTableRows>();
                    for (index iCell = 0; iCell < local.RowCount(); ++iCell)
                    {
                        const real alpha = SelectAlpha(local(iCell, 0), rhsAlpha, data.rhsFPPReduction);
                        if (data.settings.positivity.rhsFPPMode == "rhs")
                            for (int j = 0; j < 5; ++j) defect(iCell, j) *= alpha;
                        else invTau(iCell, 0) /= std::max(alpha, smallReal);
                    } });
                }
            }
            {
                const NonlinearPhaseTimer jacobianTimer(control.profile, NonlinearProfile::Jacobian);
                data.Stage([&]
                           { data.discrete.PrepareJacobian(state, auxiliary, data.inversePseudoTime, *data.problem, {mass, rhsScale, data.settings.roeDissipation, selection.Indices()}, time); });
            }
            {
                const NonlinearPhaseTimer linearTimer(control.profile, NonlinearProfile::LinearSolve);
                report.lastLinear = data.linear->Solve(data.defect);
                report.linearIterations += report.lastLinear.iterations;
                DNDS_check_throw_info(report.lastLinear.usable, "implicit linear correction is unusable");
            }
            const auto [incrementLimited, incrementAlpha] = [&]
            {
                const NonlinearPhaseTimer correctionTimer(control.profile, NonlinearProfile::Correction);
                return ApplyCorrection(state, data.linear->Solution(), data.settings.relaxation, selection.Indices());
            }();
            if (data.positivity)
            {
                report.rhsLimited += rhsLimited;
                report.incrementLimited += incrementLimited;
                report.minimumRHSAlpha = std::min(report.minimumRHSAlpha, rhsAlpha);
                report.minimumIncrementAlpha = std::min(report.minimumIncrementAlpha, incrementAlpha);
            } };
        if (singleCorrection)
        {
            auto report = NonlinearIterationReport{};
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
        return NonlinearDetail::Iterate(data.settings, data.discrete.Equation().Schema(), data.mpi.rank, stage, control, NonlinearIterationReport{}, data.reconstructionTolerance, evaluate, measure, reconstruction, advance);
    }
    void NonlinearSolver::EvaluateEquation(CFV::FVField &state, CFV::FVField &value, real mass, real rhsScale, real time, CFV::FVField *rate)
    {
        auto &data = *_data;
        data.Stage([&]
                   {
            data.discrete.Metrics().ValidateField(value, 5);
            DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time), "invalid FAS equation coefficients");
            for (const auto *input : {&state.Owned(), &state.Ghost()})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(value.Owned().Storage(), input->Storage()), "FAS equation output aliases state");
            auto check = value.Owned().Write(); check.CancelUnmodified();
            if (rate)
            {
                data.discrete.Metrics().ValidateField(*rate, 5);
                for (const auto *input : {&state.Owned(), &state.Ghost(), &value.Owned(), &data.spatial.Owned()})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(rate->Owned().Storage(), input->Storage()), "FAS spatial rate aliases operator input/output");
                auto rateCheck = rate->Owned().Write(); rateCheck.CancelUnmodified();
            } });
        data.discrete.Evaluate(state, data.spatial, *data.workspace, *data.engine, data.stream, time);
        data.Stage([&]
                   {
            const auto geometry = data.discrete.Metrics().AcquireRead(); const auto metric = geometry.HostView();
            const auto meanRead = state.Owned().Read(), spatialRead = data.spatial.Owned().Read(); auto write = value.Owned().Write();
            const auto u = meanRead.View<R::UniformTableRows>(), r = spatialRead.View<R::UniformTableRows>(); const auto output = write.View<R::UniformTableRows>();
            std::optional<R::Array<real>::WriteLease> rateWrite;
            if (rate) rateWrite.emplace(rate->Owned().Write());
            const auto spatialRate = rateWrite ? rateWrite->View<R::UniformTableRows>() : R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>{};
            for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    const real rateValue = r(iCell, j) / metric.GetCellVol(iCell);
                    output(iCell, j) = mass * u(iCell, j) - rhsScale * r(iCell, j) / metric.GetCellVol(iCell);
                    if (rate) spatialRate(iCell, j) = rateValue;
                } });
    }
    std::pair<int, real> NonlinearSolver::ReconstructionStats() const
    {
        return {_data->discrete.LastReconstructionIterations(*_data->workspace), _data->discrete.LastReconstructionUpdate(*_data->workspace)};
    }
    bool NonlinearSolver::ReconstructionConverged() const
    {
        return _data->reconstructionTolerance <= 0 || ReconstructionStats().second <= _data->reconstructionTolerance;
    }
    real NonlinearSolver::ReconstructionTolerance() const
    {
        return _data->reconstructionTolerance;
    }
    std::pair<index, real> NonlinearSolver::ApplyCorrection(CFV::FVField &state, const CFV::FVField &increment, real scale,
                                                            const std::vector<index> &updateComponents)
    {
        auto &data = *_data;
        std::array<bool, 5> active{};
        data.Stage([&]
                   {
            const Algebra::ComponentSelection selection(5, updateComponents);
            for (int j = 0; j < 5; ++j) active[j] = selection.Includes(j);
            data.discrete.Metrics().ValidateField(state, 5); data.discrete.Metrics().ValidateField(increment, 5);
            DNDS_check_throw_info(state.Owned().SharedStorage() == data.stateOwned && state.Ghost().SharedStorage() == data.stateGhost &&
                                      std::isfinite(scale) && scale > 0 && scale <= 1, "invalid bound correction state or scale");
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(state.Owned().Storage(), increment.Owned().Storage()), "implicit correction aliases state"); });
        index limited = 0;
        real minimum = 1;
        bool admissible = true;
        data.Stage([&]
                   {
            const auto correctionRead = increment.Owned().Read(), meanRead = state.Owned().Read(); auto candidateWrite = data.candidate.Owned().Write();
            const auto correction = correctionRead.View<R::UniformTableRows>(), mean = meanRead.View<R::UniformTableRows>(); const auto candidate = candidateWrite.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
            {
                FlowState u, delta;
                for (int j = 0; j < 5; ++j) { u[j] = mean(iCell, j); delta[j] = active[j] ? scale * correction(iCell, j) : 0; }
                if (data.positivity)
                {
                    const real alpha = FlowIncrementFraction(u, delta, data.discrete.ThermalConstraints(), data.settings.positivity.incrementRelax, 2, data.settings.positivity.compressionPower);
                    admissible = admissible && std::isfinite(alpha) && alpha >= 0 && alpha <= 1;
                    if (alpha < 1) { ++limited; minimum = std::min(minimum, alpha); }
                    for (auto &component : delta) component *= alpha;
                    delta = FixFlowIncrement(u, delta, data.discrete.ThermalConstraints());
                }
                for (int j = 0; j < 5; ++j)
                    if (active[j]) u[j] += delta[j];
                    else admissible = admissible && delta[j] == 0;
                admissible = admissible && ThermallyAdmissible(u, data.discrete.ThermalConstraints());
                for (int j = 0; j < 5; ++j) candidate(iCell, j) = u[j];
            } });
        // Do not acquire the trial write lease until every rank accepts the
        // candidate. A rejected correction must leave the bound state usable.
        MPI::LocalStage(data.mpi, [&]
                        { DNDS_check_throw_info(admissible, "implicit correction produces an inadmissible mean"); });
        data.Stage([&]
                   { Algebra::Copy(state, data.candidate); });
        if (data.positivity)
        {
            MPI_Allreduce(MPI_IN_PLACE, &limited, 1, MPI_INT64_T, MPI_SUM, data.mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, data.mpi.comm);
        }
        return {limited, minimum};
    }
}
