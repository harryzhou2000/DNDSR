#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct NonlinearSolver::Data
    {
        FiniteVolume discrete;
        std::shared_ptr<FiniteVolume::Workspace> workspace;
        R::EagerEngine *engine;
        R::ExecutionStream stream;
        ImplicitSettings settings;
        bool positivity;
        MPIInfo mpi;
        R::ArrayStorage::Shared stateOwned, stateGhost;
        CFV::FVField spatial, defect, candidate;
        R::Array<real> inversePseudoTime;
        std::shared_ptr<LinearProblemContext> problem;
        std::unique_ptr<LinearSolver> linear;
        Data(const FiniteVolume &fv, CFV::FVField &state, std::shared_ptr<FiniteVolume::Workspace> work,
             R::EagerEngine &executor, R::ExecutionStream lane, const DomainSettings &controls)
            : discrete(fv), workspace(std::move(work)), engine(&executor), stream(std::move(lane)), settings(controls.implicit),
              positivity(controls.numerics.positivity), mpi(fv.Metrics().Communicator()), stateOwned(state.Owned().SharedStorage()), stateGhost(state.Ghost().SharedStorage()),
              spatial(fv.Metrics().BuildField(state.Owned().Layout())), defect(fv.Metrics().BuildField(state.Owned().Layout())), candidate(fv.Metrics().BuildField(state.Owned().Layout())),
              inversePseudoTime(R::RowLayout::Dense(fv.Metrics().NumCell(), 1)),
              problem(fv.BuildLinearProblem(state, settings.diagonal == "block" ? DiagonalStorage::Block : DiagonalStorage::Entries))
        {
        }
        void PrepareLinear(const CFV::FVField &state, bool overlap)
        {
            LinearSolveSettings resolved;
            resolved.method = settings.linear.method;
            resolved.preconditioner = settings.linear.preconditioner;
            resolved.maxIterations = settings.linear.iterations;
            resolved.subspace = settings.linear.subspace;
            resolved.relativeTolerance = settings.linear.relativeTolerance;
            resolved.absoluteTolerance = settings.linear.absoluteTolerance;
            resolved.sgsSweeps = settings.sgs.sweeps;
            resolved.exchangeBetweenSweeps = settings.sgs.exchangeBetweenSweeps;
            resolved.iluFill = settings.ilu.fill;
            resolved.iluCoreFirst = settings.ilu.coreFirst;
            linear = std::make_unique<LinearSolver>(problem, state, [&](auto &field)
                                                    { return discrete.Metrics().BindHalo(field); }, resolved, LinearExecution{overlap, settings.collectiveChecks});
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
        std::string serialized, reference;
        MPI::LocalStage(mpi, [&]
                        {
            settings.implicit.Validate(); discrete.Metrics().ValidateField(state, 5);
            DNDS_check_throw_info(workspace && stream.Context().device == R::Device::Host(), "implicit stage needs a CPU workspace and stream");
            AuxiliaryRequest required; required.faceSpectralRadius = required.cellSpectralSum = true;
            discrete.RequireWorkspaceChannels(*workspace, required);
            serialized = Json{{"implicit", settings.implicit}, {"positivity", settings.numerics.positivity}, {"overlap", settings.execution.overlap}}.dump();
            DNDS_check_throw_info(serialized.size() <= size_t(INT_MAX), "implicit controls exceed MPI string range"); });
        int length = mpi.rank == 0 ? int(serialized.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        MPI::LocalStage(mpi, [&]
                        { reference = mpi.rank == 0 ? serialized : std::string(size_t(length), '\0'); });
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, mpi.comm);
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(reference == serialized, "implicit controls differ between domain ranks"); });
        MPI::LocalStage(mpi, [&]
                        { _data = std::make_shared<Data>(discrete, state, std::move(workspace), engine, std::move(stream), settings); });
        _data->PrepareLinear(state, settings.execution.overlap);
    }
    NonlinearIterationReport NonlinearSolver::Solve(CFV::FVField &state, const CFV::FVField &source, real mass, real rhsScale, real time, int)
    {
        auto &data = *_data;
        MPI::LocalStage(data.mpi, [&]
                        {
            data.discrete.Metrics().ValidateField(state, 5); data.discrete.Metrics().ValidateField(source, 5);
            DNDS_check_throw_info(state.Owned().SharedStorage() == data.stateOwned && state.Ghost().SharedStorage() == data.stateGhost &&
                                      !Algebra::Detail::Overlaps(state.Owned().Storage(), source.Owned().Storage()) &&
                                      std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time),
                                  "invalid implicit stage state/source/coefficients"); });
        const std::array<real, 3> coefficients{mass, rhsScale, time};
        auto minimum = coefficients, maximum = coefficients;
        MPI_Allreduce(MPI_IN_PLACE, minimum.data(), 3, MPI_DOUBLE, MPI_MIN, data.mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, maximum.data(), 3, MPI_DOUBLE, MPI_MAX, data.mpi.comm);
        DNDS_check_throw_info(minimum == maximum, "implicit stage coefficients differ between ranks");
        NonlinearIterationReport report;
        report.pseudoCFL = data.settings.pseudoCFL;
        auto geometry = data.discrete.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        for (index iteration = 0; iteration <= data.settings.iterations; ++iteration)
        {
            data.discrete.Evaluate(state, data.spatial, *data.workspace, *data.engine, data.stream, time);
            std::array<real, 6> norm{};
            data.Stage([&]
                       {
                const auto u = state.Owned().Read(), r = data.spatial.Owned().Read(), b = source.Owned().Read();
                auto write = data.defect.Owned().Write();
                const auto mean = u.View<R::UniformTableRows>(), residual = r.View<R::UniformTableRows>(), constant = b.View<R::UniformTableRows>();
                const auto output = write.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                {
                    const real volume = metric.GetCellVol(iCell); norm[5] += volume;
                    for (int j = 0; j < 5; ++j)
                    {
                        const real value = rhsScale * residual(iCell, j) + volume * (constant(iCell, j) - mass * mean(iCell, j));
                        DNDS_check_throw_info(std::isfinite(value), "nonfinite implicit stage defect");
                        output(iCell, j) = value; norm[j] += value * value / volume;
                    }
                } });
            MPI_Allreduce(MPI_IN_PLACE, norm.data(), 6, MPI_DOUBLE, MPI_SUM, data.mpi.comm);
            DNDS_check_throw_info(std::isfinite(norm[5]) && norm[5] > 0, "invalid global stage volume");
            bool converged = iteration >= data.settings.minIterations;
            for (int j = 0; j < 5; ++j)
            {
                report.residual[j] = std::sqrt(norm[j] / norm[5]);
                DNDS_check_throw_info(std::isfinite(report.residual[j]), "nonfinite global stage norm");
                if (iteration == 0)
                    report.initialResidual[j] = report.residual[j];
                converged = converged && report.residual[j] <= std::max(data.settings.absoluteTolerance, data.settings.relativeTolerance * report.initialResidual[j]);
            }
            report.iterations = iteration;
            if (converged || iteration == data.settings.iterations)
            {
                report.converged = converged;
                return report;
            }
            const auto &auxiliary = data.discrete.PreparedAuxValues(*data.workspace);
            index rhsLimited = 0;
            real rhsAlpha = 1;
            data.Stage([&]
                       {
                const auto signals = auxiliary.cellSpectralSum->Owned().Read(), meanRead = state.Owned().Read();
                auto inverse = data.inversePseudoTime.Write(), residualWrite = data.defect.Owned().Write();
                const auto spectral = signals.View<R::UniformTableRows>(), mean = meanRead.View<R::UniformTableRows>();
                const auto invTau = inverse.View<R::UniformTableRows>(), defect = residualWrite.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                {
                    const real volume = metric.GetCellVol(iCell);
                    real value = spectral(iCell, 0) / (report.pseudoCFL * volume);
                    DNDS_check_throw_info(std::isfinite(value) && value > 0, "invalid implicit pseudo-time spectral sum");
                    if (data.positivity && data.settings.positivity.rhsMode != "none")
                    {
                        FlowState u, delta;
                        for (int j = 0; j < 5; ++j) { u[j] = mean(iCell, j); delta[j] = defect(iCell, j) * data.settings.positivity.rhsScale / (value * volume); }
                        const real alpha = FlowIncrementFraction(u, delta, data.discrete.ThermalConstraints(), data.settings.positivity.rhsRelax, 2, data.settings.positivity.compressionPower);
                        DNDS_check_throw_info(std::isfinite(alpha) && alpha >= 0 && alpha <= 1, "invalid implicit RHS PP fraction");
                        if (alpha < 1) { ++rhsLimited; rhsAlpha = std::min(rhsAlpha, alpha); }
                        if (data.settings.positivity.rhsMode == "rhs")
                            for (int j = 0; j < 5; ++j) defect(iCell, j) *= alpha;
                        else value /= std::max(alpha, smallReal);
                    }
                    invTau(iCell, 0) = value;
                } });
            data.Stage([&]
                       { data.discrete.PrepareJacobian(state, auxiliary, data.inversePseudoTime, *data.problem, {mass, rhsScale, data.settings.roeDissipation}, time); });
            report.lastLinear = data.linear->Solve(data.defect);
            report.linearIterations += report.lastLinear.iterations;
            DNDS_check_throw_info(report.lastLinear.usable, "implicit linear correction is unusable");
            index incrementLimited = 0;
            real incrementAlpha = 1;
            bool admissible = true;
            data.Stage([&]
                       {
                const auto correctionRead = data.linear->Solution().Owned().Read(), meanRead = state.Owned().Read();
                auto candidateWrite = data.candidate.Owned().Write();
                const auto correction = correctionRead.View<R::UniformTableRows>();
                const auto mean = meanRead.View<R::UniformTableRows>();
                const auto candidate = candidateWrite.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                {
                    FlowState u, delta;
                    for (int j = 0; j < 5; ++j) { u[j] = mean(iCell, j); delta[j] = data.settings.relaxation * correction(iCell, j); }
                    if (data.positivity)
                    {
                        const real alpha = FlowIncrementFraction(u, delta, data.discrete.ThermalConstraints(), data.settings.positivity.incrementRelax, 2, data.settings.positivity.compressionPower);
                        admissible = admissible && std::isfinite(alpha) && alpha >= 0 && alpha <= 1;
                        if (alpha < 1) { ++incrementLimited; incrementAlpha = std::min(incrementAlpha, alpha); }
                        for (auto &component : delta) component *= alpha;
                        delta = FixFlowIncrement(u, delta, data.discrete.ThermalConstraints());
                    }
                    for (int j = 0; j < 5; ++j) u[j] += delta[j];
                    admissible = admissible && ThermallyAdmissible(u, data.discrete.ThermalConstraints());
                    for (int j = 0; j < 5; ++j) candidate(iCell, j) = u[j];
                } });
            // Reject before acquiring the state's write lease. A recoverable
            // bad correction must not poison the bound trial allocation and
            // turn the next smaller-dt attempt into a storage-lifetime error.
            MPI::LocalStage(data.mpi, [&]
                            { DNDS_check_throw_info(admissible, "implicit correction produces an inadmissible mean"); });
            data.Stage([&]
                       { Algebra::Copy(state, data.candidate); });
            if (data.positivity)
            {
                index counts[2]{rhsLimited, incrementLimited};
                real fractions[2]{rhsAlpha, incrementAlpha};
                MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, data.mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, fractions, 2, MPI_DOUBLE, MPI_MIN, data.mpi.comm);
                report.rhsLimited += counts[0];
                report.incrementLimited += counts[1];
                report.minimumRHSAlpha = std::min(report.minimumRHSAlpha, fractions[0]);
                report.minimumIncrementAlpha = std::min(report.minimumIncrementAlpha, fractions[1]);
            }
            report.pseudoCFL = std::min(data.settings.maxPseudoCFL, report.pseudoCFL * data.settings.pseudoGrowth);
        }
        return report;
    }
}
