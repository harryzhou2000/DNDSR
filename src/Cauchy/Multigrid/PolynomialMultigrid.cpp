#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <tuple>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct PolynomialMultigrid::Data
    {
        struct Coarse
        {
            FiniteVolume discrete;
            CFV::FVField state;
            std::shared_ptr<FiniteVolume::Workspace> workspace;
            std::unique_ptr<NonlinearSolver> solver;
            Coarse(ssp<Geom::UnstructuredMesh> mesh, DomainSettings settings, const EquationSystem &equation,
                   const R::RowLayout &layout, R::EagerEngine &engine, const R::ExecutionStream &stream)
                : discrete(std::move(mesh), settings, equation), state(discrete.Metrics().BuildField(layout))
            {
                AuxiliaryRequest channels;
                channels.faceSpectralRadius = channels.cellSpectralSum = true;
                workspace = discrete.BuildWorkspace(state, channels);
                solver = std::make_unique<NonlinearSolver>(discrete, state, workspace, engine, stream, settings);
            }
        };
        FiniteVolume fine;
        CFV::FVField state, value;
        NonlinearSolver *fineSolver;
        ImplicitSettings settings;
        MPIInfo mpi;
        std::vector<std::unique_ptr<Coarse>> coarse;
        std::unique_ptr<Multigrid::FASCycle> cycle;
        std::vector<Multigrid::FASLevel> levelDefinitions;
        std::vector<Multigrid::FASTransfer> transferDefinitions;
        std::vector<real> smootherCFL;
        real mass = 0, rhsScale = 1, time = 0;
        int stage = 0;
        NonlinearIterationReport *report = nullptr;
        Data(const FiniteVolume &discrete, CFV::FVField &input, NonlinearSolver &solver, const ImplicitSettings &controls)
            : fine(discrete), state(input), value(discrete.Metrics().BuildField(input.Owned().Layout())), fineSolver(&solver), settings(controls), mpi(discrete.Metrics().Communicator()) {}
        void Accumulate(const NonlinearIterationReport &step)
        {
            report->linearIterations += step.linearIterations;
            report->iterations += step.iterations;
            report->rhsLimited += step.rhsLimited;
            report->incrementLimited += step.incrementLimited;
            report->minimumRHSAlpha = std::min(report->minimumRHSAlpha, step.minimumRHSAlpha);
            report->minimumIncrementAlpha = std::min(report->minimumIncrementAlpha, step.minimumIncrementAlpha);
            report->lastLinear = step.lastLinear;
            report->pseudoCFL = step.pseudoCFL;
        }
        Multigrid::FASLevel Level(CFV::FVField input, NonlinearSolver &solver, index level)
        {
            auto *relax = &solver;
            const auto smooth = [this, relax, level](auto &u, const auto &source, index count, const std::vector<index> &components)
            {
                const auto step = relax->Solve(u, source, mass, rhsScale, time, stage, {count, smootherCFL[level], nullptr, components});
                smootherCFL[level] = step.pseudoCFL;
                Accumulate(step);
                return step.iterations;
            };
            return {std::move(input),
                    [this, relax](auto &u, auto &out)
                    { relax->EvaluateEquation(u, out, mass, rhsScale, time); },
                    [smooth](auto &u, const auto &source, index count)
                    { return smooth(u, source, count, {}); },
                    [this, relax](auto &u, const auto &delta)
                    {
                        const auto [limited, alpha] = relax->ApplyCorrection(u, delta, settings.multigrid.damping, settings.multigrid.updateComponents);
                        report->incrementLimited += limited;
                        report->minimumIncrementAlpha = std::min(report->minimumIncrementAlpha, alpha);
                    },
                    smooth};
        }
        void BuildCycle()
        {
            const auto &control = settings.multigrid;
            cycle = std::make_unique<Multigrid::FASCycle>(levelDefinitions, transferDefinitions, Multigrid::FASCycleSettings{control.preSmooth, control.postSmooth, control.coarseSmooth, control.updateComponents},
                                                          [this](const auto &run)
                                                          { MPI::LocalStage(mpi, run); });
        }
        FlowState Measure(const CFV::FVField &source)
        {
            fineSolver->EvaluateEquation(state, value, mass, rhsScale, time);
            std::array<real, 6> norm{};
            MPI::LocalStage(mpi, [&]
                            {
                const auto geometry = fine.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                const auto bRead = source.Owned().Read(), fRead = value.Owned().Read();
                const auto b = bRead.View<R::UniformTableRows>(), f = fRead.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < f.RowCount(); ++iCell)
                {
                    const real volume = metric.GetCellVol(iCell); norm[5] += volume;
                    for (int j = 0; j < 5; ++j)
                    {
                        const real difference = b(iCell, j) - f(iCell, j);
                        norm[j] += settings.normOrder == 1 ? volume * std::abs(difference) : volume * difference * difference;
                    }
                } });
            MPI_Allreduce(MPI_IN_PLACE, norm.data(), 6, MPI_DOUBLE, MPI_SUM, mpi.comm);
            DNDS_check_throw_info(norm[5] > 0 && std::isfinite(norm[5]), "invalid FAS global volume");
            FlowState result{};
            for (int j = 0; j < 5; ++j)
            {
                result[j] = settings.normOrder == 1 ? norm[j] / norm[5] : std::sqrt(norm[j] / norm[5]);
                DNDS_check_throw_info(std::isfinite(result[j]), "nonfinite FAS finest residual");
            }
            return result;
        }
    };
    PolynomialMultigrid::PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, CFV::FVField &state,
                                             NonlinearSolver &fineSolver, R::EagerEngine &engine, R::ExecutionStream stream, const DomainSettings &settings)
    {
        const MPIInfo mpi(fine.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            settings.implicit.Validate();
            DNDS_check_throw_info(settings.implicit.multigrid.levels > 0 && settings.numerics.reconstruction == "variational" &&
                                      settings.numerics.cfv.maxOrder >= 2, "polynomial FAS requires high-order variational reconstruction");
            _data = std::make_shared<Data>(fine, state, fineSolver, settings.implicit); });
        auto &data = *_data;
        data.smootherCFL.assign(size_t(settings.implicit.multigrid.levels + 1), settings.implicit.pseudoCFL);
        auto &levels = data.levelDefinitions;
        auto &transfers = data.transferDefinitions;
        levels.push_back(data.Level(state, fineSolver, 0));
        const auto identity = [mpi](const auto &input, auto &output)
        {
            // Same mesh/order, but each spatial operator owns a distinct DOF
            // space. Do not weaken DofAlgebra's general row-space check.
            MPI::LocalStage(mpi, [&]
                            {
                input.Validate(); output.Validate();
                DNDS_check_throw_info(input.Owned().RowCount() == output.Owned().RowCount() && input.Width() == output.Width(), "polynomial FAS identity transfer shape mismatch");
                const auto read = input.Owned().Read(); auto write = output.Owned().Write();
                const auto source = read.template View<R::UniformTableRows>(); const auto destination = write.template View<R::UniformTableRows>();
                for (index iCell = 0; iCell < source.RowCount(); ++iCell)
                    for (index j = 0; j < source.RowSize(iCell); ++j) destination(iCell, j) = source(iCell, j); });
        };
        for (int level = 0; level < settings.implicit.multigrid.levels; ++level)
        {
            auto controls = settings;
            controls.implicit.multigrid.levels = 0;
            controls.numerics.reconstruction = level == 0 ? "gradient" : "constant";
            controls.numerics.cfv.maxOrder = 1;
            // LVR's fine penalty targets this explicit limited O2 fidelity.
            // A coarse gradient operator has no high-order VR row to penalize.
            if (controls.numerics.limiter == "LVR")
                controls.numerics.limiter = "WBAP";
            if (level)
                controls.numerics.limiter = "none";
            data.coarse.push_back(std::make_unique<Data::Coarse>(mesh, controls, EquationSystem::Resolve(settings.equation), state.Owned().Layout(), engine, stream));
            DNDS_check_throw_info(data.coarse.back()->discrete.Metrics().Cells().SameOrder(fine.Metrics().Cells()), "polynomial FAS levels changed cell identity/order");
            levels.push_back(data.Level(data.coarse.back()->state, *data.coarse.back()->solver, level + 1));
            transfers.push_back({identity, identity, identity});
        }
        data.BuildCycle();
    }
    NonlinearIterationReport PolynomialMultigrid::Solve(const CFV::FVField &source, real mass, real rhsScale, real time, int stage)
    {
        auto &data = *_data;
        MPI::LocalStage(data.mpi, [&]
                        {
            data.fine.Metrics().ValidateField(source, 5);
            DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time), "invalid polynomial FAS equation coefficients");
            for (const auto *input : {&source.Owned(), &source.Ghost()})
                for (const auto *output : {&data.state.Owned(), &data.state.Ghost()})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "polynomial FAS source aliases state"); });
        std::array<real, 3> minimum{mass, rhsScale, time}, maximum = minimum;
        MPI_Allreduce(MPI_IN_PLACE, minimum.data(), 3, MPI_DOUBLE, MPI_MIN, data.mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, maximum.data(), 3, MPI_DOUBLE, MPI_MAX, data.mpi.comm);
        DNDS_check_throw_info(minimum == maximum, "polynomial FAS coefficients differ between ranks");
        data.mass = mass;
        data.rhsScale = rhsScale;
        data.time = time;
        data.stage = stage;
        if (!data.cycle)
            data.BuildCycle();
        std::fill(data.smootherCFL.begin(), data.smootherCFL.end(), data.settings.pseudoCFL);
        NonlinearIterationReport report;
        ResidualConvergence convergence(data.settings);
        data.report = &report;
        report.levelCorrections.assign(data.coarse.size() + 1, 0);
        report.initialResidual = data.Measure(source);
        for (index cycle = 0; cycle <= data.settings.multigrid.cycles; ++cycle)
        {
            report.residual = cycle ? data.Measure(source) : report.initialResidual;
            // Fine residual acceptance owns this snapshot; a coarse smoother's
            // last workspace must not be reported as the physical fine VR.
            std::tie(report.reconstructionIterations, report.reconstructionRelativeUpdate) = data.fineSolver->ReconstructionStats();
            convergence.Observe(report.residual);
            report.initialResidual = convergence.Initial();
            report.referenceResidual = convergence.Reference();
            report.residualTargetRatio = convergence.TargetRatio();
            const bool converged = report.iterations >= data.settings.minIterations && data.fineSolver->ReconstructionConverged() && convergence.Accepted();
            if (converged || cycle == data.settings.multigrid.cycles)
            {
                report.converged = converged;
                data.report = nullptr;
                return report;
            }
            Multigrid::FASCycleReport work;
            try
            {
                work = data.cycle->Apply(source);
            }
            catch (...)
            {
                data.cycle.reset();
                data.report = nullptr;
                throw;
            }
            ++report.multigridCycles;
            for (index level = 0; level < index(work.corrections.size()); ++level)
                report.levelCorrections[level] += work.corrections[level];
        }
        return report;
    }
}
