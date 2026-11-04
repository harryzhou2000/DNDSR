#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
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
        Multigrid::FASLevel Level(CFV::FVField input, NonlinearSolver &solver, index level)
        {
            auto *relax = &solver;
            const auto smooth = [this, relax, level](auto &u, const auto &source, index count, const std::vector<index> &components)
            {
                for (index correction = 0; correction < count; ++correction)
                {
                    const auto step = relax->CorrectOnce(u, source, mass, rhsScale, time, stage,
                                                         {1, smootherCFL[level], nullptr, components});
                    NonlinearDetail::AccumulateWork(*report, step);
                }
                return count;
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
        void BindEquation(const CFV::FVField &source, real equationMass, real equationScale, real equationTime, int equationStage, real pseudoCFL)
        {
            MPI::LocalStage(mpi, [&]
                            {
            fine.Metrics().ValidateField(source, 5);
            DNDS_check_throw_info(std::isfinite(equationMass) && equationMass >= 0 && std::isfinite(equationScale) && equationScale > 0 && std::isfinite(equationTime) && (pseudoCFL == 0 || (std::isfinite(pseudoCFL) && pseudoCFL >= settings.pseudoCFL && pseudoCFL <= settings.maxPseudoCFL)), "invalid polynomial FAS equation coefficients");
            for (const auto *input : {&source.Owned(), &source.Ghost()})
                for (const auto *output : {&state.Owned(), &state.Ghost()})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "polynomial FAS source aliases state"); });
            std::array<real, 4> minimum{equationMass, equationScale, equationTime, pseudoCFL}, maximum = minimum;
            MPI_Allreduce(MPI_IN_PLACE, minimum.data(), 4, MPI_DOUBLE, MPI_MIN, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, maximum.data(), 4, MPI_DOUBLE, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(minimum == maximum, "polynomial FAS coefficients differ between ranks");
            mass = equationMass;
            rhsScale = equationScale;
            time = equationTime;
            stage = equationStage;
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
    void PolynomialMultigrid::SetBoundaryInputs(const BoundaryInputs &input)
    {
        MPI::LocalStage(_data->mpi, [&]
                        {
            for(auto &coarse:_data->coarse) coarse->discrete.SetBoundaryInputs(*coarse->workspace,input); });
    }
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
    NonlinearIterationReport PolynomialMultigrid::ApplyCycle(const CFV::FVField &source, real mass, real rhsScale, real time, int stage, real pseudoCFL)
    {
        auto &data = *_data;
        data.BindEquation(source, mass, rhsScale, time, stage, pseudoCFL);
        if (!data.cycle)
            data.BuildCycle();
        const real cfl = pseudoCFL == 0 ? data.settings.pseudoCFL : pseudoCFL;
        std::fill(data.smootherCFL.begin(), data.smootherCFL.end(), cfl);
        NonlinearIterationReport report;
        report.pseudoCFL = cfl;
        data.report = &report;
        try
        {
            const auto work = data.cycle->Apply(source);
            report.multigridCycles = report.iterations = 1;
            report.levelCorrections = work.corrections;
        }
        catch (...)
        {
            data.cycle.reset();
            data.report = nullptr;
            throw;
        }
        data.report = nullptr;
        return report;
    }
    NonlinearIterationReport PolynomialMultigrid::Solve(const CFV::FVField &source, real mass, real rhsScale, real time, int stage)
    {
        auto &data = *_data;
        data.BindEquation(source, mass, rhsScale, time, stage, 0);
        // Compatibility frontend: use exactly the same finest controller as a
        // direct solve, with one V-cycle replacing one correction. The cycle
        // cap retains its existing configuration name; iterations count cycles.
        auto settings = data.settings;
        settings.iterations = settings.multigrid.cycles;
        FlowState residual{};
        return NonlinearDetail::Iterate(settings, data.fine.Equation().Schema(), data.mpi.rank, stage, {}, NonlinearIterationReport{}, data.fineSolver->ReconstructionTolerance(), [&]
                                        { residual = data.Measure(source); }, [&](auto &norms)
                                        { norms = residual; }, [&]
                                        { return data.fineSolver->ReconstructionStats(); }, [&](auto &report)
                                        {
                const auto step = ApplyCycle(source, mass, rhsScale, time, stage, report.pseudoCFL);
                NonlinearDetail::AccumulateWork(report, step); });
    }
}
