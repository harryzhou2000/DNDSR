#include "Cauchy/Core/DomainComponents.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        DomainSettings SpatialSettings(const DomainSettings &settings,
                                       const TimeSettings &time)
        {
            time.ValidateSourceIntegration();
            auto result = settings;
            if (time.sourceSplitting == "strang" &&
                settings.equation.source == "chemical")
                result.equation.source = "none";
            return result;
        }
        ssp<Geom::UnstructuredMesh> PrepareWallField(
            ssp<Geom::UnstructuredMesh> mesh, const EquationSystem &equation,
            const DomainSettings &settings)
        {
            if (!equation.HasRANS())
                return mesh;
            const auto mpi = mesh->getMPI();
            int available = mesh->HasNodeWallDist();
            available = MPI::AllreduceValue<MPI::Reduction::Sum>(available,
                                                                 mpi.comm);
            DNDS_check_throw_info(
                available == 0 || available == mpi.size,
                "RANS nodal wall-field presence differs between ranks");
            if (!available)
                mesh->BuildNodeWallDist(
                    [&](auto zone)
                    {
                        const auto found = settings.boundaries.find(
                            std::to_string(zone));
                        return found != settings.boundaries.end() &&
                               found->second.kind == "noSlipWall";
                    });
            return mesh;
        }
        R::RowLayout FlowLayout(const DomainSettings &settings,
                                index width = 5)
        {
            if (settings.execution.layout == "soa")
                return R::RowLayout::ColMajor(0, width);
            if (settings.execution.layout == "aosoa32")
                return R::RowLayout::TiledColMajor(0, width, 32);
            return R::RowLayout::Dense(0, width);
        }
        EquationDofs BuildState(
            const FiniteVolume &discrete, const EquationSystem &equation,
            const DomainSettings &settings,
            const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers,
            const std::vector<real> &initialConservative)
        {
            const auto &fv = discrete.Metrics();
            if (equation.HasMixture() || equation.HasRANS())
            {
                const auto allocate = [&]
                {
                    if (settings.execution.fieldGrouping == "coupled")
                    {
                        index components = 0;
                        for (const auto &block : equation.Schema().blocks)
                            components += index(block.components.size());
                        return EquationDofs::Coupled(
                            equation.Schema(),
                            fv.BuildField(FlowLayout(settings, components)),
                            "equations");
                    }
                    Algebra::DofBundle fields(fv.Space());
                    for (const auto &block : equation.Schema().blocks)
                        fields.Add(block.name,
                                   fv.BuildField(FlowLayout(
                                       settings,
                                       index(block.components.size()))));
                    return EquationDofs::Separate(equation.Schema(),
                                                  std::move(fields));
                };
                auto state = allocate();
                if (!initialConservative.empty())
                    discrete.InitializeConservative(state,
                                                    initialConservative);
                else
                    discrete.InitializeConstant(
                        state, settings.initialPrimitive,
                        settings.initialMassFractions,
                        settings.initialTemperature, workers,
                        settings.initialRANS);
                discrete.InitializeField(state, workers);
                return state;
            }
            auto field = fv.BuildField(FlowLayout(settings));
            if (!initialConservative.empty())
            {
                auto state = EquationDofs::Coupled(equation.Schema(),
                                                   std::move(field));
                discrete.InitializeConservative(state, initialConservative);
                discrete.InitializeField(state, workers);
                return state;
            }
            auto primitive = settings.initialPrimitive;
            if (settings.initialTemperature > 0)
                primitive[4] = primitive[0] * equation.Gas().gasConstant *
                               settings.initialTemperature;
            auto state = EquationDofs::Coupled(equation.Schema(),
                                               std::move(field));
            discrete.Initialize(state, settings.initialProfile, primitive);
            discrete.InitializeField(state, workers);
            return state;
        }
    }

    DomainProblem::DomainProblem(std::string domainName,
                                 ssp<Geom::UnstructuredMesh> mesh,
                                 const DomainSettings &settings,
                                 const TimeSettings &time,
                                 EquationSystem domainEquation)
        : name(std::move(domainName)), equation(std::move(domainEquation)),
          finiteVolume(PrepareWallField(std::move(mesh), equation, settings),
                       SpatialSettings(settings, time), equation)
    {
    }

    DomainState::DomainState(
        const DomainProblem &problem, const DomainSettings &settings,
        const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers,
        const std::vector<real> &initialConservative)
        : accepted(BuildState(problem.finiteVolume, problem.equation, settings,
                              workers, initialConservative)),
          trial(accepted.Clone()),
          residual(EquationDofOperations::NewVector(accepted)),
          initial(EquationDofOperations::NewVector(accepted)),
          reconstruction(problem.finiteVolume.BuildReconstructionState(trial))
    {
        accepted.Fields().SetName(problem.name + "/accepted", DNDS_ALLOCATION_SITE());
        trial.Fields().SetName(problem.name + "/trial", DNDS_ALLOCATION_SITE());
        residual.Fields().SetName(problem.name + "/residual", DNDS_ALLOCATION_SITE());
        initial.Fields().SetName(problem.name + "/initial", DNDS_ALLOCATION_SITE());
    }

    DomainExecution::DomainExecution(
        DomainProblem &problem, DomainState &state,
        const ExecutionSettings &settings,
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
        : context(settings, std::move(workers))
    {
        const auto where = context.ComputeStream().Context();
        if (where.device != R::Device::Host())
        {
            // Mesh/CFV setup and programmable initial-value evaluation remain
            // host operations. Place the finalized immutable plan once, then
            // copy only owned initialized values; the first halo fills ghosts.
            // This is an explicit setup transfer, never a loop fallback.
            DNDS_check_throw_info(state.accepted.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA domain placement requires combined AoS state");
            auto placed = problem.finiteVolume.Placed(where.device, where);
            auto field = placed.Metrics().BuildField(state.accepted.CoupledField().Owned().Layout(), where.device);
            field.Owned().CopyFrom(state.accepted.CoupledField().Owned(), where).Wait();
            auto accepted = EquationDofs::Coupled(problem.equation.Schema(), std::move(field), state.accepted.Fields().Name(0));
            auto trial = EquationDofOperations::NewVector(accepted);
            EquationDofOperations::Copy(context.Engine(), context.ComputeStream(), trial, accepted);
            context.Synchronize();
            state.accepted = std::move(accepted);
            state.trial = std::move(trial);
            state.initial = EquationDofOperations::NewVector(state.accepted);
            state.residual = EquationDofOperations::NewVector(state.accepted);
            problem.finiteVolume = std::move(placed);
            state.reconstruction = problem.finiteVolume.BuildReconstructionState(state.trial);
            state.accepted.Fields().SetName(problem.name + "/accepted", DNDS_ALLOCATION_SITE());
            state.trial.Fields().SetName(problem.name + "/trial", DNDS_ALLOCATION_SITE());
            state.initial.Fields().SetName(problem.name + "/initial", DNDS_ALLOCATION_SITE());
            state.residual.Fields().SetName(problem.name + "/residual", DNDS_ALLOCATION_SITE());
        }
        workspace = problem.finiteVolume.BuildWorkspace(state.trial, context.ChemicalWorkers());
    }
}
