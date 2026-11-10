// E3 observes the real composed flow5 RHS using a link-only wrapper. Production
// CFV has no profiling branch/timer. GNU-compatible --wrap is required here.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Physics/EquationSetup.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CartesianFixture.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

namespace C = DNDS::Cauchy;
namespace F = DNDS::CFV;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;

namespace PublicationProbe
{
    thread_local bool active = false;
    thread_local double seconds = 0;
    thread_local I calls = 0, logicalBytes = 0;
}
extern "C" void RealCommit(const F::VariationalReconstructionPlan *, F::FVField &, F::ReconstructionWorkspace &) asm("__real__ZNK4DNDS3CFV29VariationalReconstructionPlan20CommitReconstructionERNS_7Algebra8DofFieldIdLm18446744073709551615EEERNS0_23ReconstructionWorkspaceE");
extern "C" void WrappedCommit(const F::VariationalReconstructionPlan *, F::FVField &, F::ReconstructionWorkspace &) asm("__wrap__ZNK4DNDS3CFV29VariationalReconstructionPlan20CommitReconstructionERNS_7Algebra8DofFieldIdLm18446744073709551615EEERNS0_23ReconstructionWorkspaceE");
extern "C" void WrappedCommit(const F::VariationalReconstructionPlan *plan, F::FVField &coefficients,
                              F::ReconstructionWorkspace &work)
{
    if (!PublicationProbe::active)
    {
        RealCommit(plan, coefficients, work);
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    RealCommit(plan, coefficients, work);
    PublicationProbe::seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++PublicationProbe::calls;
    PublicationProbe::logicalBytes += 16 * coefficients.Owned().RowCount() * coefficients.Width();
}

namespace
{
    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 5 || argc == 6, "arguments: grid-width none|WBAP repetitions trials [flow5|split-mixture]");
        const I width = std::stoll(argv[1]);
        const std::string limiter = argv[2];
        const int repetitions = std::stoi(argv[3]), trials = std::stoi(argv[4]);
        const std::string profile = argc == 6 ? argv[5] : "flow5";
        const bool mixture = profile == "split-mixture";
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        DNDS_check_throw_info(width >= 4 && width <= 400 && width * width >= mpi.size &&
                                  (limiter == "none" || limiter == "WBAP") && (profile == "flow5" || mixture) &&
                                  repetitions >= 1 && repetitions <= 20 && trials >= 1 && trials <= 4,
                              "coefficient RHS assessment outside declared bounds");
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .001;
        settings.execution.overlap = true;
        settings.implicit.collectiveChecks = false;
        settings.numerics.reconstruction = "variational";
        settings.numerics.reconstructionIterations = 5;
        settings.numerics.reconstructionTolerance = 0;
        settings.numerics.limiter = limiter;
        settings.numerics.positivity = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.cfv.SORInstead = false;
        if (mixture)
        {
            settings.equation.eos = "idealGasMixture";
            settings.equation.viscosity = 0;
            settings.equation.source = "chemical";
            settings.equation.mixture.mechanismFile = "h2o2.yaml";
            settings.equation.referenceScales = {{"rho0", 1.225}, {"U0", 340}, {"L0", 2}, {"T0", 288.15}};
            settings.numerics.flux = "rusanov";
            settings.numerics.faceFallback = false;
            settings.numerics.characteristicLimiter = false;
        }
        auto physical = C::EquationSetup::Resolve(settings.equation);
        const auto &equation = physical.equation;
        C::FiniteVolume fv(mesh, settings, equation);
        std::optional<C::EquationDofs> equationMean, equationResidual;
        std::shared_ptr<C::FiniteVolume::Workspace> equationWorkspace;
        std::shared_ptr<C::FiniteVolume::ReconstructionState> equationReconstruction;
        I components = 5, fields = 1;
        std::vector<uint64_t> generations;
        R::EagerEngine engine(0);
        const auto stream = engine.CreateCPUStream();
        std::function<void()> evaluate, validate;
        if (!mixture)
        {
            equationMean.emplace(C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5)));
            {
                auto write = equationMean->WriteOwned();
                const auto values = write.Block("flow");
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const auto point = fv.Metrics().GetCellCent(iCell);
                    const auto state = equation.Gas().Conservative({1 + .01 * std::sin(6.28 * point[0]), .2, -.1, .05, 1});
                    for (int j = 0; j < 5; ++j)
                        values(iCell, j) = state[j];
                }
            }
            generations.push_back(equationMean->PhysicalField(0).Owned().Version());
            equationResidual.emplace(C::EquationDofOperations::NewVector(*equationMean));
            equationWorkspace = fv.BuildWorkspace(*equationMean);
            equationReconstruction = fv.BuildReconstructionState(*equationMean);
            evaluate = [&]
            { fv.Evaluate(*equationMean, *equationResidual, *equationReconstruction, *equationWorkspace, engine, stream); };
            validate = [&]
            {
                DNDS_check_throw_info(equationMean->PhysicalField(0).Owned().Version() == generations[0], "composed RHS mutated means");
                auto read = equationResidual->ReadOwned();
                const auto values = read.Block("flow");
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        DNDS_check_throw_info(std::isfinite(values(iCell, j)), "nonfinite composed RHS");
            };
        }
        else
        {
            DNDS::Algebra::DofBundle bundle(fv.Metrics().Space());
            components = 0;
            for (const auto &block : equation.Schema().blocks)
            {
                const auto count = I(block.components.size());
                bundle.Add(block.name, fv.Metrics().BuildField(count));
                components += count;
            }
            equationMean.emplace(C::EquationDofs::Separate(equation.Schema(), std::move(bundle)));
            fields = equationMean->PhysicalFieldNumber();
            DNDS_check_throw_info(fields == 2 && equation.Mixture().SpeciesNumber() == 10 && physical.workers,
                                  "split-mixture assessment requires h2o2's ten species and an explicit worker");
            {
                auto write = equationMean->WriteOwned();
                const auto flow = write.Block("flow"), rhoY = write.Block("species");
                auto evaluator = equation.MakePointEvaluator(physical.workers->Acquire(), {});
                std::vector<DNDS::real> fractions = equation.IndependentMassFractions({{"H2", .027}, {"O2", .22}, {"H2O", .02}});
                std::vector<DNDS::real> species(fractions.size());
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const auto point = fv.Metrics().GetCellCent(iCell);
                    const double rho = .8 + .001 * std::sin(6.28 * point[0]);
                    for (I k = 0; k < I(species.size()); ++k)
                        species[k] = rho * fractions[k];
                    const auto state = evaluator.FromPrimitive({rho, .2, -.04, .03, 0}, {species.data(), int(species.size())}, 1400. / equation.Mixture().Scales().T0);
                    for (int j = 0; j < 5; ++j)
                        flow(iCell, j) = state[j];
                    for (I k = 0; k < I(species.size()); ++k)
                        rhoY(iCell, k) = species[k];
                }
            }
            for (I iField = 0; iField < fields; ++iField)
                generations.push_back(equationMean->PhysicalField(iField).Owned().Version());
            equationResidual.emplace(C::EquationDofOperations::NewVector(*equationMean));
            equationWorkspace = fv.BuildWorkspace(*equationMean, physical.workers);
            equationReconstruction = fv.BuildReconstructionState(*equationMean);
            evaluate = [&]
            { fv.Evaluate(*equationMean, *equationResidual, *equationReconstruction, *equationWorkspace, engine, stream); };
            validate = [&]
            {
                for (I iField = 0; iField < fields; ++iField)
                    DNDS_check_throw_info(equationMean->PhysicalField(iField).Owned().Version() == generations[iField], "equation RHS mutated means");
                const auto read = equationResidual->ReadOwned();
                for (I iBlock = 0; iBlock < equationResidual->BlockNumber(); ++iBlock)
                {
                    const auto values = read.Block(iBlock);
                    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                        for (I j = 0; j < values.ComponentNumber(); ++j)
                            DNDS_check_throw_info(std::isfinite(values(iCell, j)), "nonfinite equation RHS");
                }
            };
        }
        for (int warm = 0; warm < 5; ++warm)
            evaluate();
        const I expected = fields * (5 + (limiter == "WBAP" ? 3 : 0));
        int provided = 0;
        MPI_Query_thread(&provided);
        for (int trial = 0; trial < trials; ++trial)
            for (int phase = 0; phase < 2; ++phase)
            {
                const bool instrumented = (trial % 2 == 0) == (phase == 0);
                PublicationProbe::calls = PublicationProbe::logicalBytes = 0;
                PublicationProbe::seconds = 0;
                MPI_Barrier(mpi.comm);
                PublicationProbe::active = instrumented;
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < repetitions; ++iteration)
                    evaluate();
                const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                PublicationProbe::active = false;
                validate();
                if (instrumented)
                    DNDS_check_throw_info(PublicationProbe::calls == expected * repetitions &&
                                              PublicationProbe::logicalBytes == (5 + (limiter == "WBAP" ? 3 : 0)) * repetitions * 16 * fv.Metrics().NumCell() * 9 * components,
                                          "actual production publication count/bytes differ");
                double maximum[]{total / repetitions, PublicationProbe::seconds / repetitions, PublicationProbe::seconds / total};
                double minimum = maximum[2];
                MPI_Allreduce(MPI_IN_PLACE, maximum, 3, MPI_DOUBLE, MPI_MAX, mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
                I count[]{PublicationProbe::calls, PublicationProbe::logicalBytes};
                MPI_Allreduce(MPI_IN_PLACE, count, 2, MPI_INT64_T, MPI_MAX, mpi.comm);
                if (mpi.rank == 0)
                    std::cout << "PLAN53_COPY_RHS " << J{{"np", mpi.size}, {"width", width}, {"degree", 3}, {"limiter", limiter}, {"profile", profile}, {"physical_fields", fields}, {"components", components}, {"trial", trial}, {"instrumented", instrumented}, {"repetitions", repetitions}, {"provided", provided}, {"rhs_seconds", maximum[0]}, {"publication_seconds", maximum[1]}, {"local_fraction_min", minimum}, {"local_fraction_max", maximum[2]}, {"publication_calls_max", count[0]}, {"publication_logical_bytes_max", count[1]}, {"finite", true}, {"mean_generation_unchanged", true}}.dump() << std::endl;
            }
    }
}

int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    try
    {
        Run(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "coefficient RHS assessment: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
