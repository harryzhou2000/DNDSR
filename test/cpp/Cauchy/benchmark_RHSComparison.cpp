// One component per invocation. Both RHS implementations consume the same
// distributed mesh and frozen, nonuniform flow5 means. All reconstruction
// coefficients are zero: this isolates face flux work from VR/gradient setup,
// limiting, halo exchange, pseudo-time control and nonlinear convergence.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Euler/EulerEvaluator.hpp"
#include "CartesianFixture.hpp"
#include <iostream>

namespace C = DNDS::Cauchy;
namespace E = DNDS::Euler;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(argc == 6, "arguments: grid-width degree(1|3) viscous(0|1) repetitions trials");
        const I width = std::stoll(argv[1]);
        const int degree = std::stoi(argv[2]), repetitions = std::stoi(argv[4]), trials = std::stoi(argv[5]);
        const bool viscous = std::stoi(argv[3]);
        DNDS_check_throw_info((degree == 1 || degree == 3) && repetitions > 0 && trials > 0, "invalid RHS comparison controls");
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        C::DomainSettings settings;
        settings.numerics.cfv.maxOrder = degree;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.reconstruction = degree == 1 ? "gradient" : "variational";
        settings.numerics.flux = "Roe_M1";
        settings.numerics.faceFallback = settings.numerics.positivity = false;
        settings.numerics.riemann.useStencil = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.equation.transport = viscous ? "NewtonianFourier" : "inviscid";
        settings.equation.viscosity = viscous ? .001 : 0;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto mean = fv.Metrics().BuildField(5), coefficients = fv.Metrics().BuildField(5 * fv.Reconstruction().NRecDOF());
        auto residual = fv.Metrics().BuildField(5);
        A::Fill(coefficients, 0.);

        using Eval = E::EulerEvaluator<E::NS>;
        auto old = std::make_shared<DNDS::CFV::VariationalReconstruction<2>>(mpi, mesh);
        DNDS::CFV::VRSettings oldSettings(2);
        nlohmann::ordered_json json;
        oldSettings.WriteIntoJson(json);
        json["maxOrder"] = degree;
        json["intOrder"] = settings.numerics.RHSIntegrationOrder();
        old->parseSettings(json);
        auto boundary = std::make_shared<E::BoundaryHandler<E::NS>>(5);
        Eval::InitializeFV(mesh, old, boundary);
        E::EulerEvaluatorSettings<E::NS> physics(5);
        physics.ignoreSourceTerm = true;
        physics.rsType = E::Gas::Roe_M1;
        physics.rsFixScale = physics.rsIncFScale = 1;
        auto &gas = physics.idealGasProperty;
        gas.gamma = 1.4;
        gas.Rgas = gas.rho0 = gas.U0 = gas.L0 = gas.T0 = 1;
        gas.muModel = 0;
        gas.muGas = settings.equation.viscosity;
        gas.prGas = settings.equation.prandtl;
        Eval evaluator(mesh, old, boundary, physics);
        E::ArrayDOFV<5> oldMean, oldResidual;
        E::ArrayRECV<5> oldRec;
        E::ArrayDOFV<1> beta, alpha;
        E::JacobianDiagBlock<5> source;
        old->BuildUDof(oldMean, 5);
        old->BuildUDof(oldResidual, 5);
        old->BuildURec(oldRec, 5);
        old->BuildUDof(beta, 1);
        old->BuildUDof(alpha, 1);
        oldRec.setConstant(0.);
        beta.setConstant(1.);
        alpha.setConstant(1.);
        source.SetModeAndInit(1, 5, oldMean);
        {
            auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = mean.Owned().Write();
            const auto values = write.View<R::FixedRows<5>>();
            const double pi = std::acos(-1.);
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto p = metric.GetCellBary(iCell);
                const auto state = equation.Gas().Conservative({1 + .03 * std::sin(2 * pi * p[0]), .2 + .02 * std::cos(2 * pi * p[1]), -.1, .03, 1.1});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = oldMean[iCell][j] = state[j];
            }
        }
        fv.Metrics().BindHalo(mean).Begin().Wait();
        fv.Metrics().BindHalo(coefficients).Begin().Wait();
        oldMean.trans.startPersistentPull();
        oldMean.trans.waitPersistentPull();
        oldRec.trans.startPersistentPull();
        oldRec.trans.waitPersistentPull();
        const auto nativeRHS = [&]
        {
            const auto op = fv.AcquireRHSOperator();
            const auto u = mean.Owned().Read(), ug = mean.Ghost().Read();
            const auto rec = coefficients.Owned().Read(), rg = coefficients.Ghost().Read();
            auto write = residual.Owned().Write();
            C::RHSReadViews input;
            input.mean = {u.View<C::RHSFlowRows>(), ug.View<C::RHSFlowRows>()};
            input.coefficients = {rec.View<R::UniformTableRows>(), rg.View<R::UniformTableRows>()};
            C::RHSLocalDiagnostics diagnostics;
            C::RHSWriteViews output;
            output.residual = write.View<C::RHSFlowRows>();
            diagnostics.Merge(op.ApplyBorrowedChecked(input, output));
            DNDS_check_throw_info(diagnostics.FaceFallbacks() == 0, "unexpected RHS face fallback");
        };
        const auto legacyRHS = [&]
        {
            const uint64_t flags = Eval::RHS_Dont_Update_Integration | Eval::RHS_Dont_Record_Bud_Flux | (viscous ? 0 : Eval::RHS_Ignore_Viscosity);
            evaluator.EvaluateRHS(oldResidual, source, oldMean, oldRec, oldRec, beta, alpha, false, 0., flags);
        };
        nativeRHS();
        legacyRHS();
        double error = 0, magnitude = 0;
        I nativePoints = 0, legacyPoints = 0, nativeFluxPoints = 0;
        {
            const auto read = residual.Owned().Read();
            const auto values = read.View<R::FixedRows<5>>();
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    const double value = values(iCell, j) / metric.GetCellVol(iCell);
                    error = std::max(error, std::abs(value - oldResidual[iCell][j]));
                    magnitude = std::max(magnitude, std::abs(oldResidual[iCell][j]));
                }
            for (I iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
            {
                nativePoints += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                legacyPoints += old->GetFaceQuad(iFace).GetNumPoints();
            }
            const auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
            const auto topology = topologyRead.HostView();
            for (I iCell = 0; iCell < topology.NumCell(); ++iCell)
                for (auto iFace : topology.CellFaces(iCell))
                    nativeFluxPoints += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &magnitude, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(nativePoints == legacyPoints && std::isfinite(error) && error < 1e-10 * std::max(1., magnitude), "RHS workload/output mismatch; do not time unmatched methods");
        const I legacyFluxPoints = legacyPoints;
        I globalWork[]{nativeFluxPoints, legacyFluxPoints};
        MPI_Allreduce(MPI_IN_PLACE, globalWork, 2, MPI_INT64_T, MPI_SUM, mpi.comm);
        const auto measure = [&](auto &&run)
        {
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int repetition = 0; repetition < repetitions; ++repetition)
                run();
            double elapsed = (MPI_Wtime() - start) / repetitions;
            MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return elapsed;
        };
        for (int trial = 0; trial < trials; ++trial)
        {
            double nativeSeconds, legacySeconds;
            if (trial % 2)
                legacySeconds = measure(legacyRHS), nativeSeconds = measure(nativeRHS);
            else
                nativeSeconds = measure(nativeRHS), legacySeconds = measure(legacyRHS);
            if (mpi.rank == 0)
                std::cout << "RHS_COMPONENT " << C::Json{{"degree", degree}, {"viscous", viscous}, {"cells", width * width}, {"np", mpi.size}, {"repetitions", repetitions}, {"trial", trial}, {"native_seconds", nativeSeconds}, {"legacy_seconds", legacySeconds}, {"max_error", error}, {"rhs_magnitude", magnitude}, {"face_points_rank0", nativePoints}, {"native_flux_points_global", globalWork[0]}, {"legacy_flux_points_global", globalWork[1]}, {"coefficients", "zero"}, {"includes_halo", false}, {"includes_reconstruction", false}}.dump() << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    return DNDS::MPI::Finalize();
}
