// Frozen perfect-gas SA spatial operation, including normal mean/coefficient
// halos, face convection/diffusion and volume source/destruction preparation.
// Zero polynomial coefficients: no VR iteration, PP or nonlinear solve.
// Optional auxiliary mode compares actual spectral/timestep entry points on a
// uniform state. Euler also constructs its mandatory GG gradients; native does
// not need these for SA transport. Both retain their normal halo/check protocol.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Euler/EulerEvaluator.hpp"
#include "CartesianFixture.hpp"
#include <iostream>

namespace C = DNDS::Cauchy;
namespace E = DNDS::Euler;
namespace R = DNDS::Runtime;
using I = DNDS::index;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(argc >= 4 && argc <= 6, "arguments: grid-width repetitions trials [constant|sutherland] [spatial|auxiliary|halo]");
        const I width = std::stoll(argv[1]);
        const int repetitions = std::stoi(argv[2]), trials = std::stoi(argv[3]);
        const std::string law = argc >= 5 ? argv[4] : "constant";
        const std::string operation = argc == 6 ? argv[5] : "spatial";
        DNDS_check_throw_info(operation == "spatial" || operation == "auxiliary" || operation == "halo", "unsupported SA component operation");
        DNDS_check_throw_info(law == "constant" || law == "sutherland", "unsupported spatial viscosity law");
        DNDS_check_throw_info(repetitions > 0 && trials > 0, "positive spatial comparison counts required");
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        mesh->BuildNodeWallDist([](auto)
                                { return false; });
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .001;
        settings.equation.viscosityModel = law;
        settings.equation.viscosityReferenceTemperature = 1;
        settings.equation.sutherlandConstant = law == "sutherland" ? .4 : 0;
        settings.equation.rans.model = "SA";
        settings.equation.rans.viscosityReference = .001;
        // Euler's nonreactive EvaluateDt uses (mu+mu_t)*gamma/Pr;
        // native bounds separate molecular/turbulent heat conduction. Set
        // Pr_t=Pr in this component to compare equal spectral physics.
        if (operation == "auxiliary")
            settings.equation.rans.turbulentPrandtl = settings.equation.prandtl;
        settings.numerics.reconstruction = "variational";
        settings.numerics.reconstructionIterations = 0;
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.flux = "Roe_M1";
        settings.numerics.faceFallback = settings.numerics.positivity = false;
        settings.numerics.riemann.useStencil = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.execution.overlap = settings.execution.checkAccess = false;
        settings.implicit.collectiveChecks = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(6));
        auto result = C::EquationDofOperations::NewVector(state);
        auto workspace = fv.BuildEquationWorkspace(state);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();

        using Eval = E::EulerEvaluator<E::NS_SA>;
        auto old = std::make_shared<DNDS::CFV::VariationalReconstruction<2>>(mpi, mesh);
        DNDS::CFV::VRSettings oldSettings(2);
        nlohmann::ordered_json json;
        oldSettings.WriteIntoJson(json);
        json["maxOrder"] = 3;
        json["intOrder"] = 5;
        old->parseSettings(json);
        auto boundary = std::make_shared<E::BoundaryHandler<E::NS_SA>>(6);
        Eval::InitializeFV(mesh, old, boundary);
        E::EulerEvaluatorSettings<E::NS_SA> physics(6);
        physics.ignoreSourceTerm = false;
        physics.rsType = E::Gas::Roe_M1;
        physics.rsFixScale = physics.rsIncFScale = 1;
        physics.useSourceGradFixGG = 0;
        auto &gas = physics.idealGasProperty;
        gas.gamma = 1.4;
        gas.Rgas = gas.rho0 = gas.U0 = gas.L0 = gas.T0 = 1;
        gas.muModel = law == "sutherland" ? 1 : 0;
        gas.TRef = settings.equation.viscosityReferenceTemperature;
        gas.CSutherland = settings.equation.sutherlandConstant;
        gas.muGas = settings.equation.viscosity;
        gas.prGas = settings.equation.prandtl;
        gas.prTurb = settings.equation.rans.turbulentPrandtl;
        Eval evaluator(mesh, old, boundary, physics);
        E::ArrayDOFV<6> oldMean, oldResult;
        E::ArrayRECV<6> oldRec;
        E::ArrayDOFV<1> beta, alpha, oldDt;
        E::JacobianDiagBlock<6> source;
        old->BuildUDof(oldMean, 6);
        old->BuildUDof(oldResult, 6);
        old->BuildURec(oldRec, 6);
        old->BuildUDof(beta, 1);
        old->BuildUDof(alpha, 1);
        old->BuildUDof(oldDt, 1);
        evaluator.uGradBuf.setConstant(0.);
        oldRec.setConstant(0.);
        beta.setConstant(1.);
        alpha.setConstant(1.);
        source.SetModeAndInit(1, 6, oldMean);
        {
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow"), rans = write.Block("rans");
            const double pi = std::acos(-1.);
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                const auto p = metric.GetCellBary(iCell);
                // The auxiliary oracle is uniform: otherwise Euler evaluates
                // molecular transport at the face-mean thermodynamic state,
                // while native spectra also bound the endpoint properties.
                // Uniform inputs isolate cost without equating those policies.
                const auto mean = equation.Gas().Conservative({operation == "auxiliary" ? 1 : 1 + .03 * std::sin(2 * pi * p[0]),
                                                               operation == "auxiliary" ? .2 : .2 + .02 * std::cos(2 * pi * p[1]), -.1, .03, 1.1});
                for (int j = 0; j < 5; ++j)
                    flow(iCell, j) = oldMean[iCell][j] = mean[j];
                rans(iCell, 0) = oldMean[iCell][5] = operation == "auxiliary" ? .2 : .2 * (1 + .1 * std::cos(2 * pi * p[0]));
            }
        }
        const auto version = state.PhysicalField(0).Owned().Version();
        C::AuxiliaryRequest request;
        request.meanThermodynamics = request.faceSpectralRadius = request.cellSpectralSum = request.cellWaveJump = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        auto meanHalo = fv.Metrics().BindHalo(state.PhysicalField(0));
        auto waveHalo = fv.Metrics().BindHalo(*auxiliary.cellWaveJump);
        auto coefficients = fv.Metrics().BuildField(6 * fv.Reconstruction().NRecDOF());
        DNDS::Algebra::Fill(coefficients, 0.);
        if (operation == "halo")
        {
            auto write = coefficients.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            const int modes = fv.Reconstruction().NRecDOF();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (int j = 0; j < 6; ++j)
                    for (int mode = 0; mode < modes; ++mode)
                        values(iCell, j * modes + mode) = oldRec[iCell](mode, j) =
                            .01 * (mode + 1) + .02 * j + .001 * mesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
        }
        auto coefficientHalo = fv.Metrics().BindHalo(coefficients);
        const auto coefficientVersion = coefficients.Owned().Version();
        I globalGhostRows = fv.Metrics().NumCellProc() - fv.Metrics().NumCell();
        MPI_Allreduce(MPI_IN_PLACE, &globalGhostRows, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        double nativeDt = 0, legacyDt = 0;
        const auto native = [&]
        {
            if (operation == "spatial")
                fv.Evaluate(state, result, *workspace, engine, stream);
            else if (operation == "halo")
            {
                meanHalo.Begin().Wait();
                coefficientHalo.Begin().Wait();
            }
            else
            {
                meanHalo.Begin().Wait();
                fv.PrepareAuxValues(state, auxiliary);
                waveHalo.Begin().Wait();
                nativeDt = fv.StableStep(state, auxiliary, 1);
            }
        };
        const auto legacy = [&]
        {
            oldMean.trans.startPersistentPull();
            oldMean.trans.waitPersistentPull();
            if (operation == "auxiliary")
            {
                evaluator.EvaluateDt(oldDt, oldMean, oldRec, 1, legacyDt, DNDS::veryLargeReal, true, 0., 0);
                return;
            }
            oldRec.trans.startPersistentPull();
            oldRec.trans.waitPersistentPull();
            if (operation == "halo")
                return;
            evaluator.EvaluateRHS(oldResult, source, oldMean, oldRec, oldRec, beta, alpha, false, 0.,
                                  Eval::RHS_Dont_Update_Integration | Eval::RHS_Dont_Record_Bud_Flux);
        };
        const auto check = [&]
        {
            native();
            legacy();
            double error = 0, magnitude = 0;
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            if (operation == "halo")
            {
                const auto values = state.PhysicalField(0).Ghost().Read(), rec = coefficients.Ghost().Read();
                for (I iCell = 0; iCell < values.View().RowCount(); ++iCell)
                {
                    for (int j = 0; j < 6; ++j)
                    {
                        const double actual = values.View()(iCell, j), expected = oldMean[fv.Metrics().NumCell() + iCell][j];
                        DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite halo output");
                        error = std::max(error, std::abs(actual - expected));
                        magnitude = std::max(magnitude, std::abs(expected));
                    }
                    const int modes = fv.Reconstruction().NRecDOF();
                    for (int j = 0; j < 6; ++j)
                        for (int mode = 0; mode < modes; ++mode)
                        {
                            const double actual = rec.View()(iCell, j * modes + mode);
                            const double expected = oldRec[fv.Metrics().NumCell() + iCell](mode, j);
                            DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite coefficient halo output");
                            error = std::max(error, std::abs(actual - expected));
                            magnitude = std::max(magnitude, std::abs(expected));
                        }
                }
            }
            else if (operation == "auxiliary")
            {
                const auto spectra = auxiliary.faceSpectralRadius->Read(), sums = auxiliary.cellSpectralSum->Owned().Read();
                for (I iAux = 0; iAux < I(fv.AuxiliaryFaces().size()); ++iAux)
                {
                    const I iFace = fv.AuxiliaryFaces()[iAux];
                    const double actual = spectra.View()(iAux, 0) + spectra.View()(iAux, 1), expected = evaluator.lambdaFace[iFace];
                    DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite auxiliary face output");
                    error = std::max(error, std::abs(actual - expected));
                    magnitude = std::max(magnitude, std::abs(expected));
                }
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const double actual = metric.GetCellVol(iCell) / sums.View()(iCell, 0), expected = oldDt[iCell][0];
                    DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite auxiliary cell timestep");
                    error = std::max(error, std::abs(actual - expected));
                }
                DNDS_check_throw_info(std::isfinite(nativeDt) && std::isfinite(legacyDt), "nonfinite global auxiliary timestep");
                error = std::max(error, std::abs(nativeDt - legacyDt));
            }
            else
            {
                const auto read = result.ReadOwned();
                const auto layout = read.ComponentLayout();
                const auto values = layout.View();
                for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (I j = 0; j < 6; ++j)
                    {
                        const double actual = values(iCell, j) / metric.GetCellVol(iCell), expected = oldResult[iCell][j];
                        DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite SA spatial comparison output");
                        error = std::max(error, std::abs(actual - expected));
                        magnitude = std::max(magnitude, std::abs(oldResult[iCell][j]));
                    }
            }
            double global[]{error, magnitude};
            MPI_Allreduce(MPI_IN_PLACE, global, 2, MPI_DOUBLE, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(std::isfinite(global[0]) && global[0] < 1e-10 * std::max(1., global[1]) &&
                                      version == state.PhysicalField(0).Owned().Version() && coefficientVersion == coefficients.Owned().Version(),
                                  "SA " + operation + " workload/output mismatch: error=" + std::to_string(global[0]) +
                                      " magnitude=" + std::to_string(global[1]));
            return global[0];
        };
        double error = check();
        const auto measure = [&](const auto &run)
        {
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int repeat = 0; repeat < repetitions; ++repeat)
                run();
            double elapsed = (MPI_Wtime() - start) / repetitions;
            MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return elapsed;
        };
        for (int trial = 0; trial < trials; ++trial)
        {
            double nativeSeconds, legacySeconds;
            if (trial % 2)
                legacySeconds = measure(legacy), nativeSeconds = measure(native);
            else
                nativeSeconds = measure(native), legacySeconds = measure(legacy);
            error = check();
            if (!mpi.rank)
                std::cout << "SA_SPATIAL_COMPONENT " << C::Json{{"cells", width * width}, {"np", mpi.size}, {"repetitions", repetitions}, {"operation", operation}, {"uniform_state", operation == "auxiliary"}, {"global_ghost_rows", globalGhostRows}, {"mean_width", 6}, {"coefficient_width", coefficients.Width()}, {"euler_gradient_preparation", operation == "auxiliary"}, {"turbulent_prandtl", settings.equation.rans.turbulentPrandtl}, {"trial", trial}, {"native_seconds", nativeSeconds}, {"legacy_seconds", legacySeconds}, {"max_error", error}, {"coefficients", operation == "halo" ? "nonuniform-frozen" : "zero"}, {"degree", 3}, {"integration_order", 5}, {"includes_halo", true}, {"viscosity_law", law}, {"includes_source", operation == "spatial"}, {"source_gradient", "zero-polynomial"}, {"includes_vr_or_pp", false}}.dump() << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "SA spatial comparison rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    DNDS::MPI::Finalize();
}
