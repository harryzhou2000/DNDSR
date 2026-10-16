/**
 * @file test_EulerEvaluator.cpp
 * @brief Integration test for the EulerEvaluator pipeline: fdt -> frhs -> fsolve.
 *
 * Exercises the full evaluator pipeline on three configurations:
 *   1. Isentropic Vortex (NS, 2D periodic, inviscid)
 *   2. NACA0012 (NS_SA, 2D external, viscous + SA turbulence)
 *   3. 3D Box (NS_3D, 3D periodic, inviscid)
 *
 * For each case the test:
 *   - Reads the JSON config and mesh
 *   - Initializes the evaluator and DOF arrays
 *   - Runs one Newton-like iteration: EvaluateDt -> EvaluateRHS -> LUSGSMatrixInit
 *     -> LUSGSForward -> LUSGSBackward
 *   - Measures the L1 RHS norm (golden regression value)
 *
 * This is an MPI test registered at np=1, 2, 4.
 * Golden value sentinel: 1e300 means not yet acquired.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Euler/EulerSolver.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <map>

using namespace DNDS;
using namespace DNDS::Euler;

static MPIInfo g_mpi;
static constexpr real GOLDEN_NOT_ACQUIRED = 1e300;

// ===================================================================
// Helper: resolve path relative to project root
// ===================================================================
static std::string projectRoot()
{
    std::string f(__FILE__);
    for (int i = 0; i < 4; i++) // test/cpp/Euler/file -> root
    {
        auto pos = f.rfind('/');
        if (pos == std::string::npos)
            pos = f.rfind('\\');
        if (pos != std::string::npos)
            f = f.substr(0, pos);
    }
    return f;
}

// ===================================================================
// Helper: write a temp JSON config file with mesh path fixup
// ===================================================================
/// Write a temp config by: (1) writing defaults via ConfigureFromJson(read=false),
/// (2) reading the case JSON and merge-patching, (3) applying test overrides.
/// This ensures all required config fields exist (filled with defaults).
template <EulerModel model>
static std::string writeTempConfig(const std::string &caseJsonPath, const std::string &tag,
                                   const nlohmann::ordered_json &overrides = {})
{
    auto tmpDir = std::filesystem::temp_directory_path() / "dndsr_euler_test";
    std::filesystem::create_directories(tmpDir);
    std::string defaultPath = (tmpDir / (tag + "_default.json")).string();
    std::string cfgPath = (tmpDir / (tag + ".json")).string();

    // Step 1: Write defaults
    {
        EulerSolver<model> solverTmp(g_mpi);
        solverTmp.ConfigureFromJson(defaultPath, false); // writes all defaults
    }
    MPI::Barrier(g_mpi.comm);

    // Step 2: Read defaults, merge case JSON, merge overrides
    auto fDefault = std::ifstream(defaultPath);
    DNDS_assert(fDefault);
    auto j = nlohmann::ordered_json::parse(fDefault, nullptr, true, true);
    fDefault.close();

    auto fCase = std::ifstream(caseJsonPath);
    DNDS_assert_info(fCase, ("config file not found: " + caseJsonPath).c_str());
    auto jCase = nlohmann::ordered_json::parse(fCase, nullptr, true, true);
    fCase.close();

    j.merge_patch(jCase);
    j.merge_patch(overrides);

    // Fix mesh path to be absolute
    std::string meshRel = j["dataIOControl"]["meshFile"];
    std::filesystem::path meshAbs = std::filesystem::weakly_canonical(
        std::filesystem::path(caseJsonPath).parent_path() / meshRel);
    j["dataIOControl"]["meshFile"] = meshAbs.string();

    // Disable all output
    j["outputControl"]["nDataOutC"] = 1000000;
    j["outputControl"]["nDataOut"] = 1000000;
    j["outputControl"]["nDataOutCInternal"] = 1000000;
    j["outputControl"]["nDataOutInternal"] = 1000000;
    j["outputControl"]["dataOutAtInit"] = false;
    j["timeMarchControl"]["nTimeStep"] = 1;

    // Write final config
    if (g_mpi.rank == 0)
    {
        auto fOut = std::ofstream(cfgPath);
        DNDS_assert(fOut);
        fOut << std::setw(4) << j;
        fOut.close();
    }
    MPI::Barrier(g_mpi.comm);

    return cfgPath;
}

// ===================================================================
// Template: one Newton-like step for a given EulerModel
// Returns: {rhsL1Norm_density, rhsL1Norm_energy, incrementL2Norm}
// ===================================================================
template <EulerModel model>
std::array<real, 3> runOneNewtonStep(const std::string &cfgPath)
{
    using TSolver = EulerSolver<model>;
    using TEval = EulerEvaluator<model>;
    constexpr int nVarsFixed = TEval::nVarsFixed;
    constexpr int gDim = TEval::gDim;
    int nVars = getNVars(model);

    // --- Setup ---
    TSolver solver(g_mpi);
    solver.ConfigureFromJson(cfgPath, true);
    solver.ReadMeshAndInitialize();
    CHECK_FALSE(solver.hasLimitedVRStorage());
    const auto inactive = solver.getOutputOverlapDataRefs();
    CHECK_FALSE(inactive.lvrAlpha.father);
    CHECK_FALSE(inactive.lvrBranchAlpha.father);
    CHECK_FALSE(inactive.lvrO2Beta.father);

    auto &eval = solver.getEval();
    auto &u = solver.getU();
    auto &uRec = solver.getURec();
    auto &uRecNew = solver.getURecNew();
    auto vfv = solver.getVFV();
    auto mesh = solver.getMesh();
    auto &config = solver.getConfiguration();
    auto &JSource = solver.getJSource();
    auto &betaPP = solver.getBetaPP();

    // Build working arrays
    ArrayDOFV<nVarsFixed> rhs, xinc, xincNew;
    ArrayDOFV<1> dTau, alphaPP_tmp;
    JacobianDiagBlock<nVarsFixed> JD;

    vfv->BuildUDof(rhs, nVars);
    vfv->BuildUDof(xinc, nVars);
    vfv->BuildUDof(xincNew, nVars);
    vfv->BuildUDof(dTau, 1);
    vfv->BuildUDof(alphaPP_tmp, 1);
    alphaPP_tmp.setConstant(1.0);

    JD.SetModeAndInit(
        config.eulerSettings.useScalarJacobian ? 0 : 1,
        nVars, u);

    // --- Initialize DOF (not done by ReadMeshAndInitialize) ---
    eval.InitializeUDOF(u);
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();

    // --- STEP 1: fdt (compute local pseudo-timestep) ---
    real CFL = config.implicitCFLControl.CFL;
    real curDtMin = 1e100;

    eval.FixUMaxFilter(u);
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();

    eval.EvaluateDt(dTau, u, uRec, CFL, curDtMin, 1e100,
                    config.implicitCFLControl.useLocalDt, 0.0);

    // --- STEP 2: skip reconstruction (first-order, uRec stays zero) ---
    // For a proper high-order test we'd need the boundary callback from
    // frhsOuter, which requires significant setup. A first-order RHS
    // evaluation is a valid regression test of the evaluator pipeline.

    // --- STEP 3: frhs (evaluate spatial RHS, first-order) ---
    eval.EvaluateRHS(rhs, JSource, u, uRec, uRec,
                     betaPP, alphaPP_tmp, false, 0.0);
    rhs.trans.startPersistentPull();
    rhs.trans.waitPersistentPull();

    // --- Measure RHS L1 norm (volume-weighted) ---
    Eigen::VectorXd resNorm(nVars);
    eval.EvaluateNorm(resNorm, rhs, 1, true);

    if (g_mpi.rank == 0)
    {
        std::cout << "  RHS L1 norms:";
        for (int i = 0; i < nVars; i++)
            std::cout << " " << std::scientific << std::setprecision(10) << resNorm(i);
        std::cout << std::endl;
    }

    // --- STEP 4: fsolve (one Jacobi iteration -- MPI-deterministic) ---
    // Jacobi pattern: Forward(uInc, uIncNew); Backward(uInc, uIncNew); pull uIncNew; swap.
    // Since uInc starts at zero, this is a block-diagonal solve: xincNew = D^{-1} * rhs.
    real dt = 1e100; // pseudo-steady
    real alphaDiag = 1.0;

    eval.LUSGSMatrixInit(JD, JSource, dTau, dt, alphaDiag, u, uRec, 0, 0.0);

    xinc.setConstant(0.0);
    xincNew.setConstant(0.0);
    eval.UpdateLUSGSForward(alphaDiag, 0.0, rhs, u, xinc, JD, xincNew);
    eval.UpdateLUSGSBackward(alphaDiag, 0.0, rhs, u, xinc, JD, xincNew);
    xincNew.trans.startPersistentPull();
    xincNew.trans.waitPersistentPull();
    xinc = xincNew;

    // --- Measure increment L2 norm ---
    Eigen::VectorXd incNorm(nVars);
    eval.EvaluateNorm(incNorm, xinc, 2, false);
    real incL2 = incNorm.norm();

    if (g_mpi.rank == 0)
        std::cout << "  Inc L2 norm: " << std::scientific << std::setprecision(10) << incL2 << std::endl;

    return {resNorm(0), resNorm(nVars > 4 ? 4 : nVars - 1), incL2};
}

// ===================================================================
// TEST 1: Isentropic Vortex (NS, 2D, inviscid)
// ===================================================================

TEST_CASE("EulerEvaluator pipeline: IV (NS, P1, 2D)")
{
    std::string root = projectRoot();
    std::string cfgPath = writeTempConfig<NS>(
        root + "/cases/euler/euler_config_IV.json", "iv_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
         {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});

    if (g_mpi.rank == 0)
        std::cout << "=== IV (NS, P1) ===" << std::endl;

    auto [rhsDensity, rhsEnergy, incL2] = runOneNewtonStep<NS>(cfgPath);

    CHECK(std::isfinite(rhsDensity));
    CHECK(std::isfinite(rhsEnergy));
    CHECK(std::isfinite(incL2));
    CHECK(rhsDensity > 0);

    // Golden values (Jacobi iteration is MPI-deterministic)
    real goldenRhsDensity = 1.0524823780e+01;
    real goldenIncL2 = 3.1557730986e+00;

    if (g_mpi.rank == 0)
        std::cout << "  golden candidates: rhsDensity=" << std::scientific
                  << std::setprecision(10) << rhsDensity
                  << " incL2=" << incL2 << std::endl;

    CHECK(rhsDensity == doctest::Approx(goldenRhsDensity).epsilon(1e-6));
    CHECK(incL2 == doctest::Approx(goldenIncL2).epsilon(1e-6));
}

// ===================================================================
// TEST 2: NACA0012 (NS_SA, 2D, viscous + SA)
// ===================================================================

TEST_CASE("EulerEvaluator pipeline: NACA0012 (NS_SA, P1)")
{
    std::string root = projectRoot();
    std::string cfgPath = writeTempConfig<NS_SA>(
        root + "/cases/eulerSA/eulerSA_config.json", "naca_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
         {"restartState", {{"iStep", 0}, {"iStepInternal", 0}}},
         {"dataIOControl", {{"meshFile", root + "/data/mesh/NACA0012_H2.cgns"}}}});

    if (g_mpi.rank == 0)
        std::cout << "=== NACA0012 (NS_SA, P1) ===" << std::endl;

    auto [rhsDensity, rhsEnergy, incL2] = runOneNewtonStep<NS_SA>(cfgPath);

    CHECK(std::isfinite(rhsDensity));
    CHECK(std::isfinite(rhsEnergy));
    CHECK(std::isfinite(incL2));
    CHECK(rhsDensity > 0);

    real goldenRhsDensity = 2.4001929638e-01;
    real goldenIncL2 = 9.7941775117e+01;

    if (g_mpi.rank == 0)
        std::cout << "  golden candidates: rhsDensity=" << std::scientific
                  << std::setprecision(10) << rhsDensity
                  << " incL2=" << incL2 << std::endl;

    CHECK(rhsDensity == doctest::Approx(goldenRhsDensity).epsilon(1e-6));
    CHECK(incL2 == doctest::Approx(goldenIncL2).epsilon(1e-6));
}

// ===================================================================
// TEST 3: 3D Box (NS_3D, periodic, inviscid)
// ===================================================================

TEST_CASE("EulerEvaluator pipeline: Box (NS_3D, P1)")
{
    std::string root = projectRoot();
    std::string cfgPath = writeTempConfig<NS_3D>(
        root + "/cases/euler3D/euler3D_config_Box.json", "box3d_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}, {"cacheDiffBase", false}}},
         {"dataIOControl", {{"meshFile", root + "/data/mesh/Uniform32_3D_Periodic.cgns"}}}});

    if (g_mpi.rank == 0)
        std::cout << "=== Box (NS_3D, P1) ===" << std::endl;

    auto [rhsDensity, rhsEnergy, incL2] = runOneNewtonStep<NS_3D>(cfgPath);

    CHECK(std::isfinite(rhsDensity));
    CHECK(std::isfinite(rhsEnergy));
    CHECK(std::isfinite(incL2));
    CHECK(rhsDensity > 0);

    real goldenRhsDensity = 5.8855408029e-01;
    real goldenIncL2 = 3.4532977412e+01;

    if (g_mpi.rank == 0)
        std::cout << "  golden candidates: rhsDensity=" << std::scientific
                  << std::setprecision(10) << rhsDensity
                  << " incL2=" << incL2 << std::endl;

    CHECK(rhsDensity == doctest::Approx(goldenRhsDensity).epsilon(1e-6));
    CHECK(incL2 == doctest::Approx(goldenIncL2).epsilon(1e-6));
}

// Basic end-to-end LVR fixture: a two-dimensional state, target PP, all
// branches and two halo layers, compared cellwise against MPI_COMM_SELF.
// Keep this small; it is an integration regression, not a CFD benchmark.
using LVRCellKey = std::array<long long, 2>;
using LVRCellValues = std::vector<DNDS::real>;
using LVRCells = std::map<LVRCellKey, LVRCellValues>;

static LVRCells runLVRFixture(const std::string &cfgPath, const MPIInfo &mpi,
                              const std::string &state, O2ReferenceLimiter referenceLimiter, int extraShockLayers)
{
    EulerSolver<NS> solver(mpi);
    solver.ConfigureFromJson(cfgPath, true, "",
                             {"/limiterControl/limitedVR/alphaMax", "/limiterControl/limitedVR/preset"}, {"0.5", "v3"});
    solver.ReadMeshAndInitialize();
    REQUIRE(solver.hasLimitedVRStorage());
    auto vfv = solver.getVFV();
    auto mesh = solver.getMesh();
    auto &u = solver.getU();
    auto &target = solver.getURecO2();
    auto data = solver.getOutputOverlapDataRefs();
    auto settings = solver.getConfiguration().limiterControl.limitedVR;
    CHECK(settings.gateMode == LimitedVRGateMode::PressureExtremum);
    CHECK(settings.contactExtremumHaloLayers == 2);
    CHECK(settings.alphaMax == 0.5);
    settings.o2ReferenceLimiter = referenceLimiter;
    settings.gateHaloLayers = extraShockLayers;
    settings.weakAlphaMax = state == "shock" ? 0.5 : 0;
    using TU = EulerEvaluator<NS>::TU;
    for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
    {
        const auto x = vfv->GetCellQuadraturePPhys(i, -1);
        DNDS::real rho = 1, p = 1, vx = 0.2;
        // Variation in y makes this a genuinely multidimensional velocity field.
        DNDS::real vy = state == "uniform" ? 0.1 : 0.07 * std::sin(2 * M_PI * x(1) / 10);
        if (state == "shock")
        {
            p = x(0) < 5 ? 3 : 1;
            vx = x(0) < 5 ? 0.8 : 0;
            if (x(0) > 3 && x(0) < 4)
                p = 3.08; // a supported pressure-extremum stripe
        }
        if (state == "contact")
        {
            rho = x(0) < 5 ? 10 : 1;
            if (x(0) > 5 && x(0) < 6)
                rho = 11; // a supported density extremum
        }
        u[i] << rho, rho * vx, rho * vy, 0,
            p / 0.4 + 0.5 * rho * (vx * vx + vy * vy);
    }
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
    CFV::tScalarPair indicator;
    vfv->BuildScalar(indicator);
    CFV::VariationalReconstruction<2>::TFBoundary<5> boundary =
        [](const TU &ul, const TU &, DNDS::index, DNDS::index, int,
           const Geom::tPoint &, const Geom::tPoint &, Geom::t_index)
    { return ul; };
    solver.getEval().BuildLimitedVRO2(
        u, target, solver.getURecLimited(), solver.getURecNew(), indicator,
        data.lvrO2Beta, data.lvrAlpha, data.lvrBranchAlpha, data.lvrDensityJump,
        data.lvrPressureJump, data.lvrCompression, settings, true, true, boundary);
    LVRCells cells;
    // Total alpha, shock, contact, weak, and contact halo beyond the base gate.
    std::array<DNDS::real, 5> totals{};
    for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
    {
        const auto x = vfv->GetCellQuadraturePPhys(i, -1);
        CHECK(data.lvrAlpha[i](0) >= 0);
        CHECK(data.lvrAlpha[i](0) <= 0.5);
        CHECK(data.lvrAlpha[i](0) == doctest::Approx(
                                         std::max({data.lvrBranchAlpha[i](0), data.lvrBranchAlpha[i](1), data.lvrBranchAlpha[i](3)})));
        CHECK(data.lvrO2Beta[i](0) > 0);
        CHECK(data.lvrO2Beta[i](0) <= 1);
        CHECK(target[i].allFinite());
        CHECK(target[i].bottomRows(target[i].rows() - 2).norm() == 0);
        totals[0] += data.lvrAlpha[i](0);
        totals[1] += data.lvrBranchAlpha[i](0);
        totals[2] += data.lvrBranchAlpha[i](1);
        totals[3] += data.lvrBranchAlpha[i](3);
        if (data.lvrDensityJump[i](0) < settings.contactDensityJumpStart && data.lvrBranchAlpha[i](1) > 0)
            totals[4] += 1;
        LVRCellValues values{data.lvrAlpha[i](0), data.lvrBranchAlpha[i](0),
                             data.lvrBranchAlpha[i](1), data.lvrBranchAlpha[i](2),
                             data.lvrBranchAlpha[i](3), data.lvrO2Beta[i](0)};
        // Compare physical gradients rather than basis coefficients, whose
        // local basis orientation is an implementation detail.
        const Eigen::Matrix<DNDS::real, 2, 2> d1 =
            vfv->GetIntPointDiffBaseValue(i, -1, -1, -1,
                                          std::array<int, 2>{1, 2}, 3)(EigenAll, Eigen::seqN(0, 2));
        Eigen::Matrix<DNDS::real, 2, 5> gradient = d1 * target[i].topRows(2);
        values.insert(values.end(), gradient.data(), gradient.data() + gradient.size());
        cells[{std::llround(x(0) * 1e8), std::llround(x(1) * 1e8)}] = values;
    }
    auto globalTotals = totals;
    MPI::Allreduce(totals.data(), globalTotals.data(), 5, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
    if (state == "uniform")
        CHECK(globalTotals[0] == doctest::Approx(0).epsilon(1e-10).scale(1));
    if (state == "shock")
    {
        CHECK(globalTotals[1] > 0);
        CHECK(globalTotals[3] > 0);
    }
    if (state == "contact")
    {
        CHECK(globalTotals[2] > 0);
        CHECK(globalTotals[4] > 0);
    }

    if (state == "uniform" && referenceLimiter == O2ReferenceLimiter::Barth)
    {
        // Force a negative O2 face state, then exercise the mean-based PP path.
        for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
            target[i](0, 0) = 100;
        DNDS::index nLimited = 0;
        DNDS::real minBeta = 1;
        solver.getEval().EvaluateURecBeta(u, target, data.lvrO2Beta, nLimited, minBeta,
                                          EulerEvaluator<NS>::EvaluateURecBeta_COMPRESS_TO_MEAN |
                                              EulerEvaluator<NS>::EvaluateURecBeta_LOCAL_STATS);
        if (mesh->NumCell())
        {
            CHECK(nLimited > 0);
            CHECK(minBeta < 1);
        }
        for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
        {
            const auto faces = mesh->cell2face[i];
            for (DNDS::rowsize j = 0; j < faces.size(); ++j)
            {
                const auto face = faces[j];
                const int side = mesh->CellIsFaceBack(i, face, j) ? 0 : 1;
                const auto quad = vfv->GetFaceQuad(face);
                for (int q = 0; q < quad.GetNumPoints(); ++q)
                {
                    TU uq = u[i] + (vfv->GetIntPointDiffBaseValue(i, face, side, q, 0, 1) * target[i]).transpose();
                    CHECK(uq(0) > 0);
                    CHECK(uq(4) - 0.5 * uq.segment<3>(1).squaredNorm() / uq(0) > 0);
                }
            }
        }
        settings.alphaMax = settings.contactAlphaMax = settings.weakAlphaMax = 0;
        solver.getEval().BuildLimitedVRO2(
            u, target, solver.getURecLimited(), solver.getURecNew(), indicator,
            data.lvrO2Beta, data.lvrAlpha, data.lvrBranchAlpha, data.lvrDensityJump,
            data.lvrPressureJump, data.lvrCompression, settings, true, true, boundary);
        for (DNDS::index i = 0; i < mesh->NumCell(); ++i)
        {
            CHECK(data.lvrAlpha[i](0) == 0);
            CHECK(data.lvrBranchAlpha[i].isZero());
            CHECK(target[i].isZero());
            CHECK(data.lvrO2Beta[i](0) == 1);
        }
    }
    if (state == "uniform")
    {
        // Inactive output requests must remain defined without any LVR layouts.
        ArrayDOFV<1> emptyAlpha, emptyJump, emptyPressure, emptyCompression, emptyBeta;
        ArrayDOFV<4> emptyBranches;
        ArrayDOFV<1> unused;
        OutputPicker picker;
        solver.getEval().InitializeOutputPicker(picker,
                                                {u, solver.getURec(), solver.getBetaPP(), solver.getAlphaPP(),
                                                 emptyAlpha, emptyBranches, emptyJump, emptyPressure, emptyCompression, emptyBeta},
                                                {unused, unused, unused, unused, unused});
        auto fields = picker.getSubsetList({"lvrAlpha", "lvrWeakAlpha", "lvrO2Beta"});
        if (mesh->NumCell())
        {
            CHECK(std::get<1>(fields[0])(0) == 0);
            CHECK(std::get<1>(fields[1])(0) == 0);
            CHECK(std::get<1>(fields[2])(0) == 1);
        }
    }
    return cells;
}

TEST_CASE("LVR target and branch halos agree across 2-D MPI partitions")
{
    const std::string root = projectRoot();
    const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "lvr_2d",
                                         {{"vfvSettings", {{"maxOrder", 3}, {"intOrder", 5}, {"subs2ndOrder", 1}, {"cacheDiffBase", true}}},
                                          {"limiterControl", {{"useLimiter", true}, {"limiterProcedure", "LVR"}}},
                                          {"implicitReconstructionControl", {{"recLinearScheme", 0}}},
                                          {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}}}});
    // The fixture also exercises CLI preset expansion and explicit overrides.
    for (const std::string state : {"uniform", "shock", "contact"})
        for (const auto limiter : {O2ReferenceLimiter::Barth, O2ReferenceLimiter::WBAP})
            for (const int extraShockLayers : {0, 2})
            {
                CAPTURE(state);
                CAPTURE(nlohmann::ordered_json(limiter).dump());
                CAPTURE(extraShockLayers);
                // Exercise both the three-round H2 schedule and extra shock halos.
                // Compare cellwise against an independently built serial evaluator.
                auto serial = runLVRFixture(cfg, MPIInfo(MPI_COMM_SELF), state, limiter, extraShockLayers);
                auto parallel = runLVRFixture(cfg, g_mpi, state, limiter, extraShockLayers);
                for (const auto &[key, values] : parallel)
                {
                    REQUIRE(serial.count(key) == 1);
                    const auto &expected = serial.at(key);
                    REQUIRE(values.size() == expected.size());
                    for (size_t i = 0; i < values.size(); ++i)
                        CHECK(values[i] == doctest::Approx(expected[i]).epsilon(2e-10).scale(1));
                }
            }
}

// ===================================================================
// main with MPI
// ===================================================================
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    MPI_Finalize();
    return res;
}
