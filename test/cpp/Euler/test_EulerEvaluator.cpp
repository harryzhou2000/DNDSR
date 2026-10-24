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
 *   - Repeats from identical solution/halo data using native mesh authority,
 *     rebuilding FV/wall caches after corrupting the detached legacy workspace
 *   - Compares every owned dt, RHS and increment component between the runs
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
#include <hdf5.h>
#include <unistd.h>

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

    eval.FixUMaxFilter(u);
    u.trans.startPersistentPull();
    u.trans.waitPersistentPull();
    std::vector<Eigen::VectorXd> initialU, initialBeta;
    std::vector<Eigen::MatrixXd> initialRec, initialRecNew;
    for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
    {
        initialU.emplace_back(u[cell]);
        initialBeta.emplace_back(betaPP[cell]);
        initialRec.emplace_back(uRec[cell]);
        initialRecNew.emplace_back(uRecNew[cell]);
    }
    struct PipelineResult
    {
        std::array<real, 3> norms;
        std::vector<real> owned;
        real minDt;
    };
    auto run = [&]() -> PipelineResult
    {
        // Restore the complete initial halo as well as owned cells; neither
        // run may consume solution or reconstruction values left by the other.
        for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
        {
            u[cell] = initialU.at(cell);
            betaPP[cell] = initialBeta.at(cell);
            uRec[cell] = initialRec.at(cell);
            uRecNew[cell] = initialRecNew.at(cell);
        }
        alphaPP_tmp.setConstant(1.0);
        JSource.clearValues();
        JD.SetModeAndInit(config.eulerSettings.useScalarJacobian ? 0 : 1, nVars, u);
        // --- STEP 1: fdt (compute local pseudo-timestep) ---
        const real CFL = config.implicitCFLControl.CFL;
        real curDtMin = 1e100;

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

        PipelineResult result{{resNorm(0), resNorm(nVars > 4 ? 4 : nVars - 1), incL2}, {}, curDtMin};
        for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
        {
            result.owned.push_back(dTau[cell](0));
            for (int variable = 0; variable < nVars; ++variable)
                result.owned.push_back(rhs[cell](variable));
            for (int variable = 0; variable < nVars; ++variable)
                result.owned.push_back(xinc[cell](variable));
        }
        return result;
    };
    const auto legacy = run();
    auto requirements = Geom::MeshReadRequirements::CellFlux();
    requirements.allFaceCellsLocal = true;
    requirements.wallDistance = vfv->getSettings().functionalSettings.anisotropicType ==
                                CFV::VRSettings::FunctionalSettings::AnisotropicType::WallDist;
    mesh->PublishNative(requirements);
    REQUIRE(mesh->HasNativeState());

    // Numerical queries must remain independent of the detached raw workspace.
    // Corrupt before rebuilding FV and wall caches to expose mixed helper reads.
    for (DNDS::index node = 0; node < mesh->coords.Size(); ++node)
        mesh->coords[node].array() += 37;
    auto corruptNodes = [](auto &pair)
    {
        for (DNDS::index row = 0; row < pair.Size(); ++row)
            for (DNDS::rowsize slot = 0; slot < pair.RowSize(row); ++slot)
                pair(row, slot) = UnInitIndex;
    };
    corruptNodes(mesh->cell2node);
    corruptNodes(mesh->face2node);
    for (DNDS::index face = 0; face < mesh->face2cell.Size(); ++face)
    {
        mesh->face2cell(face, 0) = 0;
        mesh->face2cell(face, 1) = UnInitIndex;
        mesh->faceElemInfo(face, 0).zone = Geom::BC_ID_DEFAULT_WALL;
    }
    if (mesh->nodeWallDist.father && mesh->nodeWallDist.son)
        for (DNDS::index node = 0; node < mesh->nodeWallDist.Size(); ++node)
            mesh->nodeWallDist[node].setConstant(73);
    mesh->isPeriodic = !mesh->isPeriodic;
    mesh->periodicInfo = Geom::Periodicity{};
    TEval::InitializeFV(mesh, vfv, solver.getBCHandler());
    eval.GetWallDist();
    const auto native = run();
    REQUIRE(native.owned.size() == legacy.owned.size());
    CHECK(native.minDt == doctest::Approx(legacy.minDt).epsilon(1e-11));
    for (size_t value = 0; value < legacy.owned.size(); ++value)
    {
        CAPTURE(value);
        CHECK(std::isfinite(native.owned[value]));
        CHECK(native.owned[value] == doctest::Approx(legacy.owned[value]).epsilon(1e-11));
    }
    return legacy.norms;
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

TEST_CASE("RANS wall cache rejects changed geometry and metric publications")
{
    const auto root = projectRoot();
    const auto cfgPath = writeTempConfig<NS_SA>(
        root + "/cases/eulerSA/eulerSA_config.json", "wall_cache_test",
        {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
         {"restartState", {{"iStep", 0}, {"iStepInternal", 0}}},
         {"eulerSettings", {{"wallDistScheme", 0}}},
         {"dataIOControl", {{"meshFile", root + "/data/mesh/Uniform_3x3_wall.cgns"}, {"meshDirectBisect", 0}, {"meshScale", 1}, {"meshRotZ", 0}}}});
    EulerSolver<NS_SA> solver(g_mpi);
    solver.ConfigureFromJson(cfgPath, true);
    solver.ReadMeshAndInitialize();
    auto mesh = solver.getMesh();
    auto vfv = solver.getVFV();
    auto &eval = solver.getEval();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());
    const auto before = eval.dWall;
    DNDS::real total = 0;
    for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
        total += before.at(cell).mean();
    MPI::AllreduceOneReal(total, MPI_SUM, g_mpi);
    REQUIRE(total > 0);

    mesh->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2);
    vfv->ConstructMetrics();
    vfv->ConstructBaseAndWeight();
    vfv->ConstructRecCoeff();
    CHECK_NOTHROW(vfv->ValidateMeshDependencies());
    CHECK_THROWS(eval.ValidateWallDistanceDependencies());
    ArrayDOFV<1> dt;
    vfv->BuildUDof(dt, 1);
    DNDS::real minDt = 1e100;
    CHECK_THROWS(eval.EvaluateDt(dt, solver.getU(), solver.getURec(), 1, minDt, 1e100, true, 0));
    eval.GetWallDist();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());
    REQUIRE(eval.dWall.size() == before.size());
    for (size_t cell = 0; cell < before.size(); ++cell)
    {
        REQUIRE(eval.dWall[cell].size() == before[cell].size());
        for (Eigen::Index point = 0; point < before[cell].size(); ++point)
            CHECK(eval.dWall[cell](point) == doctest::Approx(2 * before[cell](point)).epsilon(1e-11));
    }

    // Rebuilding quadrature/metrics in the same mesh epoch also requires new
    // distances, even when the number of rows remains unchanged.
    vfv->FinalizeMetrics();
    vfv->ConstructBaseAndWeight();
    vfv->ConstructRecCoeff();
    CHECK_THROWS(eval.ValidateMeshDependencies());
    eval.GetWallDist();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());

    const auto scheme = eval.settings.wallDistScheme;
    eval.settings.wallDistScheme = -123;
    CHECK_THROWS(eval.GetWallDist());
    eval.settings.wallDistScheme = scheme;
    CHECK_THROWS(eval.ValidateWallDistanceDependencies());
    eval.GetWallDist();
    CHECK_NOTHROW(eval.ValidateMeshDependencies());
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
                             {"/limiterControl/limitedVR/alphaMax", "/limiterControl/limitedVR/contactAlphaMax", "/limiterControl/limitedVR/preset"},
                             {"0.5", "0.5", "v3"});
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

static std::vector<double> OutputDataset(const std::string &fileName, const char *datasetName)
{
    auto file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    DNDS_check_throw_info(file >= 0, "Euler output file missing");
    auto dataset = H5Dopen2(file, datasetName, H5P_DEFAULT);
    DNDS_check_throw_info(dataset >= 0, "Euler output dataset missing");
    auto space = H5Dget_space(dataset);
    auto count = H5Sget_simple_extent_npoints(space);
    DNDS_check_throw_info(count > 0, "Euler output dataset empty");
    std::vector<double> result(static_cast<size_t>(count));
    auto error = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());
    H5Sclose(space);
    H5Dclose(dataset);
    H5Fclose(file);
    DNDS_check_throw_info(error >= 0, "Euler output dataset read failed");
    return result;
}

TEST_CASE("Euler distributed volume output reads native authority after workspace edits")
{
    const auto root = projectRoot();
    const auto cfg = writeTempConfig<NS>(root + "/cases/euler/euler_config_IV.json", "native_output",
                                         {{"vfvSettings", {{"maxOrder", 1}, {"intOrder", 3}}},
                                          {"limiterControl", {{"useLimiter", false}}},
                                          {"dataIOControl", {{"meshDirectBisect", 0}, {"meshFile", root + "/data/mesh/IV10_10.cgns"}, {"outPltMode", 1}, {"outBndData", false}, {"outVolumeData", true}, {"outPltTecplotFormat", false}, {"outPltVTKFormat", false}, {"outPltVTKHDFFormat", true}, {"outAtCellData", true}, {"outAtPointData", true}, {"allowAsyncPrintData", false}, {"outCellScalarNames", nlohmann::ordered_json::array()}}}});
    EulerSolver<NS> solver(g_mpi);
    solver.ConfigureFromJson(cfg, true);
    solver.ReadMeshAndInitialize();
    auto &eval = solver.getEval();
    eval.InitializeUDOF(solver.getU());
    auto mesh = solver.getMesh();
    for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
    {
        solver.getURec()[cell].setZero();
        solver.getURecNew()[cell].setZero();
    }
    int rootPid = int(getpid());
    MPI_Bcast(&rootPid, 1, MPI_INT, 0, g_mpi.comm);
    auto base = (std::filesystem::temp_directory_path() /
                 ("dndsr-euler-native-output-" + std::to_string(rootPid)))
                    .string();
    EulerSolver<NS>::tAdditionalCellScalarList cells, boundaries;
    auto output = [&](const std::string &suffix)
    { solver.PrintData(base + suffix, "", [](DNDS::index)
                       { return 0.0; }, cells, boundaries, eval, 0); };
    output("-legacy");
    auto requirements = Geom::MeshReadRequirements::CellFlux();
    requirements.allFaceCellsLocal = true;
    mesh->PublishNative(requirements);
    for (DNDS::index node = 0; node < mesh->coords.Size(); ++node)
        mesh->coords[node].setConstant(731);
    EulerEvaluator<NS>::InitializeFV(mesh, solver.getVFV(), solver.getBCHandler());
    output("-native");
    CHECK(mesh->HasNativeState());
    MPI_Barrier(g_mpi.comm);
    if (g_mpi.rank == 0)
    {
        for (const auto *name : {"/VTKHDF/Points", "/VTKHDF/CellData/R", "/VTKHDF/CellData/P", "/VTKHDF/CellData/Velo",
                                 "/VTKHDF/PointData/R", "/VTKHDF/PointData/P", "/VTKHDF/PointData/Velo"})
        {
            CAPTURE(name);
            const auto legacy = OutputDataset(base + "-legacy.vtkhdf", name);
            const auto native = OutputDataset(base + "-native.vtkhdf", name);
            REQUIRE(native.size() == legacy.size());
            for (size_t i = 0; i < native.size(); ++i)
                CHECK(native[i] == doctest::Approx(legacy[i]).epsilon(1e-11));
        }
        std::filesystem::remove(base + "-legacy.vtkhdf");
        std::filesystem::remove(base + "-native.vtkhdf");
    }
    MPI_Barrier(g_mpi.comm);
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
