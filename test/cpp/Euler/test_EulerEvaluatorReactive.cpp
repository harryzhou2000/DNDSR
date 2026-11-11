/**
 * @file test_EulerEvaluatorReactive.cpp
 * @brief Reactive EulerEvaluator cell-mean repair tests.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "Euler/EulerSolver.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

using namespace DNDS;
using namespace DNDS::Euler;

static MPIInfo g_mpi;

static std::string projectRoot()
{
    std::string f(__FILE__);
    for (int i = 0; i < 4; ++i)
    {
        auto pos = f.rfind('/');
        if (pos == std::string::npos)
            pos = f.rfind('\\');
        if (pos != std::string::npos)
            f = f.substr(0, pos);
    }
    return f;
}

static std::string writeReactiveConfig()
{
    constexpr int nVars = 14;
    const std::string root = projectRoot();
    const auto tmpDir = std::filesystem::temp_directory_path() / "dndsr_euler_test";
    std::filesystem::create_directories(tmpDir);
    // CTest may execute the np=1/2/4/8 registrations concurrently.  Keep the
    // generated defaults and merged cases disjoint between MPI worlds; sharing
    // fixed paths lets one solver parse another world's partially written file.
    const std::string worldSuffix = "_np" + std::to_string(g_mpi.size);
    const std::string defaultPath = (tmpDir / ("reactive_repair_default" + worldSuffix + ".json")).string();
    const std::string configPath = (tmpDir / ("reactive_repair" + worldSuffix + ".json")).string();

    {
        EulerSolver<NS_EX> solver(g_mpi, nVars);
        solver.ConfigureFromJson(defaultPath, false);
    }
    MPI::Barrier(g_mpi.comm);

    auto fDefault = std::ifstream(defaultPath);
    REQUIRE(fDefault);
    auto config = nlohmann::ordered_json::parse(fDefault, nullptr, true, true);

    auto fCase = std::ifstream(root + "/cases/eulerEX/react_test.json");
    REQUIRE(fCase);
    auto caseConfig = nlohmann::ordered_json::parse(fCase, nullptr, true, true);
    config.merge_patch(caseConfig);

    config["dataIOControl"]["meshFile"] = root + "/data/mesh/IV10_10.cgns";
    config["eulerSettings"]["reactiveFlow"]["mechanismFile"] = root + "/cases/eulerEX/h2o2.yaml";
    config["outputControl"]["dataOutAtInit"] = false;
    config["outputControl"]["nDataOut"] = 1000000;
    config["outputControl"]["nDataOutC"] = 1000000;

    if (g_mpi.rank == 0)
    {
        auto fOut = std::ofstream(configPath);
        REQUIRE(fOut);
        fOut << std::setw(4) << config;
    }
    MPI::Barrier(g_mpi.comm);
    return configPath;
}

TEST_CASE("Physical source splitting rejects globally disabled sources")
{
    EulerSolver<NS_EX>::Configuration config(14);
    config.eulerSettings.ignoreSourceTerm = false;
    CHECK(config.validate().empty());
    for (const int splitting : {1, 2})
    {
        config.timeMarchControl.sourceStrangSplitting = splitting;
        CHECK(config.validate().empty());
        config.eulerSettings.ignoreSourceTerm = true;
        const auto failures = config.validate();
        REQUIRE(failures.size() == 1);
        CHECK(failures.front().message == "physical source splitting requires ignoreSourceTerm=false");
        config.eulerSettings.ignoreSourceTerm = false;
    }
    config.timeMarchControl.sourceStrangSplitting = 0;
    config.eulerSettings.ignoreSourceTerm = true;
    CHECK(config.validate().empty());
}

TEST_CASE("EulerEvaluator repairs reactive cell-mean species")
{
    constexpr int nVars = 14;
    EulerSolver<NS_EX> solver(g_mpi, nVars);
    solver.ConfigureFromJson(writeReactiveConfig(), true);
    solver.ReadMeshAndInitialize();

    auto &eval = solver.getEval();
    auto &u = solver.getU();
    auto mesh = solver.getMesh();
    eval.InitializeUDOF(u);

    const int Ns = eval.phys().nSpecies();
    const int Ns1 = Ns - 1;
    const int Isp = nVars - Ns1;
    REQUIRE(Ns == 10);
    REQUIRE(Isp == 5);

    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        auto uCell = u[iCell];
        const real rho = uCell(0);
        uCell(Isp + 1) = -rho * 1e-12;
        uCell(Isp + 8) = rho * (0.75 + 4.3811e-10);
    }

    eval.RepairCellMeanState(u);

    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        auto uCell = u[iCell];
        const real rho = uCell(0);
        real sumRhoY = 0;
        for (int k = 0; k < Ns1; ++k)
        {
            CHECK(std::isfinite(uCell(Isp + k)));
            CHECK(uCell(Isp + k) > 0);
            sumRhoY += uCell(Isp + k);
        }
        CHECK(sumRhoY < rho);
        CHECK((rho - sumRhoY) / rho > 1e-15);

        std::vector<double> Y(static_cast<size_t>(Ns));
        Chemistry::ConstSpeciesBufferView rhoYView{uCell.data() + Isp, Ns1};
        Chemistry::SpeciesBufferView YView{Y.data(), Ns};
        CHECK_NOTHROW(eval.phys().chem().massFractions(
            rho, rhoYView, YView));
        CHECK(Y.back() > 0);

        const real T = eval.phys().temperature(uCell);
        CHECK(std::isfinite(T));
        CHECK(eval.phys().toPhysT(T) >= eval.phys().temperatureFloor());
    }

    // CompressInc must return the exact increment that can be stored. Before
    // the projection-order fix this result could remain outside the simplex;
    // AddFixedIncrement repaired it afterwards and could invalidate the energy
    // line search that had just succeeded.
    if (mesh->NumCell() > 0)
    {
        EulerEvaluator<NS_EX>::TU old = u[0];
        EulerEvaluator<NS_EX>::TU requested = old;
        requested(Isp + 1) = -old(0) * 1e-12;
        requested(Isp + 8) = old(0) * (0.75 + 4.3811e-10);
        // Materialize the Eigen expression: `auto` would retain a reference to
        // CompressInc's temporary return value and make this check undefined.
        const EulerEvaluator<NS_EX>::TU accepted = old + eval.CompressInc(old, requested - old);
        real acceptedSpecies = 0;
        for (int k = 0; k < Ns1; ++k)
        {
            CHECK(std::isfinite(accepted(Isp + k)));
            CHECK(accepted(Isp + k) > 0);
            acceptedSpecies += accepted(Isp + k);
        }
        CHECK(acceptedSpecies < accepted(0));
        const real acceptedKinetic = .5 * accepted(Eigen::seqN(1, 3)).squaredNorm() / accepted(0);
        CHECK(accepted(4) - acceptedKinetic - eval.phys().mixtureBaseInternalRhoE(accepted) > 0);
    }

    // A species-only pointwise simplex violation must not downcast the shared
    // flow/species polynomial base. The temporary point-property path repairs
    // composition, while the thermally admissible high-order coefficient is
    // retained for the common beta calculation.
    if (mesh->NumCell() > 0)
    {
        auto vr = solver.getVFV();
        ArrayRECV<EulerEvaluator<NS_EX>::nVarsFixed> rec;
        ArrayDOFV<1> beta;
        vr->BuildURec(rec, nVars);
        vr->BuildUDof(beta, 1);
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            rec[iCell].setZero();
        int selectedSpecies = 0;
        for (int k = 1; k < Ns1; ++k)
            if (u[0](Isp + k) < u[0](Isp + selectedSpecies))
                selectedSpecies = k;
        const int selectedMode = int(rec[0].rows()) - 1;
        real selectedBasis = 0;
        const auto faces = mesh->CellFaces(0);
        for (DNDS::rowsize j = 0; j < faces.size() && selectedBasis == 0; ++j)
        {
            const auto face = faces[j];
            const int side = mesh->CellIsFaceBack(0, face, j) ? 0 : 1;
            for (int q = 0; q < vr->GetFaceQuad(face).GetNumPoints(); ++q)
            {
                const auto basis = vr->GetIntPointDiffBaseValue(0, face, side, q, 0, 1);
                if (std::abs(basis(0, selectedMode)) > 1e-8)
                {
                    selectedBasis = basis(0, selectedMode);
                    break;
                }
            }
        }
        REQUIRE(selectedBasis != 0);
        const real selectedCoefficient = -2 * u[0](Isp + selectedSpecies) / selectedBasis;
        rec[0](selectedMode, Isp + selectedSpecies) = selectedCoefficient;
        CHECK(u[0](Isp + selectedSpecies) + selectedBasis * selectedCoefficient < 0);
        DNDS::index nLimited = 0;
        real minimumBeta = 1;
        eval.EvaluateURecBeta(u, rec, beta, nLimited, minimumBeta,
                              EulerEvaluator<NS_EX>::EvaluateURecBeta_LOCAL_STATS);
        CHECK(nLimited == 0);
        CHECK(beta[0](0) == 1);
        CHECK(rec[0](selectedMode, Isp + selectedSpecies) == selectedCoefficient);
    }

    // RepairCellMeanState refreshes halos after local validation.  A negative
    // test that intentionally throws before that collective is therefore a
    // single-rank contract test, not a distributed recovery protocol.
    if (g_mpi.size == 1)
    {
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            u[iCell](0) = 0;
        CHECK_THROWS_AS(eval.RepairCellMeanState(u), std::runtime_error);

        eval.InitializeUDOF(u);
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            u[iCell](4) = std::numeric_limits<real>::quiet_NaN();
        CHECK_THROWS_AS(eval.RepairCellMeanState(u), std::runtime_error);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();

    MPI_Finalize();
    return result;
}
