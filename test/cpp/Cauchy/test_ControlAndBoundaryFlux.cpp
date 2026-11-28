#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/IO/BoundaryName.hpp"
#include "Cauchy/IO/ControlServer.hpp"
#include "Cauchy/IO/OutputCoordinator.hpp"
#include "Cauchy/IO/RunControl.hpp"

#include <httplib.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#    include <sys/stat.h>
#    include <unistd.h>
#else
#    include <process.h>
#endif

namespace
{
    namespace C = DNDS::Cauchy;
    DNDS::MPIInfo mpi;

    std::string UniqueTempDir(const std::string &tag)
    {
        int pid = 0;
        static int seq = 0;
        if (mpi.rank == 0)
        {
#ifdef _WIN32
            pid = _getpid();
#else
            pid = int(getpid());
#endif
        }
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        const auto dir = std::filesystem::temp_directory_path() /
                         ("cauchy-ctrl-" + tag + "-" + std::to_string(pid) + "-" + std::to_string(++seq));
        if (mpi.rank == 0)
            std::filesystem::create_directories(dir);
        MPI_Barrier(mpi.comm);
        return dir.string();
    }

    void RemoveDir(const std::string &dir)
    {
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
        {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
        MPI_Barrier(mpi.comm);
    }

    std::vector<std::string> ReadCsvLines(const std::string &path)
    {
        std::ifstream in(path);
        REQUIRE(bool(in));
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line))
            if (!line.empty())
                lines.push_back(line);
        return lines;
    }

    void InitializeVortexLike(C::Domain &domain)
    {
        auto &means = domain.Dofs().PhysicalField(0).Owned();
        auto write = means.Write();
        auto view = write.View<DNDS::Runtime::TableRows>();
        for (DNDS::index i = 0; i < view.RowCount(); ++i)
        {
            const DNDS::real rho = 1.0 + 0.05 * std::sin(DNDS::real(i + 1));
            const DNDS::real u = 0.25 + 0.02 * std::cos(DNDS::real(i + 1));
            const DNDS::real v = 0.05;
            const DNDS::real p = 1.0;
            view(i, 0) = rho;
            view(i, 1) = rho * u;
            view(i, 2) = rho * v;
            view(i, 3) = 0.0;
            view(i, 4) = p / 0.4 + 0.5 * rho * (u * u + v * v);
        }
    }
}

TEST_CASE("BC name canonicalization produces shell-safe, collision-free names")
{
    CHECK(C::CanonicalizeBoundaryName("Wall / Upper [1]: test", 7) == "Wall_Upper_1_test");
    CHECK(C::CanonicalizeBoundaryName("   ///   ", 12) == "zone_12");
    CHECK(C::CanonicalizeBoundaryName("", 5) == "zone_5");
    CHECK(C::CanonicalizeBoundaryName("CON", 7) == "bc_CON");
    CHECK(C::CanonicalizeBoundaryName("nul.txt", 8) == "bc_nul.txt");
    const std::string longName(90, 'a');
    CHECK(C::CanonicalizeBoundaryName(longName, 9).size() == 64);

    C::DomainSettings settings;
    settings.boundaries["7"].kind = "noSlipWall";
    settings.boundaries["7"].integrateFlux = true;
    settings.boundaries["8"].kind = "slipWall";
    settings.boundaries["8"].integrateFlux = true;
    settings.boundaries["9"].kind = "farfield";
    settings.boundaries["9"].integrateFlux = false;
    settings.mesh.boundaryIds["Wall A"] = 7;
    settings.mesh.boundaryIds["wall_a"] = 8; // case-insensitive canonical collision!

    const auto specs = C::ResolveBoundaryIntegrationSpecs(settings);
    REQUIRE(specs.size() == 2);
    CHECK(specs[0].zone == 7);
    CHECK(specs[1].zone == 8);
    CHECK(specs[0].canonicalName == "Wall_A_z7");
    CHECK(specs[1].canonicalName == "wall_a_z8");
    CHECK(specs[0].canonicalName != specs[1].canonicalName);
}

TEST_CASE("RunControl live config allow-list separates output-only and steering categories")
{
    const C::Json input{
        {"defaults", {{"mesh", {{"dimension", 2}}}, {"implicit", {{"iterations", 5}, {"minIterations", 2}, {"pseudoCFL", 10.0}, {"maxPseudoCFL", 100.0}}}}},
        {"time", {{"method", "bdf1"}, {"steady", false}, {"dt", 0.01}, {"steps", 4}}},
        {"output", {{"prefix", "test"}, {"control", {{"enabled", false}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::RunControl control(config, mpi.rank);

    const auto schema = control.MutableSchema();
    REQUIRE(schema.contains("categories"));
    CHECK(schema["categories"]["output"].is_array());
    CHECK(schema["categories"]["steering"].is_array());
    CHECK(schema["categories"]["output"].size() >= 10);
    CHECK(schema["categories"]["steering"].size() >= 15);

    // Output-only patch
    {
        const auto [ok, rec] = control.EnqueueConfigPatch(
            C::Json{{"/output/checkpointInterval", 5}, {"/output/implicit/historyInterval", 2}},
            C::ControlBoundary::Iteration);
        CHECK(ok);
        CHECK(rec.category == "output");
        CHECK(rec.applyAt == "iteration");
        CHECK(rec.state == "pending");
    }

    // Steering patch with single-domain alias and maxWallSeconds alias
    {
        const auto [ok, rec] = control.EnqueueConfigPatch(
            C::Json{{"/time/steps", 8}, {"/time/maxWallSeconds", 300.0}, {"/implicit/pseudoCFL", 25.0}},
            C::ControlBoundary::Step);
        CHECK(ok);
        CHECK(rec.category == "steering");
        CHECK(rec.patch.contains("/time/wallSeconds"));
        CHECK(rec.patch.contains("/domains/0/settings/implicit/pseudoCFL"));
    }

    // Immutable structural parameter rejected
    {
        const auto [ok, rec] = control.EnqueueConfigPatch(
            C::Json{{"/domains/0/settings/mesh/dimension", 3}}, C::ControlBoundary::Step);
        CHECK_FALSE(ok);
        CHECK(rec.state == "rejected");
    }

    // Cross-field constraint rejected (minIterations > iterations)
    {
        const auto [ok, rec] = control.EnqueueConfigPatch(
            C::Json{{"/implicit/minIterations", 10}}, C::ControlBoundary::Iteration);
        CHECK_FALSE(ok);
        CHECK(rec.state == "rejected");
    }
}

TEST_CASE("ControlServer scans from 53000, handles bind failure non-fatally, and enforces bearer auth")
{
    const auto dir = UniqueTempDir("http");
    const C::Json input{
        {"defaults", {{"mesh", {{"dimension", 2}}}}},
        {"time", {{"method", "bdf1"}, {"steady", false}, {"dt", 0.01}, {"steps", 2}}},
        {"output", {{"prefix", dir + "/runA"}, {"control", {{"enabled", true}, {"host", "127.0.0.1"}, {"port", 53000}, {"portScanCount", 40}, {"registryDir", dir + "/registry"}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::RunControl controlA(config, mpi.rank);
    C::RunControl controlB(config, mpi.rank);

    C::ControlServer serverA(controlA, config.Output().control, dir + "/runA", dir + "/runA-run-manifest.json", mpi.rank);
    const bool startedA = serverA.Start();
    if (mpi.rank == 0)
    {
        CHECK(startedA);
        CHECK(serverA.Running());
        CHECK(serverA.Port() >= 53000);
        CHECK(std::filesystem::exists(serverA.ControlFilePath()));
        CHECK(std::filesystem::exists(serverA.RegistryFilePath()));
#ifndef _WIN32
        struct stat st
        {
        };
        CHECK(::stat(serverA.ControlFilePath().c_str(), &st) == 0);
        CHECK((st.st_mode & 0777) == 0600);
#endif
        // Second server requesting the same starting port must advance to next free port.
        auto settingsB = config.Output().control;
        settingsB.port = serverA.Port();
        C::ControlServer serverB(controlB, settingsB, dir + "/runB", dir + "/runB-run-manifest.json", mpi.rank);
        CHECK(serverB.Start());
        CHECK(serverB.Port() > serverA.Port());
        serverB.Stop();

        // Unbindable host fails gracefully without throwing.
        auto badSettings = config.Output().control;
        badSettings.host = "256.256.256.256";
        badSettings.registerRun = false;
        C::ControlServer badServer(controlB, badSettings, "", "", mpi.rank);
        CHECK_FALSE(badServer.Start());
        CHECK_FALSE(badServer.Running());
        CHECK_FALSE(badServer.FailureReason().empty());

        // Auth checks over real HTTP
        httplib::Client client(serverA.Host(), serverA.Port());
        auto noAuth = client.Get("/api/v1/status");
        CHECK(bool(noAuth));
        if (noAuth)
            CHECK(noAuth->status == 401);

        httplib::Headers badHeaders{{"Authorization", "Bearer wrong-token"}};
        auto wrongAuth = client.Get("/api/v1/status", badHeaders);
        CHECK(bool(wrongAuth));
        if (wrongAuth)
            CHECK(wrongAuth->status == 401);

        httplib::Headers goodHeaders{{"Authorization", "Bearer " + serverA.Token()}};
        auto okStatus = client.Get("/api/v1/status", goodHeaders);
        CHECK(bool(okStatus));
        if (okStatus)
        {
            CHECK(okStatus->status == 200);
            const auto statusJson = C::Json::parse(okStatus->body);
            CHECK(statusJson.at("endpoint").at("port") == serverA.Port());
        }

        auto okConfig = client.Get("/api/v1/config", goodHeaders);
        CHECK(bool(okConfig));
        if (okConfig)
        {
            CHECK(okConfig->status == 200);
            const auto cfgJson = C::Json::parse(okConfig->body);
            CHECK(cfgJson.at("mutableSchema").at("categories").contains("output"));
            CHECK(cfgJson.at("mutableSchema").at("categories").contains("steering"));
        }

        auto patchResp = client.Patch(
            "/api/v1/config", goodHeaders,
            C::Json{{"applyAt", "iteration"}, {"patch", {{"/output/implicit/historyInterval", 3}}}}.dump(),
            "application/json");
        CHECK(bool(patchResp));
        if (patchResp)
        {
            CHECK(patchResp->status == 202);
            CHECK(C::Json::parse(patchResp->body).at("category") == "output");
        }

        const auto regPath = serverA.RegistryFilePath();
        serverA.Stop();
        CHECK_FALSE(serverA.Running());
        CHECK_FALSE(std::filesystem::exists(regPath));
        std::ifstream cf(serverA.ControlFilePath());
        C::Json inactive;
        cf >> inactive;
        CHECK(inactive.at("active") == false);
        CHECK_FALSE(inactive.contains("token"));
    }
    else
    {
        CHECK_FALSE(startedA);
        CHECK_FALSE(serverA.Running());
    }
    RemoveDir(dir);
}

TEST_CASE("Collective live steering and iteration-vs-step stop work across MPI ranks for unsteady runs")
{
    const auto dir = UniqueTempDir("steer");
    const auto prefix = dir + "/unsteady";
    const C::Json input{
        {"defaults",
         {{"mesh", {{"dimension", 2}}},
          {"numerics", {{"reconstruction", "variational"}, {"cfv", {{"maxOrder", 2}}}}},
          {"implicit", {{"iterations", 6}, {"minIterations", 6}, {"relativeTolerance", 1e-12}, {"pseudoCFL", 10.0}, {"maxPseudoCFL", 200.0}, {"pseudoGrowth", 1.0}}}}},
        {"time", {{"method", "bdf1"}, {"steady", false}, {"dt", 0.002}, {"steps", 3}}},
        {"output", {{"prefix", prefix}, {"finalSnapshot", false}, {"implicit", {{"historyInterval", 1}, {"consoleInterval", 0}}}, {"control", {{"enabled", false}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto factory = [](DNDS::index, const auto &, const DNDS::MPIInfo &domainMPI)
    { return CauchyTest::PeriodicQuads(domainMPI, 4); };

    C::Simulation simulation(config, mpi.comm, factory);
    C::Simulation reference(config, mpi.comm, factory);
    InitializeVortexLike(simulation.GetDomain(0));
    InitializeVortexLike(reference.GetDomain(0));

    C::RunControl control(config, mpi.rank);
    int pollCount = 0;
    {
        C::OutputCoordinator outputs(simulation, config, mpi, MPI_Wtime(), &control, [&]
                                     {
            ++pollCount;
            if (mpi.rank == 0 && pollCount == 2)
            {
                // Apply steering update at iteration boundary + output update at step boundary
                auto [ok1, r1] = control.EnqueueConfigPatch(
                    C::Json{{"/implicit/pseudoCFL", 42.0}}, C::ControlBoundary::Iteration);
                REQUIRE(ok1);
                auto [ok2, r2] = control.EnqueueConfigPatch(
                    C::Json{{"/time/steps", 5}}, C::ControlBoundary::Step);
                REQUIRE(ok2);
            }
            if (mpi.rank == 0 && pollCount == 4)
                control.EnqueueStop(C::ControlBoundary::Iteration); });

        CHECK(outputs.PollStop() == C::RunStop::None);
        outputs.BeginAttempt(0.002, 0, 0.002);
        const auto outcome = simulation.Advance(0.002);
        outputs.FlushPending();
        // Unsteady run interrupted cleanly at iteration boundary!
        CHECK(outcome.interrupted);
        CHECK_FALSE(outcome.accepted);
        CHECK(simulation.Step() == 0);
        // Step-boundary patch (/time/steps) has not yet been applied until PollStop()
        CHECK(outputs.LiveTime().steps == 3);
        CHECK(outputs.PollStop() == C::RunStop::Requested);
        // Step-boundary patch applied cleanly on PollStop!
        CHECK(outputs.LiveTime().steps == 5);
        CHECK(outputs.Finalize(C::RunStop::Requested, false) == 0);

        if (mpi.rank == 0)
        {
            const auto hist = control.CurrentHistory(64);
            bool sawSteeredCFL = false;
            for (const auto &rec : hist.at("records"))
                if (rec.at("values").contains("pseudo_cfl") &&
                    std::abs(rec.at("values").at("pseudo_cfl").get<double>() - 42.0) < 1e-12)
                    sawSteeredCFL = true;
            CHECK(sawSteeredCFL);
        }
    }

    if (mpi.rank == 0)
    {
        std::ifstream sf(prefix + "-summary.json");
        REQUIRE(bool(sf));
        C::Json summary;
        sf >> summary;
        CHECK(summary.at("status") == "graceful-stop");
        CHECK(summary.at("stopBoundary") == "iteration");
        CHECK(std::filesystem::exists(summary.at("checkpoint").get<std::string>()));
    }
    RemoveDir(dir);
}

TEST_CASE("Per-boundary flux and tangential force CSV matches Euler sign, MPI invariance, and restart continuation")
{
    const auto dir = UniqueTempDir("bcflux");
    const auto prefix = dir + "/channel";
    const C::Json input{
        {"defaults",
         {{"mesh", {{"dimension", 3}, {"boundaryIds", {{"NoSlip Wall / Top", 7}, {"Slip Wall [Bottom]", 8}}}}},
          {"equation", {{"system", "compressible"}, {"eos", "perfectGas"}, {"transport", "NewtonianFourier"}, {"viscosity", 0.02}}},
          {"boundaries",
           {{"7", {{"kind", "noSlipWall"}, {"integrateFlux", true}}},
            {"8", {{"kind", "slipWall"}, {"integrateFlux", true}}}}},
          {"initialPrimitive", {1.0, 0.3, 0.05, 0.0, 1.0}},
          {"numerics", {{"reconstruction", "variational"}, {"cfv", {{"maxOrder", 1}, {"SORInstead", false}}}, {"viscousGradient", "polynomial"}}},
          {"implicit", {{"iterations", 3}, {"minIterations", 3}, {"relativeTolerance", 1e-12}}}}},
        {"time", {{"method", "bdf1"}, {"steady", false}, {"dt", 0.001}, {"steps", 2}}},
        {"output",
         {{"prefix", prefix},
          {"checkpointInterval", 1},
          {"visualizationInterval", 0},
          {"finalSnapshot", false},
          {"forceZones", {7}},
          {"implicit", {{"historyInterval", 1}, {"consoleInterval", 0}}},
          {"control", {{"enabled", false}}}}}};

    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto factory = [](DNDS::index, const auto &, const DNDS::MPIInfo &domainMPI)
    { return CFVTest::CartesianHexes(domainMPI, 2, true); };

    C::Simulation simulation(config, mpi.comm, factory);
    // Direct check of Euler sign convention vs LastBoundaryForce before stepping
    auto &domain = simulation.GetDomain(0);
    domain.Residual();
    const auto diag = domain.Diagnostics(config.Domains()[0].settings, {7});
    const auto integrals = domain.MeasureBoundaryFluxes({7, 8});
    REQUIRE(integrals.size() == 2);
    // Unit cube: zone 7 has 5 faces (area 5.0), zone 8 has 1 face (area 1.0), total 6.0
    CHECK(integrals[0].area == doctest::Approx(5.0).epsilon(1e-12));
    CHECK(integrals[1].area == doctest::Approx(1.0).epsilon(1e-12));
    // Euler sign convention: faceFlux is negative outward RHS flux = -LastBoundaryForce
    for (int d = 0; d < 3; ++d)
        CHECK(integrals[0].flux[size_t(1 + d)] == doctest::Approx(-diag[size_t(5 + d)]).epsilon(1e-12).scale(1.0));
    // Slip wall has zero tangential shear; no-slip wall has nonzero shear
    CHECK(std::abs(integrals[1].tangentialForce[0]) < 1e-11);
    CHECK(std::abs(integrals[1].tangentialForce[1]) < 1e-11);
    CHECK(std::abs(integrals[1].tangentialForce[2]) < 1e-11);
    CHECK(std::hypot(integrals[0].tangentialForce[0], integrals[0].tangentialForce[1]) > 1e-6);

    // Run 1 step with OutputCoordinator -> writes per-BC CSVs + checkpoint at step 1
    {
        C::RunControl control(config, mpi.rank);
        C::OutputCoordinator outputs(simulation, config, mpi, MPI_Wtime(), &control);
        outputs.BeginAttempt(0.001, 0, 0.001);
        REQUIRE(simulation.Advance(0.001).accepted);
        outputs.FlushPending();
        outputs.CompleteAcceptedStep(0.001, 0, {}, false, false);
        CHECK(outputs.Finalize(C::RunStop::None, false) == 0);
    }

    const std::string noslipCsv = prefix + "-D0-bc-NoSlip_Wall_Top.csv";
    const std::string slipCsv = prefix + "-D0-bc-Slip_Wall_Bottom.csv";
    size_t rowsAfterStep1 = 0;
    if (mpi.rank == 0)
    {
        CHECK(std::filesystem::exists(noslipCsv));
        CHECK(std::filesystem::exists(slipCsv));
        const auto lines = ReadCsvLines(noslipCsv);
        // Header + initial step-0 residual row (since Residual() ran before OutputCoordinator) + 3 nonlinear iterations
        CHECK(lines.size() == 5);
        rowsAfterStep1 = lines.size() - 1;
        CHECK(lines[0].find("area") != std::string::npos);
        CHECK(lines[0].find("flux_rho") != std::string::npos);
        CHECK(lines[0].find("force_t_x") != std::string::npos);

        // Verify manifest publishes canonicalized boundary integration descriptors
        std::ifstream mf(prefix + "-run-manifest.json");
        CHECK(bool(mf));
        if (mf)
        {
            C::Json manifest;
            mf >> manifest;
            const auto &bnds = manifest.at("domains").at(0).at("boundaryIntegrations");
            CHECK(bnds.size() == 2);
            if (bnds.size() == 2)
            {
                CHECK(bnds[0].at("canonicalName") == "NoSlip_Wall_Top");
                CHECK(bnds[1].at("canonicalName") == "Slip_Wall_Bottom");
            }
        }
    }

    // Restart from step-1 checkpoint into the same prefix -> per-BC CSVs append cleanly!
    C::Simulation resumed(config, mpi.comm, factory);
    resumed.ReadCheckpoint(prefix + "-D0-S1.h5");
    {
        C::RunControl control(config, mpi.rank);
        C::OutputCoordinator outputs(resumed, config, mpi, MPI_Wtime(), &control);
        outputs.BeginAttempt(0.001, 0, 0.002);
        REQUIRE(resumed.Advance(0.001).accepted);
        outputs.FlushPending();
        outputs.CompleteAcceptedStep(0.001, 0, {}, false, false);
        CHECK(outputs.Finalize(C::RunStop::None, false) == 0);
    }

    if (mpi.rank == 0)
    {
        const auto lines = ReadCsvLines(noslipCsv);
        CHECK(lines.size() == 1 + rowsAfterStep1 + 3);
    }
    RemoveDir(dir);
}

int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
