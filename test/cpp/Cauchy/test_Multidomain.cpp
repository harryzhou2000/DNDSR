#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Simulation.hpp"
#include "CartesianFixture.hpp"
#include "RotatingVortexFixture.hpp"
#include <filesystem>
#include <unistd.h>
#include <iostream>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    C::Json Domains(bool split)
    {
        std::vector<int> a, b;
        for (int rank = mpi.size - 1; rank >= 0; --rank)
        {
            if (!split || rank % 2 == 0)
                a.push_back(rank);
            if (!split || rank % 2 == 1)
                b.push_back(rank);
        }
        if (b.empty())
            b = a;
        return C::Json::array({{{"name", "outer"}, {"ranks", a}}, {{"name", "disk"}, {"ranks", b}}});
    }
    std::vector<double> Values(const C::Domain &domain)
    {
        const auto read = domain.Dofs().CoupledField().Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        std::vector<double> result;
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                result.push_back(values(iCell, j));
        return result;
    }
    std::array<double, 2> Integrals(const C::Simulation &simulation)
    {
        std::array<double, 2> value{};
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = simulation.GetDomain(iDomain);
                const auto geometry = domain.Discretization().Metrics().AcquireRead();
                const auto metric = geometry.HostView();
                const auto read = domain.Dofs().CoupledField().Owned().Read();
                const auto u = read.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < u.RowCount(); ++iCell)
                {
                    value[0] += metric.GetCellVol(iCell) * u(iCell, 0);
                    value[1] += metric.GetCellVol(iCell) * u(iCell, 4);
                }
            }
        MPI_Allreduce(MPI_IN_PLACE, value.data(), 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
        return value;
    }
    std::string CheckpointPath(const std::string &label)
    {
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() / ("cauchy-multidomain-" + label + std::to_string(pid) + ".h5")).string();
    }
    void RemoveCheckpoint(const std::string &path)
    {
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
        {
            CHECK(std::filesystem::remove(path));
            std::filesystem::remove_all(path + ".parts");
        }
        MPI_Barrier(mpi.comm);
    }
    // Independent fixture-coordinate oracle for a 4x4 torus. Dense global
    // storage is confined to this 160-value test snapshot, never mesh setup.
    std::vector<double> TorusValues(const C::Simulation &simulation)
    {
        std::vector<double> result(2 * 16 * 5, 0);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = simulation.GetDomain(iDomain);
                const auto read = domain.Dofs().CoupledField().Owned().Read();
                for (I iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
                {
                    const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
                    const I global = I(std::floor(point[0] * .4)) + 4 * I(std::floor(point[1] * .4));
                    for (int j = 0; j < 5; ++j)
                        result[(iDomain * 16 + global) * 5 + j] = read.View()(iCell, j);
                }
            }
        MPI_Allreduce(MPI_IN_PLACE, result.data(), int(result.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
        return result;
    }
}
TEST_CASE("Multi-domain BDF2 restart reserves every domain and redistributes by persistent IDs")
{
    const auto factory = [](I, const auto &, const auto &domainMPI)
    { return CauchyTest::PeriodicQuads(domainMPI, 4, 10); };
    for (bool split : {false, true})
    {
        if (split && mpi.size == 1)
            continue;
        CAPTURE(split);
        const auto path = CheckpointPath(split ? "split-" : "shared-");
        C::Json input{{"domains", Domains(split)}, {"time", {{"method", "bdf2"}}}, {"defaults", {{"initialPrimitive", {1, .1, .2, .03, 1}}, {"initialProfile", {{"kind", "isentropicVortex"}, {"strength", .2}, {"period", 10}}}, {"numerics", {{"reconstruction", "variational"}, {"limiter", "none"}, {"reconstructionIterations", 4}, {"cfv", {{"maxOrder", 2}}}}}, {"implicit", {{"iterations", 30}, {"absoluteTolerance", 1e-12}}}}}};
        C::Simulation source(C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm, factory);
        REQUIRE(source.Advance(.001).accepted);
        REQUIRE(source.Advance(.002).accepted);
        const auto saved = TorusValues(source);
        const C::Json outputController{{"schema_version", 1}, {"test_cursor", 7}};
        source.WriteCheckpoint(path, true, outputController);
        REQUIRE(source.Advance(.0015).accepted);
        const auto continuation = TorusValues(source);
        std::array<uint64_t, 2> versions{};
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (source.Local(iDomain))
                versions[iDomain] = source.GetDomain(iDomain).Dofs().CoupledField().Owned().Version();
        std::optional<R::Array<double>::ReadLease> held;
        if (source.Placement().DomainRank(1) == 0)
            held.emplace(source.GetDomain(1).Dofs().CoupledField().Owned().Read());
        CHECK_THROWS(source.ReadCheckpoint(path));
        held.reset();
        CHECK(source.Step() == 3);
        CHECK(TorusValues(source) == continuation);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (source.Local(iDomain))
                CHECK(source.GetDomain(iDomain).Dofs().CoupledField().Owned().Version() == versions[iDomain]);
        source.ReadCheckpoint(path);
        CHECK(source.Step() == 2);
        CHECK(source.Time() == .003);
        REQUIRE(source.RestoredOutputController());
        CHECK(*source.RestoredOutputController() == outputController);
        CHECK(TorusValues(source) == saved);
        REQUIRE(source.Advance(.0015).accepted);
        CHECK(TorusValues(source) == continuation);
        // Republishing uses a new child generation. Reverse sharing and use
        // SoA/P3 to exercise actual redistribution and intentional cold VR.
        source.WriteCheckpoint(path, true, outputController);
        input["domains"] = Domains(!split && mpi.size > 1);
        input["defaults"]["execution"]["layout"] = "soa";
        input["defaults"]["numerics"]["cfv"]["maxOrder"] = 3;
        C::Simulation redistributed(C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm, factory);
        redistributed.ReadCheckpoint(path);
        REQUIRE(redistributed.RestoredOutputController());
        CHECK(*redistributed.RestoredOutputController() == outputController);
        CHECK(redistributed.Step() == source.Step());
        CHECK(redistributed.Time() == source.Time());
        CHECK(TorusValues(redistributed) == continuation);
        REQUIRE(redistributed.Advance(.0005).accepted);
        RemoveCheckpoint(path);
    }
}
TEST_CASE("Coordinated domains retain single-domain time equations and atomic publication")
{
    for (const std::string method : {"ssprk3", "bdf2", "esdirk4"})
    {
        CAPTURE(method);
        C::Json input{{"domains", Domains(mpi.size > 1)}, {"time", {{"method", method}}}, {"defaults", {{"numerics", {{"reconstruction", "constant"}, {"flux", "rusanov"}}}, {"initialPrimitive", {1, .1, .2, .03, 1}}, {"initialProfile", {{"kind", "isentropicVortex"}, {"strength", .2}, {"period", 10}}}, {"implicit", {{"iterations", 30}, {"absoluteTolerance", 1e-12}}}}}};
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        const auto factory = [](I, const auto &, const auto &domainMPI)
        { return CauchyTest::PeriodicQuads(domainMPI, 4, 10); };
        C::Simulation simulation(config, mpi.comm, factory);
        std::vector<std::unique_ptr<C::Domain>> reference(2);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = config.Domains()[iDomain];
                reference[iDomain] = std::make_unique<C::Domain>(domain.placement.name,
                                                                 factory(iDomain, domain, DNDS::MPIInfo(simulation.Placement().Context(iDomain)->Communicator())), domain.settings, config.Time());
            }
        for (int step = 0; step < 2; ++step)
        {
            REQUIRE(simulation.Advance(.001).accepted);
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (reference[iDomain])
                {
                    REQUIRE(reference[iDomain]->Advance(.001).accepted);
                    CHECK(Values(simulation.GetDomain(iDomain)) == Values(*reference[iDomain]));
                    CHECK(simulation.GetDomain(iDomain).Time() == simulation.Time());
                }
        }
        // Only one rank in the second domain retains a reader. Every domain's
        // accepted field and BDF history must remain at the previous step.
        std::vector<std::vector<double>> before(2);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
                before[iDomain] = Values(simulation.GetDomain(iDomain));
        std::optional<R::Array<double>::ReadLease> held;
        if (simulation.Placement().DomainRank(1) == 0)
            held.emplace(simulation.GetDomain(1).Dofs().CoupledField().Owned().Read());
        CHECK_THROWS(simulation.Advance(.001));
        CHECK(simulation.Step() == 2);
        CHECK(simulation.Time() == .002);
        held.reset();
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
                CHECK(Values(simulation.GetDomain(iDomain)) == before[iDomain]);
        REQUIRE(simulation.Advance(.001).accepted);
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (reference[iDomain])
            {
                REQUIRE(reference[iDomain]->Advance(.001).accepted);
                CHECK(Values(simulation.GetDomain(iDomain)) == Values(*reference[iDomain]));
            }
    }
}
TEST_CASE("P3 flow5 crosses a posed disk interface with dense and sparse distributed matching")
{
    for (bool split : {false, true})
    {
        if (split && mpi.size == 1)
            continue;
        for (double omega : {0., .02})
        {
            CAPTURE(split);
            CAPTURE(omega);
            C::Json input{{"domains", Domains(split)}, {"time", {{"method", !split && omega == 0 ? "esdirk2" : "ssprk3"}}}, {"defaults", {{"initialPrimitive", {1, .1, .2, .03, 1}}, {"initialProfile", {{"kind", "isentropicVortex"}, {"strength", .5}, {"period", 10}}}, {"boundaries", {{"101", {{"kind", "interface"}}}}}, {"numerics", {{"reconstruction", "variational"}, {"flux", "rusanov"}, {"limiter", "none"}, {"positivity", false}, {"reconstructionIterations", 12}, {"cfv", {{"maxOrder", 3}, {"intOrder", 5}, {"jacobiRelax", .8}, {"boundaryDerivativeWeights", {1, 0, 0, 0}}}}}}}}, {"interfaces", C::Json::array({{{"donor", "outer"}, {"receiver", "disk"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .02}}, {{"donor", "disk"}, {"receiver", "outer"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .02}}})}};
            input["domains"][1]["overrides"] = {{"motion", {{"angularVelocity", {0, 0, omega}}, {"center", {5, 5, 0}}}}};
            const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
            { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 12, 32); };
            const auto sparseConfig = C::SimulationConfig::ResolveCollective(input, mpi.comm);
            C::Simulation sparse(sparseConfig, mpi.comm, factory);
            const auto initial = Integrals(sparse);
            for (int step = 0; step < 3; ++step)
                REQUIRE(sparse.Advance(.001).accepted);
            const auto final = Integrals(sparse);
            const auto diagnostics = sparse.GetInterfaceDiagnostics();
            CHECK(diagnostics.queryPoints > 0);
            CHECK(diagnostics.maximumGap <= .02);
            CHECK(diagnostics.mapBuilds == (omega == 0 ? 2 : 20));
            double error = 0;
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (sparse.Local(iDomain))
                {
                    const auto &domain = sparse.GetDomain(iDomain);
                    const auto &settings = sparseConfig.Domains()[iDomain].settings;
                    error = std::max(error, domain.Discretization().Error(domain.Dofs(), settings.initialProfile, settings.initialPrimitive, sparse.Time())[2][0]);
                    CHECK(domain.Discretization().Admissible(domain.Dofs()));
                }
            MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
                std::cout << "IV interfaces split=" << split << " omega=" << omega << " rho_Linf=" << error << " mass_drift=" << final[0] - initial[0] << " energy_drift=" << final[1] - initial[1] << '\n';
            CHECK(error > 1e-10);
            CHECK(error < 2e-3);
            CHECK(std::abs(final[0] - initial[0]) < 2e-3);
            CHECK(std::abs(final[1] - initial[1]) < 5e-3);
            // Same physical meshes and poses, independent dense routing. Every
            // resulting field component must agree, not merely the match count.
            for (auto &connection : input["interfaces"])
                connection["routing"] = "allGatherBoundary";
            C::Simulation dense(C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm, factory);
            for (int step = 0; step < 3; ++step)
                REQUIRE(dense.Advance(.001).accepted);
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (sparse.Local(iDomain))
                    CHECK(Values(sparse.GetDomain(iDomain)) == Values(dense.GetDomain(iDomain)));
            if (!split && omega != 0)
            {
                const auto old = Values(sparse.GetDomain(0));
                std::optional<R::Array<double>::ReadLease> held;
                if (mpi.rank == 0)
                    held.emplace(sparse.GetDomain(1).Dofs().CoupledField().Owned().Read());
                CHECK_THROWS(sparse.Advance(.001));
                held.reset();
                CHECK(sparse.Step() == 3);
                CHECK(Values(sparse.GetDomain(0)) == old);
                CHECK(sparse.GetDomain(0).StableStep(.2) > 0);
                CHECK_NOTHROW(sparse.GetDomain(0).Diagnostics(sparseConfig.Domains()[0].settings, {}));
                const auto path = CheckpointPath("rotating-");
                sparse.WriteCheckpoint(path, true);
                {
                    std::optional<R::Array<double>::ReadLease> heldRestart;
                    if (mpi.rank == 0)
                        heldRestart.emplace(sparse.GetDomain(1).Dofs().CoupledField().Owned().Read());
                    CHECK_THROWS(sparse.ReadCheckpoint(path));
                }
                // Failed publication must also restore accepted interface
                // packets; diagnostics/CFL must not consume trial-time traces.
                CHECK(sparse.Step() == 3);
                CHECK(sparse.GetDomain(0).StableStep(.2) > 0);
                REQUIRE(sparse.Advance(.001).accepted);
                const std::array<std::vector<double>, 2> continuation{Values(sparse.GetDomain(0)), Values(sparse.GetDomain(1))};
                sparse.ReadCheckpoint(path);
                CHECK(sparse.Step() == 3);
                CHECK(sparse.GetDomain(0).StableStep(.2) > 0);
                REQUIRE(sparse.Advance(.001).accepted);
                for (I iDomain = 0; iDomain < 2; ++iDomain)
                    CHECK(Values(sparse.GetDomain(iDomain)) == continuation[iDomain]);
                RemoveCheckpoint(path);
            }
        }
    }
}
TEST_CASE("P0 and physical-gradient P1 publish complete multidomain traces without modal storage")
{
    for (const std::string reconstruction : {"constant", "gradient"})
        for (const bool split : {false, true})
        {
            if (split && mpi.size == 1)
                continue;
            CAPTURE(reconstruction);
            CAPTURE(split);
            C::Json input{
                {"domains", Domains(split)},
                {"time", {{"method", "ssprk3"}}},
                {"defaults",
                 {{"initialPrimitive", {1, .1, .2, .03, 1}},
                  {"initialProfile",
                   {{"kind", "isentropicVortex"}, {"strength", .2}, {"period", 10}}},
                  {"boundaries", {{"101", {{"kind", "interface"}}}}},
                  {"numerics",
                   {{"reconstruction", reconstruction},
                    {"flux", "rusanov"},
                    {"limiter", reconstruction == "gradient" ? "Barth" : "none"},
                    {"positivity", false}}}}},
                {"interfaces",
                 C::Json::array(
                     {{{"donor", "outer"},
                       {"receiver", "disk"},
                       {"donorZone", 101},
                       {"receiverZone", 101},
                       {"maxGap", .03}},
                      {{"donor", "disk"},
                       {"receiver", "outer"},
                       {"donorZone", 101},
                       {"receiverZone", 101},
                       {"maxGap", .03}}})}};
            input["domains"][1]["overrides"] = {
                {"motion",
                 {{"angularVelocity", {0, 0, .02}}, {"center", {5, 5, 0}}}}};
            const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
            { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 8, 20); };
            C::Simulation simulation(
                C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm,
                factory);
            REQUIRE(simulation.Advance(.001).accepted);
            const auto diagnostics = simulation.GetInterfaceDiagnostics();
            CHECK(diagnostics.queryPoints > 0);
            CHECK(diagnostics.maximumGap <= .03);
            for (I iDomain = 0; iDomain < 2; ++iDomain)
                if (simulation.Local(iDomain))
                    CHECK(simulation.GetDomain(iDomain)
                              .Discretization()
                              .Admissible(simulation.GetDomain(iDomain).Dofs()));
        }
}
TEST_CASE("Split flow and SA fields cross a rotating interface as one equation packet")
{
    if (mpi.size < 2)
        return;
    C::Json input{
        {"domains", Domains(true)},
        {"time", {{"method", "ssprk3"}}},
        {"defaults",
         {{"equation",
           {{"transport", "NewtonianFourier"},
            {"viscosity", 1e-3},
            {"rans", {{"model", "SA"}}}}},
          {"execution",
           {{"fieldGrouping", "equationBlocks"}, {"layout", "soa"}}},
          {"initialPrimitive", {1, .1, .2, .03, 1}},
          {"initialRANS", {{"rhoNuTilde", .01}}},
          {"boundaries", {{"101", {{"kind", "interface"}}}}},
          {"numerics",
           {{"reconstruction", "gradient"},
            {"gradient", "leastSquares"},
            {"viscousGradient", "polynomial"},
            {"flux", "rusanov"},
            {"faceFallback", false},
            {"limiter", "Barth"},
            {"positivity", false}}}}},
        {"interfaces",
         C::Json::array(
             {{{"donor", "outer"},
               {"receiver", "disk"},
               {"donorZone", 101},
               {"receiverZone", 101},
               {"maxGap", .03}},
              {{"donor", "disk"},
               {"receiver", "outer"},
               {"donorZone", 101},
               {"receiverZone", 101},
               {"maxGap", .03}}})}};
    input["domains"][1]["overrides"] = {
        {"motion",
         {{"angularVelocity", {0, 0, .02}}, {"center", {5, 5, 0}}}}};
    const auto factory = [](I iDomain, const auto &, const auto &domainMPI)
    { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, 8, 20); };
    C::Simulation simulation(
        C::SimulationConfig::ResolveCollective(input, mpi.comm), mpi.comm,
        factory);
    REQUIRE(simulation.Advance(1e-4).accepted);
    for (I iDomain = 0; iDomain < 2; ++iDomain)
        if (simulation.Local(iDomain))
        {
            const auto &domain = simulation.GetDomain(iDomain);
            CHECK(domain.Dofs().PhysicalFieldNumber() == 2);
            const auto read = domain.Dofs().ReadOwned();
            const auto flow = read.Block("flow");
            const auto rans = read.Block("rans");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                CHECK(std::isfinite(flow.owned(iCell, 0)));
                CHECK(std::isfinite(rans.owned(iCell, 0)));
                CHECK(flow.owned(iCell, 0) > 0);
                CHECK(rans.owned(iCell, 0) >= 0);
            }
        }
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    int status;
    {
        doctest::Context context(argc, argv);
        status = context.run();
    }
    DNDS::MPI::Finalize();
    return status;
}
