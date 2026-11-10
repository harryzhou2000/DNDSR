#include "Cauchy/Core/Simulation.hpp"
#include "RotatingVortexFixture.hpp"
#include "Geom/Unstructured/UnstructuredMeshState.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
namespace
{
    C::Json Observe(const C::Simulation &simulation, const C::SimulationConfig &config, const DNDS::MPIInfo &mpi)
    {
        double sums[4]{}, maximum = 0;
        double minima[2]{1e300, 1e300};
        I cells = 0;
        for (I iDomain = 0; iDomain < 2; ++iDomain)
            if (simulation.Local(iDomain))
            {
                const auto &domain = simulation.GetDomain(iDomain);
                const auto &settings = config.Domains()[iDomain].settings;
                const auto geometry = domain.Discretization().Metrics().AcquireRead();
                const auto metric = geometry.HostView();
                const auto read = domain.Flow().Owned().Read();
                const auto u = read.View<R::UniformTableRows>();
                double volume = 0;
                for (I iCell = 0; iCell < u.RowCount(); ++iCell)
                {
                    const auto weight = metric.GetCellVol(iCell);
                    sums[0] += weight * u(iCell, 0);
                    sums[1] += weight * u(iCell, 4);
                    volume += weight;
                    C::FlowState state;
                    for (int j = 0; j < 5; ++j)
                        state[j] = u(iCell, j);
                    minima[0] = std::min(minima[0], state[0]);
                    minima[1] = std::min(minima[1], domain.Equation().Gas().Pressure(state));
                }
                sums[2] += volume;
                cells += u.RowCount();
                MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, domain.Discretization().Metrics().Communicator());
                const auto error = domain.Discretization().Error(domain.Dofs(), settings.initialProfile, settings.initialPrimitive, simulation.Time());
                if (simulation.Placement().DomainRank(iDomain) == 0)
                {
                    sums[3] += volume * error[0][0];
                    maximum = std::max(maximum, error[2][0]);
                }
            }
        MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_DOUBLE, MPI_SUM, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, minima, 2, MPI_DOUBLE, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &cells, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        const auto coupling = simulation.GetInterfaceDiagnostics();
        return {{"step", simulation.Step()}, {"time", simulation.Time()}, {"cells", cells}, {"volume", sums[2]}, {"mass", sums[0]}, {"energy", sums[1]}, {"densityL1", sums[3] / sums[2]}, {"densityLinf", maximum}, {"minimumDensity", minima[0]}, {"minimumPressure", minima[1]}, {"queryPoints", coupling.queryPoints}, {"changedDonors", coupling.changedDonors}, {"mapBuilds", coupling.mapBuilds}, {"maximumGap", coupling.maximumGap}};
    }
}
/// Bounded evidence driver, excluded from normal builds/CTest. It intentionally
/// uses the independently generated mesh fixture and emits JSON lines rather
/// than adding experiment controls to the production solver configuration.
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    int result = 0;
    try
    {
        I steps = 0;
        double dt = 0, omega = 0;
        int split = 0, subdivisions = 0, segments = 0, outerRanks = 0;
        std::string pointDistribution = "grid";
        std::string caseFile;
        I outputEvery = 0;
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            DNDS_check_throw_info(argc>=7&&argc<=11,"usage: steps dt omega split(0/1) subdivisions circleSegments [visualizationEvery] [outerRanks] [grid|poisson|gmsh] [case.json]");
            steps=std::stoll(argv[1]); dt=std::stod(argv[2]); omega=std::stod(argv[3]);
            split=std::stoi(argv[4]); subdivisions=std::stoi(argv[5]); segments=std::stoi(argv[6]);
            if(argc>=8) outputEvery=std::stoll(argv[7]);
            if(argc>=9) outerRanks=std::stoi(argv[8]);
            if(argc>=10) pointDistribution=argv[9];
            if(argc>=11) caseFile=argv[10];
            DNDS_check_throw_info(steps>=1&&steps<=100000&&std::isfinite(dt)&&dt>0&&dt<=.01&&std::isfinite(omega)&&std::abs(omega)<=4,
                                  "experiment exceeds bounded time controls");
            DNDS_check_throw_info((split==0||split==1)&&(!split||mpi.size>=2)&&subdivisions>=8&&subdivisions<=256&&segments>=16&&segments<=512,
                                  "invalid experiment placement or mesh resolution");
            DNDS_check_throw_info((outerRanks==0||(split&&outerRanks>0&&outerRanks<mpi.size))&&
                                  (pointDistribution=="grid"||pointDistribution=="poisson"||pointDistribution=="gmsh")&&
                                  (pointDistribution!="gmsh"||!caseFile.empty()),"invalid domain split or point distribution");
            DNDS_check_throw_info(outputEvery>=0&&(!outputEvery||(outputEvery<=steps&&steps/outputEvery<=500)),
                                  "invalid/beyond-bound visualization cadence"); });
        std::vector<int> a, b;
        for (int rank = mpi.size - 1; rank >= 0; --rank)
        {
            if (!split || (outerRanks ? rank < outerRanks : rank % 2 == 0))
                a.push_back(rank);
            if (!split || (outerRanks ? rank >= outerRanks : rank % 2 == 1))
                b.push_back(rank);
        }
        C::Json input{{"domains", C::Json::array({{{"name", "outer"}, {"ranks", a}}, {{"name", "disk"}, {"ranks", b}}})},
                      {"time", {{"method", "ssprk3"}, {"dt", dt}, {"steps", steps}}},
                      {"defaults", {{"initialPrimitive", {1, .1, .2, .03, 1}}, {"initialProfile", {{"kind", "isentropicVortex"}, {"strength", .5}, {"period", 10}}}, {"boundaries", {{"101", {{"kind", "interface"}}}}}, {"numerics", {{"reconstruction", "variational"}, {"flux", "rusanov"}, {"limiter", "none"}, {"positivity", false}, {"faceFallback", false}, {"reconstructionIterations", 12}, {"cfv", {{"maxOrder", 3}, {"intOrder", 5}, {"jacobiRelax", .8}, {"boundaryDerivativeWeights", {1, 0, 0, 0}}}}}}}},
                      {"interfaces", C::Json::array({{{"donor", "outer"}, {"receiver", "disk"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .02}},
                                                     {{"donor", "disk"}, {"receiver", "outer"}, {"donorZone", 101}, {"receiverZone", 101}, {"maxGap", .02}}})}};
        input["domains"][1]["overrides"] = {{"motion", {{"angularVelocity", {0, 0, omega}}, {"center", {5, 5, 0}}}}};
        // Optional experiment input keeps the original fixture reproducible.
        // Gmsh cases use the ordinary per-domain CGNS reader/mesh pipeline.
        if (!caseFile.empty())
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                std::ifstream file(caseFile);
                DNDS_check_throw_info(bool(file), "cannot read vortex experiment input");
                C::Json settings; file >> settings;
                input["defaults"]["initialPrimitive"] = settings.at("initialPrimitive");
                input["defaults"]["initialProfile"]["strength"] = settings.at("strength");
                if (pointDistribution == "gmsh")
                    for (I iDomain = 0; iDomain < 2; ++iDomain)
                    {
                        auto &mesh = input["domains"][iDomain]["overrides"]["mesh"];
                        mesh["file"] = settings.at("meshFiles").at(iDomain);
                        mesh["dimension"] = 2;
                        mesh["boundaryIds"] = {{"INTERFACE",101},
                            {"PERIODIC_1",DNDS::Geom::BC_ID_PERIODIC_1}, {"PERIODIC_1_DONOR",DNDS::Geom::BC_ID_PERIODIC_1_DONOR},
                            {"PERIODIC_2",DNDS::Geom::BC_ID_PERIODIC_2}, {"PERIODIC_2_DONOR",DNDS::Geom::BC_ID_PERIODIC_2_DONOR}};
                        if (iDomain == 0) mesh["periodicLengths"] = {10,10,0};
                    } });
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        const auto start = MPI_Wtime();
        {
            C::Simulation::MeshFactory factory;
            if (pointDistribution != "gmsh")
                factory = [&](I iDomain, const auto &, const auto &domainMPI)
                { return CauchyTest::VortexPatch(domainMPI, iDomain == 1, subdivisions, segments,
                                                 pointDistribution == "poisson" ? CauchyTest::VortexPointDistribution::PoissonDisk : CauchyTest::VortexPointDistribution::JitteredGrid); };
            C::Simulation simulation(config, mpi.comm, factory);
            double setup = MPI_Wtime() - start, advance = 0, diagnostics = 0, output = 0;
            C::Json frames = C::Json::array();
            const auto snapshot = [&]
            {
                const auto begin = MPI_Wtime();
                std::ostringstream prefix;
                prefix << "frame" << std::setw(6) << std::setfill('0') << simulation.Step();
                simulation.WriteVisualization(prefix.str());
                if (mpi.rank == 0)
                    frames.push_back({{"step", simulation.Step()}, {"time", simulation.Time()}, {"domains", {prefix.str() + "_domain0.vtkhdf", prefix.str() + "_domain1.vtkhdf"}}});
                output += MPI_Wtime() - begin;
            };
            MPI_Allreduce(MPI_IN_PLACE, &setup, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
                std::cout << C::Json{{"controls", {{"np", mpi.size}, {"steps", steps}, {"dt", dt}, {"omega", omega}, {"split", split}, {"outerRanks", outerRanks}, {"pointDistribution", pointDistribution}, {"caseFile", caseFile}, {"subdivisions", subdivisions}, {"segments", segments}, {"visualizationEvery", outputEvery}, {"setupSeconds", setup}, {"resolved", config.Root()}}}}.dump() << '\n';
            auto report = [&]
            {
                const auto begin = MPI_Wtime();
                const auto row = Observe(simulation, config, mpi);
                diagnostics += MPI_Wtime() - begin;
                if (mpi.rank == 0)
                    std::cout << row.dump() << std::endl;
            };
            report();
            if (outputEvery)
                snapshot();
            for (I iStep = 0; iStep < steps; ++iStep)
            {
                const auto begin = MPI_Wtime();
                const auto result = simulation.Advance(dt);
                DNDS_check_throw_info(result.accepted, "multidomain IV candidate rejected: " + result.reason);
                advance += MPI_Wtime() - begin;
                if (outputEvery && ((iStep + 1) % outputEvery == 0 || iStep + 1 == steps))
                    snapshot();
                if ((iStep + 1) % std::max(I(1), steps / 20) == 0 || iStep + 1 == steps)
                    report();
            }
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                std::ofstream field("state-rank"+std::to_string(mpi.rank)+".csv");
                DNDS_check_throw_info(bool(field),"cannot create rank-local final field");
                field<<"domain,cellId,rho,rhou,rhov,rhow,rhoE\n"<<std::setprecision(17);
                for(I iDomain=0;iDomain<2;++iDomain)
                    if(simulation.Local(iDomain))
                    {
                        const auto &domain=simulation.GetDomain(iDomain);
                        // Retain the immutable snapshot and prove exact row
                        // mapping before pairing persistent IDs with DOFs.
                        const auto native=domain.Discretization().Mesh().CurrentNativeState();
                        DNDS_check_throw_info(native&&native->Entities(DNDS::Geom::EntityKind::Cell).SameOrder(domain.Discretization().Metrics().Cells()),
                                              "final field and persistent cell IDs have different row order");
                        const auto read=domain.Flow().Owned().Read();
                        const auto values=read.View<R::UniformTableRows>();
                        for(I iCell=0;iCell<values.RowCount();++iCell)
                        {
                            field<<iDomain<<','<<native->OriginalEntityId(DNDS::Geom::EntityKind::Cell,iCell);
                            for(int j=0;j<5;++j) field<<','<<values(iCell,j);
                            field<<'\n';
                        }
                    }
                DNDS_check_throw_info(bool(field),"failed writing rank-local final field"); });
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0 && outputEvery)
                {
                    std::ofstream manifest("frames.json");
                    manifest << C::Json{{"referenceCoordinates", true}, {"motion", config.Root()["domains"][1]["settings"]["motion"]}, {"frames", frames}}.dump(2) << '\n';
                    DNDS_check_throw_info(bool(manifest), "failed writing visualization manifest");
                } });
            double timing[3]{advance, diagnostics, output};
            MPI_Allreduce(MPI_IN_PLACE, timing, 3, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
                std::cout << C::Json{{"complete", true}, {"steps", simulation.Step()}, {"advanceSeconds", timing[0]}, {"diagnosticSeconds", timing[1]}, {"outputSeconds", timing[2]}}.dump() << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        if (mpi.rank == 0)
            std::cerr << error.what() << '\n';
        result = 1;
    }
    MPI_Finalize();
    return result;
}
