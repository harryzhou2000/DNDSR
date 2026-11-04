#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <hdf5.h>
#include <filesystem>
#include <unistd.h>
#include <limits>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    std::string Path(const std::string &label)
    {
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() / (label + std::to_string(pid) + ".h5")).string();
    }
    C::FlowState Wave(const C::Domain &domain, DNDS::index iCell)
    {
        const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
        return domain.Equation().Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * (point[0] + point[1])), .3, -.1, .2, 1});
    }
    void Initialize(C::Domain &domain)
    {
        auto write = domain.Flow().Owned().Write();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
        {
            const auto value = Wave(domain, iCell);
            for (int j = 0; j < 5; ++j)
                write.View<R::UniformTableRows>()(iCell, j) = value[j];
        }
    }
    void Remove(const std::vector<std::string> &paths)
    {
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
            for (const auto &path : paths)
                CHECK(std::filesystem::remove(path));
        MPI_Barrier(mpi.comm);
    }
}
TEST_CASE("Older viscous archives retain primitive gradients and reject a silent polynomial method change")
{
    const auto path = Path("cauchy-viscous-policy-");
    const C::Json input{{"defaults", {{"equation", {{"transport", "NewtonianFourier"}, {"viscosity", .02}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::Domain source("flow", mesh, config.Domains()[0].settings);
    Initialize(source);
    auto old = config.Root();
    auto &numerics = old["domains"][0]["settings"]["numerics"];
    for (const auto *name : {"viscousGradient", "viscousLimited", "noGRPOnWall"})
        numerics.erase(name);
    source.WriteCheckpoint(path, old);
    C::Domain restored("flow", mesh, config.Domains()[0].settings);
    CHECK_NOTHROW(restored.ReadCheckpoint(path, config.Root()));
    {
        const auto a = source.Flow().Owned().Read(), b = restored.Flow().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(a.View<R::UniformTableRows>()(iCell, j) == b.View<R::UniformTableRows>()(iCell, j));
    }
    auto polynomialInput = input;
    polynomialInput["defaults"]["numerics"]["viscousGradient"] = "polynomial";
    const auto polynomial = C::SimulationConfig::ResolveCollective(polynomialInput, mpi.comm);
    C::Domain changed("flow", mesh, polynomial.Domains()[0].settings);
    const auto version = changed.Flow().Owned().Version();
    CHECK_THROWS(changed.ReadCheckpoint(path, polynomial.Root()));
    CHECK(changed.Flow().Owned().Version() == version);
    Remove({path});
}
TEST_CASE("Older physical-boundary archives omit inactive composition without invalidating restart")
{
    const auto path = Path("cauchy-boundary-composition-");
    const C::Json input{{"defaults", {{"mesh", {{"dimension", 3}}}, {"boundaries", {{"7", {{"kind", "extrapolate"}}}}}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    C::Domain source("flow", mesh, config.Domains()[0].settings);
    auto old = config.Root();
    old["domains"][0]["settings"]["boundaries"]["7"].erase("massFractions");
    source.WriteCheckpoint(path, old);
    C::Domain restored("flow", mesh, config.Domains()[0].settings);
    CHECK_NOTHROW(restored.ReadCheckpoint(path, config.Root()));
    const auto version = restored.Flow().Owned().Version();
    auto changed = config.Root();
    changed["domains"][0]["settings"]["boundaries"]["7"]["massFractions"] = C::Json{{"H2", .1}};
    CHECK_THROWS(restored.ReadCheckpoint(path, changed));
    CHECK(restored.Flow().Owned().Version() == version);
    Remove({path});
}
TEST_CASE("Checkpoint changes MPI partition and polynomial degree using persistent cell IDs with cold reconstruction")
{
    const auto plain = Path("cauchy-id-plain-"), polynomial = plain + ".rec.h5", cold = plain + ".cold.h5";
    C::Json input{{"defaults", {{"numerics", {{"reconstruction", "variational"}, {"reconstructionIterations", 40}, {"cfv", {{"maxOrder", 2}}}}}}}};
    MPI_Comm writerComm = MPI_COMM_NULL;
    const int writerRanks = std::max(1, mpi.size / 2);
    MPI_Comm_split(mpi.comm, mpi.rank < writerRanks ? 0 : MPI_UNDEFINED, mpi.rank, &writerComm);
    if (writerComm != MPI_COMM_NULL)
    {
        {
            const DNDS::MPIInfo writers(writerComm);
            const auto config = C::SimulationConfig::ResolveCollective(input, writerComm);
            C::Domain source("fluid", CauchyTest::PeriodicQuads(writers, 6), config.Domains()[0].settings);
            Initialize(source);
            source.Residual();
            source.WriteCheckpoint(plain, config.Root());
            source.WriteCheckpoint(polynomial, config.Root(), true);
            auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(writers);
            serializer->OpenFile(plain, true);
            CHECK_FALSE(serializer->ListCurrentPath().count("reconstruction_guess"));
            CHECK(serializer->ListCurrentPath().count("metadata_json") == 1);
            serializer->CloseFile();
            serializer->OpenFile(polynomial, true);
            const auto rec = R::ReadArray<double>(serializer, "reconstruction_guess", writers);
            serializer->CloseFile();
            double norm = 0;
            {
                const auto read = rec.Read();
                for (DNDS::index iCell = 0; iCell < rec.RowCount(); ++iCell)
                    for (DNDS::index j = 0; j < rec.RowSize(iCell); ++j)
                        norm += std::abs(read.View()(iCell, j));
            }
            MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, writerComm);
            CHECK(norm > 1e-8); // A zero fixture could not prove cold-start conversion.
        }
        MPI_Comm_free(&writerComm);
    }
    MPI_Barrier(mpi.comm);
    input["defaults"]["numerics"]["cfv"]["maxOrder"] = 3;
    // Iteration budgets/tolerances are restart controls, not stored physics.
    // Retune them together with the existing degree/layout/partition migration.
    input["defaults"]["numerics"]["reconstructionIterations"] = 80;
    input["defaults"]["numerics"]["reconstructionTolerance"] = 1e-12;
    input["defaults"]["execution"]["layout"] = "soa";
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    auto targetMesh = CauchyTest::PeriodicQuads(mpi, 6);
    std::vector<DNDS::index> before;
    for (DNDS::index iCell = 0; iCell < targetMesh->NumCell(); ++iCell)
        before.push_back(targetMesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell));
    targetMesh->ReorderLocalCells();
    DNDS::index changed = 0;
    for (DNDS::index iCell = 0; iCell < targetMesh->NumCell(); ++iCell)
        changed += before[iCell] != targetMesh->OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(changed > 0); // Prove this fixture exercises a real mesh permutation.
    C::Domain target("fluid", targetMesh, config.Domains()[0].settings);
    for (const auto &path : {plain, polynomial})
    {
        target.ReadCheckpoint(path, config.Root());
        const auto read = target.Flow().Owned().Read();
        const auto values = read.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const auto expected = Wave(target, iCell);
            for (int j = 0; j < 5; ++j)
                CHECK(values(iCell, j) == expected[j]);
        }
    }
    auto incompatible = config.Root();
    incompatible["domains"][0]["settings"]["numerics"]["flux"] = "Roe";
    const auto version = target.Flow().Owned().Version();
    CHECK_THROWS(target.ReadCheckpoint(plain, incompatible));
    CHECK(target.Flow().Owned().Version() == version);
    target.WriteCheckpoint(cold, config.Root(), true);
    auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
    serializer->OpenFile(cold, true);
    const auto coefficients = R::ReadArray<double>(serializer, "reconstruction_guess", mpi);
    serializer->CloseFile();
    {
        const auto read = coefficients.Read();
        CHECK(coefficients.Layout().Width() == 45);
        for (DNDS::index iCell = 0; iCell < coefficients.RowCount(); ++iCell)
            for (DNDS::index j = 0; j < coefficients.RowSize(iCell); ++j)
                CHECK(read.View()(iCell, j) == 0);
    }
    Remove({plain, polynomial, cold});
}
TEST_CASE("Implicit restart preserves variable BDF2 history and rebuilds live DITR stage bindings")
{
    for (const std::string method : {"bdf2", "ditr"})
    {
        const auto path = Path("cauchy-" + method + "-history-");
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 2;
        settings.numerics.reconstructionIterations = 80;
        settings.numerics.reconstructionTolerance = 1e-11;
        settings.implicit.minIterations = 0;
        settings.implicit.relativeTolerance = 1e-9;
        settings.implicit.linear.relativeTolerance = 1e-3;
        C::TimeSettings time;
        time.method = method;
        const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
        C::Domain source("flow", mesh, settings, time), restored("flow", mesh, settings, time);
        Initialize(source);
        source.Advance(.001);
        source.Advance(.002);
        const C::Json metadata{{"test", method + " native continuation"}};
        source.WriteCheckpoint(path, metadata, true);
        if (method == "ditr")
            restored.Advance(.0007); // build an old intermediate binding before replacement
        const auto oldTime = restored.Time();
        const auto oldDt = restored.PreviousTimeStep();
        const auto oldStep = restored.Epoch().acceptedStep;
        std::optional<R::Array<double>::ReadLease> held;
        if (mpi.rank == 0)
            held.emplace(restored.Flow().Owned().Read());
        CHECK_THROWS(restored.ReadCheckpoint(path, metadata));
        CHECK(restored.Time() == oldTime);
        CHECK(restored.PreviousTimeStep() == oldDt);
        CHECK(restored.Epoch().acceptedStep == oldStep);
        held.reset();
        restored.ReadCheckpoint(path, metadata);
        CHECK(restored.Time() == source.Time());
        CHECK(restored.PreviousTimeStep() == .002);
        CHECK(restored.Epoch().acceptedStep == source.Epoch().acceptedStep);
        source.Advance(.0015);
        restored.Advance(.0015);
        const auto a = source.Flow().Owned().Read(), b = restored.Flow().Owned().Read();
        for (DNDS::index iCell = 0; iCell < source.Discretization().Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(a.View()(iCell, j) == b.View()(iCell, j));
        Remove({path});
    }
}
TEST_CASE("Restart dt growth reads the accepted scalar and supports files predating it")
{
    for (const std::string method : {"bdf2", "trapz"})
    {
        const auto path = Path("cauchy-previous-step-" + method);
        const C::Json metadata{{"test", "physical timestep history"}};
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.implicit.minIterations = 0;
        C::TimeSettings time;
        time.method = method;
        time.positivity.enabled = true;
        auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
        C::Domain source("flow", mesh, settings, time), restored("flow", mesh, settings, time);
        source.Advance(.001);
        source.Advance(.002);
        source.WriteCheckpoint(path, metadata);
        restored.ReadCheckpoint(path, metadata);
        CHECK(restored.PreviousTimeStep() == .002);
        CHECK(restored.LimitPhysicalStep(.006).dt == .004);
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            if (mpi.rank == 0)
            {
                const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                DNDS_check_throw_info(file >= 0, "cannot open legacy-dt test fixture");
                const auto removed = H5Adelete(file, "last_accepted_dt");
                H5Fclose(file);
                DNDS_check_throw_info(removed >= 0, "cannot remove accepted-dt fixture attribute");
            } });
        restored.ReadCheckpoint(path, metadata);
        CHECK(restored.PreviousTimeStep() == (method == "bdf2" ? .002 : 0));
        CHECK(restored.LimitPhysicalStep(.006).dt == (method == "bdf2" ? .004 : .006));
        // Nonfinite, impossible-clock and conflicting-history scalar input
        // must fail before publishing means or changing accepted history.
        for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), .1, method == "bdf2" ? .001 : -.001})
        {
            source.WriteCheckpoint(path, metadata);
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0)
                {
                    const auto file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                    DNDS_check_throw_info(file >= 0, "cannot open invalid-dt test fixture");
                    const auto attribute = H5Aopen(file, "last_accepted_dt", H5P_DEFAULT);
                    DNDS_check_throw_info(attribute >= 0, "missing accepted-dt fixture attribute");
                    const auto written = H5Awrite(attribute, H5T_NATIVE_DOUBLE, &invalid);
                    H5Aclose(attribute); H5Fclose(file);
                    DNDS_check_throw_info(written >= 0, "cannot write invalid accepted-dt fixture");
                } });
            const auto version = restored.Flow().Owned().Version();
            const auto previous = restored.PreviousTimeStep(), clock = restored.Time();
            CHECK_THROWS(restored.ReadCheckpoint(path, metadata));
            CHECK(restored.Flow().Owned().Version() == version);
            CHECK(restored.PreviousTimeStep() == previous);
            CHECK(restored.Time() == clock);
        }
        Remove({path});
    }
}
TEST_CASE("CGNS checkpoint remains physical across independently built MPI partitions")
{
    const auto path = Path("cauchy-cgns-repartition-");
    const C::Json input{{"defaults", {{"mesh", {{"file", std::string(DNDS_TEST_SOURCE_ROOT) + "/data/mesh/IV10_20.cgns"}, {"dimension", 2}, {"periodicLengths", {10, 10, 0}}, {"boundaryIds", {{"PERIODIC_1", -1}, {"PERIODIC_1_DONOR", -4}, {"PERIODIC_2", -2}, {"PERIODIC_2_DONOR", -5}}}}}, {"numerics", {{"reconstruction", "constant"}, {"cfv", {{"maxOrder", 1}}}}}}}};
    const auto value = [](const C::Domain &domain, DNDS::index iCell)
    {
        const auto id = domain.Discretization().Mesh().OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell);
        return domain.Equation().Gas().Conservative({1 + .0001 * id, .31, -.12, .07, 1.2});
    };
    MPI_Comm writerComm = MPI_COMM_NULL;
    MPI_Comm_split(mpi.comm, mpi.rank < std::max(1, mpi.size / 2) ? 0 : MPI_UNDEFINED, mpi.rank, &writerComm);
    if (writerComm != MPI_COMM_NULL)
    {
        {
            const auto config = C::SimulationConfig::ResolveCollective(input, writerComm);
            C::Simulation simulation(config, writerComm);
            auto &domain = simulation.GetDomain(0);
            {
                auto write = domain.Flow().Owned().Write();
                for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
                {
                    const auto expected = value(domain, iCell);
                    for (int j = 0; j < 5; ++j)
                        write.View()(iCell, j) = expected[j];
                }
            }
            domain.WriteCheckpoint(path, config.Root());
        }
        MPI_Comm_free(&writerComm);
    }
    MPI_Barrier(mpi.comm);
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::Simulation restored(config, mpi.comm);
    restored.ReadCheckpoint(path);
    const auto &domain = restored.GetDomain(0);
    const auto read = domain.Flow().Owned().Read();
    for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
    {
        const auto expected = value(domain, iCell);
        for (int j = 0; j < 5; ++j)
            CHECK(read.View()(iCell, j) == expected[j]);
    }
    Remove({path});
}
TEST_CASE("Rigid motion restart retains absolute-state convention and BDF history")
{
    const auto path = Path("cauchy-rigid-frame-");
    const C::Json input{{"defaults", {{"motion", {{"angularVelocity", {0, 0, .2}}, {"center", {.4, .3, 0}}}}, {"numerics", {{"reconstruction", "constant"}}}}}, {"time", {{"method", "bdf2"}}}};
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 3);
    C::Domain source("flow", mesh, config.Domains()[0].settings, config.Time());
    source.Advance(.001);
    source.WriteCheckpoint(path, config.Root());
    {
        auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        std::string metadata;
        serializer->ReadString("metadata_json", metadata);
        const auto frame = C::Json::parse(metadata)["frame"];
        CHECK(frame["convention"] == "absolute-velocity-and-energy-in-mesh-axes");
        CHECK(frame["law"]["angularVelocity"][2] == .2);
        CHECK(frame["rotation_column_major"].size() == 9);
        serializer->CloseFile();
    }
    C::Domain restored("flow", mesh, config.Domains()[0].settings, config.Time());
    restored.ReadCheckpoint(path, config.Root());
    CHECK(restored.Time() == source.Time());
    source.Advance(.002);
    restored.Advance(.002);
    const auto a = source.Flow().Owned().Read(), b = restored.Flow().Owned().Read();
    const auto u = a.View<R::UniformTableRows>(), v = b.View<R::UniformTableRows>();
    for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(u(iCell, j) == v(iCell, j));
    Remove({path});
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    MPI_Finalize();
    return result;
}
