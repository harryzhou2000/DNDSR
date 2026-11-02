#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "CartesianFixture.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <filesystem>
#include <unistd.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;
namespace
{
    std::string Path(const char *label)
    {
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() / (std::string(label) + std::to_string(pid) + ".h5")).string();
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
TEST_CASE("Implicit BDF2 restart preserves variable-step history and compatible reconstruction continuation")
{
    const auto path = Path("cauchy-bdf-history-");
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 2;
    settings.numerics.reconstructionIterations = 80;
    settings.numerics.reconstructionTolerance = 1e-11;
    settings.implicit.minIterations = 0;
    settings.implicit.relativeTolerance = 1e-9;
    settings.implicit.linear.relativeTolerance = 1e-3;
    C::TimeSettings time;
    time.method = "bdf2";
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::Domain source("flow", mesh, settings, time), restored("flow", mesh, settings, time);
    Initialize(source);
    source.Advance(.001);
    source.Advance(.002);
    const C::Json metadata{{"test", "variable BDF2"}};
    source.WriteCheckpoint(path, metadata, true);
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(restored.Flow().Owned().Read());
    CHECK_THROWS(restored.ReadCheckpoint(path, metadata));
    CHECK(restored.Time() == 0);
    CHECK(restored.Epoch().acceptedStep == 0);
    held.reset();
    restored.ReadCheckpoint(path, metadata);
    CHECK(restored.Time() == source.Time());
    CHECK(restored.Epoch().acceptedStep == source.Epoch().acceptedStep);
    source.Advance(.0015);
    restored.Advance(.0015);
    const auto a = source.Flow().Owned().Read(), b = restored.Flow().Owned().Read();
    for (DNDS::index iCell = 0; iCell < source.Discretization().Metrics().NumCell(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(a.View()(iCell, j) == b.View()(iCell, j));
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
