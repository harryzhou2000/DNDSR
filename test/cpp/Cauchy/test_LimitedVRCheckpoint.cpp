#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/IO/EquationArchive.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "DNDS/Runtime/Array/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <filesystem>
#include <map>
#include <limits>
#include <unistd.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    std::string Path(const std::string &label)
    {
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        return (std::filesystem::temp_directory_path() / ("cauchy-lvr-" + label + std::to_string(pid) + ".h5")).string();
    }
    C::DomainSettings Settings(const std::string &variant, const std::string &layout)
    {
        C::DomainSettings settings;
        settings.execution.layout = layout;
        settings.boundaries[std::to_string(G::BC_ID_DEFAULT_WALL)].kind = "extrapolate";
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
        settings.numerics.reconstructionIterations = 2;
        settings.numerics.flux = "Roe_M9";
        settings.numerics.limiter = "LVR";
        settings.numerics.limitedVR = DNDS::CFV::LimitedVRSettings::ForVariant(variant);
        settings.numerics.positivity = true;
        settings.numerics.faceFallback = false;
        settings.implicit.iterations = 2;
        return settings;
    }
    C::Json Root(const C::DomainSettings &settings, const C::TimeSettings &time, int ranks = 0)
    {
        return C::SimulationConfig::Resolve({{"defaults", settings}, {"time", time}}, ranks ? ranks : mpi.size).Root();
    }
    void Initialize(C::Domain &domain)
    {
        auto write = domain.Flow().Owned().Write();
        const auto output = write.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < output.RowCount(); ++iCell)
        {
            const bool left = domain.Discretization().Metrics().GetCellCent(iCell)[0] < .5;
            const auto state = domain.Equation().Gas().Conservative({left ? 8. : 1.4, left ? 7. : 0., 0, 0, left ? 116.5 : 1.});
            for (int j = 0; j < 5; ++j)
                output(iCell, j) = state[j];
        }
    }
    struct WarmState
    {
        // Row-keyed independent oracle: file row order may change under a
        // local mesh permutation and physical layouts may differ on reload.
        std::map<I, std::vector<double>> rows;
        double time = 0;
        I step = 0;
        double coefficientNorm = 0, historyNorm = 0;
    };
    WarmState ReadWarm(const std::string &path, DNDS::Serializer::ArrayGlobalOffset region = DNDS::Serializer::ArrayGlobalOffset_Unknown)
    {
        auto serializer = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path, true);
        WarmState result;
        serializer->ReadReal("accepted_time", result.time);
        serializer->ReadIndex("accepted_step", result.step);
        auto ids = R::ReadArray<I>(serializer, "cell_ids", mpi, region);
        std::string metadataText;
        serializer->ReadString("metadata_json", metadataText);
        const auto metadata = C::Json::parse(metadataText);
        auto state = C::EquationArchive::Read(serializer, "state_field_", metadata.at("state_storage"), mpi, region);
        auto reconstruction = C::EquationArchive::Read(serializer, "reconstruction_field_", metadata.at("reconstruction_storage"), mpi, region);
        REQUIRE(state.fields.size() == 1);
        REQUIRE(reconstruction.fields.size() == 1);
        auto flow = std::move(state.fields[0]);
        auto coefficients = std::move(reconstruction.fields[0]);
        auto history = R::ReadArray<double>(serializer, "limited_vr_history", mpi, region);
        serializer->CloseFile();
        const auto idRead = ids.Read();
        const auto flowRead = flow.Read(), coefficientRead = coefficients.Read(), historyRead = history.Read();
        const auto id = idRead.View<R::FixedRows<1>>();
        const auto u = flowRead.View<R::UniformTableRows>(), r = coefficientRead.View<R::UniformTableRows>();
        const auto h = historyRead.View<R::FixedRows<2>>();
        REQUIRE(u.RowCount() == id.RowCount());
        REQUIRE(r.RowCount() == id.RowCount());
        REQUIRE(h.RowCount() == id.RowCount());
        for (I iCell = 0; iCell < id.RowCount(); ++iCell)
        {
            auto &row = result.rows[id(iCell, 0)];
            for (I j = 0; j < u.RowSize(iCell); ++j)
                row.push_back(u(iCell, j));
            for (I j = 0; j < r.RowSize(iCell); ++j)
            {
                row.push_back(r(iCell, j));
                result.coefficientNorm += std::abs(r(iCell, j));
            }
            for (int j = 0; j < 2; ++j)
            {
                row.push_back(h(iCell, j));
                result.historyNorm += std::abs(h(iCell, j));
            }
        }
        return result;
    }
    void Same(const WarmState &a, const WarmState &b)
    {
        CHECK(a.time == b.time);
        CHECK(a.step == b.step);
        CHECK(a.rows == b.rows);
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
TEST_CASE("LVR warm restart and output cadence preserve the complete implicit trajectory")
{
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 12).mesh;
    for (const std::string variant : {"v2", "v3"})
        for (const std::string layout : {"aos", "soa", "aosoa32"})
            for (const std::string method : {"bdf1", "ditr"})
            {
                const auto settings = Settings(variant, layout);
                C::TimeSettings time;
                time.method = method;
                time.ditr.iterations = 2;
                time.ditr.absoluteTolerance = 1e-100;
                time.ditr.relativeTolerance = 0;
                const auto root = Root(settings, time);
                const auto warm = Path("warm-"), rewritten = Path("rewritten-"), cold = Path("cold-"), coldWritten = Path("cold-written-");
                C::Domain source("fluid", mesh, settings, time), resumed("fluid", mesh, settings, time);
                Initialize(source);
                REQUIRE(source.Advance(1e-6).accepted);
                if (method == "ditr")
                {
                    CHECK(source.LastCoupledIterations() == 2);
                    CHECK(source.LastCoupledTargetRatio() > 1);
                }
                source.WriteCheckpoint(warm, root, true);
                source.WriteCheckpoint(cold, root);
                const auto accepted = ReadWarm(warm);
                double norms[]{accepted.coefficientNorm, accepted.historyNorm};
                MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
                CHECK(norms[0] > 1e-6);
                CHECK(norms[1] > .1); // An all-zero gate could not test history restore.
                resumed.ReadCheckpoint(warm, root);
                resumed.WriteCheckpoint(rewritten, root, true);
                Same(accepted, ReadWarm(rewritten));

                // Observations are frequent on one trajectory and absent on the
                // other. Neither coefficients nor retained limiter gates may drift.
                source.Diagnostics(settings, {});
                source.Diagnostics(settings, {});
                REQUIRE(source.Advance(1e-6).accepted);
                REQUIRE(resumed.Advance(1e-6).accepted);
                source.WriteCheckpoint(warm, root, true);
                resumed.WriteCheckpoint(rewritten, root, true);
                Same(ReadWarm(warm), ReadWarm(rewritten));

                const auto beforeTrial = ReadWarm(warm);
                REQUIRE(source.PrepareStep(1e-6).accepted);
                source.WriteCheckpoint(warm, root, true);
                Same(beforeTrial, ReadWarm(warm)); // Pending trial stores accepted warm state.
                source.DiscardStep();
                REQUIRE(source.Advance(1e-6).accepted);
                REQUIRE(resumed.Advance(1e-6).accepted);
                source.WriteCheckpoint(warm, root, true);
                resumed.WriteCheckpoint(rewritten, root, true);
                Same(ReadWarm(warm), ReadWarm(rewritten));

                resumed.ReadCheckpoint(cold, root);
                resumed.WriteCheckpoint(coldWritten, root, true);
                const auto coldState = ReadWarm(coldWritten);
                CHECK(coldState.coefficientNorm == 0);
                CHECK(coldState.historyNorm == 0);
                CHECK(coldState.time == accepted.time);
                REQUIRE(coldState.rows.size() == accepted.rows.size());
                for (const auto &[id, row] : accepted.rows)
                    for (int j = 0; j < 5; ++j)
                        CHECK(coldState.rows.at(id)[j] == row[j]);
                Remove({warm, rewritten, cold, coldWritten});
            }
}
TEST_CASE("LVR checkpoint remaps warm history under local permutation and cold-starts a new degree")
{
    auto settings = Settings("v3", "aosoa32");
    settings.numerics.cfv.maxOrder = 2;
    C::TimeSettings time;
    time.method = "bdf1";
    const auto root = Root(settings, time);
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 12).mesh;
    C::Domain source("fluid", mesh, settings, time);
    Initialize(source);
    REQUIRE(source.Advance(1e-6).accepted);
    const auto warm = Path("permutation-"), rewritten = Path("permutation-read-");
    source.WriteCheckpoint(warm, root, true);
    const auto expected = ReadWarm(warm);
    auto newMesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 12).mesh;
    newMesh->ReorderLocalCells();
    settings.execution.layout = "soa";
    auto newRoot = Root(settings, time);
    C::Domain permuted("fluid", newMesh, settings, time);
    permuted.ReadCheckpoint(warm, newRoot);
    permuted.WriteCheckpoint(rewritten, newRoot, true);
    Same(expected, ReadWarm(rewritten));
    // Prove a real row permutation occurred, not just a comparison of two
    // unchanged fixture orders.
    I changed = 0;
    for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
        changed += mesh->OriginalEntityId(G::EntityKind::Cell, iCell) != newMesh->OriginalEntityId(G::EntityKind::Cell, iCell);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(changed > 0);
    settings.numerics.cfv.maxOrder = 3;
    newRoot = Root(settings, time);
    C::Domain higher("fluid", newMesh, settings, time);
    higher.ReadCheckpoint(warm, newRoot);
    higher.WriteCheckpoint(rewritten, newRoot, true);
    const auto cold = ReadWarm(rewritten);
    CHECK(cold.coefficientNorm == 0);
    CHECK(cold.historyNorm == 0);
    for (const auto &[id, row] : expected.rows)
        for (int j = 0; j < 5; ++j)
            CHECK(cold.rows.at(id)[j] == row[j]);
    Remove({warm, rewritten});
}
TEST_CASE("LVR warm restart redistributes nonzero history from fewer writers into reordered strided fields")
{
    auto settings = Settings("v3", "aosoa32");
    C::TimeSettings time;
    time.method = "bdf1";
    const auto warm = Path("different-ranks-"), rewritten = Path("different-ranks-read-");
    const int writerNumber = std::max(1, mpi.size / 2);
    MPI_Comm writerComm = MPI_COMM_NULL;
    MPI_Comm_split(mpi.comm, mpi.rank < writerNumber ? 0 : MPI_UNDEFINED, mpi.rank, &writerComm);
    if (writerComm != MPI_COMM_NULL)
    {
        const DNDS::MPIInfo writers(writerComm);
        {
            auto mesh = G::OverlapTest::BuildNativeFixture(writers, false, 12).mesh;
            C::Domain source("fluid", mesh, settings, time);
            Initialize(source);
            REQUIRE(source.Advance(1e-6).accepted);
            source.WriteCheckpoint(warm, Root(settings, time, writers.size), true);
        }
        MPI_Comm_free(&writerComm);
    }
    MPI_Barrier(mpi.comm);
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 12).mesh;
    I first = mesh->NumCell() ? std::numeric_limits<I>::max() : 0;
    for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
        first = std::min(first, mesh->OriginalEntityId(G::EntityKind::Cell, iCell));
    // This independent fixture numbers row-major cell IDs in the same order
    // as the original writer's file rows, irrespective of its rank count.
    // Read only the current owner's contiguous ID window, without using the
    // checkpoint reader's ID router to manufacture its expected answer.
    const auto expected = ReadWarm(warm, {mesh->NumCell(), first});
    for (I iCell = 0; iCell < mesh->NumCell(); ++iCell)
        CHECK(expected.rows.count(mesh->OriginalEntityId(G::EntityKind::Cell, iCell)) == 1);
    double historyNorm = expected.historyNorm;
    MPI_Allreduce(MPI_IN_PLACE, &historyNorm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(historyNorm > .1);
    mesh->ReorderLocalCells();
    settings.execution.layout = "soa";
    const auto root = Root(settings, time);
    C::Domain target("fluid", mesh, settings, time);
    target.ReadCheckpoint(warm, root);
    target.WriteCheckpoint(rewritten, root, true);
    Same(expected, ReadWarm(rewritten));
    Remove({warm, rewritten});
}
TEST_CASE("LVR fine reconstruction connects to limited-gradient and constant polynomial FAS levels")
{
    const auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 12).mesh;
    auto settings = Settings("v3", "soa");
    settings.implicit.multigrid.levels = 2;
    settings.implicit.multigrid.cycles = 1;
    settings.implicit.multigrid.preSmooth = 1;
    settings.implicit.multigrid.postSmooth = 1;
    settings.implicit.multigrid.coarseSmooth = 1;
    C::TimeSettings time;
    time.method = "bdf1";
    C::Domain domain("fluid", mesh, settings, time);
    Initialize(domain);
    REQUIRE(domain.Advance(1e-6).accepted);
    CHECK(domain.Epoch().acceptedStep == 1);
    CHECK(domain.Discretization().Admissible(domain.Flow()));
    REQUIRE(domain.LastImplicitStages().size() == 1);
    const auto &report = domain.LastImplicitStages()[0];
    CHECK(report.multigridCycles == 1);
    REQUIRE(report.levelCorrections.size() == 3);
    for (const auto corrections : report.levelCorrections)
        CHECK(corrections > 0);
    const auto warm = Path("fas-levels-");
    domain.WriteCheckpoint(warm, Root(settings, time), true);
    double historyNorm = ReadWarm(warm).historyNorm;
    MPI_Allreduce(MPI_IN_PLACE, &historyNorm, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(historyNorm > .1); // Prove the fine LVR path remained active.
    Remove({warm});
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
