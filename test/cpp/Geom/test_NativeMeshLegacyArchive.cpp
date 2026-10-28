#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeMeshOverlapFixture.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "Geom/Mesh/NativeMeshRefinement.hpp"
#include "Geom/Mesh/NativeMeshLegacyArchive.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <fstream>
#include <limits>
namespace G = DNDS::Geom;
namespace S = DNDS::Serializer;
using I = DNDS::index;

static void CheckNative(const G::NativeMeshState &state, const DNDS::MPIInfo &mpi)
{
    const auto view = state.BorrowHostView();
    CHECK(state.NumGlobal(G::EntityKind::Cell) == 100);
    CHECK(state.NumGlobal(G::EntityKind::Node) == 121);
    CHECK(state.NumGlobal(G::EntityKind::Bnd) == 40);
    DNDS::real area = 0;
    for (I cell = 0; cell < view.NumCell(); ++cell)
    {
        CHECK(state.OriginalEntityId(G::EntityKind::Cell, cell) >= 0);
        const auto row = view.CellNodes(cell);
        REQUIRE(row.size() == 4);
        DNDS::real local = 0;
        for (size_t slot = 0; slot < row.size(); ++slot)
        {
            const auto a = view.Coord(row[slot]), b = view.Coord(row[(slot + 1) % row.size()]);
            local += a[0] * b[1] - a[1] * b[0];
        }
        CHECK(local / 2 == doctest::Approx(0.01));
        area += local / 2;
    }
    MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(area == doctest::Approx(1));
}

TEST_CASE("Legacy primary JSON and H5 decode directly to native owned rows")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    auto &mesh = *fixture.mesh;
    const auto root = std::filesystem::temp_directory_path() / ("dnds-native-legacy-" + std::to_string(getpid()));
    SUBCASE("JSON")
    {
        auto serializer = DNDS::make_ssp<S::SerializerJSON>();
        const auto file = root.string() + ".json";
        serializer->OpenFile(file, false);
        G::WriteCurrentPrimaryMeshSnapshot(mesh, serializer, "mesh");
        serializer->CloseFile();
        serializer->OpenFile(file, true);
        auto state = G::ReadCompatibleNativeMeshArchive(mpi, serializer, "mesh");
        CHECK(serializer->GetCurrentPath().empty());
        serializer->CloseFile();
        CheckNative(*state, mpi);
        std::ifstream input(file);
        nlohmann::json pristine;
        input >> pristine;
        for (int failure = 0; failure < 5; ++failure)
        {
            auto malformed = pristine;
            if (mpi.rank == 0)
            {
                if (failure == 0)
                    malformed["mesh"]["coords"]["father"]["size"] = -1;
                if (failure == 1)
                    malformed["mesh"]["cellElemInfo"]["father"]["array_sig"] = "bad-schema";
                if (failure == 2)
                    malformed["mesh"]["MPISize"] = 0;
                if (failure == 3)
                    malformed["mesh"]["cell2node"]["father"]["size"] = I(9999);
                if (failure == 4)
                {
                    auto &bytes = malformed["mesh"]["cellElemInfo"]["father"]["data"];
                    REQUIRE(bytes.is_array());
                    bytes.erase(bytes.end() - 1);
                }
            }
            {
                std::ofstream output(file);
                output << malformed;
            }
            serializer->OpenFile(file, true);
            CHECK_THROWS(G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh"));
            serializer->CloseFile();
        }
        {
            std::ofstream output(file);
            output << pristine;
        }
        serializer->OpenFile(file, true);
        CHECK_NOTHROW(G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh"));
        serializer->CloseFile();
        std::filesystem::remove(file);
    }
    SUBCASE("H5 matching and even split")
    {
        std::string file = root.string() + ".h5";
        int count = mpi.rank == 0 ? int(file.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
        file.resize(size_t(count));
        MPI_Bcast(file.data(), count, MPI_CHAR, 0, mpi.comm);
        auto serializer = DNDS::make_ssp<S::SerializerH5>(mpi);
        serializer->OpenFile(file, false);
        G::WriteCurrentPrimaryMeshSnapshot(mesh, serializer, "mesh");
        serializer->CloseFile();
        for (auto partition : {S::ArrayGlobalOffset_Unknown, S::ArrayGlobalOffset_EvenSplit})
        {
            serializer->OpenFile(file, true);
            auto owned = G::ReadCompatibleNativeMeshArchiveOwned(mpi, serializer, "mesh", partition);
            serializer->CloseFile();
            CHECK_FALSE(owned.retainedRead);
            for (const auto &[kind, space] : owned.spaces)
                CHECK(space.NumGhost() == 0);
            CHECK_NOTHROW(owned.coords->Write());
            auto state = G::AssembleNativeMesh(mpi, std::move(owned), 1);
            CheckNative(*state, mpi);
        }
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
            std::filesystem::remove(file);
    }
}
TEST_CASE("Legacy single-rank H5 imports with changed MPI size through native even split")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    std::string file = (std::filesystem::temp_directory_path() / ("dnds-legacy-self-" + std::to_string(getpid()) + ".h5")).string();
    int count = mpi.rank == 0 ? int(file.size()) : 0;
    MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
    file.resize(size_t(count));
    MPI_Bcast(file.data(), count, MPI_CHAR, 0, mpi.comm);
    G::FieldTransferDetail::LocalStage(mpi, [&]
                                       {
        if (mpi.rank != 0) return;
        DNDS::MPIInfo self(MPI_COMM_SELF);
        auto fixture = G::OverlapTest::BuildNativeFixture(self);
        auto serializer = DNDS::make_ssp<S::SerializerH5>(self);
        serializer->OpenFile(file, false);
        G::WriteCurrentPrimaryMeshSnapshot(*fixture.mesh, serializer, "mesh");
        serializer->CloseFile(); });
    auto serializer = DNDS::make_ssp<S::SerializerH5>(mpi);
    if (mpi.size != 1)
    {
        serializer->OpenFile(file, true);
        CHECK_THROWS(G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh"));
        serializer->CloseFile();
    }
    serializer->OpenFile(file, true);
    auto state = G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh", S::ArrayGlobalOffset_EvenSplit);
    serializer->CloseFile();
    CheckNative(*state, mpi);
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Legacy periodic primary import preserves exact ordered image bits and original IDs")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    G::NativePeriodicImportOptions periodic;
    periodic.tolerance = 1e-9; // Default unit translations match this unit-square file.
    G::AutoAppendName2ID names;
    auto primary = G::ReadNativeCGNS(mpi, (root / "data/mesh/Uniform32_Periodic.cgns").string(), 2, [&](const std::string &name)
                                     { return names(name); }, 0, {1, true}, periodic);
    auto partitioned = G::NativeMeshPreparation::Partition(*primary, mpi);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, std::move(partitioned.state));
    const auto file = (std::filesystem::temp_directory_path() / ("dnds-legacy-periodic-" + std::to_string(getpid()) + ".json")).string();
    auto serializer = DNDS::make_ssp<S::SerializerJSON>();
    serializer->OpenFile(file, false);
    G::WriteCurrentPrimaryMeshSnapshot(*mesh, serializer, "mesh");
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto state = G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh");
    serializer->CloseFile();
    CHECK(state->BorrowHostView().IsPeriodic());
    const auto view = state->BorrowHostView();
    auto original = mesh->AcquireRead();
    const auto source = original.HostView();
    CHECK(view.NumCell() == mesh->NumCell());
    DNDS::real total = 0;
    for (I cell = 0; cell < view.NumCell(); ++cell)
    {
        CHECK(state->OriginalEntityId(G::EntityKind::Cell, cell) == mesh->OriginalEntityId(G::EntityKind::Cell, cell));
        const auto bits = view.CellNodePeriodicBits(cell);
        REQUIRE(bits.size() == source.CellNodePeriodicBits(cell).size());
        for (size_t slot = 0; slot < bits.size(); ++slot)
            CHECK(uint8_t(bits[slot]) == uint8_t(source.CellNodePeriodicBits(cell)[slot]));
        G::tSmallCoords physical;
        view.GetCoordsOnCell(cell, physical);
        DNDS::real area = 0;
        for (Eigen::Index slot = 0; slot < physical.cols(); ++slot)
            area += physical(0, slot) * physical(1, (slot + 1) % physical.cols()) - physical(1, slot) * physical(0, (slot + 1) % physical.cols());
        CHECK(area > 0);
        total += area / 2;
    }
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(total == doctest::Approx(1));
    std::filesystem::remove(file);
}

TEST_CASE("Native archive helper coordinates rank-local missing and malformed JSON before retry")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    S::SerializerFactory factory("JSON");
    std::string base = (std::filesystem::temp_directory_path() / ("dnds-helper-fault-" + std::to_string(getpid()))).string();
    int length = mpi.rank == 0 ? int(base.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
    base.resize(size_t(length));
    MPI_Bcast(base.data(), length, MPI_CHAR, 0, mpi.comm);
    const auto file = std::get<0>(factory.ModifyFilePath(base, mpi, "part_%d", false));
    auto writer = factory.BuildSerializer(mpi);
    writer->OpenFile(file, false);
    fixture.mesh->WriteNativeArchive(writer, "meshPart");
    writer->CloseFile();
    std::ifstream input(file);
    const std::string pristine((std::istreambuf_iterator<char>(input)), {});
    input.close();
    for (int fault = 0; fault < 2; ++fault)
    {
        G::FieldTransferDetail::LocalStage(mpi, [&]
                                           {
            if (mpi.rank != 0) return;
            if (fault == 0) std::filesystem::rename(file, file + ".backup");
            else { std::ofstream output(file); output << "{ \"meshPart\": "; } });
        CHECK_THROWS(G::ReadNativeMeshFromArchive(mpi, factory, base));
        G::FieldTransferDetail::LocalStage(mpi, [&]
                                           {
            if (mpi.rank != 0) return;
            if (fault == 0) std::filesystem::rename(file + ".backup", file);
            else { std::ofstream output(file); output << pristine; } });
        auto restored = G::ReadNativeMeshFromArchive(mpi, factory, base);
        CHECK(restored->HasNativeState());
        CHECK(restored->NumCellGlobal() == 100);
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove_all(base + ".dir");
}

TEST_CASE("Legacy periodic high-order import rejects explicitly duplicated node IDs without renumbering")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    G::NativePeriodicImportOptions periodic;
    periodic.frames.translation[1].map() = G::tPoint{10, 0, 0};
    periodic.frames.translation[2].map() = G::tPoint{0, 10, 0};
    auto source = G::ReadNativeMeshFromCGNS(mpi, 2, (root / "data/mesh/IV10_10.cgns").string(),
                                            G::PartitionOptions{}, G::FBC_Name_2_ID_Default, periodic);
    CHECK(source->NumNodeGlobal() == 100);
    // The frozen legacy implementation produces unique compact IDs before and
    // after elevation. Deliberately corrupt a valid snapshot; do not rely on a
    // new-native fixture's ID bug to manufacture a supposed historical defect.
    auto native = G::NativeMeshRefinement::Elevate(*source->CurrentNativeState(), mpi).state;
    auto nativeFacade = G::UnstructuredMesh::FromNative(mpi, native);
    const I owned = native->NumOwned(G::EntityKind::Node);
    std::vector<I> local(size_t(owned), 0);
    I minimum = std::numeric_limits<I>::max(), maximum = -1;
    for (I row = 0; row < owned; ++row)
    {
        local[size_t(row)] = native->OriginalEntityId(G::EntityKind::Node, row);
        minimum = std::min(minimum, local[size_t(row)]);
        maximum = std::max(maximum, local[size_t(row)]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, DNDS::DNDS_MPI_INDEX, MPI_MIN, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, DNDS::DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
    REQUIRE(minimum < maximum);
    for (auto &id : local)
        if (id == maximum)
            id = minimum;
    int count = int(local.size());
    std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size) + 1, 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
    for (int rank = 0; rank < mpi.size; ++rank)
        offsets[size_t(rank) + 1] = offsets[size_t(rank)] + counts[size_t(rank)];
    std::vector<I> all(size_t(offsets.back()));
    MPI_Allgatherv(local.data(), count, DNDS::DNDS_MPI_INDEX, all.data(), counts.data(), offsets.data(), DNDS::DNDS_MPI_INDEX, mpi.comm);
    CHECK(std::set<I>(all.begin(), all.end()).size() < all.size());
    const auto file = (std::filesystem::temp_directory_path() / ("dnds-legacy-duplicates-" + std::to_string(getpid()) + ".json")).string();
    auto serializer = DNDS::make_ssp<S::SerializerJSON>();
    serializer->OpenFile(file, false);
    G::WriteCurrentPrimaryMeshSnapshot(*nativeFacade, serializer, "mesh", 1, G::CurrentMeshCheckpointPolicy::DropAuxiliary);
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    CHECK_NOTHROW(G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh"));
    serializer->CloseFile();
    nlohmann::json malformed;
    {
        std::ifstream input(file);
        input >> malformed;
    }
    malformed["mesh"]["node2nodeOrig"]["father"]["data"] = local;
    {
        std::ofstream output(file);
        output << malformed;
    }
    serializer->OpenFile(file, true);
    CHECK_THROWS(G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh"));
    serializer->CloseFile();
    serializer->OpenFile(file, false);
    G::WriteCurrentPrimaryMeshSnapshot(*source, serializer, "mesh", 1, G::CurrentMeshCheckpointPolicy::DropAuxiliary);
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    CHECK_NOTHROW(G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh"));
    serializer->CloseFile();
    std::filesystem::remove(file);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
