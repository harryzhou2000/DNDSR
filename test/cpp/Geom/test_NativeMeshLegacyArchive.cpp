#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Mesh/NativeMeshLegacyArchive.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <fstream>
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
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto &mesh = *fixture.mesh;
    const auto root = std::filesystem::temp_directory_path() / ("dnds-native-legacy-" + std::to_string(getpid()));
    SUBCASE("JSON")
    {
        auto serializer = DNDS::make_ssp<S::SerializerJSON>();
        const auto file = root.string() + ".json";
        serializer->OpenFile(file, false);
        mesh.AdjLocal2GlobalPrimary();
        mesh.WriteSerialize(serializer, "mesh");
        mesh.AdjGlobal2LocalPrimary();
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
        mesh.AdjLocal2GlobalPrimary();
        mesh.WriteSerialize(serializer, "mesh");
        mesh.AdjGlobal2LocalPrimary();
        serializer->CloseFile();
        for (auto partition : {S::ArrayGlobalOffset_Unknown, S::ArrayGlobalOffset_EvenSplit})
        {
            serializer->OpenFile(file, true);
            auto state = G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh", partition);
            serializer->CloseFile();
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
        auto fixture = G::OverlapTest::BuildFixture(self);
        auto serializer = DNDS::make_ssp<S::SerializerH5>(self);
        serializer->OpenFile(file, false);
        fixture.mesh->AdjLocal2GlobalPrimary();
        fixture.mesh->WriteSerialize(serializer, "mesh");
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
    auto mesh = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 2);
    G::UnstructuredMeshSerialRW reader(mesh, 0);
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    reader.ReadFromCGNSSerial((root / "data/mesh/Uniform32_Periodic.cgns").string());
    reader.Deduplicate1to1Periodic(1e-9);
    reader.BuildCell2Cell();
    reader.MeshPartitionCell2Cell({});
    reader.PartitionReorderToMeshCell2Cell();
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    const auto file = (std::filesystem::temp_directory_path() / ("dnds-legacy-periodic-" + std::to_string(getpid()) + ".json")).string();
    auto serializer = DNDS::make_ssp<S::SerializerJSON>();
    serializer->OpenFile(file, false);
    mesh->AdjLocal2GlobalPrimary();
    mesh->WriteSerialize(serializer, "mesh");
    mesh->AdjGlobal2LocalPrimary();
    serializer->CloseFile();
    serializer->OpenFile(file, true);
    auto state = G::NativeMeshLegacyArchive::Read(mpi, serializer, "mesh");
    serializer->CloseFile();
    CHECK(state->BorrowHostView().IsPeriodic());
    const auto view = state->BorrowHostView();
    CHECK(view.NumCell() == mesh->NumCell());
    DNDS::real total = 0;
    for (I cell = 0; cell < view.NumCell(); ++cell)
    {
        CHECK(state->OriginalEntityId(G::EntityKind::Cell, cell) == mesh->cell2cellOrig(cell, 0));
        const auto bits = view.CellNodePeriodicBits(cell);
        REQUIRE(bits.size() == mesh->cell2nodePbi[cell].size());
        for (size_t slot = 0; slot < bits.size(); ++slot)
            CHECK(uint8_t(bits[slot]) == uint8_t(mesh->cell2nodePbi[cell][DNDS::rowsize(slot)]));
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
