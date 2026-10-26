#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshOpenFOAM.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh/FieldTransfer.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include <filesystem>
#include <fstream>
#include <unistd.h>
namespace G = DNDS::Geom;
using I = DNDS::index;

static void Fixture(const std::string &path, bool tetra)
{
    std::filesystem::create_directories(path);
    auto file = [&](const char *name, const std::string &body)
    { std::ofstream out(std::filesystem::path(path) / name); out << "FoamFile { version 2.0; format ascii; class labelList; object " << name << "; }\n" << body; };
    file("points", tetra ? "4\n((0 0 0)(1 0 0)(0 1 0)(0 0 1))\n" : "8\n((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 1)(1 0 1)(1 1 1)(0 1 1))\n");
    file("faces", tetra ? "4\n(3(0 2 1)3(0 1 3)3(1 2 3)3(2 0 3))\n" : "6\n(4(0 3 2 1)4(4 5 6 7)4(0 1 5 4)4(1 2 6 5)4(2 3 7 6)4(3 0 4 7))\n");
    file("owner", tetra ? "4\n(0 0 0 0)\n" : "6\n(0 0 0 0 0 0)\n");
    file("neighbour", "0\n()\n");
    file("boundary", tetra ? "1\n(patchA { type wall; nFaces 4; startFace 0; })\n" : "1\n(patchA { type wall; nFaces 6; startFace 0; })\n");
}

TEST_CASE("Native OpenFOAM canonical Hex8 matches legacy slots and boundary remap")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    std::string path = (std::filesystem::temp_directory_path() / ("dnds-native-foam-" + std::to_string(getpid()))).string();
    int count = mpi.rank == 0 ? int(path.size()) : 0;
    MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
    path.resize(size_t(count));
    MPI_Bcast(path.data(), count, MPI_CHAR, 0, mpi.comm);
    G::FieldTransferDetail::LocalStage(mpi, [&]
                                       { if (mpi.rank == 0) Fixture(path, false); });
    const std::map<std::string, std::string> mapping{{"patchA", "named-wall"}};
    auto name = [](const std::string &value) -> G::t_index
    { DNDS_check_throw_info(value == "named-wall", "missing mapped name"); return 24; };
    const auto state = G::ReadNativeOpenFOAM(mpi, path, mapping, name, mpi.size - 1);
    CHECK(state->NumGlobal(G::EntityKind::Cell) == 1);
    CHECK(state->NumGlobal(G::EntityKind::Node) == 8);
    CHECK(state->NumGlobal(G::EntityKind::Bnd) == 6);
    const auto view = state->BorrowHostView();
    auto nodes = state->Entities(G::EntityKind::Node);
    const I expected[]{1, 2, 3, 0, 5, 6, 7, 4};
    for (I cell = 0; cell < view.NumCell(); ++cell)
    {
        CHECK(view.GetCellInfo(cell).getElemType() == G::Elem::Hex8);
        REQUIRE(view.CellNodes(cell).size() == 8);
        for (size_t slot = 0; slot < 8; ++slot)
            CHECK(nodes.LocalToGlobal(view.CellNodes(cell)[slot]) == expected[slot]);
        CHECK(state->OriginalEntityId(G::EntityKind::Cell, cell) == 0);
        G::tSmallCoords points;
        view.GetCoordsOnCell(cell, points);
        G::Elem::Quadrature quad(G::Elem::Element{G::Elem::Hex8}, 2);
        DNDS::real volume = 0;
        quad.Integration(volume, [&](DNDS::real &increment, int, G::tPoint, G::Elem::tD01Nj derivatives)
                         { increment = G::Elem::ShapeJacobianCoordD01Nj(points, derivatives).determinant(); });
        CHECK(volume == doctest::Approx(1));
    }
    for (I boundary = 0; boundary < view.NumBnd(); ++boundary)
        CHECK(view.GetBndInfo(boundary).zone == 24);
    // The old converter is the slot-order compatibility baseline; it is never
    // instantiated by the production native path.
    auto legacy = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 3);
    G::UnstructuredMeshSerialRW reader(legacy, 0);
    reader.ReadFromOpenFOAMAndConvertSerial(path, mapping, name);
    if (mpi.rank == 0)
        for (size_t slot = 0; slot < 8; ++slot)
            CHECK((*reader.cell2nodeSerial)[0][slot] == expected[slot]);
    CHECK_THROWS(G::ReadNativeOpenFOAM(mpi, path + ".missing", mapping, name));
    G::FieldTransferDetail::LocalStage(mpi, [&]
                                       { if (mpi.rank == 0) Fixture(path, true); });
    CHECK_THROWS(G::ReadNativeOpenFOAM(mpi, path, mapping, name));
    G::FieldTransferDetail::LocalStage(mpi, [&]
                                       { if (mpi.rank == 0) std::filesystem::remove_all(path); });
}
TEST_CASE("Native serial reader-owned policy and already-O2 derivation retain intended order")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    auto primary = G::ReadNativeCGNS(mpi, (root / "data/mesh/UniformSquare_10.cgns").string(), 2, [](const std::string &) -> G::t_index
                                     { return 24; }, mpi.size - 1, {1, true}, {}, G::NativeSerialDistribution::ReaderOwned);
    CHECK(primary->NumOwned(G::EntityKind::Cell) == (mpi.rank == mpi.size - 1 ? 100 : 0));
    CHECK(primary->NumOwned(G::EntityKind::Node) == (mpi.rank == mpi.size - 1 ? 121 : 0));
    auto quadratic = G::NativeMeshRefinement::Elevate(*primary, mpi, {1, true}).state;
    const auto identity = quadratic->MeshNamespace();
    G::PartitionOptions options;
    auto retained = G::DeriveAndPartitionNativeMesh(quadratic, mpi, options, 1, 0);
    CHECK(retained->MeshNamespace() == identity);
    CHECK(retained->NumGlobal(G::EntityKind::Cell) == 100);
    for (I row = 0; row < retained->NumOwned(G::EntityKind::Cell); ++row)
        CHECK(retained->BorrowHostView().GetCellInfo(row).getElemType() == G::Elem::Quad9);
    auto bisected = G::DeriveAndPartitionNativeMesh(quadratic, mpi, options, 0, 1);
    CHECK(bisected->NumGlobal(G::EntityKind::Cell) == 400);
    CHECK(bisected->NumGlobal(G::EntityKind::Node) == 441);
    for (I row = 0; row < bisected->NumOwned(G::EntityKind::Cell); ++row)
        CHECK(bisected->BorrowHostView().GetCellInfo(row).getElemType() == G::Elem::Quad4);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
