#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshArchive.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativePeriodicMatch.hpp"
#include "Geom/Mesh/RelationAccess.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include "NativePeriodicFixtures.hpp"
#include <filesystem>
#include <unistd.h>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    DNDS::MPIInfo MPI()
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        return mpi;
    }
    template <class T>
    R::Array<T> Rows(const std::vector<std::vector<T>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(T));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            std::copy(rows[row].begin(), rows[row].end(), builder.Row<T>(row).data);
        }
        return R::Array<T>::FromSharedStorage(builder.Commit());
    }
    G::NativeMeshInput Strip(const DNDS::MPIInfo &mpi, int dimension, bool boundaries = true)
    {
        const I width = dimension == 2 ? 2 : 4;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? width : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 1 : 0);
        auto bnd = nodes.CreateSibling(G::EntityKind::Bnd, boundaries && mpi.rank == mpi.size / 2 ? 2 : 0);
        std::vector<std::vector<I>> cones, boundaryCones;
        std::vector<std::vector<G::NodePeriodicBits>> cellBits, boundaryBits;
        if (mpi.rank == 0)
        {
            cones = dimension == 2 ? std::vector<std::vector<I>>{{0, 0, 1, 1}} : std::vector<std::vector<I>>{{0, 0, 1, 1, 2, 2, 3, 3}};
            cellBits = dimension == 2 ? std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {1}, {1}, {0}}} : std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}}};
        }
        if (bnd.NumOwned())
        {
            boundaryCones = dimension == 2 ? std::vector<std::vector<I>>{{0, 1}, {0, 1}} : std::vector<std::vector<I>>{{0, 1, 3, 2}, {0, 2, 3, 1}};
            boundaryBits = dimension == 2 ? std::vector<std::vector<G::NodePeriodicBits>>{{{1}, {1}}, {{0}, {0}}} : std::vector<std::vector<G::NodePeriodicBits>>{{{1}, {1}, {1}, {1}}, {{0}, {0}, {0}, {0}}};
        }
        G::NativeMeshInput input{mpi, dimension, cells, nodes, bnd, Rows(cones), Rows(boundaryCones),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(bnd.NumOwned()), 1))};
        input.periodic = true;
        input.cellNodesPbi = Rows(cellBits);
        input.boundaryNodesPbi = Rows(boundaryBits);
        {
            auto coords = input.coordinates.Write();
            for (I node = 0; node < nodes.NumOwned(); ++node)
            {
                coords.View()(size_t(node), 0) = 0;
                coords.View()(size_t(node), 1) = DNDS::real(node % 2);
                coords.View()(size_t(node), 2) = DNDS::real(node / 2);
            }
            if (cells.NumOwned())
            {
                auto info = input.cellInfo.Write();
                info.View()(0, 0) = G::ElemInfo{dimension == 2 ? G::Elem::Quad4 : G::Elem::Hex8, G::INTERNAL_ZONE};
            }
            if (bnd.NumOwned())
            {
                auto info = input.boundaryInfo.Write();
                for (size_t row = 0; row < 2; ++row)
                    info.View()(row, 0) = G::ElemInfo{dimension == 2 ? G::Elem::Line2 : G::Elem::Quad4, row == 0 ? G::BC_ID_PERIODIC_1_DONOR : G::BC_ID_PERIODIC_1};
            }
        }
        return input;
    }
    void Check(const G::NativeMeshState &state, int dimension)
    {
        CHECK(state.BorrowHostView().IsPeriodic());
        CHECK(state.NumGlobal(G::EntityKind::Face) == (dimension == 2 ? 3 : 5));
        if (dimension == 3)
            CHECK(state.NumGlobal(G::EntityKind::Edge) == 8);
        const auto &view = state.BorrowHostView();
        for (I cell = 0; cell < state.NumOwned(G::EntityKind::Cell); ++cell)
        {
            const auto faces = view.CellFaces(cell);
            const auto left = dimension == 2 ? 3 : 4, right = dimension == 2 ? 1 : 2;
            CHECK(faces[left] == faces[right]);
            CHECK(uint8_t(view.CellFacePeriodicBits(cell)[left]) != uint8_t(view.CellFacePeriodicBits(cell)[right]));
            CHECK(view.CellFaceCells(cell)[left] == cell);
            CHECK(view.CellFaceCells(cell)[right] == cell);
            const auto pair = view.FaceCells(faces[left]);
            CHECK(pair[0] == pair[1]);
            for (size_t slot = 0; slot < view.CellNodes(cell).size(); ++slot)
            {
                const auto expectedX = slot % 4 == 1 || slot % 4 == 2 ? 1. : 0.;
                CHECK(view.GetCoordNodeOnCell(cell, DNDS::rowsize(slot))[0] == expectedX);
            }
        }
        for (I boundary = 0; boundary < state.NumOwned(G::EntityKind::Bnd); ++boundary)
        {
            const auto pair = view.BoundaryCells(boundary);
            CHECK(pair[0] == pair[1]);
            CHECK(view.BoundaryFaces(boundary)[0] != DNDS::UnInitIndex);
        }
    }
}

TEST_CASE("Native periodic strip constructs exact self incidences and preserves them through archive and redistribution")
{
    const auto mpi = MPI();
    int dimension = 2;
    SUBCASE("quadrilateral") {}
    SUBCASE("hexahedron with edges") { dimension = 3; }
    auto primary = G::NativeMeshBuilder::Build(Strip(mpi, dimension));
    auto state = G::NativeMeshTopology::Build(*primary, mpi, true, dimension == 3);
    Check(*state, dimension);
    const auto name = state->MeshNamespace();
    G::ReorderInput reorder;
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
        if (state->HasEntities(kind))
            reorder.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(state->NumOwned(kind)), (mpi.rank + 1) % mpi.size)});
    state = G::NativeMeshTransition::Build(*state, mpi, reorder).state;
    primary.reset();
    Check(*state, dimension);
    CHECK(state->MeshNamespace() == name);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("native-periodic-" + std::to_string(pid) + "-r" + std::to_string(mpi.rank) + ".json")).string();
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->OpenFile(path, false);
    G::WriteNativeMeshArchive(*state, mpi, serializer, "mesh");
    serializer->CloseFile();
    state.reset();
    serializer->OpenFile(path, true);
    state = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    Check(*state, dimension);
    CHECK(state->MeshNamespace() == name);
    std::filesystem::remove(path);
}

TEST_CASE("Native periodic primary rejects invalid image metadata collectively and retains valid input")
{
    const auto mpi = MPI();
    auto input = Strip(mpi, 2);
    auto bad = input;
    if (mpi.rank == 0)
        bad.cellNodesPbi.reset();
    CHECK_THROWS(G::NativeMeshBuilder::Build(std::move(bad)));
    bad = input;
    if (mpi.rank == 0)
        bad.cellNodesPbi = Rows<G::NodePeriodicBits>({{{0}, {0}, {1}, {0}}});
    CHECK_THROWS(G::NativeMeshBuilder::Build(std::move(bad)));
    bad = input;
    if (mpi.rank == 0)
        bad.periodicInfo.translation[1].map()[0] = 2;
    if (mpi.size > 1)
        CHECK_THROWS(G::NativeMeshBuilder::Build(std::move(bad)));
    CHECK_NOTHROW(G::NativeMeshBuilder::Build(input));
}

TEST_CASE("Exact periodic frame matching handles repeated IDs whose XOR changes sorted order")
{
    CHECK(G::NativePeriodicFrame({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {{0, 2}, {0, 3}, {1, 2}, {1, 3}}) == 2);
    CHECK(G::NativePeriodicFrame({{0, 0}, {0, 2}, {1, 0}, {1, 2}}, {{0, 1}, {0, 3}, {1, 1}, {1, 3}}) == 1);
    CHECK_FALSE(G::NativePeriodicFrame({{0, 0}, {0, 1}, {1, 0}}, {{0, 2}, {0, 3}, {1, 4}}));
}

TEST_CASE("One-cell doubly periodic native torus retains two distinct self faces")
{
    const auto mpi = MPI();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 1 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 1 : 0);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, mpi.rank == mpi.size / 2 ? 4 : 0);
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries,
                             Rows<I>(cells.NumOwned() ? std::vector<std::vector<I>>{{0, 0, 0, 0}} : std::vector<std::vector<I>>{}),
                             Rows<I>(boundaries.NumOwned() ? std::vector<std::vector<I>>(4, {0, 0}) : std::vector<std::vector<I>>{}),
                             R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(boundaries.NumOwned()), 1))};
    input.periodic = true;
    input.cellNodesPbi = Rows<G::NodePeriodicBits>(cells.NumOwned() ? std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {1}, {3}, {2}}} : std::vector<std::vector<G::NodePeriodicBits>>{});
    input.boundaryNodesPbi = Rows<G::NodePeriodicBits>(boundaries.NumOwned() ? std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {1}}, {{1}, {3}}, {{3}, {2}}, {{2}, {0}}} : std::vector<std::vector<G::NodePeriodicBits>>{});
    {
        auto write = input.coordinates.Write();
        for (size_t row = 0; row < input.coordinates.Rows(); ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = 0;
    }
    if (cells.NumOwned())
    {
        auto write = input.cellInfo.Write();
        write.View()(0, 0) = G::ElemInfo{G::Elem::Quad4, G::INTERNAL_ZONE};
    }
    if (boundaries.NumOwned())
    {
        auto write = input.boundaryInfo.Write();
        const G::t_index zones[]{G::BC_ID_PERIODIC_2, G::BC_ID_PERIODIC_1_DONOR, G::BC_ID_PERIODIC_2_DONOR, G::BC_ID_PERIODIC_1};
        for (size_t row = 0; row < 4; ++row)
            write.View()(row, 0) = G::ElemInfo{G::Elem::Line2, zones[row]};
    }
    auto source = G::NativeMeshBuilder::Build(std::move(input));
    auto state = G::NativeMeshTopology::Build(*source, mpi);
    CHECK(state->NumGlobal(G::EntityKind::Face) == 2);
    if (cells.NumOwned())
    {
        const auto &view = state->BorrowHostView();
        const auto faces = view.CellFaces(0);
        CHECK(faces[0] == faces[2]);
        CHECK(faces[1] == faces[3]);
        CHECK(faces[0] != faces[1]);
        const auto bits = view.CellFacePeriodicBits(0);
        CHECK(uint8_t(bits[0]) == 0);
        CHECK(uint8_t(bits[1]) == 0);
        CHECK(uint8_t(bits[2]) == 2);
        CHECK(uint8_t(bits[3]) == 1);
    }
}

TEST_CASE("Partitioned native periodic cells distinguish internal physical and periodic faces")
{
    const auto mpi = MPI();
    auto primary = G::NativeMeshBuilder::Build(NativePeriodicFixtures::TwoCells(mpi));
    auto state = G::NativeMeshTopology::Build(*primary, mpi);
    CHECK(state->NumGlobal(G::EntityKind::Face) == 6);
    const auto &view = state->BorrowHostView();
    auto cells = state->Entities(G::EntityKind::Cell);
    for (I cell = 0; cell < cells.NumOwned(); ++cell)
    {
        const auto global = cells.LocalToGlobal(cell);
        const auto neighbors = view.CellFaceCells(cell);
        REQUIRE(neighbors.size() == 4);
        CHECK(neighbors[0] == DNDS::UnInitIndex);
        CHECK(neighbors[2] == DNDS::UnInitIndex);
        CHECK(cells.LocalToGlobal(neighbors[1]) == 1 - global);
        CHECK(cells.LocalToGlobal(neighbors[3]) == 1 - global);
        CHECK(view.CellFaces(cell)[1] != view.CellFaces(cell)[3]);
        CHECK(G::FaceIDIsPeriodic(view.GetFaceInfo(view.CellFaces(cell)[global == 0 ? 3 : 1]).zone));
        CHECK(view.GetFaceInfo(view.CellFaces(cell)[global == 0 ? 1 : 3]).zone == G::BC_ID_INTERNAL);
    }
    for (I boundary = 0; boundary < state->NumOwned(G::EntityKind::Bnd); ++boundary)
    {
        const auto pair = view.BoundaryCells(boundary);
        if (G::FaceIDIsPeriodic(view.GetBndInfo(boundary).zone))
        {
            CHECK(pair[0] != pair[1]);
            CHECK(pair[1] != DNDS::UnInitIndex);
        }
        else
            CHECK(pair[1] == DNDS::UnInitIndex);
    }
    auto bad = NativePeriodicFixtures::TwoCells(mpi);
    if (bad.boundaryInfo.Rows())
    {
        auto write = bad.boundaryInfo.Write();
        write.View()(1, 0).zone = G::BC_ID_PERIODIC_1;
    }
    auto badPrimary = G::NativeMeshBuilder::Build(std::move(bad));
    CHECK_THROWS(G::NativeMeshTopology::Build(*badPrimary, mpi));
    CHECK_NOTHROW(G::NativeMeshTopology::Build(*primary, mpi));
}

TEST_CASE("Quadratic native periodic topology keeps image-distinct spans and rejects cracked midside nodes")
{
    const auto mpi = MPI();
    auto source = G::NativeMeshBuilder::Build(NativePeriodicFixtures::QuadraticTorus(mpi));
    auto state = G::NativeMeshTopology::Build(*source, mpi);
    CHECK(state->NumGlobal(G::EntityKind::Face) == 2);
    for (I cell = 0; cell < state->NumOwned(G::EntityKind::Cell); ++cell)
    {
        const auto &view = state->BorrowHostView();
        CHECK(view.CellFaces(cell)[0] == view.CellFaces(cell)[2]);
        CHECK(view.CellFaces(cell)[1] == view.CellFaces(cell)[3]);
        CHECK(view.CellFaces(cell)[0] != view.CellFaces(cell)[1]);
        for (auto face : view.CellFaces(cell))
            CHECK(view.FaceNodes(face).size() == 3);
    }
    auto cracked = G::NativeMeshBuilder::Build(NativePeriodicFixtures::QuadraticTorus(mpi, true));
    CHECK_THROWS(G::NativeMeshTopology::Build(*cracked, mpi));
    CHECK_NOTHROW(G::NativeMeshTopology::Build(*source, mpi));
}

TEST_CASE("Native periodic construction retains a rigid rotation in physical cell images")
{
    const auto mpi = MPI();
    auto input = Strip(mpi, 2);
    input.periodicInfo.translation[1].map().setZero();
    input.periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    {
        auto write = input.coordinates.Write();
        for (size_t row = 0; row < input.coordinates.Rows(); ++row)
        {
            write.View()(row, 0) = 1 + DNDS::real(row);
            write.View()(row, 1) = 0;
            write.View()(row, 2) = 0;
        }
    }
    auto source = G::NativeMeshBuilder::Build(std::move(input));
    auto state = G::NativeMeshTopology::Build(*source, mpi);
    for (I cell = 0; cell < state->NumOwned(G::EntityKind::Cell); ++cell)
    {
        const auto &view = state->BorrowHostView();
        const DNDS::real expected[4][2]{{1, 0}, {0, 1}, {0, 2}, {2, 0}};
        for (DNDS::rowsize slot = 0; slot < 4; ++slot)
        {
            const auto point = view.GetCoordNodeOnCell(cell, slot);
            CHECK(point[0] == expected[slot][0]);
            CHECK(point[1] == expected[slot][1]);
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
