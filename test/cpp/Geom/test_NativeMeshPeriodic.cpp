#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshArchive.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativePeriodicMatch.hpp"
#include "Geom/Mesh/NativeMeshRefinement.hpp"
#include "Geom/Mesh/NativeMeshExtraction.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/RelationAccess.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include "NativePeriodicFixtures.hpp"
#include "NativeStructuredInterpolation.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include <filesystem>
#include <random>
#include <set>
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
    std::map<std::pair<int, I>, std::vector<I>> PersistentDerived(const G::NativeMeshState &state, const DNDS::MPIInfo &mpi)
    {
        // Fixed tiny test fixture only: gather records keyed by persistent IDs,
        // independent of current numbering. Production construction never does this.
        std::vector<I> packed;
        const auto &view = state.BorrowHostView();
        for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
            if (state.HasEntities(kind))
                for (I row = 0; row < state.NumOwned(kind); ++row)
                {
                    std::vector<I> values;
                    if (kind == G::EntityKind::Node)
                        for (int component = 0; component < 3; ++component)
                            values.push_back(I(std::llround(2 * view.Coord(row)[component])));
                    else
                    {
                        const auto nodes = kind == G::EntityKind::Cell ? view.CellNodes(row) : kind == G::EntityKind::Bnd ? view.BoundaryNodes(row)
                                                                                           : kind == G::EntityKind::Face  ? view.FaceNodes(row)
                                                                                                                          : view.EdgeNodes(row);
                        const auto bits = kind == G::EntityKind::Cell ? view.CellNodePeriodicBits(row) : kind == G::EntityKind::Bnd ? view.BoundaryNodePeriodicBits(row)
                                                                                                     : kind == G::EntityKind::Face  ? view.FaceNodePeriodicBits(row)
                                                                                                                                    : view.EdgeNodePeriodicBits(row);
                        values.push_back(I(nodes.size()));
                        for (size_t slot = 0; slot < nodes.size(); ++slot)
                        {
                            values.push_back(state.OriginalEntityId(G::EntityKind::Node, nodes[slot]));
                            values.push_back(uint8_t(bits[slot]));
                        }
                        if (kind == G::EntityKind::Cell)
                        {
                            const auto faces = view.CellFaces(row), neighbors = view.CellFaceCells(row);
                            values.push_back(I(faces.size()));
                            for (size_t slot = 0; slot < faces.size(); ++slot)
                            {
                                values.push_back(state.OriginalEntityId(G::EntityKind::Face, faces[slot]));
                                values.push_back(uint8_t(view.CellFacePeriodicBits(row)[slot]));
                                values.push_back(neighbors[slot] == DNDS::UnInitIndex ? DNDS::UnInitIndex : state.OriginalEntityId(G::EntityKind::Cell, neighbors[slot]));
                            }
                        }
                        if (kind == G::EntityKind::Face)
                            for (auto cell : view.FaceCells(row))
                                values.push_back(cell == DNDS::UnInitIndex ? DNDS::UnInitIndex : state.OriginalEntityId(G::EntityKind::Cell, cell));
                    }
                    if (state.Provenance() && state.Provenance()->sourceIds.count(kind))
                    {
                        auto read = state.Provenance()->sourceIds.at(kind).Read();
                        for (auto parent : read.View()[size_t(row)])
                            values.push_back(parent);
                    }
                    packed.insert(packed.end(), {I(kind), state.OriginalEntityId(kind, row), I(values.size())});
                    packed.insert(packed.end(), values.begin(), values.end());
                }
        int count = int(packed.size());
        std::vector<int> counts(size_t(mpi.size)), starts(size_t(mpi.size));
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
        int total = 0;
        for (int rank = 0; rank < mpi.size; ++rank)
        {
            starts[size_t(rank)] = total;
            total += counts[size_t(rank)];
        }
        std::vector<I> all(size_t(total), I(0));
        MPI_Allgatherv(packed.data(), count, DNDS::DNDS_MPI_INDEX, all.data(), counts.data(), starts.data(), DNDS::DNDS_MPI_INDEX, mpi.comm);
        std::map<std::pair<int, I>, std::vector<I>> result;
        for (size_t at = 0; at < all.size();)
        {
            const auto kind = int(all.at(at++));
            const auto id = all.at(at++), length = all.at(at++);
            REQUIRE(length >= 0);
            REQUIRE(size_t(length) <= all.size() - at);
            result.emplace(std::make_pair(kind, id), std::vector<I>(all.begin() + at, all.begin() + at + size_t(length)));
            at += size_t(length);
        }
        return result;
    }
}

TEST_CASE("Native periodic strip constructs exact self incidences and preserves them through archive and redistribution")
{
    const auto mpi = MPI();
    int dimension = 2;
    SUBCASE("quadrilateral") {}
    SUBCASE("hexahedron with edges") { dimension = 3; }
    auto primary = G::ConstructNativeMesh(Strip(mpi, dimension));
    auto state = G::PrepareNativeMeshTopology(*primary, mpi, true, dimension == 3);
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
    CHECK_THROWS(G::ConstructNativeMesh(std::move(bad)));
    bad = input;
    if (mpi.rank == 0)
        bad.cellNodesPbi = Rows<G::NodePeriodicBits>({{{0}, {0}, {1}, {0}}});
    CHECK_THROWS(G::ConstructNativeMesh(std::move(bad)));
    bad = input;
    if (mpi.rank == 0)
        bad.periodicInfo.translation[1].map()[0] = 2;
    if (mpi.size > 1)
        CHECK_THROWS(G::ConstructNativeMesh(std::move(bad)));
    CHECK_NOTHROW(G::ConstructNativeMesh(input));
}

TEST_CASE("Exact periodic frame matching handles repeated IDs whose XOR changes sorted order")
{
    CHECK(G::NativePeriodicFrame({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {{0, 2}, {0, 3}, {1, 2}, {1, 3}}) == 2);
    CHECK(G::NativePeriodicFrame({{0, 0}, {0, 2}, {1, 0}, {1, 2}}, {{0, 1}, {0, 3}, {1, 1}, {1, 3}}) == 1);
    CHECK_FALSE(G::NativePeriodicFrame({{0, 0}, {0, 1}, {1, 0}}, {{0, 2}, {0, 3}, {1, 4}}));
}

TEST_CASE("Periodic frame matching agrees with independent full image search")
{
    const auto reference = [](const G::NativePeriodicNodes &input, G::NativePeriodicNodes expected) -> std::optional<uint8_t>
    {
        if (input.empty() || input.size() != expected.size())
            return {};
        for (const auto &[node, bits] : input)
            if (bits >= 8)
                return {};
        for (const auto &[node, bits] : expected)
            if (bits >= 8)
                return {};
        std::sort(expected.begin(), expected.end());
        for (uint8_t mask = 0; mask < 8; ++mask)
        {
            G::NativePeriodicNodes image;
            for (const auto &[node, bits] : input)
                image.emplace_back(node, uint8_t(bits ^ mask));
            std::sort(image.begin(), image.end());
            if (image == expected)
                return mask;
        }
        return {};
    };
    std::mt19937 random(17381);
    for (size_t width = 1; width <= 10; ++width)
        for (int trial = 0; trial < 120; ++trial)
        {
            G::NativePeriodicNodes input, expected;
            const uint8_t mask = uint8_t(random() % 8);
            for (size_t node = 0; node < width; ++node)
            {
                const I id = (I(1) << 44) + I(random() % 4);
                const auto bits = uint8_t(random() % 8);
                input.emplace_back(id, bits);
                expected.emplace_back(id, uint8_t(bits ^ mask));
            }
            std::shuffle(expected.begin(), expected.end(), random);
            CHECK(G::NativePeriodicFrame(input, expected) == reference(input, expected));
            expected.back().second ^= 3;
            CHECK(G::NativePeriodicFrame(input, expected) == reference(input, expected));
            expected.back().first += 19;
            CHECK(G::NativePeriodicFrame(input, expected) == reference(input, expected));
        }
    CHECK_FALSE(G::NativePeriodicFrame({}, {}));
    CHECK_FALSE(G::NativePeriodicFrame({{1, 0}}, {}));
    CHECK_FALSE(G::NativePeriodicFrame({{1, 8}}, {{1, 0}}));
    CHECK_FALSE(G::NativePeriodicFrame({{1, 0}}, {{1, 8}}));
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
        for (size_t row = 0; row < input.coordinates.RowCount(); ++row)
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
    auto source = G::ConstructNativeMesh(std::move(input));
    auto state = G::PrepareNativeMeshTopology(*source, mpi);
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
    auto primary = G::ConstructNativeMesh(NativePeriodicFixtures::TwoCells(mpi));
    auto state = G::PrepareNativeMeshTopology(*primary, mpi);
    CHECK(state->NumGlobal(G::EntityKind::Face) == 6);
    auto coupledRead = state->Relation(G::Adj::Bnd2CellCoupled);
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
        const auto coupled = coupledRead.GlobalRow(boundary);
        REQUIRE(coupled.size() == 2);
        if (G::FaceIDIsPeriodic(view.GetBndInfo(boundary).zone))
        {
            CHECK(pair[0] == pair[1]);
            CHECK(coupled[0] != coupled[1]);
            CHECK(coupled[1] != DNDS::UnInitIndex);
            CHECK(cells.LocalToGlobal(pair[0]) == coupled[0]);
        }
        else
        {
            CHECK(pair[1] == DNDS::UnInitIndex);
            CHECK(coupled[1] == DNDS::UnInitIndex);
        }
    }
    auto bad = NativePeriodicFixtures::TwoCells(mpi);
    if (bad.boundaryInfo.RowCount())
    {
        auto write = bad.boundaryInfo.Write();
        write.View()(1, 0).zone = G::BC_ID_PERIODIC_1;
    }
    auto badPrimary = G::ConstructNativeMesh(std::move(bad));
    CHECK_THROWS(G::PrepareNativeMeshTopology(*badPrimary, mpi));
    CHECK_NOTHROW(G::PrepareNativeMeshTopology(*primary, mpi));
}

TEST_CASE("Quadratic native periodic topology keeps image-distinct spans and rejects cracked midside nodes")
{
    const auto mpi = MPI();
    auto source = G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi));
    auto state = G::PrepareNativeMeshTopology(*source, mpi);
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
    auto cracked = G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi, true));
    CHECK_THROWS(G::PrepareNativeMeshTopology(*cracked, mpi));
    CHECK_NOTHROW(G::PrepareNativeMeshTopology(*source, mpi));
}

TEST_CASE("Native periodic construction retains a rigid rotation in physical cell images")
{
    const auto mpi = MPI();
    auto input = Strip(mpi, 2);
    input.periodicInfo.translation[1].map().setZero();
    input.periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    {
        auto write = input.coordinates.Write();
        for (size_t row = 0; row < input.coordinates.RowCount(); ++row)
        {
            write.View()(row, 0) = 1 + DNDS::real(row);
            write.View()(row, 1) = 0;
            write.View()(row, 2) = 0;
        }
    }
    auto source = G::ConstructNativeMesh(std::move(input));
    auto state = G::PrepareNativeMeshTopology(*source, mpi);
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
    auto elevated = G::NativeMeshRefinement::Elevate(*source, mpi);
    const DNDS::real elevatedPoints[9][2]{{1, 0}, {0, 1}, {0, 2}, {2, 0}, {.5, .5}, {0, 1.5}, {1, 1}, {1.5, 0}, {.75, .75}};
    for (I cell = 0; cell < elevated.state->NumOwned(G::EntityKind::Cell); ++cell)
        for (DNDS::rowsize slot = 0; slot < 9; ++slot)
        {
            const auto point = elevated.state->BorrowHostView().GetCoordNodeOnCell(cell, slot);
            CHECK(point[0] == elevatedPoints[slot][0]);
            CHECK(point[1] == elevatedPoints[slot][1]);
        }
    auto split = G::NativeMeshRefinement::Bisect(*elevated.state, mpi);
    DNDS::real area = 0;
    for (I cell = 0; cell < split.state->NumOwned(G::EntityKind::Cell); ++cell)
        for (DNDS::rowsize slot = 0; slot < 4; ++slot)
        {
            const auto a = split.state->BorrowHostView().GetCoordNodeOnCell(cell, slot);
            const auto b = split.state->BorrowHostView().GetCoordNodeOnCell(cell, (slot + 1) % 4);
            area += .5 * (a[0] * b[1] - b[0] * a[1]);
        }
    MPI_Allreduce(MPI_IN_PLACE, &area, 1, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
    CHECK(area == doctest::Approx(-1.5));
}

TEST_CASE("Native periodic elevation bisection and external extraction preserve physical geometry and ancestry")
{
    const auto mpi = MPI();
    auto primary = G::ConstructNativeMesh(NativePeriodicFixtures::TwoCells(mpi));
    const auto sourceNamespace = primary->MeshNamespace();
    auto elevated = G::NativeMeshRefinement::Elevate(*primary, mpi);
    CHECK(elevated.state->BorrowHostView().IsPeriodic());
    CHECK(elevated.state->NumGlobal(G::EntityKind::Node) == 12);
    CHECK(elevated.provenance.sourceNamespace == sourceNamespace);
    auto topology = G::PrepareNativeMeshTopology(*elevated.state, mpi);
    CHECK(topology->NumGlobal(G::EntityKind::Face) == 6);
    auto bisected = G::NativeMeshRefinement::Bisect(*elevated.state, mpi);
    CHECK(bisected.state->NumGlobal(G::EntityKind::Cell) == 8);
    CHECK(bisected.state->NumGlobal(G::EntityKind::Bnd) == 12);
    const auto &view = bisected.state->BorrowHostView();
    for (I cell = 0; cell < bisected.state->NumOwned(G::EntityKind::Cell); ++cell)
    {
        DNDS::real area = 0;
        for (DNDS::rowsize slot = 0; slot < 4; ++slot)
        {
            const auto a = view.GetCoordNodeOnCell(cell, slot), b = view.GetCoordNodeOnCell(cell, (slot + 1) % 4);
            area += a[0] * b[1] - b[0] * a[1];
        }
        CHECK(area / 2 == doctest::Approx(.125));
    }
    auto mesh = G::UnstructuredMesh::FromNative(mpi, topology);
    auto retained = mesh->AcquireRead(topology->Requirements());
    G::UnstructuredMesh boundary(mpi, 1);
    mesh->ConstructBndMesh(boundary);
    CHECK(boundary.IsPeriodic());
    CHECK(boundary.NumCellGlobal() == 4);
    CHECK(boundary.CurrentNativeState()->NumGlobal(G::EntityKind::Node) == 8);
    auto cellParents = boundary.CurrentNativeState()->Provenance()->sourceIds.at(G::EntityKind::Cell).Read();
    for (I cell = 0; cell < boundary.NumCell(); ++cell)
    {
        const auto parent = boundary.cell2parentCell[size_t(cell)];
        CHECK(parent >= 2);
        CHECK(mesh->OriginalEntityId(G::EntityKind::Bnd, parent) == cellParents.View()(size_t(cell), 0));
        const auto &extractedView = boundary.CurrentNativeState()->BorrowHostView();
        const auto a = extractedView.GetCoordNodeOnCell(cell, 0), b = extractedView.GetCoordNodeOnCell(cell, 1);
        CHECK((a - b).norm() == doctest::Approx(.5));
        CHECK(extractedView.CellNodes(cell).size() == 3);
    }
    auto torus = G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi));
    auto empty = G::NativeMeshExtraction::ExtractBoundary(*torus, mpi);
    CHECK(empty.state->NumGlobal(G::EntityKind::Cell) == 0);
    CHECK(empty.state->NumGlobal(G::EntityKind::Node) == 0);
    CHECK(empty.state->BorrowHostView().IsPeriodic());
}

TEST_CASE("One-cell periodic elevation separates tangent spans with repeated canonical IDs")
{
    const auto mpi = MPI();
    auto source = G::ConstructNativeMesh(Strip(mpi, 2));
    auto result = G::NativeMeshRefinement::Elevate(*source, mpi);
    CHECK(result.state->NumGlobal(G::EntityKind::Node) == 6);
    CHECK(result.state->BorrowHostView().IsPeriodic());
    auto topology = G::PrepareNativeMeshTopology(*result.state, mpi);
    CHECK(topology->NumGlobal(G::EntityKind::Face) == 3);
    auto split = G::NativeMeshRefinement::Bisect(*result.state, mpi);
    CHECK(split.state->NumGlobal(G::EntityKind::Cell) == 4);
    auto input = NativePeriodicFixtures::QuadraticTorus(mpi);
    input.cellNodes = Rows<I>(input.cells.NumOwned() ? std::vector<std::vector<I>>{{0, 0, 0, 0}} : std::vector<std::vector<I>>{});
    input.cellNodesPbi = Rows<G::NodePeriodicBits>(input.cells.NumOwned() ? std::vector<std::vector<G::NodePeriodicBits>>{{{0}, {1}, {3}, {2}}} : std::vector<std::vector<G::NodePeriodicBits>>{});
    input.boundaryNodes = Rows<I>(std::vector<std::vector<I>>(size_t(input.boundaries.NumOwned()), {0, 0}));
    if (input.cells.NumOwned())
    {
        auto write = input.cellInfo.Write();
        write.View()(0, 0).setElemType(G::Elem::Quad4);
    }
    auto oldBoundaryBits = input.boundaryNodesPbi->Read();
    std::vector<std::vector<G::NodePeriodicBits>> bits(size_t(input.boundaries.NumOwned()));
    for (size_t row = 0; row < bits.size(); ++row)
        bits[row] = {oldBoundaryBits.View()(row, 0), oldBoundaryBits.View()(row, 1)};
    input.boundaryNodesPbi = Rows(bits);
    {
        auto write = input.boundaryInfo.Write();
        for (size_t row = 0; row < input.boundaryInfo.RowCount(); ++row)
            write.View()(row, 0).setElemType(G::Elem::Line2);
    }
    auto linear = G::ConstructNativeMesh(std::move(input));
    auto quadratic = G::NativeMeshRefinement::Elevate(*linear, mpi);
    CHECK(quadratic.state->NumGlobal(G::EntityKind::Node) == 7); // Three unused source nodes stay retained.
    CHECK(G::PrepareNativeMeshTopology(*quadratic.state, mpi)->NumGlobal(G::EntityKind::Face) == 2);
    const DNDS::real points[9][2]{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {.5, 0}, {1, .5}, {.5, 1}, {0, .5}, {.5, .5}};
    for (I cell = 0; cell < quadratic.state->NumOwned(G::EntityKind::Cell); ++cell)
        for (DNDS::rowsize slot = 0; slot < 9; ++slot)
        {
            const auto point = quadratic.state->BorrowHostView().GetCoordNodeOnCell(cell, slot);
            CHECK(point[0] == points[slot][0]);
            CHECK(point[1] == points[slot][1]);
        }
}

TEST_CASE("Three-dimensional periodic refinement preserves exact volume persistent images and ancestry across redistribution and restart")
{
    const auto mpi = MPI();
    auto source = G::ConstructNativeMesh(Strip(mpi, 3));
    auto elevated = G::NativeMeshRefinement::Elevate(*source, mpi);
    CHECK(elevated.state->NumGlobal(G::EntityKind::Node) == 18);
    for (I cell = 0; cell < elevated.state->NumOwned(G::EntityKind::Cell); ++cell)
    {
        std::set<std::array<int, 3>> lattice;
        for (DNDS::rowsize slot = 0; slot < 27; ++slot)
        {
            const auto point = elevated.state->BorrowHostView().GetCoordNodeOnCell(cell, slot);
            std::array<int, 3> key;
            for (int component = 0; component < 3; ++component)
            {
                key[size_t(component)] = int(std::llround(2 * point[component]));
                CHECK(point[component] == .5 * key[size_t(component)]);
                CHECK(key[size_t(component)] >= 0);
                CHECK(key[size_t(component)] <= 2);
            }
            lattice.insert(key);
        }
        CHECK(lattice.size() == 27);
    }
    auto children = G::NativeMeshRefinement::Bisect(*elevated.state, mpi);
    CHECK(children.provenance.sourceNamespace == elevated.state->MeshNamespace());
    CHECK(children.state->NumGlobal(G::EntityKind::Cell) == 8);
    for (I cell = 0; cell < children.state->NumOwned(G::EntityKind::Cell); ++cell)
    {
        const auto &view = children.state->BorrowHostView();
        const auto origin = view.GetCoordNodeOnCell(cell, 0);
        const G::tPoint a = view.GetCoordNodeOnCell(cell, 1) - origin;
        const G::tPoint b = view.GetCoordNodeOnCell(cell, 3) - origin;
        const G::tPoint c = view.GetCoordNodeOnCell(cell, 4) - origin;
        CHECK(a.dot(b.cross(c)) == doctest::Approx(.125));
    }
    auto current = G::PrepareNativeMeshTopology(*children.state, mpi, true, true);
    CHECK(current->NumGlobal(G::EntityKind::Face) == 32);
    CHECK(current->NumGlobal(G::EntityKind::Edge) == 42);
    const auto expected = PersistentDerived(*current, mpi);
    const auto name = current->MeshNamespace(), ancestry = current->Provenance()->sourceNamespace;
    const auto highWater = current->IdHighWater();
    source.reset();
    elevated = {};
    children = {};
    G::ReorderInput reorder;
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Face, G::EntityKind::Edge})
    {
        std::vector<DNDS::MPI_int> placements;
        for (I row = 0; row < current->NumOwned(kind); ++row)
            placements.push_back(DNDS::MPI_int((current->OriginalEntityId(kind, row) + I(kind) + 1) % mpi.size));
        reorder.explicitMaps.push_back({kind, std::move(placements)});
    }
    current = G::NativeMeshTransition::Build(*current, mpi, reorder).state;
    CHECK(PersistentDerived(*current, mpi) == expected);
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("periodic-derived-" + std::to_string(pid) + "-r" + std::to_string(mpi.rank) + ".json")).string();
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->OpenFile(path, false);
    G::WriteNativeMeshArchive(*current, mpi, serializer, "mesh");
    serializer->CloseFile();
    current.reset();
    serializer->OpenFile(path, true);
    current = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    CHECK(current->MeshNamespace() == name);
    CHECK(current->IdHighWater() == highWater);
    CHECK(current->Provenance()->sourceNamespace == ancestry);
    CHECK(PersistentDerived(*current, mpi) == expected);
    std::filesystem::remove(path);
}

TEST_CASE("Periodic parent-face flags use the published owner frame across ranks")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    // Three cells per direction avoid conflating distinct canonical node IDs
    // with repeated occurrences. Slab boundaries force remote face ownership;
    // final assembly also appends faces of ghost cells.
    auto grid = NativeStructuredInterpolation::BuildHex(mpi, 3, 7);
    auto nodes = grid.cone.Target().WithGhosts({});
    auto cells = grid.cone.Source().WithGhosts({});
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries, grid.cone.Owned(),
                             R::Array<I>(R::RowLayout::Ragged({})),
                             R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
    input.periodic = true;
    input.cellNodesPbi = grid.ownedBits;
    input.boundaryNodesPbi = R::Array<G::NodePeriodicBits>(R::RowLayout::Ragged({}));
    {
        auto info = input.cellInfo.Write();
        for (size_t row = 0; row < input.cellInfo.RowCount(); ++row)
            info.View()(row, 0) = G::ElemInfo{G::Elem::Hex8, G::INTERNAL_ZONE};
    }
    auto owned = G::NativeMeshBuilder::Build(std::move(input));
    owned = G::NativeMeshTopology::Build(std::move(owned), mpi);
    const auto recipe = G::DefaultNativeMeshGhostSpec(owned, 1, true);
    auto state = G::NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    auto read = state->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    I shifts = 0, remoteFaces = 0, checked = 0;
    for (I cell = 0; cell < view.NumCell(); ++cell)
    {
        auto element = G::Elem::Element{view.GetCellInfo(cell).getElemType()};
        for (size_t slot = 0; slot < view.CellFaces(cell).size(); ++slot)
        {
            const auto face = view.CellFaces(cell)[slot];
            std::vector<I> expectedNodes(size_t(element.ObtainFace(DNDS::Geom::t_index(slot)).GetNumNodes()));
            std::vector<G::NodePeriodicBits> expectedBits(expectedNodes.size());
            element.ExtractFaceNodes(DNDS::Geom::t_index(slot), view.CellNodes(cell), expectedNodes);
            element.ExtractFaceNodes(DNDS::Geom::t_index(slot), view.CellNodePeriodicBits(cell), expectedBits);
            std::multiset<std::pair<I, int>> expected, actual;
            for (size_t node = 0; node < expectedNodes.size(); ++node)
                expected.emplace(expectedNodes[node], uint8_t(expectedBits[node]));
            const auto shift = uint8_t(view.CellFacePeriodicBits(cell)[slot]);
            for (size_t node = 0; node < view.FaceNodes(face).size(); ++node)
                actual.emplace(view.FaceNodes(face)[node], uint8_t(view.FaceNodePeriodicBits(face)[node]) ^ shift);
            CHECK(actual == expected);
            shifts += shift != 0;
            remoteFaces += face >= view.NumFace();
            ++checked;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &shifts, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &remoteFaces, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
    CHECK(checked == 27 * 6);
    CHECK(shifts > 0);
    if (mpi.size > 1)
        CHECK(remoteFaces > 0);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
