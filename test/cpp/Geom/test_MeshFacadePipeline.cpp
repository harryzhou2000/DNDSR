#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/MeshAccess.hpp"
#include "Geom/Mesh/EntitySpace.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include "Geom/Mesh/NativeMeshSnapshot.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <vector>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

struct SerialCellOracle
{
    G::t_index type = 0, zone = 0;
    std::vector<std::array<DNDS::real, 3>> points;
};

struct PipelineFixture
{
    DNDS::ssp<G::UnstructuredMesh> mesh;
    std::vector<SerialCellOracle> cells;
};

static PipelineFixture BuildFacadePipeline(DNDS::MPIInfo mpi, bool periodic)
{
    PipelineFixture fixture;
    fixture.mesh = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 2);
    auto &mesh = *fixture.mesh;
    if (periodic)
    {
        const G::tPoint zero{0, 0, 0};
        mesh.SetPeriodicGeometry({10, 0, 0}, zero, zero, {0, 10, 0}, zero, zero, zero, zero, zero);
    }
    G::UnstructuredMeshSerialRW reader(fixture.mesh, 0);
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
                      "data" / "mesh" / (periodic ? "IV10_10.cgns" : "UniformSquare_10.cgns");
    REQUIRE(std::filesystem::exists(path));
    reader.ReadFromCGNSSerial(path.string());

    // Capture the serial CGNS geometry before periodic node deduplication or
    // partitioning. This oracle does not call a facade/legacy geometry helper.
    std::vector<DNDS::real> packed;
    if (mpi.rank == 0)
    {
        packed.push_back(DNDS::real(reader.cell2nodeSerial->Size()));
        for (I cell = 0; cell < reader.cell2nodeSerial->Size(); ++cell)
        {
            auto nodes = (*reader.cell2nodeSerial)[cell];
            const auto &info = (*reader.cellElemInfoSerial)(cell, 0);
            packed.insert(packed.end(), {DNDS::real(nodes.size()), DNDS::real(info.type), DNDS::real(info.zone)});
            for (auto node : nodes)
                for (int d = 0; d < 3; ++d)
                    packed.push_back((*reader.coordSerial)[node][d]);
        }
    }
    int count = int(packed.size());
    MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
    packed.resize(size_t(count));
    MPI_Bcast(packed.data(), count, MPI_DOUBLE, 0, mpi.comm);
    REQUIRE_FALSE(packed.empty());
    fixture.cells.resize(size_t(packed[0]));
    size_t cursor = 1;
    for (auto &cell : fixture.cells)
    {
        const auto nodes = size_t(packed.at(cursor++));
        cell.type = G::t_index(packed.at(cursor++));
        cell.zone = G::t_index(packed.at(cursor++));
        cell.points.resize(nodes);
        for (auto &point : cell.points)
            for (auto &value : point)
                value = packed.at(cursor++);
    }
    REQUIRE(cursor == packed.size());
    if (periodic)
        reader.Deduplicate1to1Periodic(1e-8);
    reader.BuildCell2Cell();
    G::UnstructuredMeshSerialRW::PartitionOptions options;
    options.metisType = "KWAY";
    options.metisUfactor = 30;
    options.metisSeed = 42;
    options.metisNcuts = 1;
    reader.MeshPartitionCell2Cell(options);
    reader.PartitionReorderToMeshCell2Cell();
    mesh.RecoverNode2CellAndNode2Bnd();
    mesh.RecoverCell2CellAndBnd2Cell();
    mesh.BuildGhostPrimary();
    mesh.AdjGlobal2LocalPrimary();
    mesh.InterpolateFace();
    if (mesh.adjN2CBState == G::Adj_PointToLocal)
        mesh.AdjLocal2GlobalN2CB();
    mesh.BuildGhostN2CB();
    mesh.AdjGlobal2LocalN2CB();
    mesh.AssertOnFaces();
    return fixture;
}

static void CheckFacadePipeline(const PipelineFixture &fixture)
{
    auto &mesh = *fixture.mesh;
    auto access = mesh.AcquireRead(G::MeshReadRequirements::CellFlux());
    const auto view = access.HostView();
    CHECK(view.NumCell() == mesh.NumCell());
    CHECK(view.NumCellGhost() == mesh.NumCellGhost());
    CHECK(view.NumNodeProc() == mesh.NumNodeProc());
    CHECK(view.NumFaceProc() == mesh.NumFaceProc());
    CHECK(view.IsPeriodic() == mesh.isPeriodic);
    I checkedSlots = 0, transformedNodes = 0;
    for (I cell = 0; cell < view.NumCellProc(); ++cell)
    {
        // Original CGNS IDs are explicitly retained setup companions, outside
        // the query view. They provide correspondence after either transition.
        const I original = mesh.cell2cellOrig(cell, 0);
        REQUIRE(original >= 0);
        REQUIRE(size_t(original) < fixture.cells.size());
        const auto &expected = fixture.cells[size_t(original)];
        const auto nodes = view.CellNodes(cell);
        REQUIRE(nodes.size() == expected.points.size());
        CHECK(view.GetCellInfo(cell).type == expected.type);
        CHECK(view.GetCellZone(cell) == expected.zone);
        G::MeshQueryCoords coords;
        view.GetCoordsOnCell(cell, coords);
        REQUIRE(size_t(coords.cols()) == nodes.size());
        for (size_t j = 0; j < nodes.size(); ++j)
        {
            REQUIRE(nodes[j] >= 0);
            REQUIRE(nodes[j] < view.NumNodeProc());
            const auto point = view.GetCoordNodeOnCell(cell, DNDS::rowsize(j));
            transformedNodes += (point - view.Coord(nodes[j])).norm() > 1e-10;
            for (int d = 0; d < 3; ++d)
            {
                CHECK(coords(d, j) == doctest::Approx(expected.points[j][d]).epsilon(1e-12));
                CHECK(point[d] == doctest::Approx(expected.points[j][d]).epsilon(1e-12));
            }
        }
        if (cell >= view.NumCell())
            continue; // Outer ghost cells need not have complete face closure.
        auto element = view.GetCellElement(cell);
        const auto faces = view.CellFaces(cell);
        REQUIRE(faces.size() == size_t(element.GetNumFaces()));
        for (size_t slot = 0; slot < faces.size(); ++slot)
        {
            const I face = faces[slot];
            REQUIRE(face >= 0);
            REQUIRE(face < view.NumFaceProc());
            auto actualNodes = view.FaceNodes(face);
            std::vector<I> actual(actualNodes.begin(), actualNodes.end());
            std::vector<I> expectedNodes(size_t(element.ObtainFace(G::t_index(slot)).GetNumNodes()));
            element.ExtractFaceNodes(G::t_index(slot), nodes, expectedNodes);
            std::sort(actual.begin(), actual.end());
            std::sort(expectedNodes.begin(), expectedNodes.end());
            CHECK(actual == expectedNodes);
            const auto adjacent = view.FaceCells(face);
            REQUIRE(adjacent.size() == 2);
            CHECK((adjacent[0] == cell || adjacent[1] == cell));
            const bool expectedBack = adjacent[0] == cell && adjacent[1] == cell
                                          ? !bool(view.CellFacePeriodicBits(cell)[slot])
                                          : adjacent[0] == cell;
            CHECK(view.CellIsFaceBack(cell, face, DNDS::rowsize(slot)) == expectedBack);
            CHECK(view.CellFaceOther(cell, face, DNDS::rowsize(slot)) == adjacent[expectedBack ? 1 : 0]);
            CHECK((adjacent[1] == DNDS::UnInitIndex) == G::FaceIDIsExternalBC(view.GetFaceZone(face)));
            ++checkedSlots;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &checkedSlots, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mesh.mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &transformedNodes, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mesh.mpi.comm);
    CHECK(checkedSlots > 0);
    CHECK((transformedNodes > 0) == mesh.isPeriodic);
}

static void CheckFacadeHalo(G::UnstructuredMesh &mesh)
{
    const auto cells = mesh.Entities(G::EntityKind::Cell);
    DNDS::ArrayPair<DNDS::ParArray<DNDS::real, 1>> legacy;
    legacy.InitPair("facade.pipeline.cellField", mesh.mpi);
    legacy.father->Resize(cells.NumOwned());
    legacy.son->Resize(0);
    legacy.TransAttach();
    G::LegacyFieldBridge::BorrowIndexing(legacy.trans, cells);
    for (I row = 0; row < cells.NumOwned(); ++row)
        legacy(row, 0) = 0.25 + DNDS::real(cells.LocalToGlobal(row));
    legacy.trans.createMPITypes();
    legacy.trans.pullOnce();
    auto route = cells.BuildRowTransferPlan();
    R::Array<DNDS::real> owned(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
    R::Array<DNDS::real> ghost(R::RowLayout::Dense(size_t(cells.NumGhost()), 1));
    {
        auto write = owned.Write();
        for (I row = 0; row < cells.NumOwned(); ++row)
            write.View()(size_t(row), 0) = legacy(row, 0);
    }
    R::PayloadPlan payload(route, owned.Layout(), ghost.Layout(), R::ElementSchema::Arithmetic<DNDS::real>());
    auto binding = payload.Bind(owned, ghost);
    binding.Begin().Wait();
    auto values = ghost.Read();
    for (I row = 0; row < cells.NumGhost(); ++row)
    {
        const auto expected = 0.25 + DNDS::real(cells.LocalToGlobal(cells.NumOwned() + row));
        CHECK((*legacy.son)(row, 0) == expected);
        CHECK(values.View()(size_t(row), 0) == expected);
    }
}

static void CheckFacadeTransitions(bool periodic)
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = BuildFacadePipeline(mpi, periodic);
    auto &mesh = *fixture.mesh;
    CheckFacadePipeline(fixture);
    CheckFacadeHalo(mesh);
    auto oldCells = mesh.Entities(G::EntityKind::Cell);
    auto oldRoute = oldCells.BuildRowTransferPlan();
    // Inject an equal-content but wrong target-map identity. Acquisition must
    // reject it and release all partial captures before the fixture is restored.
    auto correctMap = mesh.coords.trans.pLGhostMapping;
    mesh.coords.trans.pLGhostMapping = std::make_shared<DNDS::OffsetAscendIndexMapping>(*correctMap);
    CHECK_THROWS(mesh.AcquireRead(G::MeshReadRequirements::CellFlux()));
    mesh.coords.trans.pLGhostMapping = correctMap;
    CHECK(mesh.versions.CanMutate());
    mesh.ReorderLocalCells(2, 1);
    CHECK_FALSE(oldCells.Valid());
    CHECK_FALSE(oldRoute.Valid());
    CHECK_THROWS(oldCells.LocalToGlobal(0));
    CheckFacadePipeline(fixture);
    CheckFacadeHalo(mesh);

    auto beforeRedistribution = mesh.Entities(G::EntityKind::Cell);
    mesh.AdjLocal2GlobalPrimary();
    mesh.AdjLocal2GlobalN2CB();
    mesh.AdjLocal2GlobalFacial();
    mesh.AdjLocal2GlobalC2F();
    G::ReorderInput input;
    std::vector<DNDS::MPI_int> cellRanks(size_t(mesh.NumCell()));
    for (I i = 0; i < mesh.NumCell(); ++i)
        cellRanks[size_t(i)] = (mpi.rank + int(i)) % mpi.size;
    input.explicitMaps.push_back({G::EntityKind::Cell, std::move(cellRanks)});
    input.explicitMaps.push_back({G::EntityKind::Node, std::vector<DNDS::MPI_int>(size_t(mesh.NumNode()), mpi.rank)});
    input.explicitMaps.push_back({G::EntityKind::Bnd, std::vector<DNDS::MPI_int>(size_t(mesh.NumBnd()), mpi.rank)});
    input.explicitMaps.push_back({G::EntityKind::Face, std::vector<DNDS::MPI_int>(size_t(mesh.NumFace()), (mpi.rank + 1) % mpi.size)});
    mesh.ReorderEntities(input);
    CHECK_FALSE(beforeRedistribution.Valid());
    mesh.RecoverNode2CellAndNode2Bnd();
    mesh.RecoverCell2CellAndBnd2Cell();
    mesh.BuildGhostPrimary();
    mesh.AdjGlobal2LocalPrimary();
    mesh.BuildGhostFace();
    mesh.BuildGhostN2CB();
    mesh.AdjGlobal2LocalN2CB();
    mesh.AssertOnFaces();
    CheckFacadePipeline(fixture);
    CheckFacadeHalo(mesh);
}

TEST_CASE("Mesh facade checkpoint restores exact serial geometry and native query backing")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    for (bool periodic : {false, true})
    {
        auto fixture = BuildFacadePipeline(mpi, periodic);
        auto detached = G::NativeMeshSnapshot::Capture(*fixture.mesh, G::MeshReadRequirements::CellFlux());
        auto before = detached.AcquireRead();
        const auto firstPoint = G::tPoint(before.HostView().Coord(0));
        auto path = std::filesystem::temp_directory_path() /
                    ("dnds_facade_restart_np" + std::to_string(mpi.size) + (periodic ? "_periodic.h5" : "_ordinary.h5"));
        fixture.mesh->AdjLocal2GlobalPrimary();
        auto serializer = DNDS::make_ssp<DNDS::Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path.string(), false);
        fixture.mesh->WriteSerialize(serializer, "meshPart");
        serializer->CloseFile();
        // Reusing a serializer across distinct files is not part of this test.
        serializer.reset();
        auto restored = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 2);
        serializer = DNDS::make_ssp<DNDS::Serializer::SerializerH5>(mpi);
        serializer->OpenFile(path.string(), true);
        restored->ReadSerialize(serializer, "meshPart");
        serializer->CloseFile();
        restored->RecoverNode2CellAndNode2Bnd();
        restored->RecoverCell2CellAndBnd2Cell();
        restored->BuildGhostPrimary();
        restored->AdjGlobal2LocalPrimary();
        restored->InterpolateFace();
        restored->BuildGhostN2CB();
        restored->AdjGlobal2LocalN2CB();
        fixture.mesh = restored;
        CheckFacadePipeline(fixture);
        CheckFacadeHalo(*restored);
        CHECK(before.HostView().Coord(0).isApprox(firstPoint));
        CHECK_FALSE(detached.SourceStamp().Valid());
        CHECK(before.Stamp().Valid());
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
            std::filesystem::remove(path);
        MPI_Barrier(mpi.comm);
    }
}

TEST_CASE("Mesh facade real CGNS pipeline preserves serial geometry through transitions")
{
    SUBCASE("ordinary UniformSquare_10") { CheckFacadeTransitions(false); }
    SUBCASE("periodic IV10_10") { CheckFacadeTransitions(true); }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
