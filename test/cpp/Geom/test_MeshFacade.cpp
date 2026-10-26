#ifndef DNDS_MESH_FACADE_TEST_EMBEDDED
#    define DOCTEST_CONFIG_IMPLEMENT
#endif
#include "doctest.h"
#include "DNDS/ArrayPair.hpp"
#include "DNDS/Array.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/EntitySpace.hpp"
#include "MeshViewKernel.hpp"
#include <memory>
#include <optional>

namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;

static DNDS::MPIInfo FacadeMPI()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

#include "NativeMeshFacadeFixture.hpp"

static auto FacadeFixture(bool periodic = false, bool empty = false)
{
    return NativeFacadeFixture::Make(FacadeMPI(), periodic, empty).Publish(FacadeMPI());
}
static G::MeshReadRequirements AllFacadeRequirements()
{
    return NativeFacadeFixture::Requirements();
}

[[maybe_unused]] static void CheckProbeEqual(const G::Test::MeshViewProbeResult &actual, const G::Test::MeshViewProbeResult &expected)
{
    for (int i = 0; i < 15; ++i)
        CHECK(actual.counts[i] == expected.counts[i]);
    CHECK(actual.relationChecksum == expected.relationChecksum);
    CHECK(actual.elementChecksum == expected.elementChecksum);
    CHECK(actual.faceSideChecksum == expected.faceSideChecksum);
    CHECK(actual.coordinateChecksum == doctest::Approx(expected.coordinateChecksum));
    CHECK(actual.wallDistanceChecksum == doctest::Approx(expected.wallDistanceChecksum));
    CHECK(actual.periodicChecksum == doctest::Approx(expected.periodicChecksum));
    CHECK(actual.cellBatchChecksum == doctest::Approx(expected.cellBatchChecksum));
    CHECK(actual.cellExternalChecksum == doctest::Approx(expected.cellExternalChecksum));
    CHECK(actual.cellNodeChecksum == doctest::Approx(expected.cellNodeChecksum));
    CHECK(actual.faceBatchChecksum == doctest::Approx(expected.faceBatchChecksum));
    CHECK(actual.faceExternalChecksum == doctest::Approx(expected.faceExternalChecksum));
    CHECK(actual.faceNodeChecksum == doctest::Approx(expected.faceNodeChecksum));
    for (int d = 0; d < 3; ++d)
    {
        CHECK(actual.transformedCellSample[d] == doctest::Approx(expected.transformedCellSample[d]));
        CHECK(actual.transformedFaceSample[d] == doctest::Approx(expected.transformedFaceSample[d]));
    }
}

TEST_CASE("Mesh facade host acquisition exposes exact local rows and semantic helper behavior")
{
    for (bool periodic : {false, true})
    {
        auto mesh = FacadeFixture(periodic);
        auto access = mesh->AcquireRead(AllFacadeRequirements());
        const auto view = access.HostView();
        CHECK(view.Dimension() == 2);
        CHECK(view.NumCell() == 1);
        CHECK(view.NumNode() == (periodic ? 4 : 3));
        CHECK(view.NumCellGhost() == 0);
        CHECK(view.NumFace() == 3);
        CHECK(view.NumBnd() == (periodic ? 2 : 3));
        CHECK(view.NumEdge() == 3);
        CHECK(view.CellNodes(0)[2] == 2);
        CHECK(view.Coord(1)[0] == 1);
        CHECK(view.GetCellZone(0) == 11);
        CHECK(view.GetCellElement(0).GetNumNodes() == (periodic ? 4 : 3));
        for (I cell = 0; cell < view.NumCellProc(); ++cell)
        {
            G::tSmallCoords expected, actual;
            mesh->GetCoordsOnCell(cell, expected);
            view.GetCoordsOnCell(cell, actual);
            CHECK(actual.isApprox(expected));
            G::MeshQueryCoords bounded;
            view.GetCoordsOnCell(cell, bounded);
            CHECK(bounded.isApprox(expected));
            G::MeshQueryCoords externalCoords;
            view.GetCoordsOnCell(cell, externalCoords, view);
            CHECK(externalCoords.isApprox(expected));
            for (DNDS::rowsize slot = 0; slot < mesh->CellFaces(cell).size(); ++slot)
            {
                const auto face = mesh->CellFaces(cell)[slot];
                CHECK(view.CellIsFaceBack(cell, face, slot) == mesh->CellIsFaceBack(cell, face, slot));
                CHECK(view.CellFaceOther(cell, face, slot) == mesh->CellFaceOther(cell, face, slot));
            }
        }
        for (I face = 0; face < view.NumFaceProc(); ++face)
        {
            G::tSmallCoords expected, actual;
            mesh->GetCoordsOnFace(face, expected);
            view.GetCoordsOnFace(face, actual);
            CHECK(actual.isApprox(expected));
            G::MeshQueryCoords bounded;
            view.GetCoordsOnFace(face, bounded);
            CHECK(bounded.isApprox(expected));
            G::MeshQueryCoords externalCoords;
            view.GetCoordsOnFace(face, externalCoords, view);
            CHECK(externalCoords.isApprox(expected));
            CHECK(view.GetCoordNodeOnFace(face, 1).isApprox(mesh->GetCoordNodeOnFace(face, 1)));
            CHECK(view.GetCoordWallDistOnFace(face, 1).isApprox(mesh->GetCoordWallDistOnFace(face, 1)));
        }
        if (periodic)
        {
            CHECK(view.CellFaces(0)[0] == view.CellFaces(0)[2]);
            CHECK(view.CellIsFaceBack(0, 0, 0));
            CHECK_FALSE(view.CellIsFaceBack(0, 0, 2));
            CHECK(view.CellFaceOther(0, 0, 2) == 0);
            CHECK(view.GetCoordNodeOnCell(0, 2).isApprox(G::tPoint{2, 1, 0}));
            CHECK(view.GetCoordWallDistOnCell(0, 2).isApprox(G::tPoint{-2, 3, 0}));
            CHECK(view.CellOtherCellPeriodicHandle(0, 0, []
                                                   { return 1; }, []
                                                   { return 2; }, []
                                                   { return 3; }) == 2);
            CHECK(view.CellOtherCellPeriodicHandle(0, 1, []
                                                   { return 1; }, []
                                                   { return 2; }, []
                                                   { return 3; }) == 1);
        }
        const auto probe = G::Test::MeshViewProbe(view);
        CHECK(probe.coordinateChecksum == (periodic ? 19 : 11));
        CHECK_THROWS(access.DeviceView());
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
        CHECK_FALSE(mesh->versions.CanMutate());
    }
}

TEST_CASE("Mesh facade scopes retain storage after source deletion and reject invalid contexts")
{
    std::optional<G::MeshReadAccess> retained;
    {
        auto mesh = FacadeFixture();
        retained.emplace(mesh->AcquireRead(AllFacadeRequirements()));
        CHECK_FALSE(mesh->versions.CanMutate());
    }
    CHECK(retained->HostView().CellNodes(0)[2] == 2);
    CHECK(retained->HostView().Coord(2)[1] == 1);
    retained.reset();
    auto empty = FacadeFixture(false, true);
    auto access = empty->AcquireRead(AllFacadeRequirements());
    CHECK(access.HostView().NumNodeProc() == 0);
    CHECK(access.HostView().NumCellProc() == 0);
    CHECK(G::Test::MeshViewProbe(access.HostView()).relationChecksum == 0);
    CHECK_THROWS(empty->AcquireRead(G::MeshReadRequirements::Primary(), {R::Device::Host(), reinterpret_cast<void *>(1)}));
    G::UnstructuredMesh unpublished(FacadeMPI(), 2);
    CHECK_THROWS(unpublished.AcquireRead());
    CHECK_THROWS(unpublished.Entities(G::EntityKind::Cell));
    CHECK(unpublished.versions.CanMutate());
}

TEST_CASE("Native scope and detached capture expose exactly the requested optional groups")
{
    auto mesh = FacadeFixture(true);
    {
        auto primary = mesh->AcquireRead();
        const auto view = primary.HostView();
        CHECK(view.HasRelation(G::MeshRelation::CellNodes));
        CHECK_FALSE(view.HasRelation(G::MeshRelation::CellFaces));
        CHECK_FALSE(view.HasRelation(G::MeshRelation::NodeCells));
        CHECK_FALSE(view.HasRelation(G::MeshRelation::EdgeNodes));
        CHECK_FALSE(view.HasRelation(G::MeshRelation::CellFaceCells));
        CHECK_FALSE(view.HasNodeWallDist());
        CHECK(view.NumFaceProc() == 0);
        CHECK(view.CellNodePeriodicBits(0).size() == view.CellNodes(0).size());
    }
    auto primary = G::NativeMeshSnapshot::Capture(*mesh);
    auto facial = G::NativeMeshSnapshot::Capture(*mesh, G::MeshReadRequirements::CellFlux());
    auto complete = G::NativeMeshSnapshot::Capture(*mesh, AllFacadeRequirements());
    auto a = primary.AcquireRead(), b = facial.AcquireRead(), c = complete.AcquireRead();
    CHECK_FALSE(a.HostView().HasRelation(G::MeshRelation::CellFaces));
    CHECK_FALSE(a.HostView().HasNodeWallDist());
    CHECK(b.HostView().HasRelation(G::MeshRelation::FaceCells));
    CHECK_FALSE(b.HostView().HasRelation(G::MeshRelation::NodeCells));
    CHECK_FALSE(b.HostView().HasNodeWallDist());
    CHECK(c.HostView().HasRelation(G::MeshRelation::EdgeNodes));
    CHECK(c.HostView().HasNodeWallDist());
    CHECK(c.HostView().CellNodePeriodicBits(0)[2] == G::nodePB1);
    mesh.reset();
    CHECK(a.HostView().GetCellZone(0) == 11);
    CHECK(b.HostView().NumFaceProc() == 3);
}

TEST_CASE("Native facade construction rejects mismatched shapes and owned closure before publication")
{
    auto reject = [](auto change, bool periodic = false)
    {
        auto input = NativeFacadeFixture::Make(FacadeMPI(), periodic);
        change(input);
        CHECK_THROWS(input.Publish(FacadeMPI()));
        // A rejected detached candidate cannot damage a later valid publication.
        CHECK_NOTHROW(FacadeFixture(periodic));
    };
    SUBCASE("source owned and ghost split differs despite identical total")
    {
        reject([](auto &input)
               {
            auto &pair = *input.data.relations[size_t(G::MeshRelation::BoundaryCells)];
            pair.owned = R::Array<I>(R::RowLayout::Dense(2, 2));
            pair.ghost = R::Array<I>(R::RowLayout::Dense(1, 2));
            CHECK(pair.owned.Rows() + pair.ghost.Rows() == 3); });
    }
    SUBCASE("element companion split differs despite identical total")
    {
        reject([](auto &input)
               {
            input.data.cellInfo->owned = R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1));
            input.data.cellInfo->ghost = R::Array<G::ElemInfo>(R::RowLayout::Dense(1, 1)); });
    }
    SUBCASE("entity family equal extents cannot substitute a foreign domain")
    {
        reject([](auto &input)
               { input.spaces.at(G::EntityKind::Node) = G::EntitySpace::CreateOwned(G::EntityKind::Node, FacadeMPI(), 3); });
    }
    SUBCASE("missing source rows and periodic shape")
    {
        reject([](auto &input)
               { input.data.relations[size_t(G::MeshRelation::BoundaryCells)]->owned = R::Array<I>(R::RowLayout::Dense(0, 2)); });
        reject([](auto &input)
               { input.data.cellNodesPbi = NativeFacadeFixture::Rows<G::NodePeriodicBits>({{G::nodePB1}}); }, true);
    }
    SUBCASE("element type dimension and face width are validated before dispatch")
    {
        for (auto type : {G::t_index(G::Elem::UnknownElem), G::t_index(99), G::t_index(G::Elem::Hex27)})
            reject([&](auto &input)
                   { input.data.cellInfo->owned.Write().View()(0, 0).type = type; });
        reject([](auto &input)
               { input.data.relations[size_t(G::MeshRelation::CellFaces)] = NativeFacadeFixture::Rows<I>({{0, 1}}); });
    }
    SUBCASE("owned cell face and owned face first cell must be available")
    {
        for (I missing : {I(-4), DNDS::UnInitIndex})
            for (auto relation : {G::MeshRelation::CellFaces, G::MeshRelation::FaceCells})
                reject([&](auto &input)
                       { input.data.relations[size_t(relation)]->owned.Write().View()(0, 0) = missing; });
    }
    SUBCASE("zone and external sentinel must agree")
    {
        reject([](auto &input)
               { input.data.faceInfo->owned.Write().View()(0, 0).zone = G::BC_ID_INTERNAL; });
        reject([](auto &input)
               {
            input.data.faceInfo->owned.Write().View()(0, 0).zone = G::BC_ID_INTERNAL;
            input.data.relations[size_t(G::MeshRelation::FaceCells)]->owned.Write().View()(0, 1) = 0; });
        for (auto zone : {G::BC_ID_INTERNAL, G::BC_ID_DEFAULT_WALL})
            reject([&](auto &input)
                   { input.data.faceInfo->owned.Write().View()(0, 0).zone = zone; }, true);
    }
    SUBCASE("self periodic face requires exactly two opposite ordered sides")
    {
        for (auto side : {G::NodePeriodicBits{}, G::nodePB1})
            reject([&](auto &input)
                   {
                auto write = input.data.cellFacesPbi->owned.Write();
                write.View()(0, 0) = write.View()(0, 2) = side; }, true);
        reject([](auto &input)
               { input.data.relations[size_t(G::MeshRelation::CellFaces)]->owned.Write().View()(0, 2) = 1; }, true);
        auto valid = NativeFacadeFixture::Make(FacadeMPI(), true);
        valid.data.cellFacesPbi->owned.Write().View()(0, 2) = G::nodePB2;
        auto mesh = valid.Publish(FacadeMPI());
        auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        CHECK(read.HostView().CellIsFaceBack(0, 0, 0));
        CHECK_FALSE(read.HostView().CellIsFaceBack(0, 0, 2));
    }
    SUBCASE("out of range node references reject publication")
    {
        reject([](auto &input)
               { input.data.relations[size_t(G::MeshRelation::CellNodes)]->owned.Write().View()(0, 1) = 99; });
    }
}

TEST_CASE("Native snapshot validates reciprocal owned faces and partial outer ghost closure")
{
    auto input = NativeFacadeFixture::Make(FacadeMPI());
    auto &data = input.data;
    data.relations[size_t(G::MeshRelation::CellNodes)] = NativeFacadeFixture::Rows<I>({{0, 1, 2}, {0, 1, 2}});
    data.relations[size_t(G::MeshRelation::CellFaces)] = NativeFacadeFixture::Rows<I>({{0, 1, 2}, {0, 1, 2}});
    data.relations[size_t(G::MeshRelation::CellCells)] = NativeFacadeFixture::Rows<I>({{}, {}});
    data.relations[size_t(G::MeshRelation::CellEdges)] = NativeFacadeFixture::Rows<I>({{0, 1, 2}, {0, 1, 2}});
    data.relations[size_t(G::MeshRelation::CellFaceCells)] = NativeFacadeFixture::Rows<I>({{DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex}, {DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex}});
    data.cellInfo = NativeFacadeFixture::Rows<G::ElemInfo>({{{G::Elem::Tri3, 11}}, {{G::Elem::Tri3, 11}}}, 1);
    CHECK_THROWS(G::NativeMeshSnapshot::FromNative(data, AllFacadeRequirements()));
    {
        auto fc = data.relations[size_t(G::MeshRelation::FaceCells)]->owned.Write();
        auto info = data.faceInfo->owned.Write();
        for (size_t face = 0; face < 3; ++face)
        {
            fc.View()(face, 1) = 1;
            info.View()(face, 0).zone = G::BC_ID_INTERNAL;
        }
    }
    CHECK_NOTHROW(G::NativeMeshSnapshot::FromNative(data, AllFacadeRequirements()));

    auto outer = NativeFacadeFixture::Make(FacadeMPI()).data;
    outer.relations[size_t(G::MeshRelation::FaceNodes)]->ghost = NativeFacadeFixture::Rows<I>({{0, 1}}).owned;
    outer.relations[size_t(G::MeshRelation::FaceCells)]->ghost = NativeFacadeFixture::Rows<I>({{-4, -5}}, 2).owned;
    outer.relations[size_t(G::MeshRelation::FaceBoundaries)]->ghost = NativeFacadeFixture::Rows<I>({{DNDS::UnInitIndex}}, 1).owned;
    outer.faceInfo->ghost = NativeFacadeFixture::Rows<G::ElemInfo>({{{G::Elem::Line2, G::BC_ID_INTERNAL}}}, 1).owned;
    outer.relations[size_t(G::MeshRelation::NodeCells)]->owned.Write().View()(0, 0) = -4;
    {
        auto snapshot = G::NativeMeshSnapshot::FromNative(outer, AllFacadeRequirements());
        auto read = snapshot.AcquireRead();
        CHECK(read.HostView().NumFaceGhost() == 1);
        CHECK(read.HostView().FaceCells(3)[0] == -4);
        CHECK(read.HostView().FaceCells(3)[1] == -5);
        CHECK(read.HostView().NodeCells(0)[0] == -4);
        CHECK(read.HostView().FaceCells(0)[1] == DNDS::UnInitIndex);
    }
    auto allFaces = AllFacadeRequirements();
    allFaces.allFaceCellsLocal = true;
    CHECK_THROWS(G::NativeMeshSnapshot::FromNative(outer, allFaces));
    outer.relations[size_t(G::MeshRelation::CellFaces)]->owned.Write().View()(0, 0) = 3;
    CHECK_THROWS(G::NativeMeshSnapshot::FromNative(outer, AllFacadeRequirements()));
}

TEST_CASE("Native publication retains aliased storage and releases it with the final scope")
{
    auto input = NativeFacadeFixture::Make(FacadeMPI());
    auto coordinates = input.data.coords->owned;
    auto relation = input.data.relations[size_t(G::MeshRelation::CellNodes)]->owned;
    auto mesh = input.Publish(FacadeMPI());
    std::optional<G::MeshReadAccess> read;
    read.emplace(mesh->AcquireRead(AllFacadeRequirements()));
    CHECK_THROWS(coordinates.Write());
    CHECK_THROWS(relation.Write());
    const auto point = G::tPoint(read->HostView().Coord(0));
    mesh.reset();
    CHECK_THROWS(coordinates.Write());
    CHECK(read->HostView().Coord(0) == point);
    read.reset();
    CHECK_NOTHROW(coordinates.Write());
    CHECK_NOTHROW(relation.Write());
}

TEST_CASE("Mesh host view getters reject moved and sealed scopes")
{
    auto mesh = FacadeFixture();
    auto original = mesh->AcquireRead();
    auto destination = std::move(original);
    CHECK_THROWS(original.HostView());
    destination.Complete(R::Completion::Record(R::ExecutionContext::Host()));
    CHECK_THROWS(destination.HostView());
    destination.Wait();
    CHECK_THROWS(destination.HostView());
    original.Wait();
}

TEST_CASE("Native entity spaces export isolated field maps and match their native halo route")
{
    const auto mpi = FacadeMPI();
    auto owned = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, 2);
    std::vector<I> pulls;
    if (mpi.size > 1)
        pulls.push_back(I((mpi.rank + 1) % mpi.size) * 2 + 1);
    else
        CHECK_THROWS(owned.WithGhosts({1})); // Native halos cannot name a self-owned row.
    const auto space = owned.WithGhosts(pulls);
    CHECK(space.NumOwned() == 2);
    CHECK(space.NumGhost() == I(pulls.size()));
    CHECK(space.NumGlobal() == 2 * mpi.size);
    CHECK(space.LocalToGlobal(0) == 2 * mpi.rank);
    CHECK_FALSE(space.GlobalToLocal(-1).has_value());
    CHECK_THROWS(space.LocalToGlobal(-1));
    CHECK_THROWS(space.Owner(space.NumGlobal()));
    DNDS::ArrayPair<DNDS::ParArray<double, 1>> field;
    field.InitPair("field", mpi);
    field.father->Resize(2);
    field.son->Resize(0);
    field.TransAttach();
    G::LegacyFieldBridge::BorrowIndexing(field.trans, space);
    field(0, 0) = 100 + 2 * mpi.rank;
    field(1, 0) = 101 + 2 * mpi.rank;
    field.trans.createMPITypes();
    field.trans.pullOnce();
    const auto route = space.BuildRowTransferPlan();
    R::Array<double> source(R::RowLayout::Dense(2, 1)), destination(R::RowLayout::Dense(pulls.size(), 1));
    {
        auto write = source.Write();
        for (size_t row = 0; row < 2; ++row)
            write.View()(row, 0) = field(I(row), 0);
    }
    R::PayloadPlan payload(route, source.Layout(), destination.Layout(), R::ElementSchema::Arithmetic<double>());
    auto binding = payload.Bind(source, destination);
    binding.Begin().Wait();
    if (!pulls.empty())
    {
        CHECK(space.LocalToGlobal(2) == pulls[0]);
        CHECK(space.Owner(pulls[0]) == (mpi.rank + 1) % mpi.size);
        CHECK(destination.Read().View()(0, 0) == (*field.son)(0, 0));
        CHECK(destination.Read().View()(0, 0) == 100 + pulls[0]);
    }
    field.trans.pLGlobalMapping->ROffsets().back() = 0;
    CHECK(space.NumGlobal() == 2 * mpi.size);
    CHECK(space.LocalToGlobal(0) == 2 * mpi.rank);
    auto mesh = FacadeFixture();
    auto current = mesh->Entities(G::EntityKind::Cell);
    auto currentRoute = current.BuildRowTransferPlan();
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Layout);
        mutation.Commit();
    }
    CHECK_FALSE(current.Valid());
    CHECK_FALSE(currentRoute.Valid());
    CHECK_THROWS(G::LegacyFieldBridge::BorrowIndexing(field.trans, current));
    CHECK_THROWS(currentRoute.DestinationLayout(R::RowLayout::Dense(1, 1)));
    CHECK(route.Valid()); // Detached independent domain remains usable.
}

#ifndef DNDS_MESH_FACADE_TEST_EMBEDDED
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
#endif
