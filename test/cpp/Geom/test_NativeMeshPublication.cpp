#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Mesh/NativeMeshState.hpp"
#include "Solver/Direct.hpp"
#include <optional>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;

TEST_CASE("Explicit native publication owns queries independently of the raw edit workspace")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto &mesh = *fixture.mesh;
    CHECK_FALSE(mesh.HasNativeState());
    const auto originalCount = mesh.NumCell();
    const auto originalNode = mesh.cell2node(0, 0);
    const G::tPoint originalPoint = mesh.Coord(0);
    const auto originalCellPbi = mesh.getCell2NodeIndexPbiRow(0);
    const auto originalBndPbi = mesh.NumBndProc() ? mesh.getBnd2NodeIndexPbiRow(0) : std::vector<G::NodeIndexPBI>{};
    const auto originalLocalGraph = mesh.GetCell2CellFaceVLocal();
    mesh.PublishNative(G::MeshReadRequirements::CellFlux());
    CHECK(mesh.HasNativeState());
    CHECK_THROWS(mesh.template deviceView<DNDS::DeviceBackend::Host>());
    CHECK_THROWS(mesh.to_device(DNDS::DeviceBackend::Host));
    CHECK_THROWS(mesh.to_host());
    // Host placement already exists: validation must precede the cached return.
    CHECK_NOTHROW(mesh.PlaceNative(R::Device::Host(), R::ExecutionContext::Host()));
    CHECK_THROWS(mesh.PlaceNative(R::Device::Host(), R::ExecutionContext::CUDA(0)));
    CHECK_THROWS(mesh.PlaceNative(R::Device::CUDA(-1), R::ExecutionContext::CUDA(-1)));
    int invalidHostStream = 0;
    CHECK_THROWS(mesh.PlaceNative(R::Device::Host(), {R::Device::Host(), &invalidHostStream}));
    auto space = mesh.Entities(G::EntityKind::Cell);
    const auto globalCell = space.LocalToGlobal(0);
    const auto nodeSpace = mesh.Entities(G::EntityKind::Node);
    const auto globalNode = nodeSpace.LocalToGlobal(0);
    const auto bndSpace = mesh.Entities(G::EntityKind::Bnd);
    const auto globalBnd = bndSpace.NumLocal() ? bndSpace.LocalToGlobal(0) : DNDS::UnInitIndex;
    const auto faceSpace = mesh.Entities(G::EntityKind::Face);
    const auto globalFace = faceSpace.LocalToGlobal(0);
    CHECK(mesh.GetCell2CellFaceVLocal() == originalLocalGraph);
    CHECK_THROWS(mesh.IndexLocal2Global(mesh.cell2node, 0));
    CHECK(space.SameOrder(mesh.Entities(G::EntityKind::Cell)));
    std::optional<G::MeshReadAccess> first, second;
    first.emplace(mesh.AcquireRead(G::MeshReadRequirements::CellFlux()));
    second.emplace(mesh.AcquireRead(G::MeshReadRequirements::CellFlux()));
    CHECK(first->HostView().CellNodes(0).data() == second->HostView().CellNodes(0).data());
    CHECK(first->HostView().CellNodes(0).data() != mesh.cell2node.father->rowPtr(0));
    // Raw writes affect only the detached workspace in this explicit mode.
    mesh.cell2node(0, 0) = 1000000;
    mesh.coords[0] = G::tPoint{123, 456, 789};
    const auto originalBndNode = mesh.NumBndProc() ? mesh.bnd2node(0, 0) : DNDS::UnInitIndex;
    if (mesh.NumBndProc())
        mesh.bnd2node(0, 0) = 1000000;
    // Corrupt only the detached map objects; native numbering owns deep copies.
    const auto originalMapOffset = mesh.cell2node.trans.pLGlobalMapping->ROffsets()[size_t(mpi.rank)];
    mesh.cell2node.trans.pLGlobalMapping->ROffsets()[size_t(mpi.rank)] = 1000000;
    CHECK(mesh.CellIndexLocal2Global(0) == globalCell);
    CHECK(mesh.CellIndexGlobal2Local(globalCell) == 0);
    CHECK(mesh.CellIndexLocal2Global_NoSon(0) == globalCell);
    CHECK(mesh.CellIndexGlobal2Local_NoSon(globalCell) == 0);
    CHECK(mesh.NodeIndexLocal2Global(0) == globalNode);
    CHECK(mesh.NodeIndexGlobal2Local(globalNode) == 0);
    if (bndSpace.NumLocal())
    {
        CHECK(mesh.BndIndexLocal2Global(0) == globalBnd);
        CHECK(mesh.BndIndexGlobal2Local(globalBnd) == 0);
    }
    CHECK(mesh.FaceIndexLocal2Global(0) == globalFace);
    CHECK(mesh.FaceIndexGlobal2Local(globalFace) == 0);
    CHECK(mesh.CellIndexLocal2Global(DNDS::UnInitIndex) == DNDS::UnInitIndex);
    CHECK(mesh.CellIndexGlobal2Local(DNDS::UnInitIndex) == DNDS::UnInitIndex);
    CHECK(mesh.CellIndexLocal2Global(-1 - globalCell) == globalCell);
    CHECK_THROWS(mesh.CellIndexLocal2Global_NoSon(space.NumOwned()));
    CHECK(mesh.getCell2NodeIndexPbiRow(0) == originalCellPbi);
    if (mesh.NumBndProc())
        CHECK(mesh.getBnd2NodeIndexPbiRow(0) == originalBndPbi);
    CHECK(mesh.CellNodes(0)[0] == originalNode);
    CHECK(mesh.Coord(0).isApprox(originalPoint));
    CHECK(mesh.GetCell2CellFaceVLocal() == originalLocalGraph);
    CHECK(mesh.GetCellElement(0).GetNumNodes() == int(mesh.CellNodes(0).size()));
    CHECK(mesh.AcquireRead(G::MeshReadRequirements::CellFlux()).HostView().CellNodes(0)[0] == originalNode);
    CHECK(G::NativeMeshSnapshot::Capture(mesh).AcquireRead().HostView().CellNodes(0)[0] == originalNode);
    CHECK_THROWS(G::NativeMeshSnapshot::ImportLegacyWorkspace(mesh));
    CHECK(mesh.NumCell() == originalCount);
    CHECK_THROWS(mesh.UseLegacyWorkspace());
    CHECK_THROWS(mesh.TransformCoords([](const G::tPoint &point)
                                      { return point; }));
    first.reset();
    second.reset();
    mesh.UseLegacyWorkspace();
    CHECK_FALSE(mesh.HasNativeState());
    CHECK_FALSE(space.Valid());
    CHECK_THROWS(mesh.PublishNative(G::MeshReadRequirements::CellFlux()));
    CHECK_FALSE(mesh.HasNativeState());
    mesh.cell2node(0, 0) = originalNode;
    if (mesh.NumBndProc())
        mesh.bnd2node(0, 0) = originalBndNode;
    mesh.cell2node.trans.pLGlobalMapping->ROffsets()[size_t(mpi.rank)] = originalMapOffset;
    mesh.coords[0] = originalPoint;
    mesh.PublishNative(G::MeshReadRequirements::CellFlux());
    CHECK(mesh.HasNativeState());
}

TEST_CASE("Published entity maps cannot be changed through a legacy field export")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto &mesh = *fixture.mesh;
    mesh.PublishNative(G::MeshReadRequirements::CellFlux());
    auto space = mesh.Entities(G::EntityKind::Cell);
    const auto firstGlobal = space.LocalToGlobal(0);
    const auto total = space.NumGlobal();
    DNDS::ArrayPair<DNDS::ParArray<DNDS::real, 1>> field;
    field.InitPair("native.publication.export", mpi);
    field.father->Resize(space.NumOwned());
    field.son->Resize(0);
    field.TransAttach();
    G::LegacyFieldBridge::BorrowIndexing(field.trans, space);
    field.trans.pLGlobalMapping->RLengths()[size_t(mpi.rank)] = 0;
    field.trans.pLGlobalMapping->ROffsets().back() = 0;
    if (space.NumGhost())
    {
        const auto ghostGlobal = space.LocalToGlobal(space.NumOwned());
        field.trans.pLGhostMapping->ghostIndex[0] = -123;
        CHECK(space.LocalToGlobal(space.NumOwned()) == ghostGlobal);
    }
    CHECK(space.LocalToGlobal(0) == firstGlobal);
    CHECK(space.NumGlobal() == total);
    CHECK(space.Owner(firstGlobal) == mpi.rank);
    CHECK(space.SameOrder(mesh.Entities(G::EntityKind::Cell)));
}

TEST_CASE("Published periodic node rows ignore detached legacy bit edits")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto &mesh = *fixture.mesh;
    // Enable a zero-transform periodic frame so the real prepared primary
    // topology can exercise both packed periodic row families independently.
    auto initialize = [&](auto &bits, const auto &nodes, const char *name)
    {
        bits.InitPair(name, mpi);
        bits.father->Resize(nodes.father->Size(), [&](DNDS::index row)
                            { return nodes.father->RowSize(row); });
        bits.son->Resize(nodes.son->Size(), [&](DNDS::index row)
                         { return nodes.son->RowSize(row); });
        for (DNDS::index row = 0; row < nodes.Size(); ++row)
            for (DNDS::rowsize slot = 0; slot < nodes.RowSize(row); ++slot)
                bits(row, slot) = G::NodePeriodicBits{};
        bits.TransAttach();
        bits.trans.BorrowGGIndexing(nodes.trans);
    };
    initialize(mesh.cell2nodePbi, mesh.cell2node, "publication.cellBits");
    initialize(mesh.bnd2nodePbi, mesh.bnd2node, "publication.bndBits");
    mesh.isPeriodic = true;
    mesh.cell2nodePbi(0, 0) = G::nodePB1;
    if (mesh.NumBndProc())
        mesh.bnd2nodePbi(0, 0) = G::nodePB2;
    const auto cell = mesh.getCell2NodeIndexPbiRow(0);
    const auto bnd = mesh.NumBndProc() ? mesh.getBnd2NodeIndexPbiRow(0) : std::vector<G::NodeIndexPBI>{};
    mesh.PublishNative();
    auto read = mesh.AcquireRead();
    mesh.cell2nodePbi(0, 0) = G::nodePB3;
    if (mesh.NumBndProc())
        mesh.bnd2nodePbi(0, 0) = G::nodePB3;
    mesh.cell2node(0, 0) = 1000000;
    if (mesh.NumBndProc())
        mesh.bnd2node(0, 0) = 1000000;
    mesh.isPeriodic = false;
    CHECK(mesh.getCell2NodeIndexPbiRow(0) == cell);
    if (mesh.NumBndProc())
        CHECK(mesh.getBnd2NodeIndexPbiRow(0) == bnd);
    CHECK(read.HostView().CellNodePeriodicBits(0)[0] == G::nodePB1);
    if (mesh.NumBndProc())
        CHECK(read.HostView().BoundaryNodePeriodicBits(0)[0] == G::nodePB2);
}

TEST_CASE("Legacy mutation requires explicitly selecting its workspace")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto &mesh = *fixture.mesh;
    mesh.PublishNative(G::MeshReadRequirements::CellFlux());
    auto old = mesh.Entities(G::EntityKind::Cell);
    auto detached = G::NativeMeshSnapshot::Capture(mesh, G::MeshReadRequirements::CellFlux());
    const G::tPoint point = detached.AcquireRead().HostView().Coord(0);
    {
        auto read = mesh.AcquireRead();
        const auto stamp = mesh.versions.Stamp();
        CHECK_THROWS(mesh.TransformCoords([](const G::tPoint &p)
                                          { return p; }));
        CHECK(stamp.Valid());
        CHECK(read.HostView().Coord(0).isApprox(point));
    }
    CHECK(mesh.HasNativeState());
    CHECK(old.Valid());
    mesh.UseLegacyWorkspace();
    mesh.TransformCoords([](const G::tPoint &p)
                         { return (p + G::tPoint{1, 0, 0}).eval(); });
    CHECK_FALSE(mesh.HasNativeState());
    CHECK_FALSE(old.Valid());
    CHECK(detached.AcquireRead().HostView().Coord(0).isApprox(point));
    mesh.PublishNative(G::MeshReadRequirements::CellFlux());
    CHECK(mesh.AcquireRead().HostView().Coord(0).isApprox(point + G::tPoint{1, 0, 0}));
}

TEST_CASE("Legacy original-ID semantic lookup rejects absent storage and invalid rows")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    G::UnstructuredMesh empty(mpi, 2);
    CHECK_THROWS(empty.OriginalEntityId(G::EntityKind::Cell, 0));
    CHECK_THROWS(empty.OriginalEntityId(G::EntityKind::Face, 0));
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    CHECK_THROWS(fixture.mesh->OriginalEntityId(G::EntityKind::Cell, -1));
    CHECK_THROWS(fixture.mesh->OriginalEntityId(G::EntityKind::Cell, fixture.mesh->NumCellProc()));
    CHECK(fixture.mesh->OriginalEntityId(G::EntityKind::Cell, 0) == fixture.mesh->cell2cellOrig(0, 0));
}

TEST_CASE("Native publication rejects stale original-ID source correspondence")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto &mesh = *fixture.mesh;
    auto check = [&](auto &pair)
    {
        const auto originalMap = pair.trans.pLGhostMapping;
        pair.trans.pLGhostMapping = std::make_shared<DNDS::OffsetAscendIndexMapping>(*originalMap);
        CHECK_THROWS(mesh.PublishNative(G::MeshReadRequirements::CellFlux()));
        CHECK_FALSE(mesh.HasNativeState());
        CHECK(mesh.versions.CanMutate());
        pair.trans.pLGhostMapping = originalMap;
        const auto attached = pair.trans.son;
        using Array = typename std::remove_reference_t<decltype(attached)>::element_type;
        pair.trans.son = std::make_shared<Array>(*attached);
        CHECK_THROWS(mesh.PublishNative(G::MeshReadRequirements::CellFlux()));
        pair.trans.son = attached;
    };
    check(mesh.cell2cellOrig);
    check(mesh.node2nodeOrig);
    check(mesh.bnd2bndOrig);
    const auto original = mesh.OriginalEntityId(G::EntityKind::Cell, 0);
    CHECK_NOTHROW(mesh.PublishNative(G::MeshReadRequirements::CellFlux()));
    CHECK(mesh.OriginalEntityId(G::EntityKind::Cell, 0) == original);
}

TEST_CASE("Native symbolic factorization ignores detached legacy neighbor caches")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto &mesh = *fixture.mesh;
    const auto count = mesh.NumCell();
    const auto split = count / 2;
    mesh.localPartitionStarts = {0, split, count};
    mesh.cell2cellFaceVLocalParts.resize(size_t(count));
    for (int part = 0; part < 2; ++part)
        for (DNDS::index cell = mesh.LocalPartStart(part); cell < mesh.LocalPartEnd(part); ++cell)
        {
            if (cell > mesh.LocalPartStart(part))
                mesh.cell2cellFaceVLocalParts[size_t(cell)].push_back(cell - 1);
            if (cell + 1 < mesh.LocalPartEnd(part))
                mesh.cell2cellFaceVLocalParts[size_t(cell)].push_back(cell + 1);
        }
    DNDS::Direct::DirectPrecControl control;
    control.useDirectPrec = true;
    DNDS::Direct::SerialSymLUStructure legacy(mpi, count), native(mpi, count);
    mesh.ObtainSymmetricSymbolicFactorization(legacy, control);
    mesh.PublishNative(G::MeshReadRequirements::CellFlux());
    mesh.localPartitionStarts = {1234};
    mesh.cell2cellFaceVLocalParts.assign(1, {1234});
    CHECK_NOTHROW(mesh.ObtainSymmetricSymbolicFactorization(native, control));
    CHECK(native.localPartStarts == legacy.localPartStarts);
    CHECK(native.lowerTriStructure == legacy.lowerTriStructure);
    CHECK(native.upperTriStructure == legacy.upperTriStructure);
    CHECK(native.cell2cellFaceVLocal2FullRowPos == legacy.cell2cellFaceVLocal2FullRowPos);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
