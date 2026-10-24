#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/FieldTransfer.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <filesystem>
#include <map>

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;

static MPIInfo TransitionMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

static ssp<UnstructuredMesh> TransitionFixture(const MPIInfo &mpi, bool periodic = false, int dim = 2)
{
    auto mesh = make_ssp<UnstructuredMesh>(mpi, dim);
    if (periodic || dim == 3)
    {
        tPoint zero{0, 0, 0};
        if (dim == 3)
            mesh->SetPeriodicGeometry({1, 0, 0}, zero, zero, {0, 1, 0}, zero, zero, {0, 0, 1}, zero, zero);
        else
            mesh->SetPeriodicGeometry({10, 0, 0}, zero, zero, {0, 10, 0}, zero, zero, zero, zero, zero);
    }
    auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial((root / "data/mesh" / (dim == 3 ? "UP3D_32.cgns" : periodic ? "IV10_10.cgns"
                                                                                          : "UniformSquare_10.cgns"))
                                  .string());
    if (periodic || dim == 3)
        reader.Deduplicate1to1Periodic(1e-8);
    reader.BuildCell2Cell();
    UnstructuredMeshSerialRW::PartitionOptions options;
    options.metisType = "KWAY";
    options.metisSeed = 42;
    options.metisNcuts = 1;
    reader.MeshPartitionCell2Cell(options);
    reader.PartitionReorderToMeshCell2Cell();
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    mesh->BuildGhostN2CB();
    mesh->AdjGlobal2LocalN2CB();
    mesh->BuildCell2CellFace();
    mesh->AdjGlobal2LocalC2CFace();
    return mesh;
}

static ReorderInput TransitionInput(const UnstructuredMesh &mesh)
{
    ReorderInput input;
    // Cyclic placement exercises actual distributed arrivals for np > 1;
    // np=1 remains a complete identity transaction.
    input.explicitMaps.push_back({EntityKind::Cell,
                                  std::vector<MPI_int>(size_t(mesh.NumCell()), (mesh.mpi.rank + 1) % mesh.mpi.size)});
    return input;
}

static std::map<DNDS::index, std::vector<DNDS::index>> TransitionCellRows(const UnstructuredMesh &mesh)
{
    std::vector<DNDS::index> local;
    for (DNDS::index i = 0; i < mesh.NumCell(); ++i)
    {
        local.push_back(mesh.cell2cellOrig(i, 0));
        local.push_back(mesh.cell2node.RowSize(i));
        for (DNDS::rowsize j = 0; j < mesh.cell2node.RowSize(i); ++j)
        {
            local.push_back(mesh.node2nodeOrig(mesh.cell2node(i, j), 0));
            local.push_back(mesh.isPeriodic ? int64_t(uint8_t(mesh.cell2nodePbi(i, j))) : 0);
        }
    }
    int count = int(local.size());
    std::vector<int> counts(size_t(mesh.mpi.size)), offsets(size_t(mesh.mpi.size + 1), 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mesh.mpi.comm);
    for (int rank = 0; rank < mesh.mpi.size; ++rank)
        offsets[size_t(rank + 1)] = offsets[size_t(rank)] + counts[size_t(rank)];
    std::vector<DNDS::index> all(size_t(offsets.back()));
    MPI_Allgatherv(local.data(), count, DNDS_MPI_INDEX, all.data(), counts.data(), offsets.data(), DNDS_MPI_INDEX, mesh.mpi.comm);
    std::map<DNDS::index, std::vector<DNDS::index>> result;
    for (size_t p = 0; p < all.size();)
    {
        auto id = all[p++];
        auto width = size_t(all[p++]) * 2;
        result[id] = std::vector<DNDS::index>(all.begin() + p, all.begin() + p + width);
        p += width;
    }
    return result;
}

static void CheckTransitionCellRows(const UnstructuredMesh &mesh,
                                    const std::map<DNDS::index, std::vector<DNDS::index>> &expected)
{
    for (DNDS::index i = 0; i < mesh.NumCell(); ++i)
    {
        const auto &row = expected.at(mesh.cell2cellOrig(i, 0));
        REQUIRE(size_t(mesh.cell2node.RowSize(i)) * 2 == row.size());
        for (DNDS::rowsize j = 0; j < mesh.cell2node.RowSize(i); ++j)
        {
            CHECK(mesh.node2nodeOrig(mesh.cell2node(i, j), 0) == row[size_t(j) * 2]);
            CHECK((mesh.isPeriodic ? int64_t(uint8_t(mesh.cell2nodePbi(i, j))) : 0) == row[size_t(j) * 2 + 1]);
        }
    }
}

TEST_CASE("Detached transition retains source scopes and publishes matching native fields")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi);
    MeshTransitionHandle handle(source);
    auto count = size_t(source->NumCell());
    R::Field<int64_t> ids({count, 1, 1}, {37, 4, count, 0}, "cell");
    R::Field<int64_t> history({count, 1, 1}, {37, 4, count, 0}, "cell");
    {
        auto a = ids.Write();
        auto b = history.Write();
        for (size_t i = 0; i < count; ++i)
        {
            a.View<int64_t>()[i] = source->cell2cellOrig(DNDS::index(i), 0);
            b.View<int64_t>()[i] = 3 * a.View<int64_t>()[i] + 11;
        }
    }
    const auto oldIdentity = ids.Values().Identity();
    auto oldStamp = source->versions.Stamp();
    auto oldRead = source->AcquireRead(MeshReadRequirements::CellFlux());
    const auto oldNodes = oldRead.HostView().NumNode();
    auto expectedRows = TransitionCellRows(*source);
    auto input = TransitionInput(*source);
    SUBCASE("preserved faces") {}
    SUBCASE("reconstructed faces") { input.destroyKinds.insert(EntityKind::Face); }
    auto prepared = PreparedMeshTransition::Prepare(handle, input,
                                                    {RegisterField(ids, EntityKind::Cell, "ids"), RegisterField(history, EntityKind::Cell, "history")});
    CHECK(handle.Get() == source);
    CHECK(ids.Values().Identity() == oldIdentity);
    CHECK(oldStamp.Valid());
    prepared.Publish();
    auto next = handle.Get();
    CHECK(next != source);
    CHECK_FALSE(next->versions.SameController(source->versions));
    CHECK(next->coords.father->HostDataIdentity() != source->coords.father->HostDataIdentity());
    CHECK(next->cell2node.father->HostDataIdentity() != source->cell2node.father->HostDataIdentity());
    CHECK(next->cell2node.father->getRowStart().get() != source->cell2node.father->getRowStart().get());
    CHECK(oldRead.HostView().NumNode() == oldNodes);
    CHECK(oldStamp.Valid());
    CHECK(ids.Values().Identity() != oldIdentity);
    CHECK(ids.GetDistribution().epoch == 5);
    REQUIRE(ids.Layout().rows == size_t(next->NumCell()));
    for (size_t i = 0; i < ids.Layout().rows; ++i)
    {
        auto id = ids.Read().View<int64_t>()[i];
        CHECK(id == next->cell2cellOrig(DNDS::index(i), 0));
        CHECK(history.Read().View<int64_t>()[i] == 3 * id + 11);
    }
    next->AssertOnFaces();
    CheckTransitionCellRows(*next, expectedRows);
    if (source->NumNode() > 0)
    {
        tPoint coordinate = source->coords[0];
        next->TransformCoords([](tPoint x)
                              { x[0] += 1; return x; });
        CHECK((source->coords[0] - coordinate).norm() == 0);
        CHECK(oldStamp.Valid());
    }
    CHECK_THROWS(prepared.Publish());
}

TEST_CASE("Detached periodic transition retains geometry and relation companion rows")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi, true);
    MeshTransitionHandle handle(source);
    auto oldRead = source->AcquireRead(MeshReadRequirements::CellFlux());
    auto oldStamp = source->versions.Stamp();
    auto expectedRows = TransitionCellRows(*source);
    R::Array<int64_t> cellBits(R::RowLayout::Dense(size_t(source->NumCell()), 1));
    auto rowBits = [](const auto &pair, DNDS::index i)
    {
        uint64_t bits = uint64_t(pair.RowSize(i));
        for (DNDS::rowsize j = 0; j < pair.RowSize(i); ++j)
            bits = bits * 257 + uint8_t(pair(i, j));
        return int64_t(bits);
    };
    {
        auto write = cellBits.Write();
        for (DNDS::index i = 0; i < source->NumCell(); ++i)
            write.At(size_t(i), 0) = rowBits(source->cell2nodePbi, i);
    }
    auto input = TransitionInput(*source);
    SUBCASE("preserved periodic faces") {}
    SUBCASE("reconstructed periodic faces") { input.destroyKinds.insert(EntityKind::Face); }
    auto prepared = PreparedMeshTransition::Prepare(handle, input,
                                                    {RegisterArray(cellBits, EntityKind::Cell, "cell-periodic-bits")});
    prepared.Publish();
    auto next = handle.Get();
    CHECK(next->isPeriodic);
    CHECK(oldStamp.Valid());
    CHECK(next->periodicInfo.translation[1].map()[0] == source->periodicInfo.translation[1].map()[0]);
    CHECK(next->cell2nodePbi.father->Size() == next->NumCell());
    CHECK(next->face2nodePbi.father->Size() == next->NumFace());
    for (DNDS::index i = 0; i < next->NumCell(); ++i)
    {
        CHECK(next->cell2nodePbi.RowSize(i) == next->cell2node.RowSize(i));
        CHECK(next->cell2facePbi.RowSize(i) == next->cell2face.RowSize(i));
        CHECK(cellBits.Read().At(size_t(i), 0) == rowBits(next->cell2nodePbi, i));
    }
    next->AssertOnFaces();
    CheckTransitionCellRows(*next, expectedRows);
}

TEST_CASE("Detached preparation failures retain mesh and every field source")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi);
    MeshTransitionHandle handle(source);
    const auto count = size_t(source->NumCell());
    R::Field<int64_t> first({count, 1, 1}, {37, 4, count, 0}, "cell");
    R::Field<int64_t> second({count, 1, 1}, {37, 4, count, 0}, "cell");
    {
        auto a = first.Write();
        auto b = second.Write();
        for (size_t i = 0; i < count; ++i)
        {
            a.View<int64_t>()[i] = int64_t(i) + 100;
            b.View<int64_t>()[i] = int64_t(i) + 1000;
        }
    }
    auto a = RegisterField(first, EntityKind::Cell, "first");
    auto b = RegisterField(second, EntityKind::Cell, "second");
    const auto oldFirst = first.Values().Identity(), oldSecond = second.Values().Identity();
    const auto oldStamp = source->versions.Stamp();
    std::string failure;
    SUBCASE("after clone") { failure = "cloned"; }
    SUBCASE("after reorder") { failure = "reordered"; }
    SUBCASE("after complete rebuild") { failure = "rebuilt"; }
    SUBCASE("second field fails on one rank")
    {
        b.prepare = [&](const auto &, const auto &m) -> std::unique_ptr<PreparedCompanion>
        {
            if (m.rank == 0)
                throw std::runtime_error("injected second-field failure");
            return std::make_unique<FieldTransferDetail::Prepared<R::Field<int64_t>>>(second, second);
        };
    }
    CHECK_THROWS(PreparedMeshTransition::Prepare(handle, TransitionInput(*source), {a, b}, [&](const char *stage)
                                                 {
        if (mpi.rank == 0 && failure == stage)
            throw std::runtime_error("injected mesh preparation failure"); }));
    CHECK(handle.Get() == source);
    CHECK(first.Values().Identity() == oldFirst);
    CHECK(second.Values().Identity() == oldSecond);
    CHECK(first.GetDistribution().epoch == 4);
    for (size_t i = 0; i < count; ++i)
    {
        CHECK(first.Read().View<int64_t>()[i] == int64_t(i) + 100);
        CHECK(second.Read().View<int64_t>()[i] == int64_t(i) + 1000);
    }
    CHECK(oldStamp.Valid());
    CHECK(source->versions.CanMutate());
}

TEST_CASE("Detached transition rejects unsupported inventory before cloning")
{
    auto source = TransitionFixture(TransitionMPI());
    MeshTransitionHandle handle(source);
    auto stamp = source->versions.Stamp();
    source->node2parentNode.push_back(0);
    bool cloned = false;
    CHECK_THROWS(PreparedMeshTransition::Prepare(handle, TransitionInput(*source), {}, [&](const char *)
                                                 { cloned = true; }));
    CHECK_FALSE(cloned);
    CHECK(stamp.Valid());
    CHECK(handle.Get() == source);
}

TEST_CASE("Inactive registered fields still require exact owned source rows")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi);
    MeshTransitionHandle handle(source);
    auto count = size_t(source->NumFace());
    R::Field<int64_t> field({count, 1, 1}, {71, 3, count, 0}, "face");
    R::Array<int64_t> array(R::RowLayout::Dense(count, 1));
    bool valid = false;
    SUBCASE("valid owned face values retain identity") { valid = true; }
    SUBCASE("rank-local inactive field ghosts")
    {
        if (mpi.rank == 0)
            field = R::Field<int64_t>({count + 1, 1, 1}, {71, 3, count, 1}, "face");
    }
    SUBCASE("rank-local inactive field wrong rows")
    {
        if (mpi.rank == 0)
            field = R::Field<int64_t>({count + 1, 1, 1}, {71, 3, count + 1, 0}, "face");
    }
    SUBCASE("rank-local inactive array wrong rows")
    {
        if (mpi.rank == 0)
            array = R::Array<int64_t>(R::RowLayout::Dense(count + 1, 1));
    }
    auto input = TransitionInput(*source); // Face family is deliberately inactive.
    const auto fieldIdentity = field.Values().Identity(), arrayIdentity = array.Identity();
    auto stamp = source->versions.Stamp();
    auto fields = std::vector<CompanionEntry>{RegisterField(field, EntityKind::Face, "inactive-field"),
                                              RegisterArray(array, EntityKind::Face, "inactive-array")};
    if (valid)
    {
        auto prepared = PreparedMeshTransition::Prepare(handle, input, fields);
        prepared.Publish();
        CHECK(handle.Get() != source);
        CHECK(field.GetDistribution().epoch == 3);
    }
    else
    {
        CHECK_THROWS(PreparedMeshTransition::Prepare(handle, input, fields));
        CHECK(handle.Get() == source);
    }
    CHECK(field.Values().Identity() == fieldIdentity);
    CHECK(array.Identity() == arrayIdentity);
    CHECK(stamp.Valid());
}

TEST_CASE("Output export may explicitly drop only the copied boundary extraction cache")
{
    auto source = TransitionFixture(TransitionMPI());
    source->node2bndNode.assign(size_t(source->NumNodeProc()), 17);
    auto cache = source->node2bndNode;
    auto stamp = source->versions.Stamp();
    auto retained = source->AcquireRead();
    auto view = retained.HostView();
    CHECK_THROWS(CloneCurrentMeshForSetup(*source));
    auto output = CloneCurrentMeshForSetup(*source, 1, TransitionCachePolicy::Invalidate, BoundaryCachePolicy::DropBoundaryCache);
    CHECK(output->node2bndNode.empty());
    CHECK(source->node2bndNode == cache);
    CHECK(stamp.Valid());
    CHECK_FALSE(source->versions.SameController(output->versions));
    REQUIRE(output->NumNode() == view.NumNode());
    for (DNDS::index node = 0; node < output->NumNode(); ++node)
        CHECK(output->coords[node].isApprox(view.Coord(node)));
    source->node2parentNode.push_back(0);
    CHECK_THROWS(CloneCurrentMeshForSetup(*source, 1, TransitionCachePolicy::Invalidate, BoundaryCachePolicy::DropBoundaryCache));
    CHECK(source->node2bndNode == cache);
    CHECK(stamp.Valid());
}

TEST_CASE("Delayed publication rejects rank-local native field edits and replacement")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi);
    MeshTransitionHandle handle(source);
    auto count = size_t(source->NumCell());
    R::Field<int64_t> field({count, 1, 1}, {37, 4, count, 0}, "cell");
    R::Array<int64_t> array(R::RowLayout::Dense(count, 1));
    auto prepared = PreparedMeshTransition::Prepare(handle, TransitionInput(*source),
                                                    {RegisterField(field, EntityKind::Cell, "field"), RegisterArray(array, EntityKind::Cell, "array")});
    SUBCASE("field values changed")
    {
        if (mpi.rank == 0 && count)
            field.Write().View<int64_t>()[0] = 77;
    }
    SUBCASE("field allocation replaced with equal metadata")
    {
        if (mpi.rank == 0)
            field = R::Field<int64_t>({count, 1, 1}, {37, 4, count, 0}, "cell");
    }
    SUBCASE("array values changed")
    {
        if (mpi.rank == 0 && count)
            array.Write().At(0, 0) = 88;
    }
    SUBCASE("array allocation replaced with equal shape")
    {
        if (mpi.rank == 0)
            array = R::Array<int64_t>(R::RowLayout::Dense(count, 1));
    }
    const auto fieldIdentity = field.Values().Identity(), arrayIdentity = array.Identity();
    auto sourceStamp = source->versions.Stamp();
    CHECK_THROWS(prepared.Publish());
    CHECK(handle.Get() == source);
    CHECK(field.Values().Identity() == fieldIdentity);
    CHECK(array.Identity() == arrayIdentity);
    CHECK(sourceStamp.Valid());
}

TEST_CASE("Current native mesh exports independent authoritative rows and wall vectors")
{
    auto source = TransitionFixture(TransitionMPI());
    {
        auto mutation = source->BeginManagedMutation(R::Change::Interface | R::Change::Values);
        source->nodeWallDist.InitPair("transition.wall", source->mpi);
        source->nodeWallDist.father->Resize(source->NumNode());
        source->nodeWallDist.son->Resize(source->NumNodeGhost());
        for (DNDS::index i = 0; i < source->NumNodeProc(); ++i)
            source->nodeWallDist[i] = tPoint{DNDS::real(source->node2nodeOrig(i, 0)), 2, 3};
        source->nodeWallDist.TransAttach();
        source->nodeWallDist.trans.BorrowGGIndexing(source->coords.trans);
        mutation.Commit();
    }
    source->BuildVTKConnectivity();
    auto requirements = MeshReadRequirements::CellFlux();
    requirements.inverse = true;
    requirements.cellFaceNeighbors = true;
    requirements.wallDistance = true;
    source->PublishNative(requirements);
    auto retained = source->AcquireRead(requirements);
    auto view = retained.HostView();
    auto stamp = source->versions.Stamp();
    // Explicit raw workspace edits must not replace published query authority.
    if (source->coords.father->Size())
        source->coords[0][0] += 1000;
    if (source->cell2node.father->Size())
    {
        source->cell2node(0, 0) = UnInitIndex;
        source->cell2cellOrig(0, 0) = -999;
    }
    auto clone = CloneCurrentMeshForSetup(*source);
    CHECK(source->HasNativeState());
    CHECK_FALSE(clone->HasNativeState());
    CHECK(stamp.Valid());
    CHECK(clone->NumNode() == view.NumNode());
    CHECK(clone->vtkCell2nodeOffsets.size() == size_t(view.NumCell() + 1));
    CHECK(clone->vtkNCellGlobal >= view.NumCell());
    for (DNDS::index i = 0; i < clone->NumNode(); ++i)
    {
        CHECK(clone->coords[i].isApprox(view.Coord(i)));
        CHECK(clone->nodeWallDist[i].isApprox(view.NodeWallDist(i)));
        CHECK(clone->node2nodeOrig(i, 0) == source->OriginalEntityId(EntityKind::Node, i));
    }
    for (DNDS::index i = 0; i < clone->NumCell(); ++i)
    {
        CHECK(clone->cell2cellOrig(i, 0) == source->OriginalEntityId(EntityKind::Cell, i));
        REQUIRE(size_t(clone->cell2node.RowSize(i)) == view.CellNodes(i).size());
        for (DNDS::rowsize j = 0; j < clone->cell2node.RowSize(i); ++j)
            CHECK(clone->node2nodeOrig(clone->cell2node(i, j), 0) ==
                  source->OriginalEntityId(EntityKind::Node, view.CellNodes(i)[size_t(j)]));
    }
    auto invalidated = CloneCurrentMeshForSetup(*source, 1, TransitionCachePolicy::Invalidate);
    CHECK(invalidated->vtkCell2nodeOffsets.empty());
    CHECK(invalidated->localPartitionStarts.empty());
    auto checkpoint = make_ssp<Serializer::SerializerJSON>();
    checkpoint->CreatePath("/");
    CHECK_THROWS(source->WriteSerialize(checkpoint, "legacy"));
    CHECK(checkpoint->ListCurrentPath().empty());
    CHECK_THROWS(WriteCurrentMeshSnapshot(*source, checkpoint, "current"));
    CHECK_THROWS(WriteCurrentPrimaryMeshSnapshot(*source, checkpoint, "current"));
    CHECK(checkpoint->ListCurrentPath().empty());
    CHECK(source->HasNativeState());
    CHECK(stamp.Valid());
    WriteCurrentPrimaryMeshSnapshot(*source, checkpoint, "current", 1, CurrentMeshCheckpointPolicy::DropAuxiliary);
    checkpoint->GoToPath("current");
    int schema = 0, inventory = 0;
    std::string kind, auxiliary;
    checkpoint->ReadInt("checkpoint_schema", schema);
    checkpoint->ReadInt("rebuild_inventory", inventory);
    checkpoint->ReadString("checkpoint_kind", kind);
    checkpoint->ReadString("auxiliary_policy", auxiliary);
    CHECK(schema == 1);
    CHECK((inventory & 64) != 0);
    CHECK((inventory & 1) != 0);
    CHECK(kind == "DNDSR.CurrentPrimaryMeshSnapshot");
    CHECK(auxiliary == "drop_auxiliary");
    checkpoint->GoToPath("/");
    auto restored = make_ssp<UnstructuredMesh>(source->mpi, 2);
    restored->ReadSerialize(checkpoint, "current");
    CHECK(source->HasNativeState());
    CHECK(stamp.Valid());
    REQUIRE(restored->NumNode() == view.NumNode());
    for (DNDS::index i = 0; i < restored->NumNode(); ++i)
        CHECK(restored->coords[i].isApprox(view.Coord(i)));
    for (DNDS::index i = 0; i < restored->NumCell(); ++i)
        CHECK(restored->cell2cellOrig(i, 0) == source->OriginalEntityId(EntityKind::Cell, i));
}

TEST_CASE("Detached transition reconstructs 3D edges under an explicit cache policy")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi, false, 3);
    source->InterpolateEdge();
    source->RecreatePeriodicNodes();
    source->BuildVTKConnectivity();
    DNDS::index globalEdges = source->edge2node.father->globalSize();
    MeshTransitionHandle handle(source);
    auto stamp = source->versions.Stamp();
    auto input = TransitionInput(*source);
    input.destroyKinds.insert(EntityKind::Edge);
    auto prepared = PreparedMeshTransition::Prepare(handle, input, {}, {}, 1,
                                                    TransitionCachePolicy::RebuildOutput);
    prepared.Publish();
    auto next = handle.Get();
    CHECK(stamp.Valid());
    CHECK(next->adjEdgeState == Adj_PointToLocal);
    CHECK(next->edge2node.father->globalSize() == globalEdges);
    CHECK(next->vtkCell2nodeOffsets.size() == size_t(next->NumCell() + 1));
    auto requirements = MeshReadRequirements::CellFlux();
    requirements.edges = true;
    auto read = next->AcquireRead(requirements);
    auto view = read.HostView();
    for (DNDS::index cell = 0; cell < view.NumCell(); ++cell)
        for (auto edge : view.CellEdges(cell))
        {
            CHECK(edge >= 0);
            CHECK(edge < view.NumEdgeProc());
            for (auto node : view.EdgeNodes(edge))
            {
                CHECK(node >= 0);
                CHECK(node < view.NumNodeProc());
            }
        }
}

TEST_CASE("Detached transition preserves independently placed edges with exact native fields and companion slots")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi, false, 3);
    source->InterpolateEdge();
    const auto edgeSpace = source->Entities(EntityKind::Edge);
    auto owned = size_t(edgeSpace.NumOwned());
    R::Field<int64_t> ids({owned, 1, 1}, {92, 7, owned, 0}, "edge");
    // Exact source edge rows travel beside the IDs, preserving endpoint and
    // incident-cell order plus every endpoint's periodic companion bits.
    std::vector<size_t> widths(owned);
    for (size_t edge = 0; edge < owned; ++edge)
        widths[edge] = size_t(source->edge2node.RowSize(DNDS::index(edge))) * 2 +
                       size_t(source->edge2cell.RowSize(DNDS::index(edge))) + 2;
    R::Array<int64_t> exact(R::RowLayout::Ragged(widths));
    {
        auto id = ids.Write();
        auto row = exact.Write();
        for (size_t edge = 0; edge < owned; ++edge)
        {
            auto e = DNDS::index(edge);
            id.View<int64_t>()[edge] = edgeSpace.LocalToGlobal(e);
            size_t slot = 0;
            row.At(edge, slot++) = id.View<int64_t>()[edge];
            row.At(edge, slot++) = source->edge2node.RowSize(e);
            for (DNDS::rowsize j = 0; j < source->edge2node.RowSize(e); ++j)
            {
                row.At(edge, slot++) = source->OriginalEntityId(EntityKind::Node, source->edge2node(e, j));
                row.At(edge, slot++) = uint8_t(source->edge2nodePbi(e, j));
            }
            for (auto cell : source->edge2cell[e])
                row.At(edge, slot++) = source->OriginalEntityId(EntityKind::Cell, cell);
        }
    }
    std::vector<size_t> cellWidths(size_t(source->NumCell()));
    for (DNDS::index cell = 0; cell < source->NumCell(); ++cell)
        cellWidths[size_t(cell)] = size_t(source->cell2edge.RowSize(cell)) * 2;
    R::Array<int64_t> exactCells(R::RowLayout::Ragged(cellWidths));
    {
        auto rows = exactCells.Write();
        for (DNDS::index cell = 0; cell < source->NumCell(); ++cell)
            for (DNDS::rowsize slot = 0; slot < source->cell2edge.RowSize(cell); ++slot)
            {
                rows.At(size_t(cell), size_t(slot) * 2) = edgeSpace.LocalToGlobal(source->cell2edge(cell, slot));
                rows.At(size_t(cell), size_t(slot) * 2 + 1) = uint8_t(source->cell2edgePbi(cell, slot));
            }
    }
    SUBCASE("legacy current source") {}
    SUBCASE("native current source")
    {
        auto requirements = MeshReadRequirements::CellFlux();
        requirements.edges = true;
        requirements.inverse = true;
        requirements.cellFaceNeighbors = true;
        source->PublishNative(requirements);
    }
    auto input = TransitionInput(*source);
    input.explicitMaps.push_back({EntityKind::Edge,
                                  std::vector<MPI_int>(owned, (mpi.rank + 2) % mpi.size)});
    MeshTransitionHandle handle(source);
    auto stamp = source->versions.Stamp();
    auto retained = source->AcquireRead([]
                                        {
        auto r = MeshReadRequirements::CellFlux();
        r.edges = true;
        return r; }());
    auto prepared = PreparedMeshTransition::Prepare(handle, input,
                                                    {RegisterField(ids, EntityKind::Edge, "edge-original-id"), RegisterArray(exact, EntityKind::Edge, "edge-source-row"),
                                                     RegisterArray(exactCells, EntityKind::Cell, "cell-edge-source-row")});
    CHECK(handle.Get() == source);
    CHECK(stamp.Valid());
    prepared.Publish();
    auto next = handle.Get();
    CHECK(stamp.Valid());
    CHECK(ids.GetDistribution().epoch == 8);
    auto edgeIds = ids.Read();
    int edgeCount = int(ids.Layout().rows);
    std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size + 1), 0);
    MPI_Allgather(&edgeCount, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
    for (int rank = 0; rank < mpi.size; ++rank)
        offsets[size_t(rank + 1)] = offsets[size_t(rank)] + counts[size_t(rank)];
    std::vector<int64_t> oldIds(size_t(offsets.back()));
    MPI_Allgatherv(edgeIds.View<int64_t>().data, edgeCount, DNDS_MPI_INDEX, oldIds.data(), counts.data(), offsets.data(), DNDS_MPI_INDEX, mpi.comm);
    REQUIRE(ids.Layout().rows == size_t(next->edge2node.father->Size()));
    REQUIRE(exact.Rows() == ids.Layout().rows);
    auto rows = exact.Read();
    for (size_t edge = 0; edge < exact.Rows(); ++edge)
    {
        auto e = DNDS::index(edge);
        size_t slot = 0;
        CHECK(rows.At(edge, slot++) == edgeIds.View<int64_t>()[edge]);
        CHECK(rows.At(edge, slot++) == next->edge2node.RowSize(e));
        for (DNDS::rowsize j = 0; j < next->edge2node.RowSize(e); ++j)
        {
            CHECK(rows.At(edge, slot++) == next->OriginalEntityId(EntityKind::Node, next->edge2node(e, j)));
            CHECK(rows.At(edge, slot++) == uint8_t(next->edge2nodePbi(e, j)));
        }
        for (auto cell : next->edge2cell[e])
        {
            REQUIRE(cell >= 0);
            CHECK(rows.At(edge, slot++) == next->OriginalEntityId(EntityKind::Cell, cell));
        }
        CHECK(slot == exact.Layout().RowSize(edge));
    }
    auto currentEdges = next->Entities(EntityKind::Edge);
    auto cellRows = exactCells.Read();
    REQUIRE(exactCells.Rows() == size_t(next->NumCell()));
    for (DNDS::index cell = 0; cell < next->NumCell(); ++cell)
    {
        REQUIRE(exactCells.Layout().RowSize(size_t(cell)) == size_t(next->cell2edge.RowSize(cell)) * 2);
        for (DNDS::rowsize slot = 0; slot < next->cell2edge.RowSize(cell); ++slot)
        {
            auto edge = next->cell2edge(cell, slot);
            REQUIRE(edge >= 0);
            CHECK(oldIds.at(size_t(currentEdges.LocalToGlobal(edge))) == cellRows.At(size_t(cell), size_t(slot) * 2));
            CHECK(uint8_t(next->cell2edgePbi(cell, slot)) == cellRows.At(size_t(cell), size_t(slot) * 2 + 1));
        }
    }
    auto requirements = MeshReadRequirements::CellFlux();
    requirements.edges = true;
    auto read = next->AcquireRead(requirements);
    CHECK(read.HostView().NumEdge() == next->edge2node.father->Size());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
