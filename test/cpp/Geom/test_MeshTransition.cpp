#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/FieldTransfer.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshLegacyArchive.hpp"
#include "Geom/Mesh/NativeMeshExtraction.hpp"
#include "LegacyMeshFixtureCodec.hpp"
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
    std::optional<NativePeriodicImportOptions> periodicOptions;
    if (periodic || dim == 3)
    {
        periodicOptions.emplace();
        periodicOptions->tolerance = 1e-8;
        if (dim == 2)
        {
            periodicOptions->frames.translation[1].map() = tPoint{10, 0, 0};
            periodicOptions->frames.translation[2].map() = tPoint{0, 10, 0};
            periodicOptions->frames.translation[3].map().setZero();
        }
    }
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    const auto file = root / "data/mesh" / (dim == 3 ? "UP3D_32.cgns" : periodic ? "IV10_10.cgns"
                                                                                 : "UniformSquare_10.cgns");
    AutoAppendName2ID names;
    auto primary = ReadNativeCGNS(mpi, file.string(), dim, [&](const std::string &name)
                                  { return names(name); }, 0, {1, true}, periodicOptions);
    auto prepared = NativeMeshPreparation::Partition(*primary, mpi, {"KWAY", 30, 42, 0, 1, 1}).state;
    if (dim == 3)
        prepared = NativeMeshTopology::Build(*prepared, mpi, false, true);
    return UnstructuredMesh::FromNative(mpi, std::move(prepared));
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
    const auto &view = mesh.CurrentNativeState()->BorrowHostView();
    std::vector<DNDS::index> local;
    for (DNDS::index i = 0; i < view.NumCell(); ++i)
    {
        const auto nodes = view.CellNodes(i);
        local.push_back(mesh.OriginalEntityId(EntityKind::Cell, i));
        local.push_back(DNDS::index(nodes.size()));
        for (size_t slot = 0; slot < nodes.size(); ++slot)
        {
            local.push_back(mesh.OriginalEntityId(EntityKind::Node, nodes[slot]));
            local.push_back(view.IsPeriodic() ? int64_t(uint8_t(view.CellNodePeriodicBits(i)[slot])) : 0);
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

static void CheckTransitionCellRows(UnstructuredMesh &mesh,
                                    const std::map<DNDS::index, std::vector<DNDS::index>> &expected)
{
    auto retained = mesh.AcquireRead();
    auto view = retained.HostView();
    for (DNDS::index i = 0; i < view.NumCell(); ++i)
    {
        const auto &row = expected.at(mesh.OriginalEntityId(EntityKind::Cell, i));
        const auto nodes = view.CellNodes(i);
        REQUIRE(nodes.size() * 2 == row.size());
        for (size_t j = 0; j < nodes.size(); ++j)
        {
            CHECK(mesh.OriginalEntityId(EntityKind::Node, nodes[j]) == row[j * 2]);
            CHECK((view.IsPeriodic() ? int64_t(uint8_t(view.CellNodePeriodicBits(i)[j])) : 0) == row[j * 2 + 1]);
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
            a.View<int64_t>()[i] = source->OriginalEntityId(EntityKind::Cell, DNDS::index(i));
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
    {
        auto current = next->AcquireRead();
        if (source->NumNode())
            CHECK(current.HostView().Coord(0).data() != oldRead.HostView().Coord(0).data());
        if (source->NumCell())
            CHECK(current.HostView().CellNodes(0).data() != oldRead.HostView().CellNodes(0).data());
    }
    CHECK(oldRead.HostView().NumNode() == oldNodes);
    CHECK(oldStamp.Valid());
    CHECK(ids.Values().Identity() != oldIdentity);
    CHECK(ids.GetDistribution().epoch == 5);
    REQUIRE(ids.Layout().rows == size_t(next->NumCell()));
    for (size_t i = 0; i < ids.Layout().rows; ++i)
    {
        auto id = ids.Read().View<int64_t>()[i];
        CHECK(id == next->OriginalEntityId(EntityKind::Cell, DNDS::index(i)));
        CHECK(history.Read().View<int64_t>()[i] == 3 * id + 11);
    }
    CHECK(next->HasNativeState());
    CheckTransitionCellRows(*next, expectedRows);
    if (source->NumNode() > 0)
    {
        tPoint coordinate = oldRead.HostView().Coord(0);
        next->TransformCoords([](tPoint x)
                              { x[0] += 1; return x; });
        CHECK((oldRead.HostView().Coord(0) - coordinate).norm() == 0);
        CHECK(oldStamp.Valid());
    }
    CHECK_THROWS(prepared.Publish());
}

TEST_CASE("Detached periodic transition retains geometry and relation companion rows")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi, true);
    MeshTransitionHandle handle(source);
    std::optional<MeshReadAccess> oldRead{source->AcquireRead(MeshReadRequirements::CellFlux())};
    auto oldStamp = source->versions.Stamp();
    auto expectedRows = TransitionCellRows(*source);
    R::Array<int64_t> cellBits(R::RowLayout::Dense(size_t(source->NumCell()), 1));
    {
        auto write = cellBits.Write();
        const auto &view = oldRead->HostView();
        for (DNDS::index cell = 0; cell < view.NumCell(); ++cell)
        {
            uint64_t hash = view.CellNodes(cell).size();
            for (auto value : view.CellNodePeriodicBits(cell))
                hash = hash * 257 + uint8_t(value);
            write.At(size_t(cell), 0) = int64_t(hash);
        }
    }
    auto input = TransitionInput(*source);
    SUBCASE("preserved periodic faces") {}
    SUBCASE("reconstructed periodic faces") { input.destroyKinds.insert(EntityKind::Face); }
    auto prepared = PreparedMeshTransition::Prepare(handle, input,
                                                    {RegisterArray(cellBits, EntityKind::Cell, "cell-periodic-bits")});
    prepared.Publish();
    auto next = handle.Get();
    CHECK(next->IsPeriodic());
    CHECK(oldStamp.Valid());
    CHECK(next->PeriodicMetadata().translation[1].map()[0] == source->PeriodicMetadata().translation[1].map()[0]);
    auto nextRead = next->AcquireRead(MeshReadRequirements::CellFlux());
    auto nextView = nextRead.HostView();
    for (DNDS::index i = 0; i < nextView.NumCell(); ++i)
    {
        CHECK(nextView.CellNodePeriodicBits(i).size() == nextView.CellNodes(i).size());
        CHECK(nextView.CellFacePeriodicBits(i).size() == nextView.CellFaces(i).size());
        uint64_t bits = uint64_t(nextView.CellNodePeriodicBits(i).size());
        for (auto value : nextView.CellNodePeriodicBits(i))
            bits = bits * 257 + uint8_t(value);
        CHECK(cellBits.Read().At(size_t(i), 0) == int64_t(bits));
    }
    {
        std::vector<size_t> contributions(size_t(nextView.NumCellProc()), 0);
        for (DNDS::index face = 0; face < nextView.NumFaceProc(); ++face)
        {
            const auto cells = nextView.FaceCells(face);
            for (auto cell : cells)
            {
                if (cell == UnInitIndex)
                    continue;
                REQUIRE(cell >= 0);
                REQUIRE(cell < nextView.NumCellProc());
                ++contributions[size_t(cell)];
                if (face < nextView.NumFace())
                {
                    const auto reverse = nextView.CellFaces(cell);
                    CHECK(std::find(reverse.begin(), reverse.end(), face) != reverse.end());
                }
            }
        }
        for (DNDS::index cell = 0; cell < nextView.NumCell(); ++cell)
        {
            CHECK(contributions[size_t(cell)] == nextView.CellFaces(cell).size());
            for (auto face : nextView.CellFaces(cell))
            {
                REQUIRE(face >= 0);
                REQUIRE(face < nextView.NumFaceProc());
                CHECK((nextView.FaceCells(face)[0] == cell || nextView.FaceCells(face)[1] == cell));
            }
        }
    }
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

TEST_CASE("Detached transition rejects unsupported inventory before preparing candidates")
{
    auto source = TransitionFixture(TransitionMPI());
    MeshTransitionHandle handle(source);
    auto stamp = source->versions.Stamp();
    auto input = TransitionInput(*source);
    input.destroyKinds.insert(EntityKind::Node);
    bool prepared = false;
    CHECK_THROWS(PreparedMeshTransition::Prepare(handle, input, {}, [&](const char *)
                                                 { prepared = true; }));
    CHECK_FALSE(prepared);
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

TEST_CASE("Native transition invalidates destination extraction row caches without editing source")
{
    auto source = TransitionFixture(TransitionMPI());
    source->node2bndNode.assign(size_t(source->NumNodeProc()), 17);
    const auto cache = source->node2bndNode;
    const auto stamp = source->versions.Stamp();
    MeshTransitionHandle handle(source);
    auto prepared = PreparedMeshTransition::Prepare(handle, {});
    CHECK_FALSE(prepared.PreparedOutput());
    prepared.Publish();
    const auto next = handle.Get();
    CHECK(next->HasNativeState());
    CHECK(next->node2bndNode.empty());
    CHECK(source->node2bndNode == cache);
    CHECK(stamp.Valid());
    CHECK_FALSE(source->versions.SameController(next->versions));
    auto old = source->AcquireRead(), current = next->AcquireRead();
    REQUIRE(old.HostView().NumNode() == current.HostView().NumNode());
    for (DNDS::index node = 0; node < old.HostView().NumNode(); ++node)
        CHECK(old.HostView().Coord(node).isApprox(current.HostView().Coord(node)));
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

TEST_CASE("Native output preparation and primary checkpoint preserve authoritative rows and wall policy")
{
    auto source = TransitionFixture(TransitionMPI());
    const auto nodes = source->Entities(EntityKind::Node);
    R::Array<DNDS::real> owned(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
    {
        auto write = owned.Write();
        for (DNDS::index node = 0; node < nodes.NumOwned(); ++node)
        {
            write.View()(size_t(node), 0) = DNDS::real(source->OriginalEntityId(EntityKind::Node, node));
            write.View()(size_t(node), 1) = 2;
            write.View()(size_t(node), 2) = 3;
        }
    }
    RelationGhostRoute route(nodes);
    auto ghost = route.TransferRows(nodes, owned, R::ElementSchema::Arithmetic<DNDS::real>());
    ghost = R::Array<DNDS::real>::FromStorage(R::RowLayout::Dense(ghost.Rows(), 3), ghost.Storage());
    source = UnstructuredMesh::FromNative(source->mpi, source->CurrentNativeState()->WithWallDistance({owned, ghost}));
    const auto requirements = source->CurrentNativeRequirements();
    auto retained = source->AcquireRead(requirements);
    const auto view = retained.HostView();
    const auto stamp = source->versions.Stamp();
    // Explicit test-only codec preserves the old isolated-reference purpose;
    // mutating its payload never installs a second production mesh authority.
    auto reference = TestFixtureCodec::DecodePrimaryReference(*source);
    reference->TransformCoords([](const tPoint &point)
                               { return tPoint(point + tPoint{1000, 0, 0}); });
    reference->ReorderLocalCells(2);
    MeshTransitionHandle handle(source);
    auto prepared = PreparedMeshTransition::Prepare(handle, {}, {}, {}, 1, TransitionCachePolicy::RebuildOutput);
    auto output = prepared.PreparedOutput();
    REQUIRE(output);
    CHECK(output->offsets.size() == size_t(view.NumCell() + 1));
    CHECK(output->globalCells == source->NumCellGlobal());
    CHECK(output->pointSourceRows.size() == output->coordinates.Rows());
    {
        auto coordinates = output->coordinates.Read();
        for (size_t row = 0; row < output->pointSourceRows.size(); ++row)
            for (size_t component = 0; component < 3; ++component)
                CHECK(coordinates.View()(row, component) == view.Coord(output->pointSourceRows[row])[Eigen::Index(component)]);
    }
    prepared.Publish();
    auto current = handle.Get()->AcquireRead(requirements);
    for (DNDS::index row = 0; row < view.NumNode(); ++row)
    {
        CHECK(current.HostView().Coord(row).isApprox(view.Coord(row)));
        CHECK(current.HostView().NodeWallDist(row).isApprox(view.NodeWallDist(row)));
    }
    auto checkpoint = make_ssp<Serializer::SerializerJSON>();
    checkpoint->CreatePath("/");
    CHECK_THROWS(WriteCurrentPrimaryMeshSnapshot(*source, checkpoint, "current"));
    CHECK(checkpoint->ListCurrentPath().empty());
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
    auto restored = ReadCompatibleNativeMeshArchive(source->mpi, checkpoint, "current");
    REQUIRE(restored->NumOwned(EntityKind::Node) == view.NumNode());
    for (DNDS::index row = 0; row < view.NumNode(); ++row)
        CHECK(restored->BorrowHostView().Coord(row).isApprox(view.Coord(row)));
    CHECK(source->HasNativeState());
    CHECK(stamp.Valid());
    // The retained descriptor remains usable independently of source owners.
    auto detached = output->coordinates.Read();
    CHECK(detached.View().Rows() == output->pointSourceRows.size());
}

TEST_CASE("Detached transition reconstructs 3D edges under an explicit cache policy")
{
    auto mpi = TransitionMPI();
    auto source = TransitionFixture(mpi, false, 3);
    DNDS::index globalEdges = source->CurrentNativeState()->NumGlobal(EntityKind::Edge);
    MeshTransitionHandle handle(source);
    auto stamp = source->versions.Stamp();
    auto input = TransitionInput(*source);
    input.destroyKinds.insert(EntityKind::Edge);
    auto prepared = PreparedMeshTransition::Prepare(handle, input, {}, {}, 1,
                                                    TransitionCachePolicy::RebuildOutput);
    prepared.Publish();
    auto next = handle.Get();
    CHECK(stamp.Valid());
    CHECK(next->HasNativeState());
    CHECK(next->CurrentNativeState()->NumGlobal(EntityKind::Edge) == globalEdges);
    REQUIRE(prepared.PreparedOutput());
    CHECK(prepared.PreparedOutput()->offsets.size() == size_t(next->NumCell() + 1));
    for (DNDS::index row = 0; row < next->Entities(EntityKind::Edge).NumOwned(); ++row)
        CHECK(next->OriginalEntityId(EntityKind::Edge, row) >= source->CurrentNativeState()->NextEntityId(EntityKind::Edge));
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
    const auto &sourceView = source->CurrentNativeState()->BorrowHostView();
    const auto edgeSpace = source->Entities(EntityKind::Edge);
    auto owned = size_t(edgeSpace.NumOwned());
    R::Field<int64_t> ids({owned, 1, 1}, {92, 7, owned, 0}, "edge");
    // Exact source edge rows travel beside the IDs, preserving endpoint and
    // incident-cell order plus every endpoint's periodic companion bits.
    std::vector<size_t> widths(owned);
    for (size_t edge = 0; edge < owned; ++edge)
        widths[edge] = size_t(sourceView.EdgeNodes(DNDS::index(edge)).size()) * 2 +
                       size_t(sourceView.EdgeCells(DNDS::index(edge)).size()) + 2;
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
            row.At(edge, slot++) = sourceView.EdgeNodes(e).size();
            for (DNDS::rowsize j = 0; j < sourceView.EdgeNodes(e).size(); ++j)
            {
                row.At(edge, slot++) = source->OriginalEntityId(EntityKind::Node, sourceView.EdgeNodes(e)[size_t(j)]);
                row.At(edge, slot++) = uint8_t(sourceView.EdgeNodePeriodicBits(e)[size_t(j)]);
            }
            for (auto cell : sourceView.EdgeCells(e))
                row.At(edge, slot++) = source->OriginalEntityId(EntityKind::Cell, cell);
        }
    }
    std::vector<size_t> cellWidths(size_t(source->NumCell()));
    for (DNDS::index cell = 0; cell < source->NumCell(); ++cell)
        cellWidths[size_t(cell)] = size_t(sourceView.CellEdges(cell).size()) * 2;
    R::Array<int64_t> exactCells(R::RowLayout::Ragged(cellWidths));
    {
        auto rows = exactCells.Write();
        for (DNDS::index cell = 0; cell < source->NumCell(); ++cell)
            for (DNDS::rowsize slot = 0; slot < sourceView.CellEdges(cell).size(); ++slot)
            {
                rows.At(size_t(cell), size_t(slot) * 2) = edgeSpace.LocalToGlobal(sourceView.CellEdges(cell)[size_t(slot)]);
                rows.At(size_t(cell), size_t(slot) * 2 + 1) = uint8_t(sourceView.CellEdgePeriodicBits(cell)[size_t(slot)]);
            }
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
    auto requirements = MeshReadRequirements::CellFlux();
    requirements.edges = true;
    auto read = next->AcquireRead(requirements);
    auto view = read.HostView();
    REQUIRE(ids.Layout().rows == size_t(view.NumEdge()));
    REQUIRE(exact.Rows() == ids.Layout().rows);
    auto rows = exact.Read();
    for (size_t edge = 0; edge < exact.Rows(); ++edge)
    {
        auto e = DNDS::index(edge);
        size_t slot = 0;
        CHECK(rows.At(edge, slot++) == edgeIds.View<int64_t>()[edge]);
        CHECK(rows.At(edge, slot++) == view.EdgeNodes(e).size());
        for (DNDS::rowsize j = 0; j < view.EdgeNodes(e).size(); ++j)
        {
            CHECK(rows.At(edge, slot++) == next->OriginalEntityId(EntityKind::Node, view.EdgeNodes(e)[size_t(j)]));
            CHECK(rows.At(edge, slot++) == uint8_t(view.EdgeNodePeriodicBits(e)[size_t(j)]));
        }
        for (auto cell : view.EdgeCells(e))
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
        REQUIRE(exactCells.Layout().RowSize(size_t(cell)) == size_t(view.CellEdges(cell).size()) * 2);
        for (DNDS::rowsize slot = 0; slot < view.CellEdges(cell).size(); ++slot)
        {
            auto edge = view.CellEdges(cell)[size_t(slot)];
            REQUIRE(edge >= 0);
            CHECK(oldIds.at(size_t(currentEdges.LocalToGlobal(edge))) == cellRows.At(size_t(cell), size_t(slot) * 2));
            CHECK(uint8_t(view.CellEdgePeriodicBits(cell)[size_t(slot)]) == cellRows.At(size_t(cell), size_t(slot) * 2 + 1));
        }
    }
    CHECK(view.NumEdge() == next->Entities(EntityKind::Edge).NumOwned());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
