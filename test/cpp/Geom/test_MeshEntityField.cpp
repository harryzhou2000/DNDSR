#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeMeshOverlapFixture.hpp"
#include "Geom/Mesh/NativeMeshState.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include <filesystem>

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;

static MPIInfo EntityFieldMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

// Field ownership fixture: independent valid native line cells. Explicit cell
// halo row zero is retained on every nonowner, including zero-owned ranks.
static ssp<UnstructuredMesh> EntityFieldSpaces(const MPIInfo &mpi, DNDS::index owned = -1)
{
    const DNDS::index count = owned >= 0 ? owned : (mpi.rank == 0 ? 3 : (mpi.rank % 2 ? 0 : 1));
    const auto nodeBase = EntitySpace::CreateOwned(EntityKind::Node, mpi, 2 * count);
    auto cellSpace = nodeBase.CreateSibling(EntityKind::Cell, count);
    const auto boundaries = nodeBase.CreateSibling(EntityKind::Bnd, 0);
    if (mpi.rank != 0)
        cellSpace = cellSpace.WithGhosts({0});
    else
        cellSpace = cellSpace.WithGhosts({});
    const auto nodes = nodeBase.WithGhosts(mpi.rank == 0 ? std::vector<DNDS::index>{} : std::vector<DNDS::index>{0, 1});
    auto rows = [](const EntitySpace &space, size_t width, auto fill)
    {
        R::ArrayPair<DNDS::index> result{
            R::Array<DNDS::index>(R::RowLayout::Dense(size_t(space.NumOwned()), width)),
            R::Array<DNDS::index>(R::RowLayout::Dense(size_t(space.NumGhost()), width))};
        auto local = result.owned.Write(), ghost = result.ghost.Write();
        for (DNDS::index row = 0; row < space.NumLocal(); ++row)
            for (size_t slot = 0; slot < width; ++slot)
                (row < space.NumOwned() ? local.View()(size_t(row), slot) : ghost.View()(size_t(row - space.NumOwned()), slot)) = fill(row, slot);
        return result;
    };
    NativeMeshData data;
    data.dimension = 1;
    data.relations[size_t(MeshRelation::CellNodes)] = rows(cellSpace, 2, [&](DNDS::index row, size_t slot)
                                                           {
        const auto global = 2 * cellSpace.LocalToGlobal(row) + DNDS::index(slot);
        const auto local = nodes.GlobalToLocal(global);
        DNDS_check_throw_info(local, "field fixture cell cone lacks its node");
        return *local; });
    data.relations[size_t(MeshRelation::CellCells)] = rows(cellSpace, 0, [](auto, auto)
                                                           { return DNDS::UnInitIndex; });
    data.relations[size_t(MeshRelation::BoundaryNodes)] = rows(boundaries, 0, [](auto, auto)
                                                               { return DNDS::UnInitIndex; });
    data.relations[size_t(MeshRelation::BoundaryCells)] = rows(boundaries, 2, [](auto, auto)
                                                               { return DNDS::UnInitIndex; });
    data.coords = R::ArrayPair<real>{R::Array<real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                     R::Array<real>(R::RowLayout::Dense(size_t(nodes.NumGhost()), 3))};
    {
        auto local = data.coords->owned.Write(), ghost = data.coords->ghost.Write();
        for (DNDS::index row = 0; row < nodes.NumLocal(); ++row)
            for (size_t component = 0; component < 3; ++component)
                (row < nodes.NumOwned() ? local.View()(size_t(row), component) : ghost.View()(size_t(row - nodes.NumOwned()), component)) =
                    component == 0 ? double(nodes.LocalToGlobal(row)) : 0;
    }
    data.cellInfo = R::ArrayPair<ElemInfo>{R::Array<ElemInfo>(R::RowLayout::Dense(size_t(cellSpace.NumOwned()), 1)),
                                           R::Array<ElemInfo>(R::RowLayout::Dense(size_t(cellSpace.NumGhost()), 1))};
    {
        auto local = data.cellInfo->owned.Write(), ghost = data.cellInfo->ghost.Write();
        for (DNDS::index row = 0; row < cellSpace.NumLocal(); ++row)
            (row < cellSpace.NumOwned() ? local.View()(size_t(row), 0) : ghost.View()(size_t(row - cellSpace.NumOwned()), 0)) = ElemInfo{Elem::Line2, INTERNAL_ZONE};
    }
    data.bndInfo = R::ArrayPair<ElemInfo>{R::Array<ElemInfo>(R::RowLayout::Dense(0, 1)), R::Array<ElemInfo>(R::RowLayout::Dense(0, 1))};
    std::map<EntityKind, EntitySpace> spaces{{EntityKind::Cell, cellSpace}, {EntityKind::Node, nodes}, {EntityKind::Bnd, boundaries}};
    std::map<EntityKind, R::Array<DNDS::index>> ids;
    for (const auto &[kind, space] : spaces)
    {
        R::Array<DNDS::index> values(R::RowLayout::Dense(size_t(space.NumLocal()), 1));
        {
            auto write = values.Write();
            for (DNDS::index row = 0; row < space.NumLocal(); ++row)
                write.View()(size_t(row), 0) = space.LocalToGlobal(row);
        }
        ids.emplace(kind, std::move(values));
    }
    return UnstructuredMesh::FromNative(mpi, NativeMeshState::FromNative(mpi, std::move(data), std::move(spaces), std::move(ids), MeshReadRequirements::Primary()));
}

TEST_CASE("Entity-bound fields reject wrong domain/kind and retain field/mesh leases")
{
    auto mpi = EntityFieldMPI();
    auto mesh = EntityFieldSpaces(mpi);
    auto other = EntityFieldSpaces(mpi);
    auto space = mesh->Entities(EntityKind::Cell);
    R::Array<double> values(R::RowLayout::Dense(size_t(space.NumOwned()), 2));
    EntityField<double> field(space, std::move(values));
    {
        auto write = field.Write(space);
        auto view = write.View();
        for (DNDS::index row = 0; row < space.NumOwned(); ++row)
            for (size_t j = 0; j < 2; ++j)
                view(size_t(row), j) = 10 * space.LocalToGlobal(row) + double(j);
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Layout));
        write.Complete(R::Completion::Record(R::ExecutionContext::Host()));
        CHECK_THROWS(write.View());
    }
    CHECK_THROWS(field.Read(other->Entities(EntityKind::Cell)));
    CHECK_THROWS(field.Read(mesh->Entities(EntityKind::Node)));
    std::vector<DNDS::index> ghostGlobals;
    for (DNDS::index row = space.NumOwned(); row < space.NumLocal(); ++row)
        ghostGlobals.push_back(space.LocalToGlobal(row));
    auto wrongKind = space.CreateSibling(EntityKind::Node, space.NumOwned()).WithGhosts(ghostGlobals);
    CHECK(wrongKind.NumLocal() == space.NumLocal());
    CHECK(wrongKind.SameDomain(space) == false);
    CHECK_THROWS(field.Read(wrongKind));
    CHECK_THROWS(field.Read(space, {R::Device::CUDA(0), nullptr}));
    auto route = field.BuildHaloRoute(space);
    auto binding = route.Bind(field, space);
    {
        auto operation = binding.Begin();
        operation.Wait();
    }
    {
        auto ghosts = binding.GhostValues().Read();
        for (DNDS::index row = 0; row < space.NumGhost(); ++row)
            for (size_t j = 0; j < 2; ++j)
                CHECK(ghosts.View()(size_t(row), j) == 10 * space.LocalToGlobal(space.NumOwned() + row) + double(j));
    }
    {
        auto read = field.Read(space);
        auto retainedSpace = read.Space();
        auto moved = std::move(read);
        CHECK_THROWS(read.View());
        CHECK_THROWS(read.Space());
        moved.Validate(retainedSpace);
        moved.Complete(R::Completion::Record(R::ExecutionContext::Host()));
        CHECK_THROWS(moved.View());
    }
    auto movedField = std::move(field);
    CHECK_THROWS(field.Read(space));
    CHECK_THROWS(field.Write(space));
    CHECK_THROWS(field.Register("moved"));
    auto movedBinding = std::move(binding);
    CHECK_THROWS(binding.Begin());
    {
        auto operation = movedBinding.Begin();
        operation.Wait();
    }
    auto descriptor = movedField.Register("bound-consuming-rejected");
    PermutationTransfer consuming;
    consuming.isLocalOnly = true;
    CHECK_THROWS(descriptor.fn(consuming, mpi));
    CHECK_NOTHROW(movedField.Validate(space));
}

static ssp<UnstructuredMesh> EntityFieldMesh(const MPIInfo &mpi)
{
    return OverlapTest::BuildNativeFixture(mpi).mesh;
}

TEST_CASE("Detached publication binds active and inactive fields to final native destination and rejects old routes")
{
    auto mpi = EntityFieldMPI();
    auto source = EntityFieldMesh(mpi);
    auto space = source->Entities(EntityKind::Cell), faceSpace = source->Entities(EntityKind::Face);
    R::Array<double> array(R::RowLayout::Dense(size_t(space.NumOwned()), 4));
    EntityField<double> field(space, std::move(array));
    EntityField<double> faceField(faceSpace, R::Array<double>(R::RowLayout::Dense(size_t(faceSpace.NumOwned()), 1)));
    // Registration must reach Prepare's collective failure agreement even when
    // exactly one rank supplies a stale, equal-shape foreign association.
    auto foreign = EntityFieldSpaces(mpi, space.NumOwned());
    auto foreignSpace = foreign->Entities(EntityKind::Cell);
    EntityField<double> wrong(foreignSpace, R::Array<double>(R::RowLayout::Dense(size_t(space.NumOwned()), 4)));
    if (mpi.rank == 0)
    {
        auto mutation = foreign->versions.BeginMutation(R::Change::Layout);
        mutation.Commit();
    }
    MeshTransitionHandle rejectedHandle(source);
    auto bad = mpi.rank == 0 ? wrong.Register("history") : field.Register("history");
    CHECK_THROWS(PreparedMeshTransition::Prepare(rejectedHandle, ReorderInput{}, {bad}));
    CHECK(rejectedHandle.Get() == source);
    {
        auto write = field.Write(space);
        for (DNDS::index row = 0; row < space.NumOwned(); ++row)
            for (size_t j = 0; j < 4; ++j)
                write.View()(size_t(row), j) = 100 * source->OriginalEntityId(EntityKind::Cell, row) + double(j);
    }
    auto sourceRead = source->AcquireRead(MeshReadRequirements::CellFlux());
    auto oldRead = field.Read(space);
    auto oldRoute = field.BuildHaloRoute(space);
    auto oldBinding = oldRoute.Bind(field, space);
    auto stamp = source->versions.Stamp();
    MeshTransitionHandle handle(source);
    ReorderInput input;
    input.explicitMaps.push_back({EntityKind::Cell, std::vector<MPI_int>(size_t(space.NumOwned()), (mpi.rank + 1) % mpi.size)});
    auto prepared = PreparedMeshTransition::Prepare(handle, input, {field.Register("history"), faceField.Register("unchanged-face-values")});
    CHECK_THROWS(faceField.Write(faceSpace)); // Retained preparation read, even for inactive fields.
    prepared.Publish();
    auto next = handle.Get();
    CHECK(next->HasNativeState());
    auto current = next->Entities(EntityKind::Cell);
    CHECK(stamp.Valid());
    oldRead.Validate(space);
    CHECK_THROWS(oldRead.Validate(current));
    CHECK_THROWS(field.Read(space));
    CHECK_THROWS(oldBinding.Begin());
    CHECK_THROWS(oldRoute.Bind(field, current));
    faceField.Validate(next->Entities(EntityKind::Face));
    CHECK_THROWS(faceField.Validate(faceSpace));
    {
        auto read = field.Read(current);
        for (DNDS::index row = 0; row < current.NumOwned(); ++row)
            for (size_t j = 0; j < 4; ++j)
                CHECK(read.View()(size_t(row), j) == 100 * next->OriginalEntityId(EntityKind::Cell, row) + double(j));
        for (DNDS::index row = 0; row < space.NumOwned(); ++row)
            for (size_t j = 0; j < 4; ++j)
                CHECK(oldRead.View()(size_t(row), j) == 100 * source->OriginalEntityId(EntityKind::Cell, row) + double(j));
    }
    auto route = field.BuildHaloRoute(current);
    auto binding = route.Bind(field, current);
    {
        auto operation = binding.Begin();
        operation.Wait();
    }
    auto ghost = binding.GhostValues().Read();
    for (DNDS::index row = 0; row < current.NumGhost(); ++row)
        for (size_t j = 0; j < 4; ++j)
            CHECK(ghost.View()(size_t(row), j) == 100 * next->OriginalEntityId(EntityKind::Cell, current.NumOwned() + row) + double(j));
}

TEST_CASE("Entity field collective setup rejects rank-local stale/shape/schema and transfers explicit POD layouts")
{
    auto mpi = EntityFieldMPI();
    auto mesh = EntityFieldSpaces(mpi);
    auto space = mesh->Entities(EntityKind::Cell);
    CHECK_THROWS(FieldTransferDetail::LocalStage(mpi, [&]
                                                 { EntityField<double> bad(space, R::Array<double>(R::RowLayout::Dense(size_t(space.NumOwned()) + (mpi.rank == 0), 1))); }));
    CHECK_THROWS(FieldTransferDetail::LocalStage(mpi, [&]
                                                 {
        auto schema = mpi.rank == 0 ? R::ElementSchema(MPI_DOUBLE, sizeof(double), "wrong.type", R::ElementSchema::TypeIdentity<int64_t>()) :
                                      R::ElementSchema::Arithmetic<double>();
        EntityField<double> bad(space, R::Array<double>(R::RowLayout::Dense(size_t(space.NumOwned()), 1)), schema); }));
    struct Metadata
    {
        int32_t tag;
        double weight;
    };
    MPI_Datatype raw, pod;
    int blocks[] = {1, 1};
    MPI_Aint offsets[] = {offsetof(Metadata, tag), offsetof(Metadata, weight)};
    MPI_Datatype types[] = {MPI_INT32_T, MPI_DOUBLE};
    MPI_Type_create_struct(2, blocks, offsets, types, &raw);
    MPI_Type_create_resized(raw, 0, sizeof(Metadata), &pod);
    MPI_Type_commit(&pod);
    R::ElementSchema schema(pod, sizeof(Metadata), "entity.metadata.v1", R::ElementSchema::TypeIdentity<Metadata>());
    MPI_Type_free(&raw);
    MPI_Type_free(&pod);
    EntityField<Metadata> field(space, R::Array<Metadata>(R::RowLayout::Dense(size_t(space.NumOwned()), 2)), schema);
    {
        auto write = field.Write(space);
        for (DNDS::index row = 0; row < space.NumOwned(); ++row)
            for (size_t j = 0; j < 2; ++j)
                write.View()(size_t(row), j) = {int32_t(space.LocalToGlobal(row)), 0.25 + double(j)};
    }
    auto route = field.BuildHaloRoute(space);
    auto binding = route.Bind(field, space);
    {
        auto operation = binding.Begin();
        operation.Wait();
    }
    {
        auto ghost = binding.GhostValues().Read();
        for (DNDS::index row = 0; row < space.NumGhost(); ++row)
            for (size_t j = 0; j < 2; ++j)
            {
                CHECK(ghost.View()(size_t(row), j).tag == space.LocalToGlobal(space.NumOwned() + row));
                CHECK(ghost.View()(size_t(row), j).weight == 0.25 + double(j));
            }
    }
    if (mpi.size > 1)
    {
        auto otherSchema = R::ElementSchema(schema.Type(), sizeof(Metadata), mpi.rank == 0 ? "entity.metadata.v2" : "entity.metadata.v1",
                                            R::ElementSchema::TypeIdentity<Metadata>());
        EntityField<Metadata> wrong(space, R::Array<Metadata>(R::RowLayout::Dense(size_t(space.NumOwned()), 2)), otherSchema);
        CHECK_THROWS(route.Bind(wrong, space));
    }
    if (mpi.rank == 0)
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Layout);
        mutation.Commit();
    }
    CHECK_THROWS(field.BuildHaloRoute(space));
    CHECK_THROWS(binding.Begin());
    auto descriptor = field.Register("stale-descriptor");
    CHECK_THROWS(FieldTransferDetail::LocalStage(mpi, [&]
                                                 { descriptor.attachment(); }));
}

TEST_CASE("Entity CSR fields preserve variable row shape values and source leases across empty owners")
{
    const auto mpi = EntityFieldMPI();
    auto mesh = EntityFieldSpaces(mpi);
    const auto space = mesh->Entities(EntityKind::Cell);
    CHECK(space.NumOwned() == (mpi.rank == 0 ? 3 : (mpi.rank % 2 ? 0 : 1)));
    CHECK(space.NumGhost() == (mpi.rank == 0 ? 0 : 1));
    if (mpi.rank != 0)
        CHECK(space.LocalToGlobal(space.NumOwned()) == 0);
    for (size_t extra : {size_t(0), size_t(2)})
    {
        R::RowBuilder rows(size_t(space.NumOwned()), sizeof(double));
        for (DNDS::index row = 0; row < space.NumOwned(); ++row)
        {
            const auto global = space.LocalToGlobal(row);
            rows.ResizeRow(size_t(row), size_t(global % 3) + extra);
            auto output = rows.Row<double>(size_t(row));
            for (size_t slot = 0; slot < output.size; ++slot)
                output[slot] = 10 * double(global) + double(slot) / 8;
        }
        EntityField<double> field(space, R::Array<double>::FromSharedStorage(rows.Commit()));
        auto route = field.BuildHaloRoute(space);
        auto binding = route.Bind(field, space);
        {
            auto operation = binding.Begin();
            operation.Wait();
        }
        auto ghost = binding.GhostValues().Read();
        for (DNDS::index row = 0; row < space.NumGhost(); ++row)
        {
            const auto global = space.LocalToGlobal(space.NumOwned() + row);
            REQUIRE(ghost.View().RowSize(size_t(row)) == size_t(global % 3) + extra);
            for (size_t slot = 0; slot < ghost.View().RowSize(size_t(row)); ++slot)
                CHECK(ghost.View()(size_t(row), slot) == 10 * double(global) + double(slot) / 8);
        }
        auto retained = field.Read(space);
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Layout));
        for (DNDS::index row = 0; row < space.NumOwned(); ++row)
            CHECK(retained.View().RowSize(size_t(row)) == size_t(space.LocalToGlobal(row) % 3) + extra);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
