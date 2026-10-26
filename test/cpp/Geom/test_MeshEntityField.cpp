#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
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

static ssp<UnstructuredMesh> EntityFieldSpaces(const MPIInfo &mpi, DNDS::index owned = -1)
{
    auto mesh = make_ssp<UnstructuredMesh>(mpi, 2);
    const DNDS::index count = owned >= 0 ? owned : (mpi.rank == 0 ? 3 : (mpi.rank % 2 ? 0 : 1));
    mesh->coords.InitPair("bound.nodes", mpi);
    mesh->coords.father->Resize(count);
    mesh->coords.TransAttach();
    mesh->coords.trans.createFatherGlobalMapping();
    auto offset = (*mesh->coords.father->pLGlobalMapping)(mpi.rank, 0);
    for (DNDS::index row = 0; row < count; ++row)
        mesh->coords[row] = tPoint{double(offset + row), 0, 0};
    std::vector<DNDS::index> ghosts;
    if (mpi.rank != 0)
        ghosts.push_back(0);
    mesh->coords.trans.createGhostMapping(ghosts);
    mesh->coords.trans.createMPITypes();
    mesh->coords.trans.pullOnce();
    mesh->cell2node.InitPair("bound.cells", mpi);
    mesh->cell2node.father->Resize(count);
    for (DNDS::index row = 0; row < count; ++row)
    {
        mesh->cell2node.ResizeRow(row, 1);
        mesh->cell2node(row, 0) = offset + row;
    }
    mesh->cell2node.father->Compress();
    mesh->cell2node.TransAttach();
    mesh->cell2node.trans.createFatherGlobalMapping();
    mesh->cell2node.trans.createGhostMapping(ghosts);
    mesh->cell2node.trans.createMPITypes();
    mesh->cell2node.trans.pullOnce();
    mesh->cell2node.idx.markGlobal();
    mesh->adjPrimaryState = Adj_PointToGlobal;
    return mesh;
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
    ReorderInput input;
    input.explicitMaps.push_back({EntityKind::Cell, std::vector<MPI_int>(size_t(space.NumOwned()), mpi.rank)});
    CHECK_THROWS(mesh->ReorderEntitiesWithFields(input, {movedField.Register("bound-consuming-rejected")}));
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
