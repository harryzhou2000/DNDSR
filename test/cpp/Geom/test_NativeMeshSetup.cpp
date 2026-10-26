#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "NativeMeshOverlapFixture.hpp"
#include <filesystem>
#include <set>
#include <unistd.h>
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

TEST_CASE("Native setup replacement stages identity and invalidates prior associations")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = G::OverlapTest::BuildNativeFixture(mpi).mesh;
    auto destination = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 2);
    auto input = source->CurrentNativeState();
    const auto inputCells = source->Entities(G::EntityKind::Cell);
    destination->ReplaceNativeForSetup(input);
    CHECK(destination->HasNativeState());
    CHECK(destination->CurrentNativeState()->MeshNamespace() == input->MeshNamespace());
    CHECK(destination->Entities(G::EntityKind::Node).SameDomain(source->Entities(G::EntityKind::Node)));
    auto old = destination->Entities(G::EntityKind::Cell);
    R::Array<I> values(R::RowLayout::Dense(size_t(old.NumOwned()), 1));
    {
        auto write = values.Write();
        for (I row = 0; row < old.NumOwned(); ++row)
            write.View()(size_t(row), 0) = old.LocalToGlobal(row);
    }
    G::EntityField<I> field(old, values);
    auto route = field.BuildHaloRoute(old);
    auto binding = route.Bind(field, old);
    auto detached = G::NativeMeshSnapshot::Capture(*destination, G::MeshReadRequirements::CellFlux());
    std::optional<G::MeshReadAccess> held;
    if (mpi.rank == 0)
        held.emplace(destination->AcquireRead(G::MeshReadRequirements::CellFlux()));
    CHECK_THROWS(destination->ReplaceNativeForSetup(input));
    CHECK(old.Valid());
    CHECK_NOTHROW(input->Validate());
    held.reset();
    auto alternate = G::OverlapTest::BuildNativeFixture(mpi).mesh;
    destination->ReplaceNativeForSetup(alternate->CurrentNativeState());
    CHECK_FALSE(old.Valid());
    CHECK_THROWS(old.LocalToGlobal(0));
    CHECK_THROWS(binding.Begin());
    CHECK_NOTHROW(input->Validate());
    CHECK(inputCells.Valid());
    CHECK_NOTHROW(detached.AcquireRead());
    CHECK(destination->CurrentNativeState()->MeshNamespace() == alternate->CurrentNativeState()->MeshNamespace());
    source.reset();
    input.reset();
    alternate.reset();
    CHECK_NOTHROW(destination->AcquireRead(G::MeshReadRequirements::CellFlux()));
}

TEST_CASE("Native setup rejects null malformed association and preserves an existing publication on failure")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    G::UnstructuredMesh destination(mpi, 2);
    CHECK_THROWS(destination.ReplaceNativeForSetup(mpi.rank == 0 ? nullptr : fixture.mesh->CurrentNativeState()));
    CHECK_FALSE(destination.HasNativeState());
    G::UnstructuredMesh wrongDimension(mpi, 3);
    CHECK_THROWS(wrongDimension.ReplaceNativeForSetup(fixture.mesh->CurrentNativeState()));
    CHECK_FALSE(wrongDimension.HasNativeState());
    auto populated = G::UnstructuredMesh::FromNative(mpi, fixture.mesh->CurrentNativeState());
    const auto previous = populated->CurrentNativeState();
    const auto stamp = populated->versions.Stamp();
    const auto point = populated->Coord(0);
    CHECK_THROWS(populated->ReplaceNativeForSetup(mpi.rank == 0 ? nullptr : fixture.mesh->CurrentNativeState()));
    CHECK(populated->CurrentNativeState() == previous);
    CHECK(stamp.Valid());
    CHECK(populated->Coord(0) == point);
    CHECK_NOTHROW(populated->ReplaceNativeForSetup(fixture.mesh->CurrentNativeState()));
    CHECK_FALSE(stamp.Valid());
    if (mpi.size > 1)
    {
        DNDS::MPIInfo self(MPI_COMM_SELF);
        auto wrong = G::OverlapTest::BuildNativeFixture(self);
        CHECK_THROWS(destination.ReplaceNativeForSetup(wrong.mesh->CurrentNativeState()));
    }
    CHECK_NOTHROW(destination.ReplaceNativeForSetup(fixture.mesh->CurrentNativeState()));
    CHECK(destination.HasNativeState());
}

TEST_CASE("Archive helper repartition retains requested second ghost ring")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    DNDS::Serializer::SerializerFactory factory("H5");
    std::string base = (std::filesystem::temp_directory_path() / ("native-setup-closure-" + std::to_string(getpid()))).string();
    int count = mpi.rank == 0 ? int(base.size()) : 0;
    MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
    base.resize(size_t(count));
    MPI_Bcast(base.data(), count, MPI_CHAR, 0, mpi.comm);
    G::SerializeMesh(*fixture.mesh, base, factory);
    G::PartitionOptions partition;
    partition.metisSeed = 13;
    auto second = G::ReadNativeMeshFromArchive(mpi, factory, base, true, partition, 2);
    auto owned = G::NativeMeshAssembler::CaptureOwned(*second->CurrentNativeState(), mpi);
    auto first = G::UnstructuredMesh::FromNative(mpi, G::NativeMeshAssembler::Build(mpi, std::move(owned), 1));
    REQUIRE(first->NumCell() == second->NumCell());
    for (I row = 0; row < second->NumCell(); ++row)
        CHECK(first->OriginalEntityId(G::EntityKind::Cell, row) == second->OriginalEntityId(G::EntityKind::Cell, row));
    auto ids = [](const G::UnstructuredMesh &mesh)
    {
        std::set<I> result;
        for (I row = 0; row < mesh.NumCellProc(); ++row)
            result.insert(mesh.OriginalEntityId(G::EntityKind::Cell, row));
        return result;
    };
    const auto one = ids(*first), two = ids(*second);
    int extra = 0;
    for (I row = 0; row < second->NumCell(); ++row)
    {
        const I id = second->OriginalEntityId(G::EntityKind::Cell, row), x = id % 10, y = id / 10;
        for (I dx = -2; dx <= 2; ++dx)
            for (I dy = -2; dy <= 2; ++dy)
                if (std::abs(dx) + std::abs(dy) <= 2 && x + dx >= 0 && x + dx < 10 && y + dy >= 0 && y + dy < 10)
                {
                    const I required = (y + dy) * 10 + x + dx;
                    CHECK(two.count(required));
                    extra += !one.count(required);
                }
    }
    MPI_Allreduce(MPI_IN_PLACE, &extra, 1, MPI_INT, MPI_SUM, mpi.comm);
    if (mpi.size > 1)
        CHECK(extra > 0);
    const auto file = std::get<0>(factory.ModifyFilePath(base, mpi, "part_%d", true));
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(file);
}

TEST_CASE("Native helper guards reject before publishing or creating archive files")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    G::UnstructuredMesh empty(mpi, 2);
    DNDS::Serializer::SerializerFactory factory("JSON");
    std::string base = (std::filesystem::temp_directory_path() / ("native-rejected-archive-" + std::to_string(getpid()))).string();
    int count = mpi.rank == 0 ? int(base.size()) : 0;
    MPI_Bcast(&count, 1, MPI_INT, 0, mpi.comm);
    base.resize(size_t(count));
    MPI_Bcast(base.data(), count, MPI_CHAR, 0, mpi.comm);
    CHECK_THROWS(G::SerializeMesh(empty, base, factory));
    CHECK_FALSE(std::filesystem::exists(base));
    CHECK_FALSE(std::filesystem::exists(base + ".dir"));
    auto source = G::OverlapTest::BuildNativeFixture(mpi).mesh;
    auto boundary = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 1);
    auto reader = DNDS::make_ssp<G::UnstructuredMeshSerialRW>(boundary, 0);
    auto wrong = DNDS::make_ssp<G::UnstructuredMeshSerialRW>(source, 0);
    CHECK_THROWS(G::BuildBndMesh(*source, *boundary, mpi.rank == 0 ? *wrong : *reader));
    CHECK_FALSE(boundary->HasNativeState());
    CHECK_NOTHROW(G::BuildBndMesh(*source, *boundary, *reader));
    CHECK(boundary->HasNativeState());
    CHECK(reader->dataIsSerialOut);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
