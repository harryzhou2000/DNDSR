#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

TEST_CASE("Native facial partition retains ID-derived fields geometry and sparse closure")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    auto source = fixture.mesh->CurrentNativeState();
    SUBCASE("many active ranks") {}
    SUBCASE("one active rank and empty peers")
    {
        G::ReorderInput gather;
        gather.explicitMaps.push_back({G::EntityKind::Cell, std::vector<DNDS::MPI_int>(size_t(source->NumOwned(G::EntityKind::Cell)), 0)});
        source = G::NativeMeshTransition::Build(*source, mpi, gather).state;
    }
    const auto namespaceBefore = source->MeshNamespace();
    R::Array<I> field(R::RowLayout::Dense(size_t(source->NumOwned(G::EntityKind::Cell)), 1));
    {
        auto write = field.Write();
        for (size_t row = 0; row < field.Rows(); ++row)
            write.View()(row, 0) = 3 * source->OriginalEntityId(G::EntityKind::Cell, I(row)) + 7;
    }
    G::EntityField<I> bound(source->Entities(G::EntityKind::Cell), field);
    auto attached = bound.Register("original-bound-field");
    auto alternate = G::NativeMeshPreparation::PrepareTopology(*source, mpi);
    CHECK_THROWS(attached.validateSourceSpace(alternate->Entities(G::EntityKind::Cell)));
    G::NativePartitionOptions options;
    options.metisNcuts = 2;
    options.metisSeed = 41;
    auto result = G::NativeMeshPreparation::Partition(*source, mpi, options);
    CHECK(result.state->MeshNamespace() == namespaceBefore);
    for (const auto &[kind, highWater] : source->IdHighWater())
        CHECK(result.state->IdHighWater().at(kind) == highWater);
    CHECK(result.sourceSpaces.at(G::EntityKind::Cell).SameOrder(source->Entities(G::EntityKind::Cell)));
    CHECK_NOTHROW(attached.validateSourceSpace(result.sourceSpaces.at(G::EntityKind::Cell)));
    attached.preflight(result.transfers.at(G::EntityKind::Cell));
    auto staged = attached.prepare(result.transfers.at(G::EntityKind::Cell), mpi);
    staged->BindEntitySpace(result.state->Entities(G::EntityKind::Cell));
    staged->Publish();
    auto boundRead = bound.Read(result.state->Entities(G::EntityKind::Cell));
    CHECK(result.transfers.count(G::EntityKind::Cell));
    auto route = G::NativeRowTransfer(result.transfers.at(G::EntityKind::Cell), mpi);
    R::Array<I> moved(route.DestinationLayout(field.Layout()));
    R::PayloadPlan(route, field.Layout(), moved.Layout(), R::ElementSchema::Arithmetic<I>())
        .Bind(field, moved, R::TransferTransport::Direct, {R::TransferMode::Transient})
        .Begin()
        .Wait();
    auto values = moved.Read();
    const auto view = result.state->BorrowHostView();
    DNDS::real totalArea = 0;
    for (I cell = 0; cell < view.NumCell(); ++cell)
    {
        CHECK(values.View()(size_t(cell), 0) == 3 * result.state->OriginalEntityId(G::EntityKind::Cell, cell) + 7);
        CHECK(boundRead.View()(size_t(cell), 0) == values.View()(size_t(cell), 0));
        CHECK(view.CellFaces(cell).size() == 4);
        DNDS::real area = 0;
        const auto cone = view.CellNodes(cell);
        for (size_t slot = 0; slot < cone.size(); ++slot)
        {
            const auto a = view.Coord(cone[slot]), b = view.Coord(cone[(slot + 1) % cone.size()]);
            area += a[0] * b[1] - a[1] * b[0];
        }
        CHECK(area / 2 == doctest::Approx(0.01));
        totalArea += area / 2;
        for (auto face : view.CellFaces(cell))
        {
            REQUIRE(face >= 0);
            for (auto neighbor : view.FaceCells(face))
                CHECK((neighbor == DNDS::UnInitIndex || (neighbor >= 0 && neighbor < view.NumCellProc())));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &totalArea, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(totalArea == doctest::Approx(1));
    I minimum = view.NumCell(), maximum = minimum;
    MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, DNDS::DNDS_MPI_INDEX, MPI_MIN, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, DNDS::DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
    CHECK(minimum > 0);
    CHECK(maximum <= 2 * ((100 + mpi.size - 1) / mpi.size));
    auto again = G::NativeMeshPreparation::PrepareTopology(*result.state, mpi);
    CHECK(again->NumGlobal(G::EntityKind::Face) == result.state->NumGlobal(G::EntityKind::Face));
    CHECK(again->MeshNamespace() == result.state->MeshNamespace());
    CHECK(again->IdHighWater() == result.state->IdHighWater());
}

TEST_CASE("Native primary partition adds facial IDs while retaining exact original field association")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    auto source = G::ReadNativeCGNS(mpi, (root / "data/mesh/UniformSquare_10.cgns").string(), 2,
                                    [](const std::string &) -> G::t_index
                                    { return 21; });
    CHECK_FALSE(source->HasEntities(G::EntityKind::Face));
    const auto original = source->Entities(G::EntityKind::Cell);
    R::Array<I> values(R::RowLayout::Dense(size_t(original.NumOwned()), 1));
    {
        auto write = values.Write();
        for (size_t row = 0; row < values.Rows(); ++row)
            write.View()(row, 0) = 7 * source->OriginalEntityId(G::EntityKind::Cell, I(row)) + 5;
    }
    G::EntityField<I> field(original, values);
    auto entry = field.Register("primary-field");
    auto result = G::NativeMeshPreparation::Partition(*source, mpi);
    CHECK(result.state->HasEntities(G::EntityKind::Face));
    CHECK(result.state->MeshNamespace() == source->MeshNamespace());
    for (const auto &[kind, cap] : source->IdHighWater())
        CHECK(result.state->IdHighWater().at(kind) == cap);
    CHECK(result.state->IdHighWater().count(G::EntityKind::Face));
    CHECK(result.sourceSpaces.at(G::EntityKind::Cell).SameOrder(original));
    entry.validateSourceSpace(result.sourceSpaces.at(G::EntityKind::Cell));
    entry.preflight(result.transfers.at(G::EntityKind::Cell));
    auto staged = entry.prepare(result.transfers.at(G::EntityKind::Cell), mpi);
    staged->BindEntitySpace(result.state->Entities(G::EntityKind::Cell));
    staged->Publish();
    auto read = field.Read(result.state->Entities(G::EntityKind::Cell));
    for (I row = 0; row < result.state->NumOwned(G::EntityKind::Cell); ++row)
        CHECK(read.View()(size_t(row), 0) == 7 * result.state->OriginalEntityId(G::EntityKind::Cell, row) + 5);
    CHECK_NOTHROW(source->Validate());
}

TEST_CASE("Native partition invalid options reject collectively without mutating source")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    const auto source = fixture.mesh->CurrentNativeState();
    for (int failure = 0; failure < 6; ++failure)
    {
        G::NativePartitionOptions options;
        if (mpi.rank == 0)
        {
            if (failure == 0)
                options.metisType = "UNKNOWN";
            if (failure == 1)
                options.edgeWeightMethod = 2;
            if (failure == 2)
                options.metisNcuts = 0;
            if (failure == 3)
                options.metisUfactor = 0;
            if (failure == 4)
                options.ghostLayers = 0;
            if (failure == 5)
                options.metisSeed = -1;
        }
        CHECK_THROWS(G::NativeMeshPreparation::Partition(*source, mpi, options));
        CHECK_NOTHROW(source->Validate());
    }
}

TEST_CASE("Native production preparation preserves facade identity rejects readers and writes serial output")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = G::OverlapTest::BuildNativeFixture(mpi).mesh;
    auto state = source->CurrentNativeState();
    const auto original = source->versions.Stamp();
    const auto name = state->MeshNamespace();
    std::map<G::EntityKind, I> counts;
    for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face})
        counts.emplace(kind, state->NumGlobal(kind));
    G::UnstructuredMeshSerialRW reader(source, mpi.size - 1);
    G::PrepareMeshOptions options;
    options.reorderCells = 1;
    options.reorderParts = 2;
    options.buildSerialOut = true;
    {
        auto retained = source->AcquireRead();
        CHECK_THROWS(G::PrepareMesh(*source, reader, options));
        CHECK(original.Valid());
    }
    G::PrepareMesh(*source, reader, options);
    CHECK_FALSE(original.Valid());
    CHECK(source->CurrentNativeState()->MeshNamespace() == name);
    REQUIRE(reader.nativeSerialOutput);
    CHECK_NOTHROW(reader.nativeSerialOutput->ValidateSource(*source->CurrentNativeState()));
    CHECK(source->HasNativeState());
    const auto now = source->CurrentNativeState();
    for (const auto &[kind, count] : counts)
        CHECK(now->NumGlobal(kind) == count);
    auto boundary = std::make_shared<G::UnstructuredMesh>(mpi, 1);
    G::UnstructuredMeshSerialRW boundaryReader(boundary, mpi.size - 1);
    G::BuildBndMesh(*source, *boundary, boundaryReader, true);
    CHECK(boundary->HasNativeState());
    REQUIRE(boundaryReader.nativeSerialOutput);
    CHECK(boundaryReader.nativeSerialOutput->dimension == 1);
    CHECK_NOTHROW(boundaryReader.nativeSerialOutput->ValidateSource(*boundary->CurrentNativeState()));
}

TEST_CASE("Partition preserves independent existing Face ownership and exact source associations for registered fields")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = G::OverlapTest::BuildNativeFixture(mpi).mesh->CurrentNativeState();
    G::ReorderInput independentFaces;
    independentFaces.explicitMaps.push_back({G::EntityKind::Face,
                                             std::vector<DNDS::MPI_int>(size_t(source->NumOwned(G::EntityKind::Face)), mpi.size - 1)});
    source = G::NativeMeshTransition::Build(*source, mpi, independentFaces).state;
    REQUIRE(source->NumOwned(G::EntityKind::Face) == (mpi.rank == mpi.size - 1 ? source->NumGlobal(G::EntityKind::Face) : 0));
    const auto name = source->MeshNamespace();
    const auto marks = source->IdHighWater();
    auto retained = source->Snapshot(R::Device::Host()).AcquireRead();
    auto records = [](const G::NativeMeshState &state)
    {
        const auto &view = state.BorrowHostView();
        std::vector<std::vector<I>> result;
        for (I row = 0; row < state.NumOwned(G::EntityKind::Face); ++row)
        {
            std::vector<I> values{state.OriginalEntityId(G::EntityKind::Face, row), view.GetFaceInfo(row).type,
                                  view.GetFaceInfo(row).zone, I(view.FaceNodes(row).size())};
            for (auto node : view.FaceNodes(row))
                values.push_back(state.OriginalEntityId(G::EntityKind::Node, node));
            for (auto cell : view.FaceCells(row))
                values.push_back(cell == DNDS::UnInitIndex ? cell : state.OriginalEntityId(G::EntityKind::Cell, cell));
            result.push_back(std::move(values));
        }
        return result;
    };
    const auto expectedFaces = records(*source);
    R::Array<I> faceValues(R::RowLayout::Dense(size_t(source->NumOwned(G::EntityKind::Face)), 1));
    R::Array<I> cellValues(R::RowLayout::Dense(size_t(source->NumOwned(G::EntityKind::Cell)), 1));
    {
        auto faces = faceValues.Write(), cells = cellValues.Write();
        for (size_t row = 0; row < faceValues.Rows(); ++row)
            faces.View()(row, 0) = 2 * source->OriginalEntityId(G::EntityKind::Face, I(row)) + 17;
        for (size_t row = 0; row < cellValues.Rows(); ++row)
            cells.View()(row, 0) = 3 * source->OriginalEntityId(G::EntityKind::Cell, I(row)) + 7;
    }
    G::EntityField<I> faceField(source->Entities(G::EntityKind::Face), faceValues);
    G::EntityField<I> cellField(source->Entities(G::EntityKind::Cell), cellValues);
    auto faceEntry = faceField.Register("independent-face-owned-order");
    auto cellEntry = cellField.Register("partitioned-cell-order");
    auto oldFaceRead = faceField.Read(source->Entities(G::EntityKind::Face));
    G::NativePartitionOptions options;
    options.metisNcuts = 2;
    options.metisSeed = 43;
    options.ghostLayers = 2;
    auto result = G::NativeMeshPreparation::Partition(*source, mpi, options);
    CHECK(result.state->MeshNamespace() == name);
    CHECK(result.state->IdHighWater() == marks);
    for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd, G::EntityKind::Face})
    {
        REQUIRE(result.sourceSpaces.count(kind));
        CHECK(result.sourceSpaces.at(kind).SameOrder(source->Entities(kind)));
    }
    CHECK_NOTHROW(faceEntry.validateSourceSpace(result.sourceSpaces.at(G::EntityKind::Face)));
    CHECK_NOTHROW(cellEntry.validateSourceSpace(result.sourceSpaces.at(G::EntityKind::Cell)));
    CHECK_THROWS(faceEntry.validateSourceSpace(result.state->Entities(G::EntityKind::Face)));
    CHECK(records(*result.state) == expectedFaces); // Ordered nodes AND physical parent slots, not a set/count comparison.
    CHECK(records(*source) == expectedFaces);
    for (size_t row = 0; row < faceValues.Rows(); ++row)
        CHECK(oldFaceRead.View()(row, 0) == 2 * result.state->OriginalEntityId(G::EntityKind::Face, I(row)) + 17);
    cellEntry.preflight(result.transfers.at(G::EntityKind::Cell));
    auto prepared = cellEntry.prepare(result.transfers.at(G::EntityKind::Cell), mpi);
    prepared->BindEntitySpace(result.state->Entities(G::EntityKind::Cell));
    prepared->Publish();
    auto moved = cellField.Read(result.state->Entities(G::EntityKind::Cell));
    for (I row = 0; row < result.state->NumOwned(G::EntityKind::Cell); ++row)
        CHECK(moved.View()(size_t(row), 0) == 3 * result.state->OriginalEntityId(G::EntityKind::Cell, row) + 7);
    CHECK_NOTHROW(retained.HostView());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
