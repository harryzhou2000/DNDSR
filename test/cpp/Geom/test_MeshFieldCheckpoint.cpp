#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/FieldTransfer.hpp"
#include "Geom/Mesh/NativeMeshLegacyArchive.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "DNDS/Runtime/ArrayIO.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <filesystem>
#include <map>
#include <unistd.h>

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;

static ssp<UnstructuredMesh> CheckpointMesh(const MPIInfo &mpi)
{
    return OverlapTest::BuildNativeFixture(mpi).mesh;
}

static double HistoryValue(DNDS::index cellId, size_t history, size_t component)
{
    return 10000 * double(cellId + 1) + 100 * double(history) + 3 * double(component) + 0.25;
}

TEST_CASE("Retained mesh and multi-history field form a coherent matching-partition H5 restart")
{
    MPIInfo mpi;
    mpi.setWorld();
    auto source = CheckpointMesh(mpi);
    constexpr size_t histories = 4, components = 3, width = histories * components;
    R::Array<double, R::FixedRows<width>> field(R::RowLayout::Dense(size_t(source->NumCell()), width));
    {
        auto values = field.Write();
        for (DNDS::index cell = 0; cell < source->NumCell(); ++cell)
            for (size_t history = 0; history < histories; ++history)
                for (size_t component = 0; component < components; ++component)
                    values.At(size_t(cell), history * components + component) =
                        HistoryValue(source->OriginalEntityId(EntityKind::Cell, cell), history, component);
    }

    // Coordinate oracle predates redistribution/checkpoint and is keyed by
    // stable original node ID, independently of either rank-local ordering.
    std::vector<DNDS::index> localNodeIds(size_t(source->NumNode()));
    std::vector<double> localCoords(size_t(source->NumNode()) * 3);
    for (DNDS::index node = 0; node < source->NumNode(); ++node)
    {
        localNodeIds[size_t(node)] = source->OriginalEntityId(EntityKind::Node, node);
        for (size_t component = 0; component < 3; ++component)
            localCoords[size_t(node) * 3 + component] = source->CurrentNativeState()->BorrowHostView().Coord(node)[int(component)];
    }
    int localCount = int(localNodeIds.size());
    std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size + 1), 0);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
    for (int rank = 0; rank < mpi.size; ++rank)
        offsets[size_t(rank + 1)] = offsets[size_t(rank)] + counts[size_t(rank)];
    std::vector<DNDS::index> globalNodeIds(size_t(offsets.back()));
    MPI_Allgatherv(localNodeIds.data(), localCount, DNDS_MPI_INDEX, globalNodeIds.data(), counts.data(), offsets.data(), DNDS_MPI_INDEX, mpi.comm);
    for (int rank = 0; rank < mpi.size; ++rank)
    {
        counts[size_t(rank)] *= 3;
        offsets[size_t(rank)] *= 3;
    }
    std::vector<double> globalCoords(globalNodeIds.size() * 3);
    MPI_Allgatherv(localCoords.data(), localCount * 3, MPI_DOUBLE, globalCoords.data(), counts.data(), offsets.data(), MPI_DOUBLE, mpi.comm);
    std::map<DNDS::index, tPoint> coordinates;
    for (size_t node = 0; node < globalNodeIds.size(); ++node)
        coordinates.emplace(globalNodeIds[node], tPoint{globalCoords[node * 3], globalCoords[node * 3 + 1], globalCoords[node * 3 + 2]});

    MeshTransitionHandle handle(source);
    ReorderInput input;
    input.explicitMaps.push_back({EntityKind::Cell,
                                  std::vector<MPI_int>(size_t(source->NumCell()), (mpi.rank + 1) % mpi.size)});
    auto prepared = PreparedMeshTransition::Prepare(handle, input,
                                                    {RegisterArray(field, EntityKind::Cell, "flow-four-histories")});
    prepared.Publish();
    auto mesh = handle.Get();
    FieldTransferDetail::LocalStage(mpi, [&]
                                    { DNDS_check_throw_info(mesh != source, "checkpoint transition did not publish a detached mesh"); });
    auto meshStamp = mesh->versions.Stamp();
    const auto fieldIdentity = field.Identity(), fieldVersion = field.Version();

    int rootPid = int(getpid());
    MPI_Bcast(&rootPid, 1, MPI_INT, 0, mpi.comm);
    const auto filename = (std::filesystem::temp_directory_path() /
                           ("dndsr-mesh-field-checkpoint-" + std::to_string(rootPid) + "-np" + std::to_string(mpi.size) + ".h5"))
                              .string();
    auto serializer = make_ssp<Serializer::SerializerH5>(mpi);
    serializer->SetCollectiveRW(true, true);
    constexpr const char *schema = "dndsr.checkpoint.flow.f64.history4.component3.v1";
    {
        // This application-level scope stabilizes the matching mesh and field
        // through separate codec writes; a file-system transaction is not implied.
        auto meshRead = mesh->AcquireRead(MeshReadRequirements::CellFlux());
        auto fieldRead = field.Read();
        CHECK_THROWS(mesh->TransformCoords([](tPoint point)
                                           { return point; }));
        CHECK_THROWS(field.Write());
        serializer->OpenFile(filename, false);
        serializer->WriteString("application_schema", "dndsr.mesh-and-flow-history.v1");
        serializer->WriteString("field_association", "cell");
        serializer->WriteIndex("field_components", components);
        serializer->WriteIndex("field_histories", histories);
        WriteCurrentPrimaryMeshSnapshot(*mesh, serializer, "mesh");
        R::WriteArray(field, serializer, "flow_histories", mpi, schema);
        serializer->CloseFile();
        CHECK(fieldRead.View().Rows() == size_t(meshRead.HostView().NumCell()));
    }
    CHECK(meshStamp.Valid());
    CHECK(field.Identity() == fieldIdentity);
    CHECK(field.Version() == fieldVersion);

    ssp<UnstructuredMesh> restored;
    serializer->OpenFile(filename, true);
    std::string applicationSchema, association;
    DNDS::index savedComponents = 0, savedHistories = 0;
    serializer->ReadString("application_schema", applicationSchema);
    serializer->ReadString("field_association", association);
    serializer->ReadIndex("field_components", savedComponents);
    serializer->ReadIndex("field_histories", savedHistories);
    restored = UnstructuredMesh::FromNative(mpi, ReadCompatibleNativeMeshArchive(mpi, serializer, "mesh"));
    auto restartedField = R::ReadArray<double, R::FixedRows<width>>(
        serializer, "flow_histories", mpi, Serializer::ArrayGlobalOffset_Unknown, R::Device::Host(), schema);
    serializer->CloseFile();
    CHECK(applicationSchema == "dndsr.mesh-and-flow-history.v1");
    CHECK(association == "cell");
    CHECK(savedComponents == DNDS::index(components));
    CHECK(savedHistories == DNDS::index(histories));
    FieldTransferDetail::LocalStage(mpi, [&]
                                    {
        DNDS_check_throw_info(restartedField.Rows() == size_t(restored->NumCell()) &&
                                  restored->NumCell() == mesh->NumCell() && restored->NumNode() == mesh->NumNode(),
                              "checkpoint mesh and field owned partitions differ");
        auto values = restartedField.Read();
        for (DNDS::index cell = 0; cell < restored->NumCell(); ++cell)
        {
            auto id = restored->OriginalEntityId(EntityKind::Cell, cell);
            CHECK(id == mesh->OriginalEntityId(EntityKind::Cell, cell));
            for (size_t history = 0; history < histories; ++history)
                for (size_t component = 0; component < components; ++component)
                    CHECK(values.At(size_t(cell), history * components + component) == HistoryValue(id, history, component));
        }
        for (DNDS::index node = 0; node < restored->NumNode(); ++node)
            CHECK(restored->CurrentNativeState()->BorrowHostView().Coord(node).isApprox(coordinates.at(restored->OriginalEntityId(EntityKind::Node, node)))); });
    restored->PrepareNativeTopology();
    auto read = restored->AcquireRead(MeshReadRequirements::CellFlux());
    CHECK(read.HostView().NumCell() == DNDS::index(restartedField.Rows()));
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        std::error_code error;
        std::filesystem::remove(filename, error);
        CHECK_FALSE(bool(error));
    }
    MPI_Barrier(mpi.comm);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
