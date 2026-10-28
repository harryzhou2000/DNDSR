#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativeMeshRefinement.hpp"
#include "Geom/Mesh/NativeMeshArchive.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
namespace
{
    template <class T>
    R::Array<T> Rows(const std::vector<std::vector<T>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(T));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            for (size_t slot = 0; slot < rows[row].size(); ++slot)
                builder.Row<T>(row)[slot] = rows[row][slot];
        }
        return R::Array<T>::FromSharedStorage(builder.Commit());
    }
    std::shared_ptr<const G::NativeMeshState> Cube(const DNDS::MPIInfo &mpi, bool derived, bool periodic = false, bool complete = true)
    {
        const auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? (periodic ? 4 : 8) : 0);
        const auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 1 : 0);
        const auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, mpi.rank == mpi.size / 2 ? 6 : 0);
        const std::vector<I> cell = periodic ? std::vector<I>{0, 0, 1, 1, 2, 2, 3, 3} : std::vector<I>{0, 1, 2, 3, 4, 5, 6, 7};
        const std::vector<G::NodePeriodicBits> images{{0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}};
        std::vector<std::vector<G::NodePeriodicBits>> boundaryImages;
        std::vector<std::vector<I>> faces;
        G::Elem::Element hex{G::Elem::Hex8};
        R::Array<DNDS::real> coords(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
        R::Array<G::ElemInfo> info(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), bndInfo(R::RowLayout::Dense(size_t(boundaries.NumOwned()), 1));
        {
            auto write = coords.Write();
            const int xyz[8][3]{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
            for (size_t node = 0; node < coords.Rows(); ++node)
                for (size_t component = 0; component < 3; ++component)
                    write.View()(node, component) = periodic ? (component == 0 ? 0 : component == 1 ? int(node % 2)
                                                                                                    : int(node / 2))
                                                             : xyz[node][component];
            auto metadata = info.Write();
            if (cells.NumOwned())
                metadata.View()(0, 0) = {G::Elem::Hex8, 0};
            auto bnd = bndInfo.Write();
            for (I face = 0; face < boundaries.NumOwned(); ++face)
            {
                faces.emplace_back(4);
                hex.ExtractFaceNodes(DNDS::Geom::t_index(face), cell, faces.back());
                G::t_index zone = G::t_index(20 + face);
                if (periodic)
                {
                    boundaryImages.emplace_back(4);
                    hex.ExtractFaceNodes(G::t_index(face), images, boundaryImages.back());
                    const auto &bits = boundaryImages.back();
                    if (std::all_of(bits.begin(), bits.end(), [](auto bit)
                                    { return uint8_t(bit) == 0; }))
                        zone = G::BC_ID_PERIODIC_1;
                    if (std::all_of(bits.begin(), bits.end(), [](auto bit)
                                    { return uint8_t(bit) == 1; }))
                        zone = G::BC_ID_PERIODIC_1_DONOR;
                }
                bnd.View()(size_t(face), 0) = {G::Elem::Quad4, zone};
            }
        }
        G::NativeMeshInput input{mpi, 3, cells, nodes, boundaries, Rows(cells.NumOwned() ? std::vector<std::vector<I>>{cell} : std::vector<std::vector<I>>{}), Rows(faces), coords, info, bndInfo, {}};
        if (periodic)
        {
            input.periodic = true;
            input.cellNodesPbi = Rows(cells.NumOwned() ? std::vector<std::vector<G::NodePeriodicBits>>{images} : std::vector<std::vector<G::NodePeriodicBits>>{});
            input.boundaryNodesPbi = Rows(boundaryImages);
        }
        auto source = G::ConstructNativeMesh(std::move(input), {1, true});
        if (derived)
            source = G::NativeMeshRefinement::Elevate(*source, mpi, {1, true}).state;
        return complete ? G::PrepareNativeMeshTopology(*source, mpi, true, true) : source;
    }
    void CheckOutput(const G::NativeMeshOutput &out, const DNDS::MPIInfo &mpi)
    {
        CHECK(out.globalCells == 1);
        CHECK(out.globalNodes == 8);
        CHECK(out.globalConnectivity == 8);
        CHECK(out.offsets.size() == out.types.size() + 1);
        CHECK(out.connectivity.size() == 8 * out.types.size());
        const I localCells = I(out.types.size());
        std::vector<I> cellCounts(size_t(mpi.size));
        MPI_Allgather(&localCells, 1, DNDS::DNDS_MPI_INDEX, cellCounts.data(), 1, DNDS::DNDS_MPI_INDEX, mpi.comm);
        I precedingCells = 0, totalCells = 0;
        for (int rank = 0; rank < mpi.size; ++rank)
        {
            if (rank < mpi.rank)
                precedingCells += cellCounts[size_t(rank)];
            totalCells += cellCounts[size_t(rank)];
        }
        CHECK(totalCells == 1);
        CHECK(out.cellOffset == precedingCells);
        // VTKHDF offsets address the concatenated global connectivity buffer,
        // including the prefix retained by ranks with no owned cells.
        for (size_t cell = 0; cell < out.offsets.size(); ++cell)
            CHECK(out.offsets[cell] == 8 * (precedingCells + I(cell)));
        auto coordinates = out.coordinates.Read();
        std::vector<DNDS::real> local;
        for (size_t node = 0; node < out.coordinates.Rows(); ++node)
            for (size_t component = 0; component < 3; ++component)
                local.push_back(coordinates.View()(node, component));
        int count = int(local.size());
        std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size) + 1);
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
        for (int rank = 0; rank < mpi.size; ++rank)
            offsets[size_t(rank) + 1] = offsets[size_t(rank)] + counts[size_t(rank)];
        CHECK(offsets.back() == 24);
        CHECK(out.nodeOffset == offsets[size_t(mpi.rank)] / 3);
        std::vector<DNDS::real> all(size_t(offsets.back()));
        MPI_Allgatherv(local.data(), count, MPI_DOUBLE, all.data(), counts.data(), offsets.data(), MPI_DOUBLE, mpi.comm);
        const int xyz[8][3]{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        for (size_t cell = 0; cell < out.types.size(); ++cell)
        {
            CHECK(out.types[cell] == 12); // VTK_HEXAHEDRON
            for (size_t slot = 0; slot < 8; ++slot)
            {
                const auto global = out.connectivity[8 * cell + slot];
                REQUIRE(global >= 0);
                REQUIRE(global < 8);
                for (size_t component = 0; component < 3; ++component)
                    CHECK(all[size_t(global) * 3 + component] == xyz[slot][component]);
            }
        }
    }
}
TEST_CASE("Native-only repeated face/edge reconstruction reserves IDs and stages exact retained output")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = Cube(mpi, false);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, source);
    std::optional<G::MeshReadAccess> old{mesh->AcquireRead(source->Requirements())};
    const auto stamp = mesh->versions.Stamp();
    std::shared_ptr<const G::NativeMeshOutput> detached;
    for (int pass = 0; pass < 3; ++pass)
    {
        G::ReorderInput input;
        input.explicitMaps.push_back({G::EntityKind::Cell, std::vector<DNDS::MPI_int>(size_t(mesh->NumCell()), (mpi.rank + 1) % mpi.size)});
        if (pass != 1)
            input.destroyKinds.insert(G::EntityKind::Face);
        if (pass != 0)
            input.destroyKinds.insert(G::EntityKind::Edge);
        const auto before = mesh->CurrentNativeState();
        const auto faceNext = before->NextEntityId(G::EntityKind::Face), edgeNext = before->NextEntityId(G::EntityKind::Edge);
        G::MeshTransitionHandle handle(mesh);
        auto prepared = G::PreparedMeshTransition::Prepare(handle, input, {}, {}, 1, G::TransitionCachePolicy::RebuildOutput);
        detached = prepared.PreparedOutput();
        REQUIRE(detached);
        CheckOutput(*detached, mpi);
        CHECK(handle.Get() == mesh);
        prepared.Publish();
        mesh = handle.Get();
        const auto after = mesh->CurrentNativeState();
        CHECK(after->MeshNamespace() == source->MeshNamespace());
        for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
            CHECK(after->NextEntityId(kind) == before->NextEntityId(kind));
        CHECK(after->NumGlobal(G::EntityKind::Face) == 6);
        CHECK(after->NumGlobal(G::EntityKind::Edge) == 12);
        CHECK(after->NextEntityId(G::EntityKind::Face) == faceNext + (input.destroyKinds.count(G::EntityKind::Face) ? 6 : 0));
        CHECK(after->NextEntityId(G::EntityKind::Edge) == edgeNext + (input.destroyKinds.count(G::EntityKind::Edge) ? 12 : 0));
        for (auto kind : input.destroyKinds)
            for (I row = 0; row < after->NumOwned(kind); ++row)
                CHECK(after->OriginalEntityId(kind, row) >= before->NextEntityId(kind));
    }
    CHECK(stamp.Valid());
    CHECK(old->HostView().NumCell() == source->NumOwned(G::EntityKind::Cell));
    old.reset();
    mesh.reset();
    source.reset();
    CheckOutput(*detached, mpi);
}
TEST_CASE("Native transition reconstruction retains surviving derived ancestry through archive")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = Cube(mpi, true);
    REQUIRE(source->Provenance());
    const auto ancestor = source->Provenance()->sourceNamespace;
    auto input = G::ReorderInput{};
    input.destroyKinds = {G::EntityKind::Face, G::EntityKind::Edge};
    auto result = G::NativeMeshTransition::Build(*source, mpi, input);
    REQUIRE(result.state->Provenance());
    CHECK(result.state->Provenance()->sourceNamespace == ancestor);
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
    {
        auto before = source->Provenance()->sourceIds.at(kind).Read();
        auto after = result.state->Provenance()->sourceIds.at(kind).Read();
        REQUIRE(before.View().Rows() == after.View().Rows());
        for (size_t row = 0; row < before.View().Rows(); ++row)
        {
            REQUIRE(before.View()[row].size() == after.View()[row].size());
            for (size_t slot = 0; slot < before.View()[row].size(); ++slot)
                CHECK(before.View()(row, slot) == after.View()(row, slot));
        }
    }
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->CreatePath("/");
    G::WriteNativeMeshArchive(*result.state, mpi, serializer, "mesh");
    source.reset();
    result.state.reset();
    auto restart = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    REQUIRE(restart->Provenance());
    CHECK(restart->Provenance()->sourceNamespace == ancestor);
    CHECK(restart->NumGlobal(G::EntityKind::Node) == 27);
}
TEST_CASE("Face-only periodic reconstruction preserves retained Edge IDs and every ordered frame slot")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto source = Cube(mpi, false, true);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, source);
    auto retained = mesh->AcquireRead(source->Requirements());
    const auto before = retained.HostView();
    std::array<I, 24> expected{};
    if (before.NumCell())
        for (size_t slot = 0; slot < 12; ++slot)
        {
            expected[2 * slot] = source->OriginalEntityId(G::EntityKind::Edge, before.CellEdges(0)[slot]);
            expected[2 * slot + 1] = uint8_t(before.CellEdgePeriodicBits(0)[slot]);
        }
    MPI_Bcast(expected.data(), 24, DNDS::DNDS_MPI_INDEX, 0, mpi.comm);
    const auto edgeNext = source->NextEntityId(G::EntityKind::Edge), faceNext = source->NextEntityId(G::EntityKind::Face);
    G::ReorderInput input;
    input.destroyKinds.insert(G::EntityKind::Face);
    input.explicitMaps.push_back({G::EntityKind::Cell, std::vector<DNDS::MPI_int>(size_t(before.NumCell()), (mpi.rank + 1) % mpi.size)});
    G::MeshTransitionHandle handle(mesh);
    auto prepared = G::PreparedMeshTransition::Prepare(handle, input, {}, {}, 1, G::TransitionCachePolicy::RebuildOutput);
    REQUIRE(prepared.PreparedOutput());
    CheckOutput(*prepared.PreparedOutput(), mpi);
    prepared.Publish();
    auto result = handle.Get()->CurrentNativeState();
    const auto &after = result->BorrowHostView();
    CHECK(result->NextEntityId(G::EntityKind::Edge) == edgeNext);
    CHECK(result->NextEntityId(G::EntityKind::Face) == faceNext + 5);
    REQUIRE(after.NumEdge() == before.NumEdge());
    for (I edge = 0; edge < before.NumEdge(); ++edge)
    {
        CHECK(result->OriginalEntityId(G::EntityKind::Edge, edge) == source->OriginalEntityId(G::EntityKind::Edge, edge));
        REQUIRE(after.EdgeNodes(edge).size() == before.EdgeNodes(edge).size());
        for (size_t slot = 0; slot < before.EdgeNodes(edge).size(); ++slot)
        {
            CHECK(result->OriginalEntityId(G::EntityKind::Node, after.EdgeNodes(edge)[slot]) == source->OriginalEntityId(G::EntityKind::Node, before.EdgeNodes(edge)[slot]));
            CHECK(uint8_t(after.EdgeNodePeriodicBits(edge)[slot]) == uint8_t(before.EdgeNodePeriodicBits(edge)[slot]));
        }
        REQUIRE(after.EdgeCells(edge).size() == before.EdgeCells(edge).size());
        for (size_t slot = 0; slot < before.EdgeCells(edge).size(); ++slot)
            CHECK(result->OriginalEntityId(G::EntityKind::Cell, after.EdgeCells(edge)[slot]) == source->OriginalEntityId(G::EntityKind::Cell, before.EdgeCells(edge)[slot]));
    }
    for (I cell = 0; cell < after.NumCell(); ++cell)
        for (size_t slot = 0; slot < 12; ++slot)
        {
            CHECK(result->OriginalEntityId(G::EntityKind::Edge, after.CellEdges(cell)[slot]) == expected[2 * slot]);
            CHECK(uint8_t(after.CellEdgePeriodicBits(cell)[slot]) == expected[2 * slot + 1]);
        }
    CHECK(retained.HostView().NumCell() == before.NumCell());
}
TEST_CASE("Generated native ID floors agree collectively and never modify retained family high-water marks")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto primary = Cube(mpi, false, false, false);
    const auto oldNodes = primary->Entities(G::EntityKind::Node);
    CHECK_THROWS(G::PrepareNativeMeshTopology(*primary, mpi, true, false, 1, {{G::EntityKind::Node, 10}}));
    CHECK_THROWS(G::PrepareNativeMeshTopology(*primary, mpi, true, false, 1, {{G::EntityKind::Edge, 10}}));
    CHECK_THROWS(G::PrepareNativeMeshTopology(*primary, mpi, true, false, 1, {{G::EntityKind::Face, std::numeric_limits<I>::max()}}));
    if (mpi.size > 1)
        CHECK_THROWS(G::PrepareNativeMeshTopology(*primary, mpi, true, false, 1, {{G::EntityKind::Face, 100 + mpi.rank}}));
    auto large = G::PrepareNativeMeshTopology(*primary, mpi, true, true, 1, {{G::EntityKind::Face, 1000000000000LL}, {G::EntityKind::Edge, 2000000000000LL}});
    CHECK(large->NextEntityId(G::EntityKind::Face) == 1000000000006LL);
    CHECK(large->NextEntityId(G::EntityKind::Edge) == 2000000000012LL);
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
        CHECK(large->NextEntityId(kind) == primary->NextEntityId(kind));
    CHECK_THROWS(G::PrepareNativeMeshTopology(*large, mpi, true, false, 1, {{G::EntityKind::Face, 0}}));
    auto retry = G::PrepareNativeMeshTopology(*primary, mpi, true, true);
    CHECK(retry->NextEntityId(G::EntityKind::Face) == 6);
    CHECK(retry->NextEntityId(G::EntityKind::Edge) == 12);
    CHECK(oldNodes.Valid());
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
