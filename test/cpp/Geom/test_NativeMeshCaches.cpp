#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "LegacyMeshFixtureCodec.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "Solver/Direct.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    std::shared_ptr<const G::NativeMeshState> Strip(const DNDS::MPIInfo &mpi, bool emptyPeers = false)
    {
        const bool active = !emptyPeers || mpi.rank == 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? 8 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? 3 : 0);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        R::RowBuilder cones(size_t(cells.NumOwned()), sizeof(I));
        const I first = emptyPeers ? 0 : 8 * ((mpi.rank + 1) % mpi.size);
        for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
        {
            cones.ResizeRow(row, 4);
            const I values[]{first + I(row), first + I(row) + 1, first + I(row) + 5, first + I(row) + 4};
            std::copy(std::begin(values), std::end(values), cones.Row<I>(row).data);
        }
        R::Array<DNDS::real> coordinates(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3, 5));
        R::Array<G::ElemInfo> metadata(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
        {
            auto points = coordinates.Write();
            for (size_t row = 0; row < size_t(nodes.NumOwned()); ++row)
            {
                points.View()(row, 0) = DNDS::real(row % 4) + 5 * mpi.rank + DNDS::real(row) / 32;
                points.View()(row, 1) = DNDS::real(row / 4) + DNDS::real(row % 4) / 16;
                points.View()(row, 2) = 0.125 * mpi.rank + DNDS::real(row + 1) / 256;
                points.View().data[row * 5 + 3] = 1234;
                points.View().data[row * 5 + 4] = -5678;
            }
            auto info = metadata.Write();
            for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
                info.View()(row, 0) = G::ElemInfo{G::Elem::Quad4, 0};
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, R::Array<I>::FromSharedStorage(cones.Commit()),
                                 R::Array<I>(), coordinates, metadata, R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        return G::NativeMeshBuilder::Build(std::move(input), {1, true});
    }

    void CheckHash(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi)
    {
        auto native = G::UnstructuredMesh::FromNative(mpi, state);
        G::UnstructuredMesh legacy(mpi, 2);
        legacy.coords.InitPair("coordinate hash reference", mpi);
        auto read = native->AcquireRead();
        const auto view = read.HostView();
        legacy.coords.father->Resize(view.NumNode());
        legacy.coords.son->Resize(view.NumNodeGhost());
        for (I row = 0; row < view.NumNodeProc(); ++row)
            legacy.coords[row] = view.Coord(row);
        const auto expected = legacy.coords.hash();
        CHECK(native->CoordinateHash() == expected);
        CHECK(legacy.CoordinateHash() == expected);
        CHECK_FALSE(native->coords.father);
        auto retained = native->AcquireRead();
        CHECK(native->CoordinateHash() == expected);
    }
}

TEST_CASE("Coordinate compatibility hash matches legacy owned and ghost rows before and after redistribution")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto state = Strip(mpi);
    CheckHash(state, mpi);
    G::ReorderInput input;
    input.explicitMaps = {{G::EntityKind::Cell, std::vector<DNDS::MPI_int>(3, (mpi.rank + 1) % mpi.size)},
                          {G::EntityKind::Node, std::vector<DNDS::MPI_int>(8, (mpi.rank + 2) % mpi.size)}};
    auto moved = G::NativeMeshTransition::Build(*state, mpi, input);
    CheckHash(moved.state, mpi);
    CheckHash(Strip(mpi, true), mpi);
}

TEST_CASE("Native face graph and direct ordering match exact legacy face slots without legacy storage")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto primary = Strip(mpi);
    auto missingFaces = G::UnstructuredMesh::FromNative(mpi, primary);
    CHECK_THROWS(missingFaces->GetCell2CellFaceVLocal());
    auto state = G::NativeMeshTopology::Build(*primary, mpi);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto legacy = G::TestFixtureCodec::DecodeLegacy(*mesh);
    const G::tLocalMatStruct expected{{1}, {2, 0}, {1}};
    CHECK(mesh->GetCell2CellFaceVLocal() == expected);
    CHECK(mesh->GetCell2CellFaceVLocal(true) == expected);
    CHECK(legacy->GetCell2CellFaceVLocal() == expected);
    const auto cells = mesh->Entities(G::EntityKind::Cell);
    const auto stamp = mesh->versions.Stamp();
    auto retained = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
    for (int ordering : {0, -1, 1, 2, 3, 4})
    {
        CAPTURE(ordering);
        DNDS::Direct::DirectPrecControl control;
        control.useDirectPrec = true;
        control.orderingCode = ordering;
        DNDS::Direct::SerialSymLUStructure actual(mpi, 3), reference(mpi, 3);
        mesh->ObtainLocalFactFillOrdering(actual, control);
        legacy->ObtainLocalFactFillOrdering(reference, control);
        CHECK(actual.localFillOrderingNew2Old == reference.localFillOrderingNew2Old);
        CHECK(actual.localFillOrderingOld2New == reference.localFillOrderingOld2New);
        CHECK(mesh->LocalCellNeighbors() == expected);
        mesh->ObtainSymmetricSymbolicFactorization(actual, control);
        legacy->ObtainSymmetricSymbolicFactorization(reference, control);
        CHECK(actual.lowerTriStructure == reference.lowerTriStructure);
        CHECK(actual.upperTriStructure == reference.upperTriStructure);
        CHECK(actual.cell2cellFaceVLocal2FullRowPos == reference.cell2cellFaceVLocal2FullRowPos);
    }
    CHECK(stamp.Valid());
    CHECK(mesh->Entities(G::EntityKind::Cell).SameOrder(cells));
    CHECK(retained.HostView().NumCell() == 3);
    CHECK_FALSE(mesh->cell2node.father);
    CHECK(mesh->cell2cellFaceVLocalParts.empty());
    CHECK_THROWS(state->WithLocalCellSetup({0, 1, 3}, expected));
    CHECK_THROWS(state->WithLocalCellSetup({0, 3}, {{1}, {2, 0}}));
    CHECK_THROWS(state->WithLocalCellSetup({0, 3}, {{3}, {2, 0}, {1}}));
    CHECK_THROWS(state->WithLocalCellSetup({0, 3}, {{0}, {2, 0}, {1}}));

    // Native partition metadata preserves cross-part filtering without reading
    // detached legacy facade offsets. Earlier checks retain the legacy numeric
    // ordering/factorization reference.
    auto partitioned = G::UnstructuredMesh::FromNative(mpi, state->WithLocalPartitions({0, 1, 3}));
    const G::tLocalMatStruct filtered{{}, {2}, {1}};
    CHECK(partitioned->GetCell2CellFaceVLocal() == expected);
    CHECK(partitioned->GetCell2CellFaceVLocal(true) == filtered);
    DNDS::Direct::DirectPrecControl control;
    control.useDirectPrec = true;
    control.orderingCode = -1;
    DNDS::Direct::SerialSymLUStructure ordering(mpi, 3);
    partitioned->ObtainLocalFactFillOrdering(ordering, control);
    CHECK(ordering.localFillOrderingNew2Old == std::vector<I>{0, 2, 1});
    CHECK(partitioned->LocalCellNeighbors() == filtered);
    CHECK(partitioned->LocalPartStart(1) == 1);
    CHECK(partitioned->localPartitionStarts.empty());

    if (mpi.size > 1)
    {
        // Move each strip's middle cell away from both endpoints. Every real
        // face neighbor is now a ghost, despite three owned rows on each rank.
        G::ReorderInput input;
        input.explicitMaps = {{G::EntityKind::Cell, {mpi.rank, (mpi.rank + 1) % mpi.size, mpi.rank}},
                              {G::EntityKind::Node, std::vector<DNDS::MPI_int>(8, mpi.rank)}};
        auto moved = G::NativeMeshTransition::Build(*state, mpi, input);
        auto current = G::UnstructuredMesh::FromNative(mpi, moved.state);
        CHECK(current->NumCell() == 3);
        CHECK(current->NumCellGhost() > 0);
        CHECK(current->GetCell2CellFaceVLocal() == G::tLocalMatStruct(3));
        DNDS::Direct::SerialSymLUStructure structure(mpi, 3);
        control.orderingCode = 0;
        current->ObtainLocalFactFillOrdering(structure, control);
        current->ObtainSymmetricSymbolicFactorization(structure, control);
        CHECK(current->LocalCellNeighbors() == G::tLocalMatStruct(3));
    }
}

TEST_CASE("Native and legacy direct setup keep empty cell owners in RCM collectives")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto state = G::NativeMeshTopology::Build(*Strip(mpi, true), mpi);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto legacy = G::TestFixtureCodec::DecodeLegacy(*mesh);
    const G::tLocalMatStruct expected = mpi.rank == 0 ? G::tLocalMatStruct{{1}, {2, 0}, {1}} : G::tLocalMatStruct{};
    CHECK(mesh->GetCell2CellFaceVLocal() == expected);
    for (int code : {0, 3, 4})
    {
        DNDS::Direct::DirectPrecControl control;
        control.useDirectPrec = true;
        control.orderingCode = code;
        DNDS::Direct::SerialSymLUStructure structure(mpi, mesh->NumCell()), reference(mpi, mesh->NumCell());
        mesh->ObtainLocalFactFillOrdering(structure, control);
        legacy->ObtainLocalFactFillOrdering(reference, control);
        mesh->ObtainSymmetricSymbolicFactorization(structure, control);
        legacy->ObtainSymmetricSymbolicFactorization(reference, control);
        CHECK(structure.localFillOrderingNew2Old == reference.localFillOrderingNew2Old);
        CHECK(mesh->LocalCellNeighbors() == expected);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
