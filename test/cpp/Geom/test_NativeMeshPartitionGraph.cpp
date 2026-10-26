#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshPartitionGraph.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    std::shared_ptr<const G::NativeMeshState> WeightedStrip(const DNDS::MPIInfo &mpi, bool readerOwned)
    {
        const bool active = !readerOwned || mpi.rank == mpi.size - 1;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? 8 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? 3 : 0);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        R::RowBuilder cones(size_t(cells.NumOwned()), sizeof(I));
        const I first = readerOwned ? 0 : 8 * mpi.rank;
        for (size_t cell = 0; cell < size_t(cells.NumOwned()); ++cell)
        {
            cones.ResizeRow(cell, 4);
            const I values[]{first + I(cell), first + I(cell) + 1, first + I(cell) + 5, first + I(cell) + 4};
            std::copy(std::begin(values), std::end(values), cones.Row<I>(cell).data);
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, R::Array<I>::FromSharedStorage(cones.Commit()), R::Array<I>(),
                                 R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        {
            auto points = input.coordinates.Write();
            const DNDS::real heights[]{1, 2, 4, 3};
            for (size_t row = 0; row < size_t(nodes.NumOwned()); ++row)
            {
                points.View()(row, 0) = 5 * mpi.rank + row % 4;
                points.View()(row, 1) = row < 4 ? 0 : heights[row % 4] * (mpi.rank + 1);
                points.View()(row, 2) = 0;
            }
            auto info = input.cellInfo.Write();
            for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
                info.View()(row, 0) = G::ElemInfo{G::Elem::Quad4, G::t_index(100 * mpi.rank + int(row))};
        }
        return G::NativeMeshBuilder::Build(std::move(input));
    }
}

TEST_CASE("Native partition uses exact global CSR and normalized nonuniform physical face diameters")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool readerOwned : {false, true})
    {
        auto source = WeightedStrip(mpi, readerOwned);
        auto prepared = G::NativeMeshTopology::Build(*source, mpi);
        const auto graph = G::detail::BuildNativePartitionGraph(*prepared, mpi, 1);
        const auto unweighted = G::detail::BuildNativePartitionGraph(*prepared, mpi, 0);
        CHECK(unweighted.offsets == graph.offsets);
        CHECK(unweighted.adjacency == graph.adjacency);
        CHECK(unweighted.weights.empty());
        const bool active = !readerOwned || mpi.rank == mpi.size - 1;
        if (active)
        {
            const idx_t first = readerOwned ? 0 : 3 * mpi.rank;
            // Shared edges have lengths 2*(rank+1), 4*(rank+1).
            // The maximum over all ranks is exactly 4*mpi.size.
            const I numerator = readerOwned ? 1 : mpi.rank + 1;
            const I denominator = readerOwned ? 1 : mpi.size;
            const I cap = std::min<I>(INT_MAX, (I(std::numeric_limits<idx_t>::max()) - 1) / (4 * (readerOwned ? 1 : mpi.size)));
            const idx_t half = idx_t((cap - 1) * numerator / (2 * denominator) + 1);
            const idx_t full = idx_t((cap - 1) * numerator / denominator + 1);
            CHECK(graph.offsets == std::vector<idx_t>{0, 1, 3, 4});
            CHECK(graph.adjacency == std::vector<idx_t>{first + 1, first, first + 2, first + 1});
            CHECK(graph.weights == std::vector<idx_t>{half, half, full, full});
        }
        else
        {
            CHECK(graph.offsets == std::vector<idx_t>{0});
            CHECK(graph.adjacency.empty());
            CHECK(graph.weights.empty());
        }
        G::NativePartitionOptions options;
        options.edgeWeightMethod = 1;
        options.metisNcuts = 2;
        if (readerOwned)
            options.metisType = "RB";
        auto result = G::NativeMeshPreparation::Partition(*source, mpi, options);
        CHECK(result.state->MeshNamespace() == source->MeshNamespace());
        CHECK(result.state->NumGlobal(G::EntityKind::Cell) == source->NumGlobal(G::EntityKind::Cell));
        CHECK(result.state->NumGlobal(G::EntityKind::Node) == source->NumGlobal(G::EntityKind::Node));
        for (const auto &[kind, highWater] : source->IdHighWater())
            CHECK(result.state->IdHighWater().at(kind) == highWater);
        CHECK(result.transfers.count(G::EntityKind::Face));
        const auto view = result.state->BorrowHostView();
        for (I cell = 0; cell < view.NumCell(); ++cell)
        {
            const auto original = result.state->OriginalEntityId(G::EntityKind::Cell, cell);
            const I owner = readerOwned ? mpi.size - 1 : original / 3;
            const I sourceCell = original % 3;
            CHECK(view.GetCellInfo(cell).zone == 100 * owner + sourceCell);
            const I first = readerOwned ? 0 : 8 * owner;
            const I expected[]{first + sourceCell, first + sourceCell + 1, first + sourceCell + 5, first + sourceCell + 4};
            for (size_t slot = 0; slot < 4; ++slot)
                CHECK(result.state->OriginalEntityId(G::EntityKind::Node, view.CellNodes(cell)[slot]) == expected[slot]);
        }
        if (readerOwned)
        {
            CHECK(result.state->NumGlobal(G::EntityKind::Face) == 10);
            std::array<I, 50> originalFaces{};
            if (mpi.rank == mpi.size - 1)
            {
                const auto originalView = prepared->BorrowHostView();
                REQUIRE(originalView.NumFace() == 10);
                for (I face = 0; face < 10; ++face)
                {
                    const auto id = prepared->OriginalEntityId(G::EntityKind::Face, face);
                    REQUIRE(id >= 0);
                    REQUIRE(id < 10);
                    auto row = originalFaces.data() + 5 * id;
                    row[0] = id;
                    for (size_t slot = 0; slot < 2; ++slot)
                    {
                        row[1 + slot] = prepared->OriginalEntityId(G::EntityKind::Node, originalView.FaceNodes(face)[slot]);
                        const auto cell = originalView.FaceCells(face)[slot];
                        row[3 + slot] = cell == DNDS::UnInitIndex ? cell : prepared->OriginalEntityId(G::EntityKind::Cell, cell);
                    }
                }
            }
            MPI_Bcast(originalFaces.data(), int(originalFaces.size()), DNDS::DNDS_MPI_INDEX, mpi.size - 1, mpi.comm);
            if (mpi.size > 1 && result.state->NumOwned(G::EntityKind::Cell) == 0)
                CHECK(result.state->NumOwned(G::EntityKind::Face) == 0);
            if (mpi.size > 1 && mpi.rank == mpi.size - 1)
                CHECK(result.state->NumOwned(G::EntityKind::Face) < 10);
            for (I face = 0; face < view.NumFace(); ++face)
            {
                const auto id = result.state->OriginalEntityId(G::EntityKind::Face, face);
                REQUIRE(id >= 0);
                REQUIRE(id < 10);
                const auto original = originalFaces.data() + 5 * id;
                CHECK(original[0] == id);
                const auto support = view.FaceCells(face);
                I minimumCellOwner = mpi.size;
                for (I cell : support)
                    if (cell != DNDS::UnInitIndex)
                    {
                        REQUIRE(cell >= 0);
                        const auto cells = result.state->Entities(G::EntityKind::Cell);
                        minimumCellOwner = std::min<I>(minimumCellOwner, cells.Owner(cells.LocalToGlobal(cell)));
                    }
                CHECK(minimumCellOwner == mpi.rank);
                std::vector<I> nodes;
                for (size_t slot = 0; slot < 2; ++slot)
                {
                    const auto node = result.state->OriginalEntityId(G::EntityKind::Node, view.FaceNodes(face)[slot]);
                    nodes.push_back(node);
                    CHECK(node == original[1 + slot]);
                    const auto cell = support[slot];
                    CHECK((cell == DNDS::UnInitIndex ? cell : result.state->OriginalEntityId(G::EntityKind::Cell, cell)) == original[3 + slot]);
                }
                std::sort(nodes.begin(), nodes.end());
                REQUIRE(nodes.size() == 2);
                const I a = nodes[0], b = nodes[1];
                CHECK(((a < 4 && b == a + 1) || (a >= 4 && b == a + 1) || b == a + 4));
            }
        }
        if (!readerOwned && mpi.size > 1)
        {
            options.metisType = "RB";
            CHECK_THROWS(G::NativeMeshPreparation::Partition(*source, mpi, options));
            options.metisType = "KWAY";
            CHECK_NOTHROW(G::NativeMeshPreparation::Partition(*source, mpi, options));
        }
        CHECK_THROWS(G::detail::BuildNativePartitionGraph(*prepared, mpi, mpi.rank == 0 ? 2 : 1));
        CHECK_NOTHROW(G::detail::BuildNativePartitionGraph(*prepared, mpi, 1));
        CHECK_NOTHROW(source->Validate());
    }
}

TEST_CASE("Several physical faces of one cell pair contribute their maximum diameter")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == 0 ? 13 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 5 : 0);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    R::RowBuilder cones(size_t(cells.NumOwned()), sizeof(I));
    const I inputCones[5][4]{{0, 1, 2, 3}, {1, 0, 4, 2}, {5, 6, 10, 9}, {6, 7, 11, 10}, {7, 8, 12, 11}};
    for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
    {
        cones.ResizeRow(row, 4);
        std::copy(std::begin(inputCones[row]), std::end(inputCones[row]), cones.Row<I>(row).data);
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, R::Array<I>::FromSharedStorage(cones.Commit()), R::Array<I>(),
                             R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
    {
        auto points = input.coordinates.Write();
        const DNDS::real coordinates[13][3]{{0, 0, 0}, {1, 0, 0}, {1, 2, 0}, {0, 2, 0}, {2, -1, 0}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}, {13, 0, 0}, {10, 1, 0}, {11, 2, 0}, {12, 4, 0}, {13, 3, 0}};
        for (size_t row = 0; row < size_t(nodes.NumOwned()); ++row)
            for (size_t j = 0; j < 3; ++j)
                points.View()(row, j) = coordinates[row][j];
        auto info = input.cellInfo.Write();
        for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
            info.View()(row, 0) = G::ElemInfo{G::Elem::Quad4, 0};
    }
    auto source = G::NativeMeshTopology::Build(*G::NativeMeshBuilder::Build(std::move(input)), mpi);
    auto graph = G::detail::BuildNativePartitionGraph(*source, mpi, 1);
    if (mpi.rank == 0)
    {
        const idx_t cap = idx_t(std::min<I>(INT_MAX, (I(std::numeric_limits<idx_t>::max()) - 1) / 6));
        const idx_t half = idx_t((I(cap) - 1) / 2 + 1);
        CHECK(graph.offsets == std::vector<idx_t>{0, 1, 2, 3, 5, 6});
        CHECK(graph.adjacency == std::vector<idx_t>{1, 0, 3, 2, 4, 3});
        CHECK(graph.weights == std::vector<idx_t>{half, half, half, half, cap, cap});
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
