#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshSerialOutput.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    std::shared_ptr<const G::NativeMeshState> Strip(const DNDS::MPIInfo &mpi, bool emptyPeers, bool periodic = false)
    {
        const bool active = !emptyPeers || mpi.rank == 0;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, active ? 3 : 0);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, active ? 1 : 0);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
        R::RowBuilder cones(size_t(cells.NumOwned()), sizeof(I));
        if (active)
        {
            cones.ResizeRow(0, 3);
            const I first = emptyPeers ? 0 : 3 * ((mpi.rank + 1) % mpi.size);
            for (size_t slot = 0; slot < 3; ++slot)
                cones.Row<I>(0).data[slot] = first + I(slot);
        }
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, R::Array<I>::FromSharedStorage(cones.Commit()),
                                 R::Array<I>(), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)),
                                 R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
        auto coords = input.coordinates.Write();
        for (I row = 0; row < nodes.NumOwned(); ++row)
            for (size_t j = 0; j < 3; ++j)
                coords.View()(size_t(row), j) = 100 * mpi.rank + 10 * row + DNDS::real(j);
        if (active)
            input.cellInfo.Write().View()(0, 0) = G::ElemInfo{G::Elem::Tri3, 11 + mpi.rank};
        coords.Wait();
        if (periodic)
        {
            // Physical frame payload support, without claiming an interface
            // mesh: every cone carries the same translated image frame.
            input.periodic = true;
            input.periodicInfo.translation[0].map() = G::tPoint(2, 3, 4);
            input.cellNodesPbi.emplace(R::RowLayout::Dense(size_t(cells.NumOwned()), 3));
            input.boundaryNodesPbi.emplace(R::RowLayout::Dense(0, 0));
            auto bits = input.cellNodesPbi->Write();
            for (size_t row = 0; row < size_t(cells.NumOwned()); ++row)
                for (size_t slot = 0; slot < 3; ++slot)
                    bits.View()(row, slot) = G::nodePB1;
        }
        return G::ConstructNativeMesh(std::move(input));
    }

    void CheckField(const G::NativeMeshSerialOutput &output, const DNDS::MPIInfo &mpi, bool node, I owned)
    {
        DNDS::ArrayAdjacencyPair<2> fields;
        fields.InitPair("native serial field compatibility", mpi);
        fields.father->Resize(owned);
        fields.TransAttach();
        const auto &global = node ? output.nodeGlobal : output.cellGlobal;
        for (I row = 0; row < owned; ++row)
        {
            const auto id = (*global)(mpi.rank, row);
            fields(row, 0) = 17 * id + 2;
            fields(row, 1) = 19 * id - 3;
        }
        if (node)
            output.BorrowNodeIndexing(fields.trans);
        else
            output.BorrowCellIndexing(fields.trans);
        fields.trans.createMPITypes();
        fields.trans.pullOnce();
        REQUIRE(fields.son->Size() == (mpi.rank == output.readerRank ? global->globalSize() : 0));
        for (I row = 0; row < fields.son->Size(); ++row)
        {
            CHECK((*fields.son)(row, 0) == 17 * row + 2);
            CHECK((*fields.son)(row, 1) == 19 * row - 3);
        }
    }
}

TEST_CASE("Native serial output gathers exact global rows to a selectable reader with maps-only field compatibility")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (bool emptyPeers : {false, true})
        for (bool periodic : {false, true})
        {
            auto source = Strip(mpi, emptyPeers, periodic);
            const int reader = mpi.size - 1;
            auto output = G::NativeMeshSerialOutput::Build(*source, mpi, reader);
            CHECK_NOTHROW(output.ValidateSource(*source));
            auto setup = source->BindEpoch(R::VersionTracker{});
            CHECK_NOTHROW(output.ValidateSource(*setup));
            const I globalCells = emptyPeers ? 1 : mpi.size, globalNodes = 3 * globalCells;
            REQUIRE(output.cellNodes.Rows() == (mpi.rank == reader ? size_t(globalCells) : 0));
            REQUIRE(output.coordinates.Rows() == (mpi.rank == reader ? size_t(globalNodes) : 0));
            CHECK(output.cellInfo.Rows() == output.cellNodes.Rows());
            CHECK(bool(output.cellNodesPbi) == periodic);
            CHECK(output.periodic == periodic);
            if (periodic)
            {
                CHECK(output.periodicInfo.translation[0].map() == G::tPoint(2, 3, 4));
                auto bits = output.cellNodesPbi->Read();
                for (size_t row = 0; row < output.cellNodesPbi->Rows(); ++row)
                    for (size_t slot = 0; slot < 3; ++slot)
                        CHECK(bits.View()(row, slot) == G::nodePB1);
            }
            if (mpi.rank == reader)
            {
                auto cones = output.cellNodes.Read();
                auto coords = output.coordinates.Read();
                auto info = output.cellInfo.Read();
                for (I cell = 0; cell < globalCells; ++cell)
                {
                    const I nodeRank = emptyPeers ? 0 : (cell + 1) % mpi.size;
                    CHECK(cones.View().RowSize(size_t(cell)) == 3);
                    CHECK(info.View()(size_t(cell), 0).type == I(G::Elem::Tri3));
                    CHECK(info.View()(size_t(cell), 0).zone == 11 + cell);
                    for (size_t slot = 0; slot < 3; ++slot)
                        CHECK(cones.View()(size_t(cell), slot) == 3 * nodeRank + I(slot));
                }
                for (I node = 0; node < globalNodes; ++node)
                    for (size_t j = 0; j < 3; ++j)
                        CHECK(coords.View()(size_t(node), j) == 100 * (node / 3) + 10 * (node % 3) + DNDS::real(j));
            }
            CheckField(output, mpi, false, source->NumOwned(G::EntityKind::Cell));
            CheckField(output, mpi, true, source->NumOwned(G::EntityKind::Node));
            if (mpi.size > 1)
                CHECK_THROWS(G::NativeMeshSerialOutput::Build(*source, mpi, mpi.rank));
            CHECK_THROWS(G::NativeMeshSerialOutput::Build(*source, mpi, mpi.rank == 0 ? -1 : reader));
            auto retry = G::NativeMeshSerialOutput::Build(*source, mpi, reader);
            CHECK(retry.cellNodes.Rows() == output.cellNodes.Rows());
            auto retained = output;
            setup.reset();
            source.reset();
            output = {};
            CHECK(retained.coordinates.Read().View().Rows() == (mpi.rank == reader ? size_t(globalNodes) : 0));
        }
}

TEST_CASE("Native serial output retains old geometry and explicitly refreshes after publication")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, Strip(mpi, false));
    auto before = G::NativeMeshSerialOutput::Build(*mesh->CurrentNativeState(), mpi);
    mesh->TransformCoords([](const G::tPoint &point)
                          { return G::tPoint(point + G::tPoint(2, 3, 4)); });
    auto after = G::NativeMeshSerialOutput::Build(*mesh->CurrentNativeState(), mpi);
    CHECK_THROWS(before.ValidateSource(*mesh->CurrentNativeState()));
    CHECK_NOTHROW(after.ValidateSource(*mesh->CurrentNativeState()));
    auto oldPoints = before.coordinates.Read(), newPoints = after.coordinates.Read();
    REQUIRE(oldPoints.View().Rows() == newPoints.View().Rows());
    for (size_t row = 0; row < oldPoints.View().Rows(); ++row)
        for (size_t j = 0; j < 3; ++j)
            CHECK(newPoints.View()(row, j) == oldPoints.View()(row, j) + 2 + j);
    // Geometry changes leave numbering associations valid for field schedules.
    CheckField(before, mpi, true, mesh->NumNode());
    const auto ownedNodes = mesh->NumNode();
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Topology);
        mutation.Commit();
    }
    DNDS::ArrayAdjacencyPair<2> field;
    field.InitPair("stale native serial field", mpi);
    field.father->Resize(ownedNodes);
    field.TransAttach();
    CHECK_THROWS(before.BorrowNodeIndexing(field.trans));
    CHECK(mesh->HasNativeState());
}

TEST_CASE("Serial output facade rejects absent native authority and retains successful descriptors after failure")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto native = G::UnstructuredMesh::FromNative(mpi, Strip(mpi, false));
    auto empty = std::make_shared<G::UnstructuredMesh>(mpi, 2);
    const auto stamp = empty->versions.Stamp();
    G::UnstructuredMeshSerialRW reader(empty, mpi.size - 1);
    CHECK_THROWS(reader.BuildSerialOut());
    CHECK(stamp.Valid());
    CHECK_FALSE(reader.nativeSerialOutput);
    if (mpi.size > 1)
    {
        reader.mesh = mpi.rank == 0 ? empty : native;
        CHECK_THROWS(reader.BuildSerialOut());
        CHECK_FALSE(reader.nativeSerialOutput);
    }
    reader.mesh = native;
    reader.BuildSerialOut();
    REQUIRE(reader.nativeSerialOutput);
    CHECK(reader.dataIsSerialOut);
    CHECK_FALSE(reader.dataIsSerialIn);
    const auto prepared = reader.nativeSerialOutput;
    const auto nodes = prepared->coordinates.Read();
    CHECK(nodes.View().Rows() == (mpi.rank == reader.mRank ? size_t(3 * mpi.size) : 0));
    reader.mRank = mpi.rank == 0 ? -1 : mpi.size - 1;
    CHECK_THROWS(reader.BuildSerialOut());
    CHECK(reader.nativeSerialOutput == prepared);
    CHECK(reader.dataIsSerialOut);
    reader.mRank = mpi.size - 1;
    reader.BuildSerialOut();
    CHECK(reader.nativeSerialOutput != prepared);
    CHECK_NOTHROW(prepared->ValidateSource(*native->CurrentNativeState()));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
