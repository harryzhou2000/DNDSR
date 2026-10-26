#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "NativePeriodicFixtures.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    R::Array<I> Rows(const std::vector<std::vector<I>> &rows, size_t width = 0)
    {
        R::RowBuilder builder(rows.size(), sizeof(I), width ? R::RowLayoutKind::Table : R::RowLayoutKind::CSR, width);
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size());
            std::copy(rows[row].begin(), rows[row].end(), builder.Row<I>(row).data);
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }
    enum class Defect
    {
        None,
        DuplicateFace,
        MissingCellSupport,
        MissingBoundarySupport,
        WrongSupportNode,
        FaceZoneSides
    };

    // Direct local publication supplies a known incidence oracle on every rank.
    // Optional synthetic outer rows isolate closure policy from mesh assembly.
    std::shared_ptr<G::UnstructuredMesh> Fixture(const DNDS::MPIInfo &mpi, Defect defect = Defect::None,
                                                 bool faces = true, bool inverse = true, bool outer = false)
    {
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 5); // One intentionally unused node.
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, 1);
        if (outer && mpi.size > 1)
            cells = cells.WithGhosts({I((mpi.rank + 1) % mpi.size)});
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 1);
        auto faceSpace = nodes.CreateSibling(G::EntityKind::Face, faces ? 4 : 0);
        G::NativeMeshData data;
        data.dimension = 2;
        data.coords = R::ArrayPair<DNDS::real>{R::Array<DNDS::real>(R::RowLayout::Dense(5, 3)), R::Array<DNDS::real>(R::RowLayout::Dense(0, 3))};
        {
            auto write = data.coords->owned.Write();
            const DNDS::real points[5][3]{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {9, 9, 0}};
            for (size_t row = 0; row < 5; ++row)
                for (size_t component = 0; component < 3; ++component)
                    write.View()(row, component) = points[row][component];
        }
        auto info = [](size_t count, G::Elem::ElemType type, G::t_index zone)
        {
            R::Array<G::ElemInfo> result(R::RowLayout::Dense(count, 1));
            auto write = result.Write();
            for (size_t row = 0; row < count; ++row)
                write.View()(row, 0) = G::ElemInfo{type, zone};
            return result;
        };
        data.cellInfo = R::ArrayPair<G::ElemInfo>{info(1, G::Elem::Quad4, G::INTERNAL_ZONE), info(size_t(cells.NumGhost()), G::Elem::Quad4, G::INTERNAL_ZONE)};
        data.bndInfo = R::ArrayPair<G::ElemInfo>{info(1, G::Elem::Line2, G::BC_ID_DEFAULT_WALL), info(0, G::Elem::Line2, G::BC_ID_DEFAULT_WALL)};
        auto relation = [&](G::MeshRelation kind, std::vector<std::vector<I>> owned, std::vector<std::vector<I>> ghost = {}, size_t width = 0)
        { data.relations[size_t(kind)] = R::ArrayPair<I>{Rows(owned, width), Rows(ghost, width)}; };
        const bool ghost = cells.NumGhost() != 0;
        relation(G::MeshRelation::CellNodes, {{0, 1, 2, 3}}, ghost ? std::vector<std::vector<I>>{{0, 1, 2, 3}} : std::vector<std::vector<I>>{});
        relation(G::MeshRelation::CellCells, {{}}, ghost ? std::vector<std::vector<I>>{{}} : std::vector<std::vector<I>>{});
        relation(G::MeshRelation::BoundaryNodes, {{0, 1}});
        relation(G::MeshRelation::BoundaryCells, {{0, DNDS::UnInitIndex}}, {}, 2);
        if (inverse)
        {
            std::vector<std::vector<I>> support{{0}, {0}, {0}, {0}, {}};
            if (defect == Defect::MissingCellSupport)
                support[0].clear();
            if (defect == Defect::WrongSupportNode)
                support[4] = {0};
            // A separate unavailable support is permitted without requiring a
            // complete node star. Never use it as a local index in the check.
            if (!outer && mpi.size > 1)
                support[0].push_back(-1 - I((mpi.rank + 1) % mpi.size));
            relation(G::MeshRelation::NodeCells, support);
            std::vector<std::vector<I>> boundarySupport = defect == Defect::MissingBoundarySupport
                                                              ? std::vector<std::vector<I>>{{}, {0}, {}, {}, {}}
                                                              : std::vector<std::vector<I>>{{0}, {0}, {}, {}, {}};
            if (mpi.size > 1)
                boundarySupport[1].push_back(-1 - I((mpi.rank + 1) % mpi.size));
            relation(G::MeshRelation::NodeBoundaries, boundarySupport);
        }
        if (faces)
        {
            data.faceInfo = R::ArrayPair<G::ElemInfo>{info(4, G::Elem::Line2, G::BC_ID_DEFAULT_WALL), info(0, G::Elem::Line2, G::BC_ID_DEFAULT_WALL)};
            if (defect == Defect::FaceZoneSides)
                data.faceInfo->owned.Write().View()(0, 0).zone = G::INTERNAL_ZONE;
            std::vector<I> cellFaces = defect == Defect::DuplicateFace ? std::vector<I>{0, 0, 2, 3} : std::vector<I>{0, 1, 2, 3};
            std::vector<std::vector<I>> outerFaces;
            if (ghost)
            {
                const I first = 4 * ((mpi.rank + 1) % mpi.size);
                outerFaces = {{-1 - first, -2 - first, -3 - first, -4 - first}};
            }
            relation(G::MeshRelation::CellFaces, {cellFaces}, outerFaces);
            relation(G::MeshRelation::FaceNodes, {{0, 1}, {1, 2}, {2, 3}, {3, 0}});
            relation(G::MeshRelation::FaceCells, {{0, DNDS::UnInitIndex}, {0, DNDS::UnInitIndex}, {0, DNDS::UnInitIndex}, {0, DNDS::UnInitIndex}}, {}, 2);
            relation(G::MeshRelation::FaceBoundaries, {{0}, {DNDS::UnInitIndex}, {DNDS::UnInitIndex}, {DNDS::UnInitIndex}}, {}, 1);
            relation(G::MeshRelation::BoundaryFaces, {{0}}, {}, 1);
        }
        auto requirements = G::MeshReadRequirements::Primary();
        requirements.faces = faces;
        requirements.inverse = inverse;
        std::map<G::EntityKind, G::EntitySpace> spaces{{G::EntityKind::Node, nodes}, {G::EntityKind::Cell, cells}, {G::EntityKind::Bnd, boundaries}};
        if (faces)
            spaces.emplace(G::EntityKind::Face, faceSpace);
        std::map<G::EntityKind, R::Array<I>> ids;
        for (const auto &[kind, space] : spaces)
        {
            R::Array<I> values(R::RowLayout::Dense(size_t(space.NumLocal()), 1));
            {
                auto write = values.Write();
                for (I row = 0; row < space.NumLocal(); ++row)
                    write.View()(size_t(row), 0) = 10000 + 13 * space.LocalToGlobal(row) + I(kind);
            }
            ids.emplace(kind, std::move(values));
        }
        return G::UnstructuredMesh::FromNative(mpi, G::NativeMeshState::FromNative(mpi, std::move(data), std::move(spaces), std::move(ids), requirements));
    }
}

TEST_CASE("Native mesh assertions validate real incidences and reject missing groups and reciprocal defects")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto valid = Fixture(mpi);
    const auto state = valid->CurrentNativeState();
    const auto stamp = valid->versions.Stamp();
    auto retained = valid->AcquireRead(valid->CurrentNativeRequirements());
    CHECK_NOTHROW(valid->AssertOnFaces());
    CHECK_NOTHROW(valid->AssertOnN2CB());
    CHECK(stamp.Valid());
    CHECK(valid->CurrentNativeState() == state);
    CHECK_FALSE(valid->coords.father);
    CHECK_FALSE(valid->node2cell.father);
    auto noFaces = Fixture(mpi, Defect::None, false);
    auto noInverse = Fixture(mpi, Defect::None, true, false);
    auto duplicateFace = Fixture(mpi, Defect::DuplicateFace);
    auto missingCell = Fixture(mpi, Defect::MissingCellSupport);
    auto missingBoundary = Fixture(mpi, Defect::MissingBoundarySupport);
    auto wrongNode = Fixture(mpi, Defect::WrongSupportNode);
    CHECK_THROWS(noFaces->AssertOnFaces());
    CHECK_THROWS(noInverse->AssertOnN2CB());
    CHECK_THROWS(duplicateFace->AssertOnFaces());
    CHECK_THROWS(missingCell->AssertOnN2CB());
    CHECK_THROWS(missingBoundary->AssertOnN2CB());
    CHECK_THROWS(wrongNode->AssertOnN2CB());
    // Invalid internal/external side sentinels fail during native publication,
    // before any assertion method can receive the inconsistent face group.
    CHECK_THROWS(Fixture(mpi, Defect::FaceZoneSides));
    CHECK_NOTHROW(valid->AssertOnFaces());
    CHECK_NOTHROW(valid->AssertOnN2CB());
}

TEST_CASE("Native face assertions permit unavailable outer ghost cell faces and preserve self periodic multiplicity")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto outer = Fixture(mpi, Defect::None, true, true, true);
    if (mpi.size > 1)
    {
        auto read = outer->AcquireRead(G::MeshReadRequirements::CellFlux());
        REQUIRE(read.HostView().NumCellGhost() == 1);
        for (auto face : read.HostView().CellFaces(1))
            CHECK(face < 0);
    }
    CHECK_NOTHROW(outer->AssertOnFaces());
    CHECK_NOTHROW(outer->AssertOnN2CB());
    auto primary = G::NativeMeshBuilder::Build(NativePeriodicFixtures::QuadraticTorus(mpi), {1, true});
    auto state = G::NativeMeshTopology::Build(*primary, mpi);
    auto periodic = G::UnstructuredMesh::FromNative(mpi, state);
    auto read = periodic->AcquireRead(state->Requirements());
    if (periodic->NumCell())
    {
        const auto faces = read.HostView().CellFaces(0);
        CHECK(faces[0] == faces[2]);
        CHECK(faces[1] == faces[3]);
    }
    CHECK_NOTHROW(periodic->AssertOnFaces());
    CHECK_NOTHROW(periodic->AssertOnN2CB());
    CHECK(periodic->CurrentNativeState()->MeshNamespace() == state->MeshNamespace());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
