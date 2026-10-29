#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/AdjacencyRegistry.hpp"

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;

static MPIInfo RegistryMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

template <class T>
static R::ArrayPair<T> TablePair(size_t rows, size_t width)
{
    return {R::Array<T>(R::RowLayout::Dense(rows, width)), R::Array<T>(R::RowLayout::Dense(0, width))};
}

static R::Array<DNDS::index> IndexRows(const std::vector<std::vector<DNDS::index>> &rows)
{
    R::RowBuilder builder(rows.size(), sizeof(DNDS::index));
    for (size_t row = 0; row < rows.size(); ++row)
    {
        builder.ResizeRow(row, rows[row].size());
        auto output = builder.Row<DNDS::index>(row);
        for (size_t slot = 0; slot < rows[row].size(); ++slot)
            output[slot] = rows[row][slot];
    }
    return R::Array<DNDS::index>::FromSharedStorage(builder.Commit());
}

struct RegistryTriangle
{
    NativeMeshData data;
    std::map<EntityKind, EntitySpace> spaces;
    std::map<EntityKind, R::Array<DNDS::index>> ids;

    explicit RegistryTriangle(const MPIInfo &mpi)
    {
        auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, 3);
        spaces.emplace(EntityKind::Node, nodes);
        spaces.emplace(EntityKind::Cell, nodes.CreateSibling(EntityKind::Cell, 1));
        spaces.emplace(EntityKind::Bnd, nodes.CreateSibling(EntityKind::Bnd, 0));
        data.dimension = 2;
        data.coords = TablePair<DNDS::real>(3, 3);
        {
            auto write = data.coords->owned.Write();
            const DNDS::real coordinates[3][3]{{DNDS::real(2 * mpi.rank), 0, 0},
                                               {DNDS::real(2 * mpi.rank + 1), 0, 0},
                                               {DNDS::real(2 * mpi.rank), 1, 0}};
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    write.View()(i, j) = coordinates[i][j];
        }
        data.cellInfo = TablePair<ElemInfo>(1, 1);
        {
            auto write = data.cellInfo->owned.Write();
            write.View()(0, 0) = ElemInfo{Elem::Tri3, INTERNAL_ZONE};
        }
        data.bndInfo = TablePair<ElemInfo>(0, 1);
        data.relations[size_t(MeshRelation::CellNodes)] = R::ArrayPair<DNDS::index>{IndexRows({{0, 1, 2}}), IndexRows({})};
        data.relations[size_t(MeshRelation::CellCells)] = R::ArrayPair<DNDS::index>{IndexRows({{}}), IndexRows({})};
        data.relations[size_t(MeshRelation::BoundaryNodes)] = R::ArrayPair<DNDS::index>{IndexRows({}), IndexRows({})};
        data.relations[size_t(MeshRelation::BoundaryCells)] = TablePair<DNDS::index>(0, 2);
        for (const auto &[kind, space] : spaces)
        {
            R::Array<DNDS::index> values(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
            auto write = values.Write();
            for (DNDS::index row = 0; row < space.NumOwned(); ++row)
                write.View()(size_t(row), 0) = 1000000000000LL + 17 * space.LocalToGlobal(row) + int(kind);
            ids.emplace(kind, std::move(values));
        }
    }
};

static AdjacencyDescriptor RegistryInverse()
{
    return Adj::Cell2Face.Inverted();
}

static AdjacencyDescriptor RegistryComposition(RelationSelection selection)
{
    return AdjacencyDescriptor::Composition(Adj::Face2Node, Adj::Cell2Node.Inverted(), selection, false, "registry.selection", 1);
}

static void AddRegistry(RegistryTriangle &input, const MPIInfo &mpi)
{
    // Three independently defined face->cell variants intentionally coexist.
    // Registry storage is tested with distinct supplied rows; the registry
    // preserves definitions and does not recompute relation mathematics.
    auto anchor = input.spaces.at(EntityKind::Node);
    input.spaces.emplace(EntityKind::Face, anchor.CreateSibling(EntityKind::Face, 3));
    input.data.faceInfo = TablePair<ElemInfo>(3, 1);
    {
        auto write = input.data.faceInfo->owned.Write();
        for (size_t i = 0; i < 3; ++i)
            write.View()(i, 0) = ElemInfo{Elem::Line2, BC_ID_DEFAULT_WALL};
    }
    input.data.relations[size_t(MeshRelation::CellFaces)] = R::ArrayPair<DNDS::index>{IndexRows({{0, 1, 2}}), IndexRows({})};
    input.data.relations[size_t(MeshRelation::FaceNodes)] = R::ArrayPair<DNDS::index>{IndexRows({{0, 1}, {1, 2}, {2, 0}}), IndexRows({})};
    input.data.relations[size_t(MeshRelation::FaceCells)] = TablePair<DNDS::index>(3, 2);
    {
        auto write = input.data.relations[size_t(MeshRelation::FaceCells)]->owned.Write();
        for (size_t i = 0; i < 3; ++i)
        {
            write.View()(i, 0) = 0;
            write.View()(i, 1) = UnInitIndex;
        }
    }
    input.data.relations[size_t(MeshRelation::FaceBoundaries)] = R::ArrayPair<DNDS::index>{IndexRows({{UnInitIndex}, {UnInitIndex}, {UnInitIndex}}), IndexRows({})};
    input.data.relations[size_t(MeshRelation::BoundaryFaces)] = R::ArrayPair<DNDS::index>{IndexRows({}), IndexRows({})};
    R::Array<DNDS::index> ids(R::RowLayout::Dense(3, 1));
    {
        auto write = ids.Write();
        for (size_t i = 0; i < 3; ++i)
            write.View()(i, 0) = 9000000000000LL + 3 * mpi.rank + i;
    }
    input.ids.emplace(EntityKind::Face, std::move(ids));
    const auto global = input.spaces.at(EntityKind::Cell).LocalToGlobal(0);
    input.data.namedRelations.emplace(RegistryInverse(), NativeNamedRelation{
                                                             {IndexRows({{0}, {0}, {0}}), IndexRows({})}, Adj_PointToLocal});
    input.data.namedRelations.emplace(RegistryComposition(RelationSelection::Any), NativeNamedRelation{
                                                                                       {IndexRows({{global}, {global}, {}}), IndexRows({})}, Adj_PointToGlobal});
    input.data.namedRelations.emplace(RegistryComposition(RelationSelection::All), NativeNamedRelation{
                                                                                       {IndexRows({{global}, {}, {}}), IndexRows({})}, Adj_PointToGlobal});
}

static MeshReadRequirements RegistryRequirements()
{
    auto requirements = MeshReadRequirements::Primary();
    requirements.faces = true;
    return requirements;
}

TEST_CASE("Named native relation variants retain exact definitions arrays and canonical orders")
{
    const auto mpi = RegistryMPI();
    RegistryTriangle input(mpi);
    AddRegistry(input, mpi);
    auto retained = input.data.namedRelations.at(RegistryInverse()).rows.owned;
    auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, RegistryRequirements());
    CHECK(state->RelationKeys().size() == 3);
    CHECK_THROWS(retained.Write());
    R::VersionTracker owner2, owner3;
    auto second = state->BindEpoch(owner2);
    auto third = second->BindEpoch(owner3);
    for (const auto &publication : {state, second, third})
    {
        auto inverse = publication->Relation(RegistryInverse());
        CHECK(inverse.Source().SameOrder(publication->Entities(EntityKind::Face)));
        CHECK(inverse.Target().SameOrder(publication->Entities(EntityKind::Cell)));
        CHECK(inverse.Raw(0, 0) == 0);
        CHECK(inverse.Raw(2, 0) == 0);
        CHECK(inverse.GlobalRow(1) == std::vector<DNDS::index>{mpi.rank});
        auto any = publication->Relation(RegistryComposition(RelationSelection::Any));
        auto all = publication->Relation(RegistryComposition(RelationSelection::All));
        CHECK(any.GlobalRow(0) == std::vector<DNDS::index>{mpi.rank});
        CHECK(any.GlobalRow(1) == std::vector<DNDS::index>{mpi.rank});
        CHECK(any.RowSize(2) == 0);
        CHECK(all.GlobalRow(0) == std::vector<DNDS::index>{mpi.rank});
        CHECK(all.RowSize(1) == 0);
        auto absent = RegistryComposition(RelationSelection::Exact);
        CHECK_THROWS(publication->Relation(absent));
    }
    auto escaped = third->Relation(RegistryInverse());
    state.reset();
    second.reset();
    third.reset();
    CHECK(escaped.Raw(2, 0) == 0);
    CHECK_THROWS(retained.Write());
}

TEST_CASE("Native mesh registration retains named inverse and selection variants beside facade adapters")
{
    const auto mpi = RegistryMPI();
    RegistryTriangle input(mpi);
    AddRegistry(input, mpi);
    const DNDS::index next = (mpi.rank + 1) % mpi.size;
    input.data.namedRelations.at(RegistryComposition(RelationSelection::Any)).rows.owned = IndexRows({{next}, {}, {}});
    auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, RegistryRequirements());
    auto mesh = UnstructuredMesh::FromNative(mpi, state);
    CHECK(mesh->NativeRelationKeys().size() == 3);
    CHECK(mesh->ReadNativeRelation(RegistryInverse()).GlobalRow(0) == std::vector<DNDS::index>{mpi.rank});
    CHECK(RegistryInverse() != Adj::Face2Cell);
    CHECK(RegistryInverse().provenance == "inverse:" + Adj::Cell2Face.Signature());
    AdjacencyRegistry setup(mpi);
    setup.RegisterMesh(*mesh, RegistryRequirements());
    mesh.reset();
    state.reset();
    // The setup registry alone retains all three named readers and the facade
    // adapters. No endpoint fallback can satisfy these exact descriptor hops.
    for (const auto &key : {RegistryInverse(), RegistryComposition(RelationSelection::Any), RegistryComposition(RelationSelection::All), Adj::Face2Cell})
    {
        GhostSpec spec{{{EntityKind::Face, {key}, EntityKind::Cell}}};
        auto discovery = setup.Evaluate(CompiledGhostTree::compile(spec));
        if (mpi.size > 1 && key == RegistryComposition(RelationSelection::Any))
            CHECK(discovery.ghostIndices.at(EntityKind::Cell) == std::vector<DNDS::index>{next});
        else
            CHECK(discovery.ghostIndices.empty());
    }
}

TEST_CASE("Mathematical inverse definitions never alias legacy physical slots")
{
    const std::vector<AdjacencyDescriptor> cones{Adj::Cell2Node, Adj::Cell2Face, Adj::Cell2Edge, Adj::Face2Node,
                                                 Adj::Face2Edge, Adj::Edge2Node, Adj::Bnd2Node};
    for (const auto &cone : cones)
    {
        const auto inverse = cone.Inverted();
        CHECK(inverse != AdjacencyDescriptor(cone.to, cone.from));
        CHECK(inverse.selection == RelationSelection::Incidence);
        CHECK(inverse.order == RelationOrder::SortedUnique);
        CHECK(inverse.Path() == std::vector<EntityKind>{cone.to, cone.from});
        CHECK(inverse.provenance == "inverse:" + cone.Signature());
    }
}

TEST_CASE("Named native relation publication rejects invalid rank-local state and collective definition mismatch")
{
    const auto mpi = RegistryMPI();
    RegistryTriangle input(mpi);
    AddRegistry(input, mpi);
    auto retained = input.data.namedRelations.at(RegistryInverse()).rows.owned;
    SUBCASE("invalid local index")
    {
        if (mpi.rank == 0)
        {
            auto write = retained.Write();
            write.View()(0, 0) = 1;
        }
    }
    SUBCASE("unspecified index representation")
    {
        if (mpi.rank == 0)
            input.data.namedRelations.at(RegistryInverse()).state = Adj_Unknown;
    }
    SUBCASE("source row extent")
    {
        if (mpi.rank == 0)
            input.data.namedRelations.at(RegistryInverse()).rows.owned = IndexRows({{0}});
    }
    SUBCASE("descriptor schedule differs")
    {
        if (mpi.size == 1)
            return;
        if (mpi.rank == 0)
            input.data.namedRelations.erase(RegistryComposition(RelationSelection::All));
    }
    SUBCASE("duplicate sorted target")
    {
        if (mpi.rank == 0)
            input.data.namedRelations.at(RegistryComposition(RelationSelection::Any)).rows.owned = IndexRows({{0, 0}, {}, {}});
    }
    SUBCASE("absent sorted target")
    {
        if (mpi.rank == 0)
            input.data.namedRelations.at(RegistryComposition(RelationSelection::Any)).rows.owned = IndexRows({{UnInitIndex}, {}, {}});
    }
    SUBCASE("invalid descriptor")
    {
        if (mpi.rank == 0)
        {
            auto relation = input.data.namedRelations.at(RegistryInverse());
            auto invalid = RegistryComposition(RelationSelection::Any);
            invalid.path.clear();
            input.data.namedRelations.emplace(invalid, std::move(relation));
        }
    }
    CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, RegistryRequirements()));
    CHECK_NOTHROW(retained.Write());
}

TEST_CASE("Duplicate facade definitions require identical decoded ordered slots")
{
    const auto mpi = RegistryMPI();
    RegistryTriangle input(mpi);
    AddRegistry(input, mpi);
    // Matching global encoding is accepted against the fixed local-index cone.
    const DNDS::index offset = input.spaces.at(EntityKind::Node).LocalToGlobal(0);
    auto retained = IndexRows({{offset, offset + 1, offset + 2}});
    input.data.namedRelations.emplace(Adj::Cell2Node, NativeNamedRelation{
                                                          {retained, IndexRows({})}, Adj_PointToGlobal});
    input.data.namedRelations.emplace(Adj::Face2Cell, NativeNamedRelation{
                                                          {IndexRows({{0, UnInitIndex}, {0, UnInitIndex}, {0, UnInitIndex}}), IndexRows({})}, Adj_PointToLocal});
    bool valid = true;
    SUBCASE("matching ordered slots in a different encoding") {}
    SUBCASE("only rank zero changes cone order")
    {
        valid = false;
        if (mpi.rank == 0)
        {
            auto write = retained.Write();
            std::swap(write.View()(0, 0), write.View()(0, 1));
        }
    }
    SUBCASE("only rank zero drops the absent support slot")
    {
        valid = false;
        if (mpi.rank == 0)
            input.data.namedRelations.at(Adj::Face2Cell).rows.owned = IndexRows({{0}, {0}, {0}});
    }
    SUBCASE("only rank zero changes the absent support slot")
    {
        valid = false;
        if (mpi.rank == 0)
        {
            auto write = input.data.namedRelations.at(Adj::Face2Cell).rows.owned.Write();
            write.View()(0, 1) = 0;
        }
    }
    if (!valid)
    {
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, RegistryRequirements()));
        CHECK_NOTHROW(retained.Write());
        CHECK(input.spaces.at(EntityKind::Node).Valid());
    }
    else
    {
        auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, RegistryRequirements());
        CHECK(state->Relation(Adj::Cell2Node).GlobalRow(0) == std::vector<DNDS::index>{offset, offset + 1, offset + 2});
        CHECK(state->Relation(Adj::Face2Cell).GlobalRow(0) == std::vector<DNDS::index>{mpi.rank, UnInitIndex});
        CHECK_THROWS(retained.Write());
    }
}

TEST_CASE("Named relation access rejects stale publication epochs")
{
    const auto mpi = RegistryMPI();
    RegistryTriangle input(mpi);
    AddRegistry(input, mpi);
    auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, RegistryRequirements());
    R::VersionTracker owner;
    auto rebound = state->BindEpoch(owner);
    {
        auto mutation = owner.BeginMutation(R::Change::Topology);
        mutation.Commit();
    }
    CHECK_THROWS(rebound->Relation(RegistryInverse()));
    CHECK_NOTHROW(state->Relation(RegistryInverse()));
}

TEST_CASE("Sorted native relation order uses globals across owned and ghost local indices")
{
    const auto mpi = RegistryMPI();
    if (mpi.size < 2)
        return;
    auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, 3);
    auto cells = nodes.CreateSibling(EntityKind::Cell, 1);
    const DNDS::index own = nodes.LocalToGlobal(0);
    const DNDS::index remote = 3 * ((mpi.rank + mpi.size - 1) % mpi.size);
    nodes = nodes.WithGhosts({remote});
    const auto lower = std::min(own, remote), upper = std::max(own, remote);
    AdjacencyDescriptor key{EntityKind::Cell, EntityKind::Node};
    key.path = {EntityKind::Cell, EntityKind::Node};
    key.selection = RelationSelection::Incidence;
    key.order = RelationOrder::SortedUnique;
    key.periodic = RelationPeriodicPolicy::EntityIdentity;
    auto rows = IndexRows({{*nodes.GlobalToLocal(lower), *nodes.GlobalToLocal(upper)}});
    MeshAdjacencyRead read(key, cells, nodes, Adj_PointToLocal, rows);
    CHECK(read.GlobalRow(0) == std::vector<DNDS::index>{lower, upper});
    if (mpi.rank > 0)
        CHECK(read.Raw(0, 0) > read.Raw(0, 1));
    auto reversed = IndexRows({{upper, lower}});
    CHECK_THROWS(MeshAdjacencyRead(key, cells, nodes, Adj_PointToGlobal, reversed));
}

TEST_CASE("Native removeSelf descriptor rejects self incidences")
{
    const auto mpi = RegistryMPI();
    auto cells = EntitySpace::CreateOwned(EntityKind::Cell, mpi, 1);
    auto key = AdjacencyDescriptor::Composition(Adj::Cell2Node, Adj::Cell2Node.Inverted(), RelationSelection::Any, true);
    auto rows = IndexRows({{cells.LocalToGlobal(0)}});
    CHECK_THROWS(MeshAdjacencyRead(key, cells, cells, Adj_PointToGlobal, rows));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
