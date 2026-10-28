#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "Geom/Mesh/NativeMeshOutput.hpp"
#include "Geom/Mesh/NativeMeshSmoothing.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeConnectivity.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <filesystem>
#include <set>
#include <map>
#include <numeric>
#include <chrono>
using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
struct MeshConfig
{
    const char *name;          ///< Human-readable label
    const char *file;          ///< Filename relative to data/mesh/
    int dim;                   ///< Spatial dimension
    bool periodic;             ///< Has periodic boundaries
    tPoint translation1;       ///< Periodic translation axis 1
    tPoint translation2;       ///< Periodic translation axis 2
    tPoint translation3;       ///< Periodic translation axis 3
    DNDS::index expectedCells; ///< Expected global cell count (-1 = skip check)
    DNDS::index expectedBnds;  ///< Expected global bnd count (-1 = skip check)
};

static const MeshConfig g_configs[] = {
    {"UniformSquare_10", "UniformSquare_10.cgns", 2, false, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 100, 40},
    {"IV10_10", "IV10_10.cgns", 2, true, {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, 100, -1},
    {"NACA0012_H2", "NACA0012_H2.cgns", 2, false, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 20816, 484},
    {"IV10U_10", "IV10U_10.cgns", 2, true, {10, 0, 0}, {0, 10, 0}, {0, 0, 10}, 322, -1},
    {"Ball2", "Ball2.cgns", 3, false, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 958994, 16402}, // 3D mixed-element mesh (Ball2 is np=8 only)
};
static constexpr int N_CONFIGS = sizeof(g_configs) / sizeof(g_configs[0]);

static MPIInfo g_mpi;
static ssp<UnstructuredMesh> g_full[N_CONFIGS];
static ssp<UnstructuredMesh> g_edges;
static std::string meshPath(const std::string &name)
{
    return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "data/mesh" / name).string();
}
static MeshReadRequirements FullRequirements()
{
    auto r = MeshReadRequirements::CellFlux();
    r.inverse = true;
    r.cellFaceNeighbors = true;
    r.allFaceCellsLocal = true;
    return r;
}
static R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
{
    R::RowBuilder builder(rows.size(), sizeof(I));
    for (size_t row = 0; row < rows.size(); ++row)
    {
        builder.ResizeRow(row, rows[row].size());
        auto target = builder.Row<I>(row);
        std::copy(rows[row].begin(), rows[row].end(), target.data);
    }
    return R::Array<I>::FromSharedStorage(builder.Commit());
}
static std::optional<NativePeriodicImportOptions> PeriodicOptions(const MeshConfig &cfg)
{
    if (!cfg.periodic)
        return {};
    NativePeriodicImportOptions p;
    p.tolerance = 1e-8;
    p.frames.translation[1].map() = cfg.translation1;
    p.frames.translation[2].map() = cfg.translation2;
    p.frames.translation[3].map() = cfg.translation3;
    return p;
}
static std::shared_ptr<const NativeMeshState> Complete(std::shared_ptr<const NativeMeshState> state, int layers = 1)
{
    auto data = NativeMeshAssembler::CaptureOwned(*state, g_mpi);
    const auto &view = state->BorrowHostView();
    const auto cells = state->Entities(EntityKind::Cell);
    std::vector<std::vector<I>> other(size_t(cells.NumOwned()));
    for (I cell = 0; cell < cells.NumOwned(); ++cell)
        for (size_t slot = 0; slot < view.CellFaces(cell).size(); ++slot)
        {
            const auto adjacent = view.CellFaceOther(cell, view.CellFaces(cell)[slot], DNDS::rowsize(slot));
            other[size_t(cell)].push_back(adjacent == UnInitIndex ? UnInitIndex : (adjacent >= 0 ? cells.LocalToGlobal(adjacent) : -1 - adjacent));
        }
    data.relations[size_t(MeshRelation::CellFaceCells)] = Rows(other);
    data.requirements = FullRequirements();
    return AssembleNativeMesh(g_mpi, std::move(data), layers);
}
static ssp<UnstructuredMesh> ReadConfigured(const MeshConfig &cfg, int layers = 1)
{
    AutoAppendName2ID names;
    auto state = ReadNativeCGNS(g_mpi, meshPath(cfg.file), cfg.dim, [&](const std::string &name)
                                { return names(name); }, 0, {layers, true}, PeriodicOptions(cfg), NativeSerialDistribution::ReaderOwned);
    NativePartitionOptions p;
    p.metisUfactor = 30;
    p.metisSeed = 42;
    p.metisNcuts = 1;
    p.ghostLayers = layers;
    state = NativeMeshPreparation::Partition(*state, g_mpi, p).state;
    return UnstructuredMesh::FromNative(g_mpi, Complete(std::move(state), layers));
}
static ssp<UnstructuredMesh> nativeBoundarySource(int id, const MeshConfig &) { return g_full[id]; }
static ssp<UnstructuredMesh> buildFullMesh(int, const MeshConfig &cfg, int parts = 1)
{
    auto mesh = ReadConfigured(cfg);
    if (parts > 1)
        mesh->ReorderLocalCells(parts);
    return mesh;
}
static std::vector<I> Global(const EntitySpace &space, MeshRowSpan<I> row)
{
    std::vector<I> result;
    for (auto value : row)
        result.push_back(value == UnInitIndex ? UnInitIndex : (value >= 0 ? space.LocalToGlobal(value) : -1 - value));
    return result;
}
static std::set<I> Set(const std::vector<I> &values) { return {values.begin(), values.end()}; }
static void FaceIncidence(UnstructuredMesh &mesh)
{
    auto retained = mesh.AcquireRead(FullRequirements());
    const auto v = retained.HostView();
    const auto nodes = mesh.Entities(EntityKind::Node);
    for (I cell = 0; cell < v.NumCell(); ++cell)
    {
        auto e = v.GetCellElement(cell);
        const auto cone = v.CellNodes(cell);
        const auto faces = v.CellFaces(cell);
        REQUIRE(faces.size() == size_t(e.GetNumFaces()));
        for (size_t slot = 0; slot < faces.size(); ++slot)
        {
            const auto face = faces[slot];
            REQUIRE(face >= 0);
            REQUIRE(face < v.NumFaceProc());
            auto ef = e.ObtainFace(DNDS::rowsize(slot));
            std::vector<I> expected(size_t(ef.GetNumNodes()));
            e.ExtractFaceNodes(DNDS::rowsize(slot), cone, expected);
            for (auto &node : expected)
                node = nodes.LocalToGlobal(node);
            const auto actual = Global(nodes, v.FaceNodes(face));
            CHECK(Set(actual) == Set(expected));
            if (v.CellIsFaceBack(cell, face, DNDS::rowsize(slot)))
                CHECK(actual == expected);
            const auto parents = v.FaceCells(face);
            CHECK((parents[0] == cell || parents[1] == cell));
            CHECK(v.GetFaceInfo(face).getElemType() == ef.type);
        }
    }
}
/// Check required membership directly from complete inverse rows, separately
/// from reassembly equality. A missing boundary on BOTH publications must fail.
/// Only locally stored rows are visited; there is no global-sized oracle table.
static void CheckRequiredBoundaryCoverage(UnstructuredMesh &mesh)
{
    auto retained = mesh.AcquireRead(FullRequirements());
    const auto view = retained.HostView();
    const auto boundaries = mesh.Entities(EntityKind::Bnd);
    auto requireStored = [&](const std::vector<I> &globals)
    {
        for (const auto global : globals)
            if (global != UnInitIndex)
            {
                CAPTURE(global);
                CHECK(boundaries.GlobalToLocal(global).has_value());
            }
    };
    auto visitNode = [&](I node)
    {
        REQUIRE(node >= 0);
        REQUIRE(node < view.NumNodeProc());
        requireStored(Global(boundaries, view.NodeBoundaries(node)));
    };
    for (I boundary = 0; boundary < view.NumBnd(); ++boundary)
        for (const auto node : view.BoundaryNodes(boundary))
            visitNode(node);
    for (I node = 0; node < view.NumNode(); ++node)
        visitNode(node);
    for (I face = 0; face < view.NumFace(); ++face)
        requireStored(Global(boundaries, view.Relation(MeshRelation::FaceBoundaries, face)));
}
static std::array<AdjKind, size_t(MeshRelation::Count)> Bindings()
{
    return {Adj::Cell2Node, Adj::Bnd2Node, Adj::Bnd2Cell, Adj::Cell2Cell, Adj::Node2Cell, Adj::Node2Bnd, Adj::Cell2Face, Adj::Face2Node, Adj::Face2Cell, Adj::Face2Bnd, Adj::Bnd2Face, Adj::Cell2Edge, Adj::Edge2Node, Adj::Edge2Cell, Adj::Cell2CellFace};
}
static void Roundtrip(UnstructuredMesh &mesh, std::initializer_list<MeshRelation> slots)
{
    const auto stamp = mesh.versions.Stamp();
    auto retained = mesh.AcquireRead(FullRequirements());
    const auto view = retained.HostView();
    const auto kinds = Bindings();
    for (auto slot : slots)
    {
        auto key = kinds[size_t(slot)];
        key.semanticName = "pipeline.convert." + std::to_string(size_t(slot));
        key.order = RelationOrder::Topological;
        key.selection = RelationSelection::Incidence;
        key.path = {key.from, key.to};
        key.periodic = RelationPeriodicPolicy::EntityIdentity;
        const auto from = mesh.Entities(key.from), to = mesh.Entities(key.to);
        std::vector<std::vector<I>> rows(size_t(from.NumOwned())), local(rows.size());
        for (I row = 0; row < from.NumOwned(); ++row)
        {
            rows[size_t(row)] = Global(to, view.Relation(slot, row));
            auto r = view.Relation(slot, row);
            local[size_t(row)] = {r.begin(), r.end()};
        }
        auto values = Rows(rows);
        RelationReadAccess original(key, from, to, Adj_PointToGlobal, values);
        auto localized = NativeRelation::Convert(original, Adj_PointToLocal);
        auto localizedRead = localized.Read();
        auto restored = NativeRelation::Convert(localizedRead, Adj_PointToGlobal);
        auto restoredRead = restored.Read();
        for (I row = 0; row < from.NumOwned(); ++row)
        {
            CHECK(restoredRead.GlobalRow(row) == rows[size_t(row)]);
            std::vector<I> raw;
            for (size_t j = 0; j < localizedRead.RowSize(row); ++j)
                raw.push_back(localizedRead.Raw(row, j));
            CHECK(raw == local[size_t(row)]);
        }
    }
    CHECK(stamp.Valid());
}
#define FOR_EACH_MESH_CONFIG(...)             \
    for (int _ci = 0; _ci < N_CONFIGS; ++_ci) \
    {                                         \
        if (_ci == 4 && g_mpi.size != 8)      \
            continue;                         \
        CAPTURE(_ci);                         \
        const auto &cfg = g_configs[_ci];     \
        auto m = g_full[_ci];                 \
        SUBCASE(cfg.name) { __VA_ARGS__ }     \
    }
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();

    // Build one full-pipeline mesh per config.
    for (int ic = 0; ic < N_CONFIGS; ic++)
    {
        // Ball2 (index 4) is only tested with np=8 — skip the expensive
        // build at other rank counts to avoid wasting ~60s on serial
        // read+partition of a 958K-cell 3D mesh that no test will use.
        if (ic == 4 && g_mpi.size != 8)
            continue;
        g_full[ic] = buildFullMesh(ic, g_configs[ic]);
    }

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int res = ctx.run();

    for (auto &m : g_full)
        m.reset();
    g_edges.reset();
    MPI_Finalize();
    return res;
}

TEST_CASE("Pipeline: all five published relation groups are Local"){FOR_EACH_MESH_CONFIG({auto read=m->AcquireRead(FullRequirements());auto v=read.HostView();for(size_t slot=0;slot<size_t(MeshRelation::Count);++slot)if(v.HasRelation(MeshRelation(slot)))CHECK(v.RelationState(MeshRelation(slot))==Adj_PointToLocal); })} TEST_CASE("Pipeline: global cell count matches expected"){FOR_EACH_MESH_CONFIG({ CHECK(m->NumCellGlobal() == cfg.expectedCells); })} TEST_CASE("Pipeline: global bnd count matches expected"){FOR_EACH_MESH_CONFIG({if(cfg.expectedBnds>=0)CHECK(m->NumBndGlobal()==cfg.expectedBnds); })} TEST_CASE("InterpolateFace: face count is positive"){FOR_EACH_MESH_CONFIG({ CHECK(m->NumFaceGlobal() > 0); })} TEST_CASE("InterpolateFace: face2cell has exactly 2 entries per face"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I f=0;f<v.NumFace();++f)CHECK(v.FaceCells(f).size()==2); })} TEST_CASE("InterpolateFace: face2cell owner is a valid local cell"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I f=0;f<v.NumFace();++f){CHECK(v.FaceCells(f)[0]>=0);CHECK(v.FaceCells(f)[0]<v.NumCellProc());} })} TEST_CASE("InterpolateFace: face2node indices in valid range"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I f=0;f<v.NumFace();++f)for(auto n:v.FaceNodes(f)){CHECK(n>=0);CHECK(n<v.NumNodeProc());} })} TEST_CASE("InterpolateFace: cell2face row sizes match expected face count"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I c=0;c<v.NumCell();++c)CHECK(v.CellFaces(c).size()==size_t(v.GetCellElement(c).GetNumFaces())); })} TEST_CASE("InterpolateFace: cell2face entries are valid face indices"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I c=0;c<v.NumCell();++c)for(auto f:v.CellFaces(c)){CHECK(f>=0);CHECK(f<v.NumFaceProc());} })} TEST_CASE("InterpolateFace: bnd2face maps to valid faces"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I b=0;b<v.NumBnd();++b)for(auto f:v.Relation(MeshRelation::BoundaryFaces,b)){CHECK(f>=0);CHECK(f<v.NumFaceProc());} })} TEST_CASE("InterpolateFace: face element types are valid"){FOR_EACH_MESH_CONFIG({for(I f=0;f<m->NumFace();++f)CHECK(m->GetFaceElement(f).type!=Elem::UnknownElem); })} TEST_CASE("InterpolateFace: independent element slots and exact back-face cones agree")
{
    FOR_EACH_MESH_CONFIG({ FaceIncidence(*m); })
}

static uint8_t FaceFrame(const UnstructuredMeshView &v, I cell, I face)
{
    const auto cn = v.CellNodes(cell), fn = v.FaceNodes(face);
    const auto cb = v.CellNodePeriodicBits(cell), fb = v.FaceNodePeriodicBits(face);
    std::optional<uint8_t> result;
    for (size_t j = 0; j < fn.size(); ++j)
    {
        auto it = std::find(cn.begin(), cn.end(), fn[j]);
        REQUIRE(it != cn.end());
        const auto frame = uint8_t(cb[size_t(it - cn.begin())]) ^ uint8_t(fb[j]);
        if (result)
            CHECK(frame == *result);
        else
            result = frame;
    }
    REQUIRE(result);
    return *result;
}
TEST_CASE("Periodic: face pbi has uniform XOR between owner and neighbor (collaborating check)"){
    FOR_EACH_MESH_CONFIG({if(!cfg.periodic)continue;auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();
    for(I f=0;f<v.NumFace();++f){auto parents=v.FaceCells(f);REQUIRE(parents[0]>=0);FaceFrame(v,parents[0],f);if(parents[1]>=0)FaceFrame(v,parents[1],f);} })} TEST_CASE("Periodic: face2nodePbi is consistent with owner cell's cell2nodePbi")
{
    FOR_EACH_MESH_CONFIG({if(!cfg.periodic)continue;auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();
    for(I cell=0;cell<v.NumCell();++cell)for(size_t j=0;j<v.CellFaces(cell).size();++j)
        CHECK(FaceFrame(v,cell,v.CellFaces(cell)[j])==uint8_t(v.CellFacePeriodicBits(cell)[j])); })
}
static std::set<std::pair<I, uint8_t>> OutputImages(UnstructuredMesh &mesh)
{
    auto r = mesh.AcquireRead(FullRequirements());
    auto v = r.HostView();
    auto nodes = mesh.Entities(EntityKind::Node);
    std::set<std::pair<I, uint8_t>> local;
    for (I n = 0; n < nodes.NumOwned(); ++n)
        local.emplace(nodes.LocalToGlobal(n), 0);
    for (I c = 0; c < v.NumCell(); ++c)
        for (size_t j = 0; j < v.CellNodes(c).size(); ++j)
            local.emplace(nodes.LocalToGlobal(v.CellNodes(c)[j]), uint8_t(v.CellNodePeriodicBits(c)[j]));
    std::vector<I> flat;
    for (const auto &[node, bits] : local)
    {
        flat.push_back(node);
        flat.push_back(bits);
    }
    int count = int(flat.size());
    std::vector<int> counts(size_t(g_mpi.size)), offsets(size_t(g_mpi.size) + 1, 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, g_mpi.comm);
    for (int p = 0; p < g_mpi.size; ++p)
        offsets[size_t(p) + 1] = offsets[size_t(p)] + counts[size_t(p)];
    std::vector<I> global(size_t(offsets.back()));
    MPI_Allgatherv(flat.data(), count, DNDS_MPI_INDEX, global.data(), counts.data(), offsets.data(), DNDS_MPI_INDEX, g_mpi.comm);
    std::set<std::pair<I, uint8_t>> result;
    for (size_t i = 0; i < global.size(); i += 2)
        result.emplace(global[i], uint8_t(global[i + 1]));
    return result;
}
TEST_CASE("Periodic: native output image count matches independent node-frame keys"){
    FOR_EACH_MESH_CONFIG({if(!cfg.periodic)continue;const auto keys=OutputImages(*m);auto output=NativeMeshOutput::Build(*m->CurrentNativeState(),g_mpi);CHECK(output.globalNodes==I(keys.size()));
    auto read=output.coordinates.Read();auto nodes=m->Entities(EntityKind::Node);
    std::vector<tPoint> expected,actual;
    for(const auto &[global,bits]:keys)if(nodes.Owner(global)==g_mpi.rank)
    {auto row=nodes.GlobalToLocal(global);REQUIRE(row);expected.push_back(m->Periodic().GetCoordByBits(m->Coord(*row),NodePeriodicBits{bits}));}
    for(size_t row=0;row<output.coordinates.Rows();++row){tPoint p;for(int d=0;d<3;++d)p[d]=read.View()(row,size_t(d));actual.push_back(p);}
    auto order=[](const tPoint&a,const tPoint&b){return std::array<DNDS::real,3>{a[0],a[1],a[2]}<std::array<DNDS::real,3>{b[0],b[1],b[2]};};
    std::sort(expected.begin(),expected.end(),order);std::sort(actual.begin(),actual.end(),order);REQUIRE(expected.size()==actual.size());for(size_t j=0;j<actual.size();++j)CHECK(actual[j].isApprox(expected[j],1e-13)); })} TEST_CASE("Periodic: ordered incidence PBI and physical geometry survive native reassembly"){
    FOR_EACH_MESH_CONFIG({if(!cfg.periodic)continue;auto data=NativeMeshAssembler::CaptureOwned(*m->CurrentNativeState(),g_mpi);auto next=AssembleNativeMesh(g_mpi,std::move(data),1);
    auto old=m->AcquireRead(FullRequirements());auto fresh=next->Snapshot(R::Device::Host()).AcquireRead();auto a=old.HostView(),b=fresh.HostView();
    REQUIRE(a.NumFaceProc()==b.NumFaceProc());for(I f=0;f<a.NumFaceProc();++f){CHECK(Global(m->Entities(EntityKind::Node),a.FaceNodes(f))==Global(next->Entities(EntityKind::Node),b.FaceNodes(f)));auto ab=a.FaceNodePeriodicBits(f),bb=b.FaceNodePeriodicBits(f);REQUIRE(ab.size()==bb.size());for(size_t j=0;j<ab.size();++j)CHECK(uint8_t(ab[j])==uint8_t(bb[j]));} })} TEST_CASE("bnd2cell: independent boundary facet incidence on all mesh configs"){
    FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();
    for(I boundary=0;boundary<v.NumBnd();++boundary){const auto bnodes=v.BoundaryNodes(boundary);const auto parents=v.BoundaryCells(boundary);REQUIRE(parents.size()==2);REQUIRE(parents[0]>=0);for(auto parent:parents)if(parent!=UnInitIndex){REQUIRE(parent>=0);REQUIRE(parent<v.NumCellProc());auto cone=v.CellNodes(parent);for(auto node:bnodes)CHECK(std::find(cone.begin(),cone.end(),node)!=cone.end());}} })} TEST_CASE("BuildGhostPrimary: sparse closure and all global relations survive native reassembly")
{
    FOR_EACH_MESH_CONFIG({auto next=AssembleNativeMesh(g_mpi,NativeMeshAssembler::CaptureOwned(*m->CurrentNativeState(),g_mpi),1);
    for(auto kind:{EntityKind::Cell,EntityKind::Node,EntityKind::Face,EntityKind::Bnd}){const auto a=m->Entities(kind),b=next->Entities(kind);CHECK(a.NumOwned()==b.NumOwned());CHECK(a.NumGhost()==b.NumGhost());std::set<I>x,y;for(I row=a.NumOwned();row<a.NumLocal();++row)x.insert(a.LocalToGlobal(row));for(I row=b.NumOwned();row<b.NumLocal();++row)y.insert(b.LocalToGlobal(row));CHECK(x==y);}
    auto a=m->AcquireRead(FullRequirements());auto b=next->Snapshot(R::Device::Host()).AcquireRead();const auto av=a.HostView(),bv=b.HostView();const auto kinds=Bindings();
    for(size_t slot=0;slot<size_t(MeshRelation::Count);++slot)if(av.HasRelation(MeshRelation(slot))){const auto from=m->Entities(kinds[slot].from);for(I row=0;row<from.NumOwned();++row)CHECK(Global(m->Entities(kinds[slot].to),av.Relation(MeshRelation(slot),row))==Global(next->Entities(kinds[slot].to),bv.Relation(MeshRelation(slot),row)));}
    FaceIncidence(*m); CheckRequiredBoundaryCoverage(*m); })
}

static std::map<I, std::set<I>> LocalNodeStars(const UnstructuredMeshView &v, const EntitySpace &cells, const EntitySpace &nodes)
{
    std::map<I, std::set<I>> result;
    for (I cell = 0; cell < v.NumCellProc(); ++cell)
        for (auto node : v.CellNodes(cell))
            result[nodes.LocalToGlobal(node)].insert(cells.LocalToGlobal(cell));
    return result;
}
TEST_CASE("Benchmark: native inverse compose and face topology with verified results")
{
    const auto &cfg = g_configs[2];
    AutoAppendName2ID names;
    auto owned = ReadNativeCGNSOwned(g_mpi, meshPath(cfg.file), cfg.dim, [&](const std::string &name)
                                     { return names(name); }, 0, {1, true}, PeriodicOptions(cfg), NativeSerialDistribution::ReaderOwned);
    NativePartitionOptions partition;
    partition.metisUfactor = 30;
    partition.metisSeed = 42;
    partition.metisNcuts = 1;
    owned = NativeMeshPreparation::PartitionOwned(owned, g_mpi, partition).owned;
    const auto recipe = DefaultNativeMeshGhostSpec(owned, 1, true);
    auto source = NativeMeshAssembler::Build(g_mpi, std::move(owned), recipe);
    auto sourceMesh = UnstructuredMesh::FromNative(g_mpi, source);
    REQUIRE(sourceMesh->NumCellGlobal() == 20816);
    double inverseTime = 0, composeTime = 0, topologyTime = 0;
    for (int run = 0; run < 3; ++run)
    {
        auto cone = sourceMesh->ReadNativeRelation(Adj::Cell2Node);
        MPI_Barrier(g_mpi.comm);
        const auto t0 = MPI_Wtime();
        auto inverse = NativeConnectivity::Inverse(cone, g_mpi);
        MPI_Barrier(g_mpi.comm);
        inverseTime += MPI_Wtime() - t0;
        auto complete = inverse.WithGhostRows(RelationGhostRoute(source->Entities(EntityKind::Node)));
        auto inverseRead = inverse.Read();
        auto retained = sourceMesh->AcquireRead();
        auto v = retained.HostView();
        const auto cells = source->Entities(EntityKind::Cell), nodes = source->Entities(EntityKind::Node);
        const auto stars = LocalNodeStars(v, cells, nodes);
        for (I node = 0; node < nodes.NumOwned(); ++node)
            CHECK(Set(inverseRead.GlobalRow(node)) == stars.at(nodes.LocalToGlobal(node)));
        MPI_Barrier(g_mpi.comm);
        const auto t1 = MPI_Wtime();
        auto adjacency = NativeConnectivity::Compose(cone, complete.Read(), true);
        MPI_Barrier(g_mpi.comm);
        composeTime += MPI_Wtime() - t1;
        auto adjacencyRead = adjacency.Read();
        for (I cell = 0; cell < cells.NumOwned(); ++cell)
        {
            std::set<I> expected;
            for (auto node : v.CellNodes(cell))
            {
                const auto &star = stars.at(nodes.LocalToGlobal(node));
                expected.insert(star.begin(), star.end());
            }
            expected.erase(cells.LocalToGlobal(cell));
            CHECK(Set(adjacencyRead.GlobalRow(cell)) == expected);
        }
        MPI_Barrier(g_mpi.comm);
        const auto t2 = MPI_Wtime();
        auto topology = PrepareNativeMeshTopology(*source, g_mpi, true, false);
        MPI_Barrier(g_mpi.comm);
        topologyTime += MPI_Wtime() - t2;
        CHECK(topology->NumGlobal(EntityKind::Cell) == 20816);
        CHECK(topology->NumGlobal(EntityKind::Face) > 0);
        auto full = UnstructuredMesh::FromNative(g_mpi, Complete(topology));
        FaceIncidence(*full);
    }
    for (auto &elapsed : std::array<std::reference_wrapper<double>, 3>{inverseTime, composeTime, topologyTime})
        MPI_Allreduce(MPI_IN_PLACE, &elapsed.get(), 1, MPI_DOUBLE, MPI_MAX, g_mpi.comm);
    if (g_mpi.rank == 0)
        std::cout << "Native setup mean over 3 runs: inverse " << inverseTime / 3 << " compose " << composeTime / 3 << " face topology " << topologyTime / 3 << std::endl;
}
TEST_CASE("N2CB: every owned node has at least one adjacent cell"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I node=0;node<v.NumNode();++node)CHECK(!v.Relation(MeshRelation::NodeCells,node).empty()); })} TEST_CASE("N2CB: owned node stars match independent local cone incidence"){
    FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();const auto cells=m->Entities(EntityKind::Cell),nodes=m->Entities(EntityKind::Node);const auto stars=LocalNodeStars(v,cells,nodes);
    for(I node=0;node<v.NumNode();++node){CHECK(Set(Global(cells,v.Relation(MeshRelation::NodeCells,node)))==stars.at(nodes.LocalToGlobal(node)));for(auto cell:v.Relation(MeshRelation::NodeCells,node)){CHECK(cell>=0);CHECK(cell<v.NumCellProc());}} })} TEST_CASE("Cell face neighbors: row sizes match cell2face"){FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I cell=0;cell<v.NumCell();++cell)CHECK(v.Relation(MeshRelation::CellFaceCells,cell).size()==v.CellFaces(cell).size()); })} TEST_CASE("Cell face neighbors: each slot names the opposite face cell or boundary sentinel"){
    FOR_EACH_MESH_CONFIG({auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I cell=0;cell<v.NumCell();++cell){const auto neighbors=v.Relation(MeshRelation::CellFaceCells,cell);for(size_t slot=0;slot<neighbors.size();++slot)CHECK(neighbors[slot]==v.CellFaceOther(cell,v.CellFaces(cell)[slot],DNDS::rowsize(slot)));} })} TEST_CASE("ConstructBndMesh: correct dimensions and native local state"){
    FOR_EACH_MESH_CONFIG({
        if (cfg.periodic)
            continue; // periodic meshes may have no real boundaries after deduplication
        auto source = nativeBoundarySource(_ci, cfg);
        UnstructuredMesh bMesh(g_mpi, source->getDim() - 1);
        source->ConstructBndMesh(bMesh);
        CHECK(bMesh.getDim() == source->getDim() - 1);
        CHECK(bMesh.NumCell() == source->NumBnd());
        CHECK(bMesh.HasNativeState());
        CHECK(bMesh.ReadNativeRelation(Adj::Cell2Node).State() == Adj_PointToLocal);
    })}

TEST_CASE("ConstructBndMesh: node2parentNode valid"){
    FOR_EACH_MESH_CONFIG({
        if (cfg.periodic)
            continue;
        auto source = nativeBoundarySource(_ci, cfg);
        UnstructuredMesh bMesh(g_mpi, source->getDim() - 1);
        source->ConstructBndMesh(bMesh);
        CHECK(bMesh.node2parentNode.size() ==
              static_cast<size_t>(bMesh.NumNode() + bMesh.NumNodeGhost()));
        for (DNDS::index i = 0; i < bMesh.NumNode(); i++)
        {
            DNDS::index pn = bMesh.node2parentNode[i];
            CHECK(pn >= 0);
            CHECK(pn < source->NumNode() + source->NumNodeGhost());
            CHECK(bMesh.Coord(i).isApprox(source->Coord(pn), 1e-13));
        }
    })}

TEST_CASE("Boundary adjacency round-trip: native detached local and global states")
{
    FOR_EACH_MESH_CONFIG(
        auto source = nativeBoundarySource(_ci, cfg);
        UnstructuredMesh bMesh(g_mpi, source->getDim() - 1);
        source->ConstructBndMesh(bMesh);
        const auto stamp = bMesh.versions.Stamp();
        auto original = bMesh.ReadNativeRelation(Adj::Cell2Node);
        REQUIRE(original.State() == Adj_PointToLocal);
        std::vector<std::vector<DNDS::index>> snapshot(size_t(original.Rows()));
        for (DNDS::index row = 0; row < original.Rows(); ++row) for (size_t slot = 0; slot < original.RowSize(row); ++slot)
            snapshot[size_t(row)]
                .push_back(original.Raw(row, slot));
        auto globals = NativeRelation::Convert(original, Adj_PointToGlobal);
        auto globalRead = globals.Read();
        CHECK(globalRead.State() == Adj_PointToGlobal);
        for (DNDS::index row = 0; row < globalRead.Rows(); ++row) for (size_t slot = 0; slot < globalRead.RowSize(row); ++slot) {
            const auto global = globalRead.Raw(row, slot);
            CHECK(global >= 0);
            CHECK(global < globalRead.Target().NumGlobal());
        } auto locals = NativeRelation::Convert(globalRead, Adj_PointToLocal, false);
        auto localRead = locals.Read();
        CHECK(localRead.State() == Adj_PointToLocal);
        CHECK(localRead.Rows() == original.Rows());
        CHECK(stamp.Valid()); // Detached conversion never toggles live query storage.
        CHECK(bMesh.ReadNativeRelation(Adj::Cell2Node).State() == Adj_PointToLocal);
        for (DNDS::index row = 0; row < localRead.Rows(); ++row) {
            REQUIRE(localRead.RowSize(row) == snapshot[size_t(row)].size());
            for (size_t slot = 0; slot < localRead.RowSize(row); ++slot)
                CHECK(localRead.Raw(row, slot) == snapshot[size_t(row)][slot]);
        })
}

// ===========================================================================
// Ball2: 3D mixed-element mesh (np=8 only)
// ===========================================================================

TEST_CASE("Ball2: mixed 3D element types")
{
    // Ball2 (config 4) is only tested with np=8
    if (g_mpi.size != 8)
        return;

    auto m = g_full[4]; // Ball2 is index 4
    DNDS_assert(m != nullptr);

    // Count element types locally
    std::map<Elem::ElemType, int> typeCounts;
    for (DNDS::index iC = 0; iC < m->NumCell(); iC++)
    {
        auto elem = m->GetCellElement(iC);
        typeCounts[elem.type]++;
    }

    // Gather all unique element types from all ranks
    std::vector<int> localTypes;
    for (const auto &[type, count] : typeCounts)
        localTypes.push_back(static_cast<int>(type));

    int nLocalTypes = localTypes.size();
    std::vector<int> allRankSizes(g_mpi.size);
    MPI_Gather(&nLocalTypes, 1, MPI_INT, allRankSizes.data(), 1, MPI_INT, 0, g_mpi.comm);

    std::vector<int> allTypes;
    std::vector<int> displs;
    int totalTypes = 0;

    if (g_mpi.rank == 0)
    {
        displs.resize(g_mpi.size);
        for (int i = 0; i < g_mpi.size; i++)
        {
            displs[i] = totalTypes;
            totalTypes += allRankSizes[i];
        }
        allTypes.resize(totalTypes);
    }

    MPI_Gatherv(localTypes.data(), nLocalTypes, MPI_INT,
                allTypes.data(), allRankSizes.data(), displs.data(), MPI_INT, 0, g_mpi.comm);

    // Build set of unique types (on rank 0)
    std::set<Elem::ElemType> uniqueTypes;
    if (g_mpi.rank == 0)
    {
        for (int typeInt : allTypes)
            uniqueTypes.insert(static_cast<Elem::ElemType>(typeInt));
    }

    // Broadcast unique types to all ranks
    int nUnique = uniqueTypes.size();
    MPI_Bcast(&nUnique, 1, MPI_INT, 0, g_mpi.comm);
    std::vector<int> uniqueTypesVec;
    if (g_mpi.rank == 0)
    {
        for (auto type : uniqueTypes)
            uniqueTypesVec.push_back(static_cast<int>(type));
    }
    uniqueTypesVec.resize(nUnique);
    MPI_Bcast(uniqueTypesVec.data(), nUnique, MPI_INT, 0, g_mpi.comm);

    // All ranks iterate over the same types in the same order
    std::map<Elem::ElemType, int> globalCounts;
    for (int typeInt : uniqueTypesVec)
    {
        Elem::ElemType type = static_cast<Elem::ElemType>(typeInt);
        int localCount = typeCounts.count(type) ? typeCounts[type] : 0;
        int globalCount;
        MPI_Reduce(&localCount, &globalCount, 1, MPI_INT, MPI_SUM, 0, g_mpi.comm);
        if (g_mpi.rank == 0)
            globalCounts[type] = globalCount;
    }

    if (g_mpi.rank == 0)
    {
        std::cout << "\nBall2 element type counts:" << std::endl;
        for (const auto &[type, count] : globalCounts)
        {
            std::cout << "  Type " << static_cast<int>(type) << ": " << count << std::endl;
        }

        // Verify expected total
        int total = 0;
        for (const auto &[type, count] : globalCounts)
            total += count;
        CHECK(total == 958994);
    }
}

// ===========================================================================

TEST_CASE("Native output connectivity: arrays populated correctly"){
    FOR_EACH_MESH_CONFIG({auto output=NativeMeshOutput::Build(*m->CurrentNativeState(),g_mpi);CHECK(output.offsets.size()==size_t(m->NumCell()+1));CHECK(output.types.size()==size_t(m->NumCell()));const I local=I(output.connectivity.size());I prefix=0;MPI_Exscan(&local,&prefix,1,MPI_INT64_T,MPI_SUM,g_mpi.comm);if(g_mpi.rank==0)prefix=0;CHECK(output.offsets.front()==prefix);CHECK(output.offsets.back()==prefix+local);for(size_t j=1;j<output.offsets.size();++j)CHECK(output.offsets[j]>=output.offsets[j-1]);for(auto type:output.types)CHECK(type!=0);for(auto point:output.connectivity){CHECK(point>=0);CHECK(point<output.globalNodes);}CHECK(output.globalCells==cfg.expectedCells); })} TEST_CASE("Native output connectivity: repeat build replaces buffers without accumulating entries"){
    FOR_EACH_MESH_CONFIG({auto a=NativeMeshOutput::Build(*m->CurrentNativeState(),g_mpi),b=NativeMeshOutput::Build(*m->CurrentNativeState(),g_mpi);CHECK(a.connectivity==b.connectivity);CHECK(a.offsets==b.offsets);CHECK(a.types==b.types);CHECK(a.pointSourceRows==b.pointSourceRows);CHECK(a.coordinates.Rows()==b.coordinates.Rows());auto ar=a.coordinates.Read(),br=b.coordinates.Read();for(size_t row=0;row<a.coordinates.Rows();++row)for(size_t d=0;d<3;++d)CHECK(ar.View()(row,d)==br.View()(row,d)); })} TEST_CASE("ReorderLocalCells: all published relation groups remain local"){
    FOR_EACH_MESH_CONFIG({m->ReorderLocalCells(2);auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(size_t slot=0;slot<size_t(MeshRelation::Count);++slot)if(v.HasRelation(MeshRelation(slot)))CHECK(v.RelationState(MeshRelation(slot))==Adj_PointToLocal); })} TEST_CASE("ReorderLocalCells: cell count preserved"){FOR_EACH_MESH_CONFIG({const auto old=m->NumCellGlobal();m->ReorderLocalCells(2);CHECK(m->NumCellGlobal()==old);CHECK(old==cfg.expectedCells); })} TEST_CASE("ReorderLocalCells: partition starts valid"){FOR_EACH_MESH_CONFIG({m->ReorderLocalCells(2);CHECK(m->NLocalParts()==2);CHECK(m->LocalPartStart(0)==0);CHECK(m->LocalPartEnd(1)==m->NumCell());for(int part=0;part<2;++part)CHECK(m->LocalPartStart(part)<=m->LocalPartEnd(part)); })} TEST_CASE("ReorderLocalCells: cell2node still valid"){FOR_EACH_MESH_CONFIG({m->ReorderLocalCells(2);auto r=m->AcquireRead(FullRequirements());auto v=r.HostView();for(I c=0;c<v.NumCell();++c)for(auto n:v.CellNodes(c)){CHECK(n>=0);CHECK(n<v.NumNodeProc());}FaceIncidence(*m); })} TEST_CASE("ReorderLocalCells: face count preserved"){FOR_EACH_MESH_CONFIG({const auto old=m->NumFaceGlobal();m->ReorderLocalCells(2);CHECK(m->NumFaceGlobal()==old); })} TEST_CASE("AdjFacial detached round-trip: face2node and face2cell identity"){FOR_EACH_MESH_CONFIG({ Roundtrip(*m, {MeshRelation::FaceNodes, MeshRelation::FaceCells}); })} TEST_CASE("AdjC2F detached round-trip: cell2face and bnd2face identity"){FOR_EACH_MESH_CONFIG({ Roundtrip(*m, {MeshRelation::CellFaces, MeshRelation::BoundaryFaces}); })} TEST_CASE("AdjN2CB detached round-trip: node2cell and node2bnd identity"){FOR_EACH_MESH_CONFIG({ Roundtrip(*m, {MeshRelation::NodeCells, MeshRelation::NodeBoundaries}); })} TEST_CASE("AdjC2CFace detached round-trip: ordered face-neighbor identity"){FOR_EACH_MESH_CONFIG({ Roundtrip(*m, {MeshRelation::CellFaceCells}); })} TEST_CASE("AdjPrimary detached round-trip and serial output preserve source publication")
{
    FOR_EACH_MESH_CONFIG({const auto stamp=m->versions.Stamp();Roundtrip(*m,{MeshRelation::CellNodes,MeshRelation::CellCells,MeshRelation::BoundaryNodes,MeshRelation::BoundaryCells});UnstructuredMeshSerialRW output(m,0);output.BuildSerialOut();CHECK(output.dataIsSerialOut);CHECK(output.nativeSerialOutput);CHECK(stamp.Valid()); })
}
static ssp<UnstructuredMesh> NativeElevationFixture(const MeshConfig &cfg)
{
    NativePeriodicImportOptions periodic;
    periodic.frames.translation[1].map() = cfg.translation1;
    periodic.frames.translation[2].map() = cfg.translation2;
    periodic.frames.translation[3].map() = cfg.translation3;
    periodic.tolerance = 1e-8;
    PartitionOptions options;
    options.metisUfactor = 30;
    options.metisSeed = 42;
    options.metisNcuts = 1;
    AutoAppendName2ID names;
    return ReadNativeMeshFromCGNS(g_mpi, cfg.dim, meshPath(cfg.file), options, [&](const std::string &name)
                                  { return names(name); }, cfg.periodic ? std::optional<NativePeriodicImportOptions>(periodic) : std::nullopt);
}

// Elevation/Bisection tests
// ===========================================================================

TEST_CASE("Elevation: O2 mesh cell count equals O1 cell count")
{
    // Only test with config 0 (UniformSquare_10) for now
    const auto &cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = NativeElevationFixture(cfg);

    // Build O2 mesh via elevation
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Set up MPI state for mO2

    DNDS::index nCellO1 = mO1->NumCellGlobal();
    DNDS::index nCellO2 = mO2->NumCellGlobal();

    // Elevation preserves cell count
    CHECK(nCellO2 == nCellO1);

    // All cells should be O2 type
    for (DNDS::index iC = 0; iC < mO2->NumCell(); iC++)
    {
        auto elem = mO2->GetCellElement(iC);
        CHECK(elem.GetOrder() == 2);
    }
}

TEST_CASE("Elevation: O2 mesh has more nodes than O1")
{
    const auto &cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = NativeElevationFixture(cfg);

    // Build O2 mesh via elevation
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Set up MPI state for mO2

    DNDS::index nNodeO1 = mO1->NumNodeGlobal();
    DNDS::index nNodeO2 = mO2->NumNodeGlobal();

    // Elevation adds nodes
    CHECK(nNodeO2 > nNodeO1);
}

TEST_CASE("Bisection: O1 mesh has more cells than O2")
{
    const auto &cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = NativeElevationFixture(cfg);

    // Build O2 mesh
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Build O1 mesh via bisection

    auto mO1B = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO1B->BuildBisectO1FormO2(*mO2);

    DNDS::index nCellO2 = mO2->NumCellGlobal();
    DNDS::index nCellO1B = mO1B->NumCellGlobal();

    // Bisection increases cell count
    CHECK(nCellO1B > nCellO2);

    // All cells should be O1 (linear) element types
    for (DNDS::index iC = 0; iC < mO1B->NumCell(); iC++)
    {
        auto elem = mO1B->GetCellElement(iC);
        CHECK(elem.GetOrder() == 1);
    }
}

TEST_CASE("Bisection: global node count is preserved from O2 mesh")
{
    const auto &cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = NativeElevationFixture(cfg);

    // Build O2 mesh
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Build O1 mesh via bisection

    auto mO1B = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO1B->BuildBisectO1FormO2(*mO2);

    // No new nodes created during bisection
    DNDS::index nNodeO2 = mO2->NumNodeGlobal();
    DNDS::index nNodeO1B = mO1B->NumNodeGlobal();
    CHECK(nNodeO1B == nNodeO2);
}

TEST_CASE("Elevation+Bisection: exact cell count progression")
{
    // UniformSquare_10: 100 Quad4 cells
    // After elevation: 100 Quad9 cells (same count, higher order)
    // After bisection: 400 Quad4 cells (4 sub-cells per Quad9)
    const auto &cfg = g_configs[0];

    // Build fresh O1 mesh
    auto mO1 = NativeElevationFixture(cfg);

    // Build O2 mesh
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Build O1 mesh via bisection

    auto mO1B = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO1B->BuildBisectO1FormO2(*mO2);

    DNDS::index nCellOrig = mO1->NumCellGlobal();
    DNDS::index nCellElevated = mO2->NumCellGlobal();
    DNDS::index nCellBisected = mO1B->NumCellGlobal();

    CHECK(nCellOrig == 100);
    CHECK(nCellElevated == 100);
    CHECK(nCellBisected == 400);
}

// ===========================================================================
// Elevation + Boundary/Internal Smooth tests (NACA0012_H2, config 2)
// ===========================================================================

/**
 * \brief Build an O2 mesh with full face interpolation, ready for smoothing.
 *
 * Sequence: read O1 -> partition -> ghost -> elevate -> rebuild topology
 * -> face interpolation -> N2CB ghost.
 */
static ssp<UnstructuredMesh> buildO2MeshWithFaces(
    const MeshConfig &cfg, ssp<UnstructuredMesh> &mO1Out)
{
    // Build O1 mesh
    auto mO1 = NativeElevationFixture(cfg);

    // Build O2 mesh via elevation
    auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
    mO2->BuildO2FromO1Elevation(*mO1);

    // Rebuild full topology for O2

    mO2->PrepareNativeTopology();

    mO1Out = mO1;
    return mO2;
}

TEST_CASE("BoundarySmooth: NACA0012 moves boundary O2 nodes")
{
    // NACA0012_H2 is config 2 — non-periodic, curved airfoil boundary
    const auto &cfg = g_configs[2];

    ssp<UnstructuredMesh> mO1;
    auto mO2 = buildO2MeshWithFaces(cfg, mO1);

    // Snapshot pre-smooth coordinates
    std::vector<tPoint> preSmooth(mO2->NumNode());
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        preSmooth[i] = mO2->Coord(i);

    // Run boundary smooth — all external BCs are smoothed
    mO2->ElevatedNodesGetBoundarySmooth(
        [](Geom::t_index bndId)
        { return Geom::FaceIDIsExternalBC(bndId); });

    // The retained native smoothing workspace records actual boundary movement.
    CHECK(mO2->BoundarySmoothingWorkspace()->TotalMoved() > 0);

    // The displacement array covers every owned node.
    CHECK(mO2->BoundarySmoothingWorkspace()->Displacement().owned.Rows() == size_t(mO2->NumNode()));

    // All coordinates must be finite (no NaN/Inf)
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
    {
        for (int d = 0; d < 3; d++)
            CHECK(std::isfinite(mO2->Coord(i)(d)));
    }
}

TEST_CASE("BoundarySmooth: no NaN in displacement field")
{
    const auto &cfg = g_configs[2]; // NACA0012_H2

    ssp<UnstructuredMesh> mO1;
    auto mO2 = buildO2MeshWithFaces(cfg, mO1);

    mO2->ElevatedNodesGetBoundarySmooth(
        [](Geom::t_index bndId)
        { return Geom::FaceIDIsExternalBC(bndId); });

    // Prescribed native constraints have finite displacement values.
    DNDS::index nMovedLocal = 0;
    auto diagnostics = mO2->BoundarySmoothingWorkspace();
    auto displacement = diagnostics->Displacement().owned.Read();
    auto roles = diagnostics->Constraints().owned.Read();
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
    {
        tPoint disp;
        for (int component = 0; component < 3; ++component)
            disp[component] = displacement.View()(size_t(i), size_t(component));
        // Native explicit roles replace legacy displacement sentinels.
        if (NativeSmoothingConstraint(roles.View()(size_t(i), 0)) == NativeSmoothingConstraint::Prescribed)
        {
            for (int d = 0; d < 3; d++)
                CHECK(std::isfinite(disp(d)));
            nMovedLocal++;
        }
    }

    DNDS::index nMovedGlobal = 0;
    MPI_Allreduce(&nMovedLocal, &nMovedGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    CHECK(nMovedGlobal > 0);
}

TEST_CASE("InternalSmooth V2: coordinates remain finite after solve")
{
    const auto &cfg = g_configs[2]; // NACA0012_H2

    ssp<UnstructuredMesh> mO1;
    auto mO2 = buildO2MeshWithFaces(cfg, mO1);

    // Set up elevation info for fast convergence (small problem)
    mO2->elevationInfo.nIter = 10;
    mO2->elevationInfo.nSearch = 20;
    mO2->elevationInfo.RBFRadius = 5.0;

    // Run boundary smooth
    mO2->ElevatedNodesGetBoundarySmooth(
        [](Geom::t_index bndId)
        { return Geom::FaceIDIsExternalBC(bndId); });

    if (mO2->BoundarySmoothingWorkspace()->TotalMoved() == 0)
        return; // nothing to smooth

    // Save pre-smooth coords
    std::vector<tPoint> preSmoothCoords(mO2->NumNode());
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        preSmoothCoords[i] = mO2->Coord(i);

    // Run internal smooth V2
    mO2->ElevatedNodesSolveInternalSmoothV2();

    // All coordinates must be finite
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        for (int d = 0; d < 3; d++)
            CHECK(std::isfinite(mO2->Coord(i)(d)));

    // At least some coordinates should have changed
    DNDS::index nChangedLocal = 0;
    for (DNDS::index i = 0; i < mO2->NumNode(); i++)
        if ((mO2->Coord(i) - preSmoothCoords[i]).norm() > 1e-15)
            nChangedLocal++;

    DNDS::index nChangedGlobal = 0;
    MPI_Allreduce(&nChangedLocal, &nChangedGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    CHECK(nChangedGlobal > 0);
}

TEST_CASE("Elevation: O2 coordinates have no NaN")
{
    // Test on all non-periodic configs (0, 2)
    for (int ci : {0, 2})
    {
        const auto &cfg = g_configs[ci];
        SUBCASE(cfg.name)
        {
            auto mO1 = NativeElevationFixture(cfg);

            auto mO2 = std::make_shared<UnstructuredMesh>(g_mpi, cfg.dim);
            mO2->BuildO2FromO1Elevation(*mO1);

            // All O2 coordinates must be finite
            for (DNDS::index i = 0; i < mO2->NumNode(); i++)
                for (int d = 0; d < 3; d++)
                    CHECK(std::isfinite(mO2->Coord(i)(d)));
        }
    }
}

// ===========================================================================

static ssp<UnstructuredMesh> buildMeshWithGhostLayers(const MeshConfig &cfg, int layers) { return ReadConfigured(cfg, layers); }
TEST_CASE("MultiLayerGhost: ghost counts increase with layer count")
{
    std::array<I, 3> previous{-1, -1, -1};
    for (int layers = 1; layers <= 3; ++layers)
    {
        auto mesh = buildMeshWithGhostLayers(g_configs[0], layers);
        const std::array<I, 3> counts{mesh->NumCellGhost(), mesh->NumNodeGhost(), mesh->NumBndGhost()};
        for (size_t j = 0; j < 3; ++j)
            CHECK(counts[j] >= previous[j]);
        if (g_mpi.size >= 2 && layers > 1)
        {
            CHECK(counts[0] > previous[0]);
            CHECK(counts[1] > previous[1]);
        }
        previous = counts;
    }
}
TEST_CASE("MultiLayerGhost: 2-layer cell2cell neighbors are all local")
{
    auto mesh = buildMeshWithGhostLayers(g_configs[0], 2);
    auto r = mesh->AcquireRead(FullRequirements());
    auto v = r.HostView();
    for (I cell = 0; cell < v.NumCell(); ++cell)
        for (auto first : v.CellCells(cell))
        {
            REQUIRE(first >= 0);
            REQUIRE(first < v.NumCellProc());
            for (auto second : v.CellCells(first))
            {
                CHECK(second >= 0);
                CHECK(second < v.NumCellProc());
            }
        }
}
TEST_CASE("MultiLayerGhost: global cell count preserved across layers")
{
    for (int layers = 1; layers <= 3; ++layers)
    {
        auto mesh = buildMeshWithGhostLayers(g_configs[0], layers);
        CHECK(mesh->NumCellGlobal() == 100);
    }
}

static ssp<UnstructuredMesh> TinyEdges()
{
    if (g_edges)
        return g_edges;
    const I nc = g_mpi.size == 1 ? 2 : (g_mpi.rank == 0 || g_mpi.rank == g_mpi.size - 1 ? 1 : 0);
    const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, g_mpi, g_mpi.rank == 0 ? 5 : 0);
    const auto cells = nodes.CreateSibling(EntityKind::Cell, nc), bnds = nodes.CreateSibling(EntityKind::Bnd, g_mpi.rank == 0 ? 6 : 0);
    const std::array<std::array<I, 4>, 2> cones{{{{0, 1, 2, 3}}, {{0, 2, 1, 4}}}};
    const std::array<std::array<I, 3>, 6> boundary{{{{0, 1, 3}}, {{1, 2, 3}}, {{2, 0, 3}}, {{0, 2, 4}}, {{2, 1, 4}}, {{1, 0, 4}}}};
    std::vector<std::vector<I>> cr, br;
    for (I row = 0; row < nc; ++row)
    {
        auto global = cells.LocalToGlobal(row);
        cr.emplace_back(cones[size_t(global)].begin(), cones[size_t(global)].end());
    }
    if (g_mpi.rank == 0)
        for (const auto &r : boundary)
            br.emplace_back(r.begin(), r.end());
    R::Array<DNDS::real> xyz(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
    {
        auto w = xyz.Write();
        const std::array<tPoint, 5> values{tPoint{0, 0, 0}, tPoint{1, 0, 0}, tPoint{0, 1, 0}, tPoint{0, 0, 1}, tPoint{0, 0, -1}};
        for (size_t row = 0; row < xyz.Rows(); ++row)
            for (size_t d = 0; d < 3; ++d)
                w.View()(row, d) = values[row][Eigen::Index(d)];
    }
    R::Array<ElemInfo> ci(R::RowLayout::Dense(size_t(nc), 1)), bi(R::RowLayout::Dense(size_t(bnds.NumOwned()), 1));
    {
        auto w = ci.Write();
        for (size_t row = 0; row < ci.Rows(); ++row)
            w.View()(row, 0) = ElemInfo{Elem::Tet4, INTERNAL_ZONE};
    }
    {
        auto w = bi.Write();
        for (size_t row = 0; row < bi.Rows(); ++row)
            w.View()(row, 0) = ElemInfo{Elem::Tri3, BC_ID_DEFAULT_WALL};
    }
    NativeMeshInput input{g_mpi, 3, cells, nodes, bnds, Rows(cr), Rows(br), xyz, ci, bi, {}};
    auto state = ConstructNativeMesh(std::move(input), {1, true});
    state = PrepareNativeMeshTopology(*state, g_mpi, true, true);
    g_edges = UnstructuredMesh::FromNative(g_mpi, state);
    return g_edges;
}
TEST_CASE("InterpolateEdge: edge count is positive")
{
    auto mesh = TinyEdges();
    CHECK(mesh->Entities(EntityKind::Edge).NumGlobal() == 9);
    CHECK(mesh->NumCellGlobal() == 2);
    CHECK(mesh->NumFaceGlobal() == 7);
}
TEST_CASE("InterpolateEdge: edge2node indices in valid range")
{
    auto mesh = TinyEdges();
    auto q = MeshReadRequirements::CellFlux();
    q.edges = true;
    auto r = mesh->AcquireRead(q);
    auto v = r.HostView();
    for (I edge = 0; edge < v.NumEdge(); ++edge)
    {
        CHECK(v.EdgeNodes(edge).size() == 2);
        for (auto node : v.EdgeNodes(edge))
        {
            CHECK(node >= 0);
            CHECK(node < v.NumNodeProc());
        }
    }
}
TEST_CASE("InterpolateEdge: edge2cell has independently expected parents per edge")
{
    auto mesh = TinyEdges();
    auto q = MeshReadRequirements::CellFlux();
    q.edges = true;
    auto r = mesh->AcquireRead(q);
    auto v = r.HostView();
    auto cells = mesh->Entities(EntityKind::Cell);
    const std::array<std::set<I>, 2> vertices{{{0, 1, 2, 3}, {0, 1, 2, 4}}};
    auto nodes = mesh->Entities(EntityKind::Node);
    for (I edge = 0; edge < v.NumEdge(); ++edge)
    {
        auto endpoints = Global(nodes, v.EdgeNodes(edge));
        std::set<I> expected;
        for (I cell = 0; cell < 2; ++cell)
            if (vertices[size_t(cell)].count(endpoints[0]) && vertices[size_t(cell)].count(endpoints[1]))
                expected.insert(cell);
        CHECK(Set(Global(cells, v.EdgeCells(edge))) == expected);
    }
}
TEST_CASE("InterpolateEdge: cell2edge row sizes match expected edge count")
{
    auto mesh = TinyEdges();
    auto q = MeshReadRequirements::CellFlux();
    q.edges = true;
    auto r = mesh->AcquireRead(q);
    auto v = r.HostView();
    for (I cell = 0; cell < v.NumCell(); ++cell)
        CHECK(v.CellEdges(cell).size() == size_t(v.GetCellElement(cell).GetNumEdges()));
}
TEST_CASE("InterpolateEdge: cell2edge entries have exact element-slot endpoint IDs")
{
    auto mesh = TinyEdges();
    auto q = MeshReadRequirements::CellFlux();
    q.edges = true;
    auto r = mesh->AcquireRead(q);
    auto v = r.HostView();
    auto nodes = mesh->Entities(EntityKind::Node);
    for (I cell = 0; cell < v.NumCell(); ++cell)
    {
        auto element = v.GetCellElement(cell);
        const auto cone = v.CellNodes(cell);
        for (size_t slot = 0; slot < v.CellEdges(cell).size(); ++slot)
        {
            const auto edge = v.CellEdges(cell)[slot];
            REQUIRE(edge >= 0);
            REQUIRE(edge < v.NumEdgeProc());
            std::vector<I> expected(2);
            element.ExtractEdgeNodes(DNDS::rowsize(slot), cone, expected);
            for (auto &n : expected)
                n = nodes.LocalToGlobal(n);
            CHECK(Set(Global(nodes, v.EdgeNodes(edge))) == Set(expected));
        }
    }
}
TEST_CASE("InterpolateEdge: edge adjacency state is Local after pipeline")
{
    auto mesh = TinyEdges();
    auto q = MeshReadRequirements::CellFlux();
    q.edges = true;
    auto r = mesh->AcquireRead(q);
    for (auto slot : {MeshRelation::CellEdges, MeshRelation::EdgeNodes, MeshRelation::EdgeCells})
        CHECK(r.HostView().RelationState(slot) == Adj_PointToLocal);
}
TEST_CASE("InterpolateEdge: edge elem types are valid Line2 or Line3")
{
    auto mesh = TinyEdges();
    auto q = MeshReadRequirements::CellFlux();
    q.edges = true;
    auto r = mesh->AcquireRead(q);
    auto v = r.HostView();
    for (I edge = 0; edge < v.NumEdge(); ++edge)
        CHECK(v.GetEdgeInfo(edge).getElemType() == Elem::Line2);
}
