#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Mesh/NativeMeshPreparation.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Geom/Mesh/NativeMeshPartitionGraph.hpp"
#include "NativePeriodicFixtures.hpp"
#include "NativeReorderChecks.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"

#include <algorithm>
#include <iterator>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

TEST_CASE("Owned graph rejects mismatched weight branches before coordinate exchange")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto owned = G::NativeMeshBuilder::Build(NativePeriodicFixtures::TwoCells(mpi));
    const int method = mpi.size == 1 ? -1 : mpi.rank == 0 ? 0
                                                          : 1;
    CHECK_THROWS(G::detail::BuildOwnedPartitionGraph(owned, mpi, method));
    CHECK_NOTHROW(owned.coords->Write());
    owned.dimension = 1;
    CHECK_THROWS(G::detail::BuildOwnedPartitionGraph(owned, mpi, 0));
}

TEST_CASE("Owned CGNS partition permutation and faces need only one final assembly")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    for (const bool periodic : {false, true})
        for (const bool readerOwned : {false, true})
            for (const int weight : {0, 1})
            {
                CAPTURE(periodic);
                CAPTURE(readerOwned);
                CAPTURE(weight);
                std::optional<G::NativePeriodicImportOptions> frames;
                if (periodic)
                {
                    frames.emplace();
                    frames->frames.translation[1].map() = G::tPoint{10, 0, 0};
                    frames->frames.translation[2].map() = G::tPoint{0, 10, 0};
                }
                const auto path = (NativeReorderChecks::Root() / "data/mesh" /
                                   (periodic ? "IV10_10.cgns" : "UniformSquare_10.cgns"))
                                      .string();
                G::ResetNativeSerialCopyCounters();
                auto owned = G::ReadNativeCGNSOwned(mpi, path, 2, G::FBC_Name_2_ID_Default, mpi.size - 1,
                                                    {1, true}, frames, readerOwned ? G::NativeSerialDistribution::ReaderOwned : G::NativeSerialDistribution::Even);
                const auto copyCounters = G::GetNativeSerialCopyCounters();
                if (readerOwned && mpi.rank == mpi.size - 1)
                    CHECK(copyCounters.bulkCalls + copyCounters.rowCalls > 0);
                if (!readerOwned || mpi.rank != mpi.size - 1)
                    CHECK(copyCounters.bulkCalls + copyCounters.rowCalls == 0);
                auto checkOwned = [&](const G::NativeOwnedMeshData &data)
                {
                    CHECK_FALSE(data.retainedRead);
                    for (const auto &[kind, space] : data.spaces)
                        CHECK(space.NumGhost() == 0);
                };
                checkOwned(owned);
                CHECK_FALSE(owned.spaces.count(G::EntityKind::Face));
                const auto recipe = G::DefaultNativeMeshGhostSpec(owned, 1, true);
                // A separate completed-state reference exercises the older
                // call sequence. It is never an input to the owned pipeline.
                auto reference = G::NativeMeshAssembler::Build(mpi, owned, recipe);
                auto facialReference = G::NativeMeshPreparation::PrepareTopology(*reference, mpi);
                const auto actualGraph = G::detail::BuildOwnedPartitionGraph(owned, mpi, weight);
                const auto expectedGraph = G::detail::BuildNativePartitionGraph(*facialReference, mpi, weight);
                CHECK(actualGraph.offsets == expectedGraph.offsets);
                CHECK(actualGraph.adjacency == expectedGraph.adjacency);
                CHECK(actualGraph.weights == expectedGraph.weights);
                G::NativePartitionOptions options;
                options.metisSeed = 43;
                options.metisNcuts = 1;
                options.edgeWeightMethod = weight;
                options.deferNewFacesUntilPreparation = true;
                auto partition = G::NativeMeshPreparation::PartitionOwned(owned, mpi, options);
                checkOwned(partition.owned);
                CHECK_FALSE(partition.owned.spaces.count(G::EntityKind::Face));
                const auto unfinalized = G::NativeMeshAssembler::Build(
                    mpi, partition.owned, G::DefaultNativeMeshGhostSpec(partition.owned, 1, true));
                const auto unfinalizedKeys = unfinalized->RelationKeys();
                CHECK(std::find(unfinalizedKeys.begin(), unfinalizedKeys.end(),
                                G::NativeSetupFacialCellNeighborsKey()) == unfinalizedKeys.end());
                reference = G::NativeMeshPreparation::Partition(*reference, mpi, options).state;
                auto permutation = G::NativeMeshTransition::ReorderOwnedCells(partition.owned, mpi, 2, 1);
                checkOwned(permutation.owned);
                CHECK_FALSE(permutation.owned.spaces.count(G::EntityKind::Face));
                reference = G::NativeMeshTransition::ReorderLocalCells(*reference, mpi, 2, 1).state;
                auto topology = G::NativeMeshTopology::Build(std::move(permutation.owned), mpi);
                checkOwned(topology);
                CHECK(topology.spaces.count(G::EntityKind::Face) == 1);
                auto finalRecipe = G::DefaultNativeMeshGhostSpec(topology, 1, true);
                auto actual = G::NativeMeshAssembler::Build(mpi, std::move(topology), finalRecipe);
                reference = G::NativeMeshPreparation::PrepareTopology(*reference, mpi);
                for (const auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
                {
                    CHECK(actual->NumOwned(kind) == reference->NumOwned(kind));
                    NativeReorderChecks::Equal(NativeReorderChecks::Records(*actual, kind), NativeReorderChecks::Records(*reference, kind));
                    std::set<I> actualGhosts, expectedGhosts;
                    for (I row = actual->NumOwned(kind); row < actual->Entities(kind).NumLocal(); ++row)
                        actualGhosts.insert(actual->OriginalEntityId(kind, row));
                    for (I row = reference->NumOwned(kind); row < reference->Entities(kind).NumLocal(); ++row)
                        expectedGhosts.insert(reference->OriginalEntityId(kind, row));
                    CHECK(actualGhosts == expectedGhosts);
                }
                CHECK(actual->NLocalParts() == 2);
                for (int part = 0; part < 2; ++part)
                    CHECK(actual->LocalPartStart(part) == reference->LocalPartStart(part));
            }
}

TEST_CASE("Periodic default boundary ownership follows physical side while explicit support follows both")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    if (mpi.size < 2)
        return;
    auto source = G::ConstructNativeMesh(NativePeriodicFixtures::TwoCells(mpi), {1, true});
    for (const bool withFaces : {false, true})
        for (const bool missingNamedSide : {false, true})
        {
            auto input = withFaces ? G::NativeMeshPreparation::PrepareTopology(*source, mpi) : source;
            if (missingNamedSide)
            {
                auto owned = G::NativeMeshAssembler::CaptureOwned(*input, mpi);
                owned.namedRelations.erase(G::NativeBoundaryPlacementRelation(*input));
                input = G::AssembleNativeMesh(mpi, std::move(owned));
                CHECK_THROWS(input->Relation(G::NativeBoundaryPlacementRelation(*input)));
            }
            const auto &view = input->BorrowHostView();
            // Reverse the two cell owners: main side goes to larger rank, partner
            // to rank zero. Min-over-both would incorrectly move the main with it.
            G::ReorderInput reorder;
            std::vector<DNDS::MPI_int> ranks;
            for (I cell = 0; cell < view.NumCell(); ++cell)
                ranks.push_back(input->OriginalEntityId(G::EntityKind::Cell, cell) == 0 ? mpi.size - 1 : 0);
            reorder.explicitMaps.push_back({G::EntityKind::Cell, ranks});
            auto result = G::NativeMeshTransition::Build(*input, mpi, reorder).state;
            auto inspect = [&](const G::NativeMeshState &state, bool explicitBoth)
            {
                const auto &actual = state.BorrowHostView();
                auto coupled = state.Relation(G::Adj::Bnd2CellCoupled);
                for (I boundary = 0; boundary < actual.NumBnd(); ++boundary)
                {
                    const I id = state.OriginalEntityId(G::EntityKind::Bnd, boundary);
                    const I side = id == 0 || id == 2 || id == 4 ? 0 : 1;
                    const int expectedOwner = explicitBoth && id < 2 ? 0 : (side == 0 ? mpi.size - 1 : 0);
                    CHECK(mpi.rank == expectedOwner);
                    const auto pair = actual.BoundaryCells(boundary);
                    CHECK(state.OriginalEntityId(G::EntityKind::Cell, pair[0]) == side);
                    if (id < 2)
                    {
                        REQUIRE(pair[1] != DNDS::UnInitIndex);
                        CHECK(pair[1] == pair[0]);
                        const auto partner = coupled.ReadEntry(boundary, 1);
                        REQUIRE(!partner.absent);
                        const auto local = state.Entities(G::EntityKind::Cell).GlobalToLocal(partner.global);
                        REQUIRE(local);
                        CHECK(state.OriginalEntityId(G::EntityKind::Cell, *local) == 1 - side);
                    }
                }
                CHECK(state.NumGlobal(G::EntityKind::Bnd) == 6);
                CHECK(state.IdHighWater() == input->IdHighWater());
            };
            inspect(*result, false);
            // Explicit FollowSpec retains its documented minimum-over-relation
            // meaning; callers can deliberately request coupled support ownership.
            reorder.follows = {{G::EntityKind::Node, G::EntityKind::Cell, G::Adj::Node2Cell},
                               {G::EntityKind::Bnd, G::EntityKind::Cell, G::Adj::Bnd2CellCoupled}};
            inspect(*G::NativeMeshTransition::Build(*input, mpi, reorder).state, true);
        }
}

TEST_CASE("Source helpers retain exact legacy serial METIS graph policy and cell ownership")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const int readerRank = mpi.size - 1;
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    const auto path = (root / "data/mesh/UniformSquare_10.cgns").string();
    const auto name = [](const std::string &) -> G::t_index
    { return 21; };
    std::vector<idx_t> offsets, adjacency;
    G::FieldTransferDetail::LocalStage(mpi, [&]
                                       {
        if (mpi.rank != readerRank) return;
        auto serial = G::ReadNativeCGNSSerial(path, 2, name);
        auto nodes = serial.cellNodes.Read();
        auto info = serial.cellInfo.Read();
        std::vector<std::vector<I>> vertices(serial.cellNodes.RowCount());
        for (size_t cell = 0; cell < vertices.size(); ++cell)
        {
            const auto count = G::Elem::Element{info.View()(cell,0).getElemType()}.GetNumVertices();
            for (int slot = 0; slot < count; ++slot) vertices[cell].push_back(nodes.View()(cell,size_t(slot)));
            std::sort(vertices[cell].begin(),vertices[cell].end());
        }
        // Independent legacy BuildCell2Cell oracle: ordered vertex
        // intersections, before any native faces or native partition graph.
        offsets.push_back(0);
        for (size_t cell = 0; cell < vertices.size(); ++cell)
        {
            for (size_t other = 0; other < vertices.size(); ++other)
            {
                if (cell == other) continue;
                std::vector<I> common;
                std::set_intersection(vertices[cell].begin(),vertices[cell].end(),vertices[other].begin(),vertices[other].end(),std::back_inserter(common));
                if (common.size() >= 2) adjacency.push_back(idx_t(other));
            }
            offsets.push_back(idx_t(adjacency.size()));
        } });
    auto primary = G::ReadNativeCGNS(mpi, path, 2, name, readerRank, {1, true}, {}, G::NativeSerialDistribution::ReaderOwned);
    CHECK(primary->NumOwned(G::EntityKind::Cell) == (mpi.rank == readerRank ? 100 : 0));
    auto complete = G::NativeMeshPreparation::PrepareTopology(*primary, mpi);
    const auto graph = G::detail::BuildNativePartitionGraph(*complete, mpi);
    if (mpi.rank == readerRank)
    {
        CHECK(graph.offsets == offsets);
        CHECK(graph.adjacency == adjacency);
    }
    for (const std::string method : {"KWAY", "RB"})
        for (const int cuts : {0, 2})
        {
            G::PartitionOptions options;
            options.metisType = method;
            options.metisSeed = 41;
            options.metisNcuts = cuts;
            options.metisUfactor = 30;
            std::vector<int> expected(100, 0);
            G::FieldTransferDetail::LocalStage(mpi, [&]
                                               {
                if (mpi.rank != readerRank || mpi.size == 1) return;
                std::array<idx_t,METIS_NOPTIONS> settings{};
                METIS_SetDefaultOptions(settings.data());
                settings[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
                settings[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
                settings[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
                settings[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
                settings[METIS_OPTION_NCUTS] = std::max(cuts,1);
                settings[METIS_OPTION_NITER] = 10;
                settings[METIS_OPTION_UFACTOR] = 30;
                settings[METIS_OPTION_MINCONN] = 1;
                settings[METIS_OPTION_CONTIG] = 1;
                settings[METIS_OPTION_SEED] = 41;
                settings[METIS_OPTION_NUMBERING] = 0;
                std::vector<idx_t> parts(100);
                idx_t count = 100, constraints = 1, nParts = mpi.size, cut = 0;
                auto partition = method == "RB" ? METIS_PartGraphRecursive : METIS_PartGraphKway;
                DNDS_check_throw_info(partition(&count,&constraints,offsets.data(),adjacency.data(),nullptr,nullptr,nullptr,&nParts,nullptr,nullptr,settings.data(),&cut,parts.data()) == METIS_OK,"legacy serial partition oracle failed");
                for (size_t cell = 0; cell < parts.size(); ++cell) expected[cell] = int(parts[cell]); });
            MPI_Bcast(expected.data(), int(expected.size()), MPI_INT, readerRank, mpi.comm);
            auto actual = G::ReadNativeMeshFromCGNS(mpi, 2, path, options, name, {}, 0, 0, readerRank)->CurrentNativeState();
            std::vector<I> wanted;
            for (size_t cell = 0; cell < expected.size(); ++cell)
                if (expected[cell] == mpi.rank)
                    wanted.push_back(I(cell));
            REQUIRE(actual->NumOwned(G::EntityKind::Cell) == I(wanted.size()));
            for (size_t cell = 0; cell < wanted.size(); ++cell)
                CHECK(actual->OriginalEntityId(G::EntityKind::Cell, I(cell)) == wanted[cell]);
        }
}

TEST_CASE("Prepared reordered faces follow independent post-permutation parent-major slots")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    for (const bool periodic : {false, true})
    {
        const int readerRank = mpi.size - 1;
        std::array<std::map<G::EntityKind, std::vector<I>>, 3> ownedIdsByMode;
        std::array<std::vector<std::vector<I>>, 3> faceGeometryByMode;
        for (const int inputMode : {0, 1, 2})
        {
            const bool initialFaces = inputMode == 1;
            G::AutoAppendName2ID names;
            G::NativePeriodicImportOptions images;
            images.frames.translation[1].map() = G::tPoint{10, 0, 0};
            images.frames.translation[2].map() = G::tPoint{0, 10, 0};
            images.frames.translation[3].map() = G::tPoint{0, 0, 10};
            const auto path = (root / "data/mesh" / (periodic ? "IV10_10.cgns" : "UniformSquare_10.cgns")).string();
            DNDS::ssp<G::UnstructuredMesh> mesh;
            if (inputMode != 0)
                mesh = G::ReadNativeMeshFromCGNS(mpi, 2, path, {}, [&](const std::string &name)
                                                 { return names(name); }, periodic ? std::optional<G::NativePeriodicImportOptions>(images) : std::nullopt, 0, 0, readerRank, inputMode == 2);
            else
                mesh = G::UnstructuredMesh::FromNative(mpi, G::ReadNativeCGNS(mpi, path, 2, [&](const std::string &name)
                                                                              { return names(name); }, readerRank, {1, true}, periodic ? std::optional<G::NativePeriodicImportOptions>(images) : std::nullopt));
            auto before = mesh->CurrentNativeState();
            CHECK(before->Requirements().faces == initialFaces);
            if (inputMode == 2)
            {
                CHECK_FALSE(before->HasEntities(G::EntityKind::Face));
                CHECK_FALSE(before->Requirements().faces);
            }
            G::UnstructuredMeshSerialRW output(mesh, readerRank);
            G::PrepareMeshOptions options;
            options.reorderCells = 1;
            options.reorderParts = 2;
            options.buildSerialOut = false;
            G::PrepareMesh(*mesh, output, options);
            const auto state = mesh->CurrentNativeState();
            const auto &view = state->BorrowHostView();
            CHECK(state->MeshNamespace() == before->MeshNamespace());
            for (const auto &[kind, highWater] : before->IdHighWater())
                CHECK(state->IdHighWater().at(kind) == highWater);
            CHECK(state->NLocalParts() == 2);
            CHECK(state->LocalPartStart(0) == 0);
            CHECK(state->LocalPartEnd(1) == state->NumOwned(G::EntityKind::Cell));
            auto actualCellIds = std::vector<I>{}, previousCellIds = std::vector<I>{};
            for (I cell = 0; cell < view.NumCell(); ++cell)
                actualCellIds.push_back(state->OriginalEntityId(G::EntityKind::Cell, cell));
            for (I cell = 0; cell < before->NumOwned(G::EntityKind::Cell); ++cell)
                previousCellIds.push_back(before->OriginalEntityId(G::EntityKind::Cell, cell));
            std::sort(actualCellIds.begin(), actualCellIds.end());
            std::sort(previousCellIds.begin(), previousCellIds.end());
            CHECK(actualCellIds == previousCellIds);
            for (const auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
            {
                auto &ids = ownedIdsByMode[size_t(inputMode)][kind];
                for (I row = 0; row < state->NumOwned(kind); ++row)
                    ids.push_back(state->OriginalEntityId(kind, row));
                std::sort(ids.begin(), ids.end());
            }
            struct Face
            {
                std::vector<I> nodes, parents;
                std::vector<uint8_t> bits;
                int owner;
            };
            std::map<std::vector<I>, size_t> lookup;
            std::vector<Face> expected;
            const auto cells = state->Entities(G::EntityKind::Cell);
            for (I cell = 0; cell < view.NumCellProc(); ++cell)
            {
                auto element = view.GetCellElement(cell);
                std::vector<I> parentNodes;
                std::vector<uint8_t> parentBits;
                for (size_t slot = 0; slot < view.CellNodes(cell).size(); ++slot)
                {
                    parentNodes.push_back(state->OriginalEntityId(G::EntityKind::Node, view.CellNodes(cell)[slot]));
                    parentBits.push_back(periodic ? uint8_t(view.CellNodePeriodicBits(cell)[slot]) : 0);
                }
                for (int slot = 0; slot < element.GetNumFaces(); ++slot)
                {
                    const auto faceElement = element.ObtainFace(slot);
                    std::vector<I> nodes(size_t(faceElement.GetNumNodes()));
                    std::vector<uint8_t> bits(nodes.size());
                    element.ExtractFaceNodes(slot, parentNodes, nodes);
                    element.ExtractFaceNodes(slot, parentBits, bits);
                    // An independent image-invariant key; the retained first
                    // occurrence still records the exact unsorted physical slots.
                    std::vector<I> key;
                    for (uint8_t mask = 0; mask < (periodic ? 8 : 1); ++mask)
                    {
                        std::vector<std::pair<I, uint8_t>> pairs;
                        for (size_t node = 0; node < nodes.size(); ++node)
                            pairs.emplace_back(nodes[node], uint8_t(bits[node] ^ mask));
                        std::sort(pairs.begin(), pairs.end());
                        std::vector<I> candidate{I(faceElement.type)};
                        for (auto [node, image] : pairs)
                        {
                            candidate.push_back(node);
                            candidate.push_back(image);
                        }
                        if (key.empty() || candidate < key)
                            key = std::move(candidate);
                    }
                    const int owner = cells.Owner(cells.LocalToGlobal(cell));
                    const auto [found, inserted] = lookup.emplace(key, expected.size());
                    if (inserted)
                        expected.push_back({nodes, {}, bits, owner});
                    auto &face = expected[found->second];
                    face.parents.push_back(state->OriginalEntityId(G::EntityKind::Cell, cell));
                    face.owner = std::min(face.owner, owner);
                }
            }
            I owned = 0;
            for (auto &face : expected)
            {
                if (face.owner != mpi.rank)
                    continue;
                REQUIRE(owned < view.NumFace());
                std::vector<I> nodes, parents;
                std::vector<uint8_t> bits;
                for (size_t slot = 0; slot < view.FaceNodes(owned).size(); ++slot)
                {
                    nodes.push_back(state->OriginalEntityId(G::EntityKind::Node, view.FaceNodes(owned)[slot]));
                    bits.push_back(periodic ? uint8_t(view.FaceNodePeriodicBits(owned)[slot]) : 0);
                }
                for (auto cell : view.FaceCells(owned))
                    parents.push_back(cell == DNDS::UnInitIndex ? cell : state->OriginalEntityId(G::EntityKind::Cell, cell));
                if (face.parents.size() == 1)
                    face.parents.push_back(DNDS::UnInitIndex);
                CHECK(nodes == face.nodes);
                CHECK(bits == face.bits);
                CHECK(parents == face.parents);
                ++owned;
            }
            CHECK(owned == view.NumFace());
            CHECK_NOTHROW(state->BuildLocalCellNeighbors(true));

            auto &faceGeometry = faceGeometryByMode[size_t(inputMode)];
            for (I face = 0; face < view.NumFace(); ++face)
            {
                std::vector<std::pair<I, uint8_t>> nodes;
                for (size_t slot = 0; slot < view.FaceNodes(face).size(); ++slot)
                    nodes.emplace_back(state->OriginalEntityId(G::EntityKind::Node, view.FaceNodes(face)[slot]),
                                       periodic ? uint8_t(view.FaceNodePeriodicBits(face)[slot]) : 0);
                std::sort(nodes.begin(), nodes.end());
                std::vector<I> parents;
                for (auto cell : view.FaceCells(face))
                    parents.push_back(cell == DNDS::UnInitIndex ? cell : state->OriginalEntityId(G::EntityKind::Cell, cell));
                std::sort(parents.begin(), parents.end());
                std::vector<I> signature{I(nodes.size())};
                for (const auto &[node, bits] : nodes)
                {
                    signature.push_back(node);
                    signature.push_back(I(bits));
                }
                signature.push_back(I(parents.size()));
                signature.insert(signature.end(), parents.begin(), parents.end());
                faceGeometry.push_back(std::move(signature));
            }
            std::sort(faceGeometry.begin(), faceGeometry.end());
        }
        CHECK(ownedIdsByMode[1] == ownedIdsByMode[2]);
        CHECK(faceGeometryByMode[1] == faceGeometryByMode[2]);
    }
}

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
        for (size_t row = 0; row < field.RowCount(); ++row)
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
        for (size_t row = 0; row < values.RowCount(); ++row)
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
        for (size_t row = 0; row < faceValues.RowCount(); ++row)
            faces.View()(row, 0) = 2 * source->OriginalEntityId(G::EntityKind::Face, I(row)) + 17;
        for (size_t row = 0; row < cellValues.RowCount(); ++row)
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
    for (size_t row = 0; row < faceValues.RowCount(); ++row)
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
