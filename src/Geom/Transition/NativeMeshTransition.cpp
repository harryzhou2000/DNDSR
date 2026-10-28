#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Connectivity/NativeConnectivity.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Transition/CellPermutation.hpp"
#include "Geom/Unstructured/ReaderPartitionGraph.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>

namespace DNDS::Geom
{
    namespace
    {
        using RArray = Runtime::Array<index>;
        const std::array<AdjKind, size_t(MeshRelation::Count)> &Bindings()
        {
            static const std::array<AdjKind, size_t(MeshRelation::Count)> result{
                Adj::Cell2Node, Adj::Bnd2Node, Adj::Bnd2Cell, Adj::Cell2Cell,
                Adj::Node2Cell, Adj::Node2Bnd, Adj::Cell2Face, Adj::Face2Node,
                Adj::Face2Cell, Adj::Face2Bnd, Adj::Bnd2Face, Adj::Cell2Edge,
                Adj::Edge2Node, Adj::Edge2Cell, Adj::Cell2CellFace};
            return result;
        }
        AdjKind FacadeKey(size_t slot)
        {
            auto key = Bindings()[slot];
            key.path = key.Path();
            key.selection = RelationSelection::Incidence;
            key.order = RelationOrder::Topological;
            key.periodic = RelationPeriodicPolicy::EntityIdentity;
            key.semanticName = "geom.transition.facade." + std::to_string(slot);
            key.semanticVersion = 1;
            return key;
        }
        RArray Rows(const std::vector<std::vector<index>> &rows)
        {
            return Runtime::BuildArrayRows<index>(rows.size(), [&](size_t row)
                                                  { return rows[row].size(); }, [&](size_t row, auto output)
                                                  { std::copy(rows[row].begin(), rows[row].end(), output.begin()); });
        }
        /// Move row payloads. allowLocalPermutation must be rank-uniform and is
        /// true only for a full local permutation (or a one-rank route). The
        /// local branch preserves independent output storage while avoiding
        /// self-MPI and payload-plan construction.
        template <class T>
        Runtime::Array<T> Move(const Runtime::Array<T> &source, const Runtime::RowTransferPlan &route,
                               const Runtime::ElementSchema &schema, const MPIInfo &mpi,
                               bool allowLocalPermutation)
        {
            const auto &send = route.SendRows(), &receive = route.ReceiveRows();
            bool localOnly = allowLocalPermutation && source.GetDevice() == Runtime::Device::Host() && !route.HasCollisions() &&
                             send.size() == size_t(mpi.size) && receive.size() == size_t(mpi.size);
            for (int peer = 0; localOnly && peer < mpi.size; ++peer)
                if (peer != mpi.rank && (!send[size_t(peer)].empty() || !receive[size_t(peer)].empty()))
                    localOnly = false;
            if (localOnly)
            {
                const auto &sendRows = send[size_t(mpi.rank)], &receiveRows = receive[size_t(mpi.rank)];
                localOnly = sendRows.size() == source.Rows() &&
                            receiveRows.size() == source.Rows() && route.DestinationRows() == source.Rows();
                if (localOnly)
                {
                    bool identity = true;
                    for (size_t row = 0; row < source.Rows(); ++row)
                        identity = identity && sendRows[row] == receiveRows[row];
                    if (identity)
                        return source.Clone();

                    std::vector<size_t> sizes(source.Rows());
                    for (size_t row = 0; row < source.Rows(); ++row)
                        sizes[receiveRows[row]] = source.RowSize(sendRows[row]);
                    const auto sourceLayout = source.Layout();
                    auto localLayout = !sourceLayout.IsVariable()
                                           ? Runtime::RowLayout::Dense(source.Rows(), sourceLayout.Width(), sourceLayout.Stride())
                                       : sourceLayout.IsPadded()
                                           ? Runtime::RowLayout::Padded(sizes, sourceLayout.Stride())
                                           : Runtime::RowLayout::Ragged(sizes);
                    Runtime::Array<T> output(localLayout);
                    auto input = source.Read(), target = output.Write();
                    auto inputView = input.View(), outputView = target.View();
                    for (size_t row = 0; row < source.Rows(); ++row)
                    {
                        const auto from = sendRows[row], to = receiveRows[row];
                        std::copy(inputView.Row(from), inputView.Row(from) + sizes[to], outputView.Row(to));
                    }
                    return output;
                }
            }

            auto layout = route.DestinationLayout(source.Layout());
            if (!source.Layout().IsVariable())
                layout = Runtime::RowLayout::Dense(route.DestinationRows(), source.Layout().Width());
            Runtime::Array<T> output(layout);
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = std::make_shared<Runtime::CommunicationContext>(route.Communicator());
            Runtime::PayloadPlan(route, source.Layout(), output.Layout(), schema)
                .Bind(source, output, Runtime::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            return output;
        }
        void Agree(const MPIInfo &mpi, const std::string &signature)
        {
            int length = int(signature.size());
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            std::string expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0');
            MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(signature == expected, "native transition schedules differ"); });
        }
    }

    NativeMeshTransitionResult NativeMeshTransition::Build(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                           const ReorderInput &input, int ghostLayers)
    {
        // Reconstruction first removes only the requested native families.
        // Retained relations, IDs and ancestry then use the normal exact move.
        // New target IDs begin at the old family's high-water mark.
        const std::set<EntityKind> requested(input.destroyKinds.begin(), input.destroyKinds.end());
        std::string reconstruction;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            DNDS_check_throw_info(ghostLayers >= 1,"native transition requires positive closure");
            for (auto kind : requested)
            {
                DNDS_check_throw_info((kind == EntityKind::Face || kind == EntityKind::Edge) && source.HasEntities(kind),
                                      "only existing native Face or Edge families can be reconstructed");
                reconstruction += std::to_string(int(kind))+";";
            }
            for (const auto &map : input.explicitMaps)
                DNDS_check_throw_info(!input.destroyKinds.count(map.kind),"cannot place a reconstructed family");
            for (const auto &follow : input.follows)
                DNDS_check_throw_info(!input.destroyKinds.count(follow.follower) && !input.destroyKinds.count(follow.leader),
                                      "cannot follow through a reconstructed family"); });
        Agree(mpi, reconstruction);
        if (input.destroyKinds.empty())
            return BuildImpl(source, mpi, input, ghostLayers, nullptr);
        auto stripped = NativeMeshAssembler::CaptureOwned(source, mpi);
        const auto &bindings = Bindings();
        std::map<EntityKind, index> starts;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (auto kind : requested)
            {
                starts.emplace(kind,source.NextEntityId(kind));
                stripped.spaces.erase(kind); stripped.ids.erase(kind); stripped.identity.nextIds.erase(kind);
                for (size_t slot = 0; slot < stripped.relations.size(); ++slot)
                    if (bindings[slot].from == kind || bindings[slot].to == kind) stripped.relations[slot].reset();
                for (auto item = stripped.namedRelations.begin(); item != stripped.namedRelations.end();)
                {
                    const auto path = item->first.Path();
                    if (std::find(path.begin(),path.end(),kind) != path.end()) item = stripped.namedRelations.erase(item);
                    else ++item;
                }
                if (stripped.provenance)
                {
                    stripped.provenance->sourceIds.erase(kind); stripped.provenance->sourceKinds.erase(kind);
                    if (stripped.provenance->sourceIds.empty()) stripped.provenance.reset();
                }
                if (kind == EntityKind::Face)
                {
                    stripped.faceInfo.reset(); stripped.cellFacesPbi.reset(); stripped.faceNodesPbi.reset();
                    stripped.relations[size_t(MeshRelation::CellFaceCells)].reset();
                    stripped.requirements.faces = stripped.requirements.cellFaceNeighbors = stripped.requirements.allFaceCellsLocal = false;
                }
                else
                {
                    stripped.edgeInfo.reset(); stripped.cellEdgesPbi.reset(); stripped.edgeNodesPbi.reset();
                    stripped.requirements.edges = false;
                }
            } });
        auto retained = input;
        retained.destroyKinds.clear();
        auto moved = BuildOwnedImpl(stripped, mpi, retained, nullptr);
        // Field source associations are the original publication, not the
        // temporary owned-family projection with equivalent numbering.
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            moved.sourceSpaces.clear();
            for (const auto &[kind,family] : source._spaces) moved.sourceSpaces.emplace(kind,family.identity); });
        moved.owned = NativeMeshTopology::Build(std::move(moved.owned), mpi, input.destroyKinds.count(EntityKind::Face),
                                                input.destroyKinds.count(EntityKind::Edge), starts);
        NativeMeshTransitionResult result;
        result.transfers = std::move(moved.transfers);
        result.sourceSpaces = std::move(moved.sourceSpaces);
        result.state = AssembleNativeMesh(mpi, std::move(moved.owned), ghostLayers);
        return result;
    }

    NativeMeshTransitionResult NativeMeshTransition::ReorderLocalCells(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                                       int nParts, int nPartsInner, int ghostLayers)
    {
        nParts = std::max(nParts, 1);
        nPartsInner = std::max(nPartsInner, 1);
        detail::CellPermutationResult permutation;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            DNDS_check_throw_info(int64_t(nParts) * nPartsInner < std::numeric_limits<int>::max() && ghostLayers >= 1,
                                  "native local reorder partition count or closure is invalid");
            source.CheckRequirements(source.Requirements().faces ? MeshReadRequirements::CellFlux() : MeshReadRequirements::Primary());
            const auto &view = source.BorrowHostView();
            // Private-cell sorting below intentionally uses full node-neighbor
            // connectivity, including remote vertex-only contacts.
            for (index cell = 0; cell < view.NumCell(); ++cell)
                for (auto other : view.CellCells(cell))
                    DNDS_check_throw_info(other >= 0 && other < view.NumCellProc(),
                                          "native local reorder requires local owned CellCells closure"); });
        Agree(mpi, "local:" + std::to_string(nParts) + ":" + std::to_string(nPartsInner) + ":" + std::to_string(ghostLayers));
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto graph = source.Requirements().faces ? source.BuildLocalCellNeighbors()
                                                     : detail::BuildPrimaryLocalCellNeighbors(source);
            // Source setup historically reorders before face interpolation:
            // its vertex-derived facial graph is sorted and duplicate-free.
            // Face-slot order, including repeated periodic neighbors, must
            // not change the METIS/RCM permutation for that setup path.
            for (auto &row : graph)
            {
                std::sort(row.begin(), row.end());
                row.erase(std::unique(row.begin(), row.end()), row.end());
            }
            struct CellRows
            {
                const UnstructuredMeshView &view;
                auto operator[](index row) const { return view.CellCells(row); }
            } rows{source.BorrowHostView()};
            if (source.NumOwned(EntityKind::Cell) == 0)
                permutation.localPartitionStarts.assign(size_t(nParts) + 1, 0);
            else
                permutation = detail::ComputeCellPermutation(graph, rows, source.NumOwned(EntityKind::Cell), nParts, nPartsInner); });
        auto result = BuildImpl(source, mpi, {}, ghostLayers, &permutation.cellOld2New);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { result.state = result.state->WithLocalPartitions(std::move(permutation.localPartitionStarts)); });
        return result;
    }

    NativeMeshTransitionResult NativeMeshTransition::BuildImpl(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                               const ReorderInput &input, int ghostLayers,
                                                               const std::vector<index> *cellOld2New)
    {
        auto moved = BuildOwnedImpl(NativeMeshAssembler::CaptureOwned(source, mpi), mpi, input, cellOld2New);
        NativeMeshTransitionResult result;
        result.transfers = std::move(moved.transfers);
        // Owned capture has deliberately detached ghost mappings. External
        // fields must still validate against the original published spaces.
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (const auto &[kind, family] : source._spaces)
                result.sourceSpaces.emplace(kind, family.identity); });
        result.state = AssembleNativeMesh(mpi, std::move(moved.owned), ghostLayers);
        return result;
    }

    NativeOwnedMeshTransitionResult NativeMeshTransition::ReorderOwnedCells(
        const NativeOwnedMeshData &source, const MPIInfo &mpi, int nParts, int nPartsInner)
    {
        nParts = std::max(nParts, 1);
        nPartsInner = std::max(nPartsInner, 1);
        Agree(mpi, "owned-local:" + std::to_string(nParts) + ":" + std::to_string(nPartsInner));
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(int64_t(nParts) * nPartsInner < std::numeric_limits<int>::max(), "owned local partition count overflows"); });
        auto cachedFacial = source.namedRelations.find(NativeSetupFacialCellNeighborsKey());
        std::optional<detail::NativePartitionGraph> facial;
        if (cachedFacial != source.namedRelations.end())
        {
            facial.emplace();
            auto rows = cachedFacial->second.Read();
            DNDS_check_throw_info(rows.View().Rows() == size_t(source.spaces.at(EntityKind::Cell).NumOwned()),
                                  "cached facial graph row count differs from owned cells");
            facial->offsets.push_back(0);
            std::vector<index> globals;
            for (size_t row = 0; row < rows.View().Rows(); ++row)
            {
                globals.clear();
                for (auto global : rows.View()[row])
                {
                    DNDS_check_throw_info(global >= 0 && global < source.spaces.at(EntityKind::Cell).NumGlobal(),
                                          "cached facial graph contains an invalid cell");
                    globals.push_back(global);
                }
                std::sort(globals.begin(), globals.end());
                globals.erase(std::unique(globals.begin(), globals.end()), globals.end());
                for (auto global : globals)
                    facial->adjacency.push_back(METIS::indexToIdx(global));
                facial->offsets.push_back(METIS::indexToIdx(facial->adjacency.size()));
            }
        }
        else
            facial.emplace(detail::BuildOwnedPartitionGraph(source, mpi));
        detail::CellPermutationResult permutation;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            const auto cells = source.spaces.at(EntityKind::Cell);
            tLocalMatStruct graph(size_t(cells.NumOwned())), neighbors(size_t(cells.NumOwned()));
            auto all = source.relations.at(size_t(MeshRelation::CellCells))->Read();
            for (index cell = 0; cell < cells.NumOwned(); ++cell)
            {
                for (idx_t slot = facial->offsets[size_t(cell)]; slot < facial->offsets[size_t(cell + 1)]; ++slot)
                    if (const auto local = cells.GlobalToLocal(index(facial->adjacency[size_t(slot)])))
                        graph[size_t(cell)].push_back(*local);
                for (auto global : all.View()[size_t(cell)])
                {
                    DNDS_check_throw_info(global >= 0 && global < cells.NumGlobal(), "owned cell neighbor is unavailable");
                    const auto local = cells.GlobalToLocal(global);
                    neighbors[size_t(cell)].push_back(local ? *local : cells.NumOwned());
                }
            }
            if (cells.NumOwned() == 0)
                permutation.localPartitionStarts.assign(size_t(nParts) + 1, 0);
            else
                permutation = detail::ComputeCellPermutation(graph, neighbors, cells.NumOwned(), nParts, nPartsInner); });
        auto result = BuildOwnedImpl(source, mpi, {}, &permutation.cellOld2New);
        result.owned.namedRelations.erase(NativeSetupFacialCellNeighborsKey());
        result.owned.localPartitionStarts = std::move(permutation.localPartitionStarts);
        return result;
    }

    NativeOwnedMeshTransitionResult NativeMeshTransition::BuildOwned(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                                     const ReorderInput &input)
    {
        return BuildOwnedImpl(source, mpi, input, nullptr);
    }

    NativeOwnedMeshTransitionResult NativeMeshTransition::BuildOwnedImpl(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                                         const ReorderInput &input,
                                                                         const std::vector<index> *cellOld2New)
    {
        NativeOwnedMeshTransitionResult result;
        std::map<EntityKind, EntitySpace> oldSpaces, spaces;
        std::map<AdjKind, NativeNamedRelation> original;
        std::map<EntityKind, std::vector<MPI_int>> placements;
        std::string schedule;
        RArray noGhostRows;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            noGhostRows = Rows({});
            DNDS_check_throw_info(input.destroyKinds.empty(), "owned transition requires retained entity kinds");
            int rank = -1, size = 0;
            MPI_Comm_rank(mpi.comm, &rank);
            MPI_Comm_size(mpi.comm, &size);
            DNDS_check_throw_info(rank == mpi.rank && size == mpi.size, "native transition MPIInfo differs from communicator");
            for (const auto &[kind, family] : source.spaces)
            {
                family.Validate();
                EntitySpace::CheckCommunicator(family._mpi.comm, mpi.comm);
                DNDS_check_throw_info(family.Kind() == kind && family.NumGhost() == 0,
                                      "owned transition requires matching owned-only entity spaces");
                oldSpaces.emplace(kind, family);
            }
            for (size_t slot = 0; slot < source.relations.size(); ++slot)
                if (source.relations[slot])
                    original.emplace(FacadeKey(slot), NativeNamedRelation{{*source.relations[slot], noGhostRows}, Adj_PointToGlobal});
            for (const auto &[key, relation] : source.namedRelations)
                DNDS_check_throw_info(original.emplace(key, NativeNamedRelation{{relation, noGhostRows}, Adj_PointToGlobal}).second,
                                      "reserved native transition descriptor collision");
            for (const auto &map : input.explicitMaps)
            {
                DNDS_check_throw_info(oldSpaces.count(map.kind) && map.targetRanks.size() == size_t(oldSpaces.at(map.kind).NumOwned()) &&
                                          placements.emplace(map.kind, map.targetRanks).second, "invalid native placement family or extent");
                for (auto target : map.targetRanks) DNDS_check_throw_info(target >= 0 && target < mpi.size, "invalid native placement rank");
                schedule += "m" + std::to_string(int(map.kind)) + ";";
            }
            for (const auto &follow : input.follows)
                schedule += "f" + std::to_string(int(follow.follower)) + ":" + std::to_string(int(follow.leader)) + ":" + follow.follower2leader.Signature();
            schedule += cellOld2New ? "local" : "partition";
            for (const auto &[key, unused] : original) schedule += key.Signature(); });
        Agree(mpi, schedule);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { result.sourceSpaces = oldSpaces; });
        // Missing published inverse slots need not force a legacy recovery.
        // Their sparse native inverses are setup scratch only.
        for (auto slot : {MeshRelation::NodeCells, MeshRelation::NodeBoundaries})
            if (!original.count(FacadeKey(size_t(slot))))
            {
                const auto coneSlot = slot == MeshRelation::NodeCells ? MeshRelation::CellNodes : MeshRelation::BoundaryNodes;
                const auto &cone = original.at(FacadeKey(size_t(coneSlot)));
                const auto &key = Bindings()[size_t(coneSlot)];
                RelationReadAccess read(key, oldSpaces.at(key.from), oldSpaces.at(key.to), cone.state,
                                        cone.rows.owned, &cone.rows.ghost);
                auto inverse = NativeConnectivity::Inverse(read, mpi);
                original.emplace(FacadeKey(size_t(slot)), NativeNamedRelation{
                                                              {inverse.Owned(), Rows({})}, Adj_PointToGlobal});
            }
        const auto boundaryPlacement = source.periodic
                                           ? AdjKind::Composition(Adj::Bnd2Node, Adj::Cell2Node.Inverted(), RelationSelection::Exact, false,
                                                                  "geom.boundary.physical-frame-face", 1, RelationPeriodicPolicy::FrameMatched)
                                           : Adj::Bnd2Cell;
        if (source.periodic && !original.count(boundaryPlacement))
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                // Full snapshots/archives may predate the optional named
                // exact relation. The physical facade contract already puts
                // the boundary's own side in slot 0. Publish a distinct exact
                // side relation; never change/truncate its coupled support.
                const auto &physical = original.at(FacadeKey(size_t(MeshRelation::BoundaryCells)));
                RelationReadAccess read(Adj::Bnd2Cell, oldSpaces.at(EntityKind::Bnd), oldSpaces.at(EntityKind::Cell),
                                        physical.state, physical.rows.owned, &physical.rows.ghost);
                std::vector<std::vector<index>> sides(size_t(read.NumOwned()));
                for (index boundary = 0; boundary < read.NumOwned(); ++boundary)
                {
                    DNDS_check_throw_info(read.RowSize(boundary) >= 1 && !read.ReadEntry(boundary,0).absent,
                                          "periodic boundary placement requires its physical side in slot zero");
                    sides[size_t(boundary)] = {read.ReadEntry(boundary,0).global};
                }
                original.emplace(boundaryPlacement,NativeNamedRelation{{Rows(sides),Rows({})},Adj_PointToGlobal}); });
        std::vector<FollowSpec> follows = input.follows;
        if (follows.empty() && placements.count(EntityKind::Cell))
            for (auto kind : {EntityKind::Node, EntityKind::Bnd})
                if (!placements.count(kind))
                    follows.push_back({kind, EntityKind::Cell, kind == EntityKind::Node ? Adj::Node2Cell : boundaryPlacement});
        for (const auto &follow : follows)
        {
            std::optional<RelationReadAccess> relation;
            std::vector<index> requested;
            std::vector<Runtime::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(placements.count(follow.leader) && !placements.count(follow.follower) &&
                                          follow.follower2leader.from == follow.follower && follow.follower2leader.to == follow.leader,
                                      "invalid native follow dependency");
                auto found = original.find(follow.follower2leader);
                if (found == original.end())
                    for (size_t slot = 0; slot < Bindings().size(); ++slot)
                        if (Bindings()[slot] == follow.follower2leader) { found = original.find(FacadeKey(slot)); break; }
                DNDS_check_throw_info(found != original.end(), "native follow relation is unavailable");
                relation.emplace(found->first, oldSpaces.at(follow.follower), oldSpaces.at(follow.leader), found->second.state,
                                 found->second.rows.owned, &found->second.rows.ghost);
                std::set<index> globals;
                for (index row = 0; row < relation->NumOwned(); ++row)
                    for (auto value : relation->GlobalRow(row)) if (value != UnInitIndex) globals.insert(value);
                requested.assign(globals.begin(), globals.end());
                const auto &leader = oldSpaces.at(follow.leader);
                for (size_t row = 0; row < requested.size(); ++row)
                    requests.push_back({leader.Owner(requested[row]), leader.OwnerLocalRow(requested[row]), row}); });
            RArray ranks(Runtime::RowLayout::Dense(placements.at(follow.leader).size(), 1));
            {
                auto write = ranks.Write();
                for (size_t i = 0; i < ranks.Rows(); ++i)
                    write.View()(i, 0) = placements.at(follow.leader)[i];
            }
            Runtime::RowTransferPlan route(mpi.comm, ranks.Rows(), requested.size(), requests);
            auto received = Move(ranks, route, Runtime::ElementSchema::Arithmetic<index>(), mpi, false);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto read = received.Read();
                std::map<index, MPI_int> targets;
                for (size_t i = 0; i < requested.size(); ++i) targets.emplace(requested[i], MPI_int(read.View()(i, 0)));
                std::vector<MPI_int> assignment(size_t(relation->NumOwned()), mpi.rank);
                for (index row = 0; row < relation->NumOwned(); ++row)
                {
                    int target = mpi.size;
                    for (auto global : relation->GlobalRow(row)) if (global != UnInitIndex) target = std::min(target, int(targets.at(global)));
                    if (target < mpi.size) assignment[size_t(row)] = target;
                }
                placements.emplace(follow.follower, std::move(assignment)); });
        }
        std::map<EntityKind, Runtime::RowTransferPlan> routes;
        std::map<EntityKind, PermutationTransfer> allTransfers;
        const auto anchor = oldSpaces.begin()->second;
        for (const auto &[kind, old] : oldSpaces)
        {
            auto assigned = placements.find(kind);
            const auto targets = assigned == placements.end() ? std::vector<MPI_int>(size_t(old.NumOwned()), mpi.rank) : assigned->second;
            PermutationTransfer transfer;
            if (cellOld2New)
            {
                std::vector<index> identity(size_t(old.NumOwned()));
                std::iota(identity.begin(), identity.end(), index(0));
                transfer = PermutationTransfer::fromLocalPermutation(kind == EntityKind::Cell ? *cellOld2New : identity,
                                                                     old._global, mpi);
            }
            else
                transfer = PermutationTransfer::fromPartition(targets, old._global, mpi);
            auto route = NativeRowTransfer(transfer, mpi);
            spaces.emplace(kind, anchor.CreateSibling(kind, index(route.DestinationRows())));
            routes.emplace(kind, std::move(route));
            if (cellOld2New || assigned != placements.end())
                result.transfers.emplace(kind, transfer);
            allTransfers.emplace(kind, std::move(transfer));
        }
        // SharedStorage identifies the complete immutable shape/value record,
        // unlike the allocation identity, which can back distinct row views.
        using PayloadKey = std::tuple<uintptr_t, uintptr_t, EntityKind, EntityKind, MeshAdjState>;
        auto payloadKey = [](const AdjKind &key, const NativeNamedRelation &values)
        {
            return PayloadKey{reinterpret_cast<uintptr_t>(values.rows.owned.SharedStorage().get()),
                              reinterpret_cast<uintptr_t>(values.rows.ghost.SharedStorage().get()),
                              key.from, key.to, values.state};
        };
        std::map<EntityKind, std::map<index, index>> remap;
        for (const auto &[kind, old] : oldSpaces)
        {
            // Only Cell numbering changes during a local cell permutation.
            // Per-descriptor validation below still checks every input row;
            // building identity maps for other target families adds no proof.
            if (cellOld2New && kind != EntityKind::Cell)
                continue;
            std::vector<index> globals;
            std::vector<Runtime::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::set<index> referenced;
                std::set<PayloadKey> scanned;
                for (const auto &[key, values] : original) if (key.to == kind)
                {
                    RelationReadAccess read(key, oldSpaces.at(key.from), old, values.state, values.rows.owned, &values.rows.ghost);
                    // Validate every descriptor even when its exact payload was
                    // already scanned under another semantic name.
                    if (!scanned.insert(payloadKey(key, values)).second) continue;
                    for (index row = 0; row < read.NumOwned(); ++row)
                        for (size_t slot = 0; slot < read.RowSize(row); ++slot)
                        {
                            const auto global = read.ConvertedIndex(row, slot, Adj_PointToGlobal);
                            if (global != UnInitIndex) referenced.insert(global);
                        }
                }
                const auto first = (*old._global)(mpi.rank, 0);
                const auto &newGlobals = allTransfers.at(kind).newGlobalIndices;
                for (auto global : referenced)
                {
                    if (global >= first && global - first < old.NumOwned())
                        remap[kind].emplace(global, newGlobals[size_t(global - first)]);
                    else
                    {
                        requests.push_back({old.Owner(global), old.OwnerLocalRow(global), globals.size()});
                        globals.push_back(global);
                    }
                } });
            RArray values(Runtime::RowLayout::Dense(size_t(old.NumOwned()), 1));
            {
                auto write = values.Write();
                for (size_t i = 0; i < values.Rows(); ++i)
                    write.View()(i, 0) = allTransfers.at(kind).newGlobalIndices[i];
            }
            Runtime::RowTransferPlan route(mpi.comm, values.Rows(), globals.size(), requests);
            auto received = Move(values, route, Runtime::ElementSchema::Arithmetic<index>(), mpi, false);
            auto read = received.Read();
            for (size_t i = 0; i < globals.size(); ++i)
                remap[kind].emplace(globals[i], read.View()(i, 0));
        }
        std::map<AdjKind, RArray> movedRelations;
        using MoveKey = std::pair<PayloadKey, bool>;
        std::map<MoveKey, int> movedPayloads;
        std::vector<RArray> movedByOrdinal;
        for (const auto &[key, values] : original)
        {
            std::optional<RelationReadAccess> read;
            MoveKey identity;
            int candidate = -1;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                read.emplace(key, oldSpaces.at(key.from), oldSpaces.at(key.to), values.state, values.rows.owned, &values.rows.ghost);
                identity = {payloadKey(key, values), key.order == RelationOrder::SortedUnique};
                auto found = movedPayloads.find(identity);
                if (found != movedPayloads.end()) candidate = found->second; });
            // Local alias inventories can differ. All ranks must reuse the
            // SAME earlier collective output, not merely find some local hit.
            int candidates[2]{candidate, -candidate};
            MPI_Allreduce(MPI_IN_PLACE, candidates, 2, MPI_INT, MPI_MIN, mpi.comm);
            if (candidates[0] >= 0 && candidates[0] == -candidates[1])
            {
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    auto moved = movedByOrdinal.at(size_t(candidates[0]));
                    movedRelations.emplace(key, moved);
                    movedByOrdinal.push_back(std::move(moved)); });
                continue;
            }
            std::optional<RArray> globalRows;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { globalRows.emplace(Runtime::BuildArrayRows<index>(size_t(read->NumOwned()), [&](size_t row)
                                                                                                { return read->RowSize(index(row)); }, [&](size_t row, auto output)
                                                                                                {
                        for (size_t slot = 0; slot < output.size(); ++slot)
                        {
                            const auto entry = read->ConvertedIndex(index(row), slot, Adj_PointToGlobal);
                            output[slot] = entry == UnInitIndex || (cellOld2New && key.to != EntityKind::Cell)
                                               ? entry : remap.at(key.to).at(entry);
                        }
                        if (key.order == RelationOrder::SortedUnique && output.size() > 1) std::sort(output.begin(), output.end()); })); });
            auto moved = cellOld2New && key.from != EntityKind::Cell
                             ? *globalRows
                             : Move(*globalRows, routes.at(key.from), Runtime::ElementSchema::Arithmetic<index>(), mpi,
                                    cellOld2New || mpi.size == 1);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                movedPayloads.emplace(identity, int(movedByOrdinal.size()));
                movedRelations.emplace(key, moved);
                movedByOrdinal.push_back(std::move(moved)); });
        }
        NativeOwnedMeshData owned;
        owned.identity = source.identity;
        owned.requirements = source.requirements;
        owned.audit = source.audit;
        owned.ghostSpec = source.ghostSpec;
        owned.dimension = source.dimension;
        owned.periodic = source.periodic;
        owned.periodicInfo = source.periodicInfo;
        owned.spaces = spaces;
        if (source.provenance)
        {
            owned.provenance = *source.provenance;
            for (auto &[kind, values] : owned.provenance->sourceIds)
                if (!cellOld2New || kind == EntityKind::Cell)
                    values = Move(values, routes.at(kind), Runtime::ElementSchema::Arithmetic<index>(), mpi,
                                  cellOld2New || mpi.size == 1);
        }
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (size_t slot = 0; slot < source.relations.size(); ++slot)
                if (source.relations[slot])
                {
                    auto rows = movedRelations.at(FacadeKey(slot));
                    const auto &shape = source.relations[slot]->Layout();
                    if (!shape.IsVariable())
                        rows = RArray::FromStorage(Runtime::RowLayout::Dense(rows.Rows(), shape.Width()), rows.Storage());
                    owned.relations[slot] = std::move(rows);
                }
            for (const auto &[key, unused] : source.namedRelations)
                owned.namedRelations.emplace(key, movedRelations.at(key)); });
        auto companion = [&](auto &destination, const auto &originalPair, EntityKind kind, const Runtime::ElementSchema &schema)
        {
            if (originalPair)
                destination = cellOld2New && kind != EntityKind::Cell ? *originalPair
                                                                      : Move(*originalPair, routes.at(kind), schema, mpi,
                                                                             cellOld2New || mpi.size == 1);
        };
        companion(owned.coords, source.coords, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        companion(owned.nodeWallDist, source.nodeWallDist, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        const auto elementSchema = Runtime::ElementSchema::For<ElemInfo>();
        companion(owned.cellInfo, source.cellInfo, EntityKind::Cell, elementSchema);
        companion(owned.bndInfo, source.bndInfo, EntityKind::Bnd, elementSchema);
        companion(owned.faceInfo, source.faceInfo, EntityKind::Face, elementSchema);
        companion(owned.edgeInfo, source.edgeInfo, EntityKind::Edge, elementSchema);
        const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
        companion(owned.cellNodesPbi, source.cellNodesPbi, EntityKind::Cell, bits);
        companion(owned.bndNodesPbi, source.bndNodesPbi, EntityKind::Bnd, bits);
        companion(owned.cellFacesPbi, source.cellFacesPbi, EntityKind::Cell, bits);
        companion(owned.faceNodesPbi, source.faceNodesPbi, EntityKind::Face, bits);
        companion(owned.cellEdgesPbi, source.cellEdgesPbi, EntityKind::Cell, bits);
        companion(owned.edgeNodesPbi, source.edgeNodesPbi, EntityKind::Edge, bits);
        for (const auto &[kind, old] : oldSpaces)
        {
            const auto &originalIds = source.ids.at(kind);
            if (cellOld2New && kind != EntityKind::Cell)
            {
                // The combined ID allocation begins with the owned prefix.
                // Reuse that immutable prefix; assembly builds its new ghosts.
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    const auto layout = Runtime::RowLayout::Dense(size_t(old.NumOwned()), 1, originalIds.Layout().Stride());
                    owned.ids.emplace(kind, RArray::FromStorage(layout, originalIds.Storage().Subspan(
                                                                          0, CheckedSize::Multiply(layout.Elements(), sizeof(index))))); });
                continue;
            }
            const auto &baseRoute = routes.at(kind);
            auto idRoute = Runtime::RowTransferPlan::Known(std::make_shared<Runtime::CommunicationContext>(mpi.comm),
                                                           originalIds.Rows(), baseRoute.DestinationRows(), baseRoute.SendRows(), baseRoute.ReceiveRows());
            owned.ids.emplace(kind, Move(originalIds, idRoute, Runtime::ElementSchema::Arithmetic<index>(), mpi,
                                         cellOld2New || mpi.size == 1));
        }
        owned.retainedRead = source.retainedRead;
        owned.retainedProvenance = source.retainedProvenance;
        result.owned = std::move(owned);
        return result;
    }
}
