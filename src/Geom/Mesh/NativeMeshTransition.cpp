#include "NativeMeshTransition.hpp"
#include "NativeMeshAssembler.hpp"
#include "NativeMeshTopology.hpp"
#include "NativeConnectivity.hpp"
#include "FieldTransfer.hpp"
#include "Mesh_CellPermutation.hpp"
#include <limits>
#include <numeric>
#include <set>

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
            Runtime::RowBuilder builder(rows.size(), sizeof(index));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                auto output = builder.Row<index>(row);
                std::copy(rows[row].begin(), rows[row].end(), output.data);
            }
            return RArray::FromSharedStorage(builder.Commit());
        }
        template <class T>
        Runtime::Array<T> Move(const Runtime::Array<T> &source, const Runtime::RowTransferPlan &route,
                               const Runtime::ElementSchema &schema)
        {
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

    NativeMeshTransitionResult NativeMeshTransition::Build(const NativeMeshState &source, const MPIInfo &mpi,
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
        auto native = NativeMeshAssembler::Build(mpi, std::move(stripped), ghostLayers);
        auto retained = input;
        retained.destroyKinds.clear();
        auto result = BuildImpl(*native, mpi, retained, ghostLayers, nullptr);
        // Field source associations are the original publication, not the
        // temporary owned-family projection with equivalent numbering.
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            result.sourceSpaces.clear();
            for (const auto &[kind,family] : source._spaces) result.sourceSpaces.emplace(kind,family.identity); });
        result.state = NativeMeshTopology::Build(*result.state, mpi, input.destroyKinds.count(EntityKind::Face),
                                                 input.destroyKinds.count(EntityKind::Edge), ghostLayers, starts);
        return result;
    }

    NativeMeshTransitionResult NativeMeshTransition::ReorderLocalCells(const NativeMeshState &source, const MPIInfo &mpi,
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
            source.CheckRequirements(MeshReadRequirements::CellFlux()); });
        Agree(mpi, "local:" + std::to_string(nParts) + ":" + std::to_string(nPartsInner) + ":" + std::to_string(ghostLayers));
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto graph = source.BuildLocalCellNeighbors();
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

    NativeMeshTransitionResult NativeMeshTransition::BuildImpl(const NativeMeshState &source, const MPIInfo &mpi,
                                                               const ReorderInput &input, int ghostLayers,
                                                               const std::vector<index> *cellOld2New)
    {
        NativeMeshTransitionResult result;
        std::map<EntityKind, EntitySpace> oldSpaces, spaces;
        std::map<AdjKind, NativeNamedRelation> original;
        std::map<EntityKind, std::vector<MPI_int>> placements;
        std::string schedule;
        const NativeMeshSnapshot::State *backing = nullptr;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { backing = &source.Snapshot(Runtime::Device::Host()).Live(); });
        const auto &snapshot = *backing;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            source.Validate();
            DNDS_check_throw_info(ghostLayers >= 1 && input.destroyKinds.empty(), "native transition requires positive closure and retained entity kinds");
            int rank = -1, size = 0;
            MPI_Comm_rank(mpi.comm, &rank);
            MPI_Comm_size(mpi.comm, &size);
            DNDS_check_throw_info(rank == mpi.rank && size == mpi.size, "native transition MPIInfo differs from communicator");
            for (const auto &[kind, family] : source._spaces)
            {
                EntitySpace::CheckCommunicator(family.identity._mpi.comm, mpi.comm);
                oldSpaces.emplace(kind, family.identity);
            }
            for (size_t slot = 0; slot < snapshot.relations.size(); ++slot)
                if (snapshot.relations[slot])
                    original.emplace(FacadeKey(slot), NativeNamedRelation{*snapshot.relations[slot], Adj_PointToLocal});
            for (const auto &[key, relation] : source._namedRelations)
                DNDS_check_throw_info(original.emplace(key, relation.data).second, "reserved native transition descriptor collision");
            for (const auto &map : input.explicitMaps)
            {
                DNDS_check_throw_info(oldSpaces.count(map.kind) && map.targetRanks.size() == size_t(oldSpaces.at(map.kind).NumOwned()) &&
                                          placements.emplace(map.kind, map.targetRanks).second, "invalid native placement family or extent");
                for (auto target : map.targetRanks) DNDS_check_throw_info(target >= 0 && target < mpi.size, "invalid native placement rank");
                schedule += "m" + std::to_string(int(map.kind)) + ";";
            }
            for (const auto &follow : input.follows)
                schedule += "f" + std::to_string(int(follow.follower)) + ":" + std::to_string(int(follow.leader)) + ":" + follow.follower2leader.Signature();
            schedule += "layers" + std::to_string(ghostLayers) + (cellOld2New ? "local" : "partition");
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
        std::vector<FollowSpec> follows = input.follows;
        if (follows.empty() && placements.count(EntityKind::Cell))
            for (auto kind : {EntityKind::Node, EntityKind::Bnd})
                if (!placements.count(kind))
                    follows.push_back({kind, EntityKind::Cell, kind == EntityKind::Node ? Adj::Node2Cell : Adj::Bnd2Cell});
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
            auto received = Move(ranks, route, Runtime::ElementSchema::Arithmetic<index>());
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
                                                                     source._spaces.at(kind).global, mpi);
            }
            else
                transfer = PermutationTransfer::fromPartition(targets, source._spaces.at(kind).global, mpi);
            auto route = NativeRowTransfer(transfer, mpi);
            spaces.emplace(kind, anchor.CreateSibling(kind, index(route.DestinationRows())));
            routes.emplace(kind, std::move(route));
            if (cellOld2New || assigned != placements.end())
                result.transfers.emplace(kind, transfer);
            allTransfers.emplace(kind, std::move(transfer));
        }
        std::map<EntityKind, std::map<index, index>> remap;
        for (const auto &[kind, old] : oldSpaces)
        {
            std::vector<index> globals;
            std::vector<Runtime::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::set<index> referenced;
                for (const auto &[key, values] : original) if (key.to == kind)
                {
                    RelationReadAccess read(key, oldSpaces.at(key.from), old, values.state, values.rows.owned, &values.rows.ghost);
                    for (index row = 0; row < read.NumOwned(); ++row)
                        for (auto global : read.GlobalRow(row)) if (global != UnInitIndex) referenced.insert(global);
                }
                globals.assign(referenced.begin(), referenced.end());
                for (size_t i = 0; i < globals.size(); ++i) requests.push_back({old.Owner(globals[i]), old.OwnerLocalRow(globals[i]), i}); });
            RArray values(Runtime::RowLayout::Dense(size_t(old.NumOwned()), 1));
            {
                auto write = values.Write();
                for (size_t i = 0; i < values.Rows(); ++i)
                    write.View()(i, 0) = allTransfers.at(kind).newGlobalIndices[i];
            }
            Runtime::RowTransferPlan route(mpi.comm, values.Rows(), globals.size(), requests);
            auto received = Move(values, route, Runtime::ElementSchema::Arithmetic<index>());
            auto read = received.Read();
            for (size_t i = 0; i < globals.size(); ++i)
                remap[kind].emplace(globals[i], read.View()(i, 0));
        }
        std::map<AdjKind, RArray> movedRelations;
        for (const auto &[key, values] : original)
        {
            RArray globalRows = [&]
            {
                std::vector<std::vector<index>> rows;
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    RelationReadAccess read(key, oldSpaces.at(key.from), oldSpaces.at(key.to), values.state, values.rows.owned, &values.rows.ghost);
                    rows.resize(size_t(read.NumOwned()));
                    for (index row = 0; row < read.NumOwned(); ++row)
                    {
                        auto &entries = rows[size_t(row)];
                        entries = read.GlobalRow(row);
                        for (auto &entry : entries) if (entry != UnInitIndex) entry = remap.at(key.to).at(entry);
                        if (key.order == RelationOrder::SortedUnique) std::sort(entries.begin(), entries.end());
                    } });
                return Rows(rows);
            }();
            auto moved = Move(globalRows, routes.at(key.from), Runtime::ElementSchema::Arithmetic<index>());
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { movedRelations.emplace(key, std::move(moved)); });
        }
        NativeOwnedMeshData owned;
        owned.identity = {source.MeshNamespace(), source.IdHighWater()};
        owned.requirements = source.Requirements();
        owned.dimension = snapshot.semantics.dim;
        owned.periodic = snapshot.semantics.isPeriodic;
        owned.periodicInfo = snapshot.semantics.periodicInfo;
        owned.spaces = spaces;
        if (source.Provenance())
        {
            owned.provenance = *source.Provenance();
            for (auto &[kind, values] : owned.provenance->sourceIds)
                values = Move(values, routes.at(kind), Runtime::ElementSchema::Arithmetic<index>());
        }
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (size_t slot = 0; slot < snapshot.relations.size(); ++slot)
                if (snapshot.relations[slot])
                {
                    auto rows = movedRelations.at(FacadeKey(slot));
                    const auto &shape = snapshot.relations[slot]->owned.Layout();
                    if (!shape.IsVariable())
                        rows = RArray::FromStorage(Runtime::RowLayout::Dense(rows.Rows(), shape.Width()), rows.Storage());
                    owned.relations[slot] = std::move(rows);
                }
            for (const auto &[key, unused] : source._namedRelations)
                owned.namedRelations.emplace(key, movedRelations.at(key)); });
        auto companion = [&](auto &destination, const auto &originalPair, EntityKind kind, const Runtime::ElementSchema &schema)
        { if (originalPair) destination = Move(originalPair->owned, routes.at(kind), schema); };
        companion(owned.coords, snapshot.coords, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        companion(owned.nodeWallDist, snapshot.nodeWallDist, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        const auto elementSchema = Runtime::ElementSchema::For<ElemInfo>();
        companion(owned.cellInfo, snapshot.cellInfo, EntityKind::Cell, elementSchema);
        companion(owned.bndInfo, snapshot.bndInfo, EntityKind::Bnd, elementSchema);
        companion(owned.faceInfo, snapshot.faceInfo, EntityKind::Face, elementSchema);
        companion(owned.edgeInfo, snapshot.edgeInfo, EntityKind::Edge, elementSchema);
        const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
        companion(owned.cellNodesPbi, snapshot.cellNodesPbi, EntityKind::Cell, bits);
        companion(owned.bndNodesPbi, snapshot.bndNodesPbi, EntityKind::Bnd, bits);
        companion(owned.cellFacesPbi, snapshot.cellFacesPbi, EntityKind::Cell, bits);
        companion(owned.faceNodesPbi, snapshot.faceNodesPbi, EntityKind::Face, bits);
        companion(owned.cellEdgesPbi, snapshot.cellEdgesPbi, EntityKind::Cell, bits);
        companion(owned.edgeNodesPbi, snapshot.edgeNodesPbi, EntityKind::Edge, bits);
        for (const auto &[kind, old] : oldSpaces)
        {
            const auto &originalIds = source._originalIds.at(kind);
            const auto &baseRoute = routes.at(kind);
            auto idRoute = Runtime::RowTransferPlan::Known(std::make_shared<Runtime::CommunicationContext>(mpi.comm),
                                                           originalIds.Rows(), baseRoute.DestinationRows(), baseRoute.SendRows(), baseRoute.ReceiveRows());
            owned.ids.emplace(kind, Move(originalIds, idRoute, Runtime::ElementSchema::Arithmetic<index>()));
        }
        result.state = NativeMeshAssembler::Build(mpi, std::move(owned), ghostLayers);
        return result;
    }
}
