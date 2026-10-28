#include "Geom/Connectivity/NativeConnectivity.hpp"
#include "Geom/Connectivity/NativePeriodicMatch.hpp"
#include "Geom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include <set>

namespace DNDS::Geom
{
    namespace
    {
        template <class T>
        Runtime::Array<T> BuildRows(const std::vector<std::vector<T>> &rows, size_t first, size_t count)
        {
            return Runtime::BuildArrayRows<T>(count, [&](size_t row)
                                              { return rows[first + row].size(); }, [&](size_t row, auto output)
                                              {
                const auto &values = rows[first + row];
                std::copy(values.begin(), values.end(), output.begin()); });
        }
        Runtime::Array<index> Transfer(const Runtime::RowTransferPlan &route, const Runtime::Array<index> &source,
                                       const std::shared_ptr<Runtime::CommunicationContext> &context)
        {
            Runtime::Array<index> output(route.DestinationLayout(source.Layout()));
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = context;
            Runtime::PayloadPlan payload(route, source.Layout(), output.Layout(), Runtime::ElementSchema::Arithmetic<index>());
            payload.Bind(source, output, Runtime::TransferTransport::Direct, options).Begin().Wait();
            return output;
        }
        void SortUnique(std::vector<index> &values)
        {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
    }

    RelationPeriodicBits::RelationPeriodicBits(const RelationReadAccess &relation, Runtime::Array<NodePeriodicBits> owned,
                                               std::optional<Runtime::Array<NodePeriodicBits>> ghost)
        : _source(relation.Source()), _target(relation.Target()), _kind(relation.Kind()),
          _owned(std::move(owned)), _ghost(std::move(ghost)), _ownedRead(_owned.Read())
    {
        DNDS_check_throw_info(_owned.GetDevice() == Runtime::Device::Host() &&
                                  (!_ghost || _ghost->GetDevice() == Runtime::Device::Host()),
                              "periodic relation setup requires host placement");
        _view.owned = _ownedRead->View();
        if (_ghost)
        {
            _ghostRead.emplace(_ghost->Read());
            _view.ghost = _ghostRead->View();
        }
        Validate(relation);
    }
    void RelationPeriodicBits::Validate(const RelationReadAccess &relation) const
    {
        if (_meshRead)
            (void)_meshRead->HostView();
        DNDS_check_throw_info(_source.SameOrder(relation.Source()) && _target.SameOrder(relation.Target()) && _kind == relation.Kind(),
                              "periodic companion belongs to another relation/order");
        DNDS_check_throw_info(_view.owned.Rows() == size_t(relation.NumOwned()) &&
                                  _view.owned.Rows() + _view.ghost.Rows() == size_t(relation.Rows()),
                              "periodic companion row extent differs from relation");
        for (index row = 0; row < relation.Rows(); ++row)
            DNDS_check_throw_info(_view.RowSize(size_t(row)) == relation.RowSize(row), "periodic companion slots differ from relation");
    }

    void NativeConnectivity::ValidateCollectively(const std::function<void()> &validate) const
    {
        int valid = 1;
        try
        {
            validate();
        }
        catch (...)
        {
            valid = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _mpi.comm);
        DNDS_check_throw_info(valid, "native connectivity setup validation failed on a rank");
    }
    void NativeConnectivity::Register(RelationReadAccess relation)
    {
        relation.Validate();
        relation.CheckCommunicator(_mpi.comm);
        const auto kind = relation.Kind();
        DNDS_check_throw_info(!_relations.count(kind), "duplicate native setup relation");
        for (const auto *space : {&relation.Source(), &relation.Target()})
        {
            auto found = _spaces.find(space->Kind());
            DNDS_check_throw_info(found == _spaces.end() || found->second.SameOrder(*space),
                                  "native setup registry entity order mismatch");
        }
        _spaces.emplace(relation.Source().Kind(), relation.Source());
        _spaces.emplace(relation.Target().Kind(), relation.Target());
        _relations.emplace(kind, std::make_shared<RelationReadAccess>(std::move(relation)));
    }
    std::shared_ptr<const RelationReadAccess> NativeConnectivity::Find(const AdjKind &kind) const
    {
        kind.Validate();
        const auto found = _relations.find(kind);
        return found == _relations.end() ? nullptr : found->second;
    }

    Runtime::Array<index> NativeConnectivity::Fetch(const RelationReadAccess &relation, const std::vector<index> &globals) const
    {
        std::vector<Runtime::RowRequest> requests;
        ValidateCollectively([&]
                             {
            relation.CheckActive();
            relation.Source().Validate();
            relation.Target().Validate();
            if (!_fetchValues.count(relation.Kind()))
            {
                // Registration retains immutable input leases. Share globally
                // indexed native storage; convert local numbering once per
                // relation, rather than rebuilding every owned row per hop.
                const auto *owned = dynamic_cast<const RelationReadAccess::ArrayHold *>(relation._owned.get());
                if (relation.State() == Adj_PointToGlobal && owned)
                    _fetchValues.emplace(relation.Kind(), owned->array);
                else
                {
                    _fetchValues.emplace(relation.Kind(), Runtime::BuildArrayRows<index>(size_t(relation.NumOwned()),
                        [&](size_t row) { return relation.RowSize(index(row)); }, [&](size_t row, auto output)
                        {
                            for (size_t slot = 0; slot < output.size(); ++slot)
                                output[slot] = relation.ConvertedIndex(index(row), slot, Adj_PointToGlobal);
                        }));
                }
            }
            for (size_t row = 0; row < globals.size(); ++row)
                requests.push_back({relation.Source().Owner(globals[row]), relation.Source().OwnerLocalRow(globals[row]), row}); });
        const auto &values = _fetchValues.at(relation.Kind());
        auto context = std::make_shared<Runtime::CommunicationContext>(_mpi.comm);
        Runtime::RowTransferPlan route(context, size_t(relation.NumOwned()), globals.size(), requests,
                                       {relation.Source().Stamp(), relation.Target().Stamp()});
        return Transfer(route, values, context);
    }
    GhostResult NativeConnectivity::Evaluate(const CompiledGhostTree &tree) const
    {
        // Compare the compiled schedule before entering its per-hop collectives.
        // Equal schedules are a collective caller requirement, checked here.
        auto schedule = tree.dump();
        int length = _mpi.rank == 0 ? int(schedule.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, _mpi.comm);
        std::string canonical(size_t(length), ' ');
        if (_mpi.rank == 0)
            canonical = schedule;
        MPI_Bcast(canonical.data(), length, MPI_CHAR, 0, _mpi.comm);
        ValidateCollectively([&]
                             {
            DNDS_check_throw_info(schedule == canonical, "ghost traversal schedule differs between ranks");
            for (auto kind : tree.requiredAdjs())
            {
                auto found = _relations.find(kind);
                DNDS_check_throw_info(found != _relations.end(), "native ghost traversal relation is missing");
                found->second->Validate();
            } });
        if (tree.levels.empty())
            return {};
        std::vector<std::vector<index>> sets(size_t(tree.totalNodes));
        for (const auto &root : tree.levels.front())
        {
            auto &space = _spaces.at(root.kind);
            for (index row = 0; row < space.NumOwned(); ++row)
                sets[size_t(root.nodeId)].push_back(space.LocalToGlobal(row));
        }
        GhostResult result;
        for (int level = 0; level <= tree.maxLevel; ++level)
        {
            for (const auto &entry : tree.levels[size_t(level)])
            {
                if (!entry.collect)
                    continue;
                const auto &space = _spaces.at(entry.kind);
                auto &output = result.ghostIndices[entry.kind];
                for (auto global : sets[size_t(entry.nodeId)])
                    if (space.Owner(global) != _mpi.rank)
                        output.push_back(global);
                SortUnique(output);
            }
            if (level == tree.maxLevel)
                break;
            for (const auto &entry : tree.levels[size_t(level + 1)])
            {
                const auto &relation = *_relations.at(entry.hop);
                auto &output = sets[size_t(entry.nodeId)];
                std::vector<index> remote;
                ValidateCollectively([&]
                                     {
                    relation.CheckActive();
                    const auto &source = relation.Source();
                    const auto first = (*source._global)(_mpi.rank, 0);
                    // Local incidence is already retained and validated. A DSL
                    // traversal needs its union, not another transferred copy
                    // of every local row. Only remote parents need a route.
                    for (auto global : sets[size_t(entry.parentId)])
                    {
                        if (global >= first && global - first < source.NumOwned())
                        {
                            const auto row = global - first;
                            for (size_t slot = 0; slot < relation.RowSize(row); ++slot)
                            {
                                const auto value = relation.ConvertedIndex(row, slot, Adj_PointToGlobal);
                                if (value != UnInitIndex) output.push_back(value);
                            }
                        }
                        else remote.push_back(global);
                    } });
                int needsRemote = !remote.empty();
                MPI_Allreduce(MPI_IN_PLACE, &needsRemote, 1, MPI_INT, MPI_MAX, _mpi.comm);
                if (needsRemote)
                {
                    // All ranks participate when any peer needs a row, even
                    // ranks with no requests of their own. An entirely local
                    // hop needs neither a payload copy nor an empty route.
                    auto rows = Fetch(relation, remote);
                    auto read = rows.Read();
                    auto view = read.View();
                    for (size_t row = 0; row < view.Rows(); ++row)
                        for (size_t slot = 0; slot < view.RowSize(row); ++slot)
                            if (view(row, slot) != UnInitIndex)
                                output.push_back(view(row, slot));
                }
                SortUnique(output);
            }
        }
        int mask = 0;
        for (auto it = result.ghostIndices.begin(); it != result.ghostIndices.end();)
            if (it->second.empty())
                it = result.ghostIndices.erase(it);
            else
            {
                mask |= 1 << int(it->first);
                ++it;
            }
        MPI_Allreduce(MPI_IN_PLACE, &mask, 1, MPI_INT, MPI_BOR, _mpi.comm);
        for (int kind = 0; kind < int(EntityKind::NUM_KINDS); ++kind)
            if (mask & (1 << kind))
                result.activeKinds.insert(EntityKind(kind));
        return result;
    }

    EntitySpace NativeConnectivity::MakeSpace(EntityKind kind, index owned, const EntitySpace &anchor)
    {
        auto global = std::make_shared<GlobalOffsetsMapping>();
        global->setMPIAlignBcast(anchor._mpi, owned);
        std::vector<index> empty;
        auto ghosts = std::make_shared<OffsetAscendIndexMapping>((*global)(anchor._mpi.rank, 0), owned, empty, *global, anchor._mpi);
        Runtime::VersionTracker versions;
        EntitySpace result(kind, anchor._mpi, owned, 0, anchor._identity, versions.Stamp(), global, ghosts, versions);
        result._publishedMaps = true;
        return result;
    }

    NativeInterpolation NativeConnectivity::Interpolate(const RelationReadAccess &cone, EntityKind entityKind,
                                                        const SubEntityQueryPbi &query, const OwnershipResolverMulti &resolver,
                                                        const RelationPeriodicBits *periodic)
    {
        const auto mpi = cone.MPI();
        NativeConnectivity setup(mpi);
        std::optional<ConnectivityRows::SubEntityRows> rows;
        std::vector<std::vector<NodePeriodicBits>> nodeBits, parentBits;

        std::vector<bool> owned;
        setup.ValidateCollectively([&]
                                   {
            cone.Validate();
            DNDS_check_throw_info(cone.Target().Kind() == EntityKind::Node && entityKind != cone.Source().Kind() &&
                                      (entityKind == EntityKind::Face || entityKind == EntityKind::Edge) && bool(query.numSubEntities) && bool(query.describe) &&
                                      bool(query.extractNodes) && bool(resolver), "invalid native interpolation query/kinds");
            DNDS_check_throw_info(cone.Rows() == cone.Source().NumLocal(), "interpolation requires every parent ghost row");
            if (periodic)
            {
                periodic->Validate(cone);
                DNDS_check_throw_info(bool(query.extractPbi), "periodic interpolation requires ordered bit extraction");
            }
            for (index row = 0; row < cone.Rows(); ++row)
            {
                DNDS_check_throw_info(query.numSubEntities(row) >= 0, "negative subentity count");
                for (int sub = 0; sub < query.numSubEntities(row); ++sub)
                {
                    const auto desc = query.describe(row, sub);
                    DNDS_check_throw_info(desc.nVertices > 0 && desc.nVertices <= desc.nNodes && desc.nNodes <= 10,
                                          "invalid subentity node/vertex count");
                }
                for (size_t slot = 0; slot < cone.RowSize(row); ++slot)
                {
                    auto entry = cone.ReadEntry(row, slot);
                    DNDS_check_throw_info(!entry.absent && !entry.unavailable,
                                          "interpolation requires local node closure");
                }
            }
            SubEntityQueryPbi enumeration = query;
            if (periodic)
                enumeration.matchExtra = [&](index parent, int sub, index entity, index originalParent, int originalSub)
                {
                    if (query.matchExtra && !query.matchExtra(parent, sub, entity, originalParent, originalSub)) return false;
                    auto extract = [&](index p, int slot)
                    {
                        const auto desc = query.describe(p, slot);
                        std::vector<index> nodes(size_t(desc.nNodes));
                        std::vector<NodePeriodicBits> bits(size_t(desc.nNodes));
                        query.extractNodes(p, slot, [&](int j) { return cone.ConvertedIndex(p, size_t(j), Adj_PointToGlobal); }, nodes.data());
                        query.extractPbi(p, slot, [&](int j) { return (*periodic)(p, size_t(j)); }, bits.data());
                        std::vector<std::pair<index, uint8_t>> result;
                        for (size_t j = 0; j < nodes.size(); ++j) result.emplace_back(nodes[j], uint8_t(bits[j]));
                        return result;
                    };
                    auto current = extract(parent, sub), original = extract(originalParent, originalSub);
                    return bool(NativePeriodicFrame(std::move(current), std::move(original)));
                };
            rows.emplace(ConnectivityRows::EnumerateSubEntities([&](index row, int slot)
            {
                DNDS_check_throw_info(slot >= 0 && size_t(slot) < cone.RowSize(row), "node extraction slot outside cone");
                return cone.ConvertedIndex(row, size_t(slot), Adj_PointToLocal, false);
            }, enumeration, cone.Rows(), cone.Target().NumLocal()));
            nodeBits.resize(size_t(rows->nEntities));
            parentBits.resize(size_t(cone.Rows()));
            if (periodic)
            {
                for (index parent = 0; parent < cone.Rows(); ++parent)
                {
                    parentBits[size_t(parent)].resize(rows->parent2entity[size_t(parent)].size());
                    for (size_t sub = 0; sub < rows->parent2entity[size_t(parent)].size(); ++sub)
                    {
                        auto entity = rows->parent2entity[size_t(parent)][sub];
                        auto desc = query.describe(parent, int(sub));
                        std::vector<NodePeriodicBits> bits(size_t(desc.nNodes));
                        std::vector<index> nodes(size_t(desc.nNodes));
                        query.extractPbi(parent, int(sub), [&](int slot)
                        {
                            DNDS_check_throw_info(slot >= 0 && size_t(slot) < cone.RowSize(parent), "periodic extraction slot outside cone");
                            return (*periodic)(parent, size_t(slot));
                        }, bits.data());
                        query.extractNodes(parent, int(sub), [&](int slot)
                        { return cone.ConvertedIndex(parent, size_t(slot), Adj_PointToLocal, false); }, nodes.data());
                        auto &stored = nodeBits[size_t(entity)];
                        if (stored.empty()) stored = bits;
                        const auto &entityNodes = rows->entity2node[size_t(entity)];
                        DNDS_check_throw_info(entityNodes.size() == nodes.size(), "periodic subentity node count mismatch");
                        std::vector<std::pair<index, uint8_t>> current, original;
                        for (size_t node = 0; node < nodes.size(); ++node)
                        {
                            current.emplace_back(nodes[node], uint8_t(bits[node]));
                            original.emplace_back(entityNodes[node], uint8_t(stored[node]));
                        }
                        // Repeated periodic node IDs are distinguished by their
                        // per-node frame bits, matching the legacy sorted-pair math.
                        auto uniform = NativePeriodicFrame(std::move(current), std::move(original));
                        DNDS_check_throw_info(uniform, "periodic subentity has nonuniform frame transform");
                        parentBits[size_t(parent)][sub] = NodePeriodicBits{*uniform};
                    }
                }
            }
            owned.resize(size_t(rows->nEntities));
            for (index entity = 0; entity < rows->nEntities; ++entity)
            {
                const auto &parents = rows->entity2parent[size_t(entity)];
                std::vector<MPI_int> ranks;
                bool anyOwned = false, anyGhost = false;
                for (auto parent : parents)
                {
                    ranks.push_back(cone.Source().Owner(cone.Source().LocalToGlobal(parent)));
                    anyOwned = anyOwned || parent < cone.NumOwned();
                    anyGhost = anyGhost || parent >= cone.NumOwned();
                }
                owned[size_t(entity)] = anyOwned && (!anyGhost || resolver(parents, ranks, cone.NumOwned()).owned);
            } });
        int entityKinds[]{int(entityKind), -int(entityKind)};
        MPI_Allreduce(MPI_IN_PLACE, entityKinds, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(entityKinds[0] == -entityKinds[1], "interpolation entity kind differs between ranks");
        int periodicFlags[]{int(bool(periodic)), -int(bool(periodic))};
        MPI_Allreduce(MPI_IN_PLACE, periodicFlags, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(periodicFlags[0] == -periodicFlags[1], "periodic interpolation differs between ranks");
        std::vector<index> oldToOwned(size_t(rows->nEntities), UnInitIndex);
        index nOwned = 0;
        for (index entity = 0; entity < rows->nEntities; ++entity)
            if (owned[size_t(entity)])
                oldToOwned[size_t(entity)] = nOwned++;
        auto entities = MakeSpace(entityKind, nOwned, cone.Source());
        std::vector<std::vector<index>> e2n(static_cast<size_t>(nOwned)), e2p(static_cast<size_t>(nOwned));
        std::vector<std::vector<NodePeriodicBits>> e2nBits(static_cast<size_t>(nOwned));
        std::vector<std::vector<ElemInfo>> info(static_cast<size_t>(nOwned));
        std::vector<std::vector<index>> messages(size_t(mpi.size));
        for (index entity = 0; entity < rows->nEntities; ++entity)
        {
            auto output = oldToOwned[size_t(entity)];
            if (output == UnInitIndex)
                continue;
            const auto global = entities.LocalToGlobal(output);
            for (auto node : rows->entity2node[size_t(entity)])
                e2n[size_t(output)].push_back(cone.Target().LocalToGlobal(node));
            for (auto parent : rows->entity2parent[size_t(entity)])
            {
                auto parentGlobal = cone.Source().LocalToGlobal(parent);
                e2p[size_t(output)].push_back(parentGlobal);
                if (parent >= cone.NumOwned())
                {
                    const auto &slots = rows->parent2entity[size_t(parent)];
                    for (size_t slot = 0; slot < slots.size(); ++slot)
                        if (slots[slot] == entity)
                        {
                            auto &message = messages[size_t(cone.Source().Owner(parentGlobal))];
                            message.insert(message.end(), {parentGlobal, global, index(slot), periodic ? index(uint8_t(parentBits[size_t(parent)][slot])) : 0});
                        }
                }
            }
            e2nBits[size_t(output)] = nodeBits[size_t(entity)];
            info[size_t(output)].push_back(rows->entityElemInfo[size_t(entity)]);
        }
        // One ragged batch per peer, owner-directed and direct typed MPI.
        auto batches = BuildRows(messages, 0, messages.size());
        std::vector<int> sendPresence(size_t(mpi.size)), receivePresence(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
            sendPresence[size_t(peer)] = !messages[size_t(peer)].empty();
        MPI_Alltoall(sendPresence.data(), 1, MPI_INT, receivePresence.data(), 1, MPI_INT, mpi.comm);
        std::vector<std::vector<size_t>> send(size_t(mpi.size)), receive(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            if (sendPresence[size_t(peer)])
                send[size_t(peer)].push_back(size_t(peer));
            if (receivePresence[size_t(peer)])
                receive[size_t(peer)].push_back(size_t(peer));
        }
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        auto route = Runtime::RowTransferPlan::Known(context, size_t(mpi.size), size_t(mpi.size), std::move(send), std::move(receive),
                                                     {cone.Source().Stamp(), cone.Target().Stamp()});
        auto incoming = Transfer(route, batches, context);
        std::vector<std::vector<index>> p2e(size_t(cone.Rows()));
        for (index parent = 0; parent < cone.Rows(); ++parent)
            for (auto entity : rows->parent2entity[size_t(parent)])
            {
                const auto output = oldToOwned[size_t(entity)];
                p2e[size_t(parent)].push_back(output == UnInitIndex ? UnInitIndex : entities.LocalToGlobal(output));
            }
        setup.ValidateCollectively([&]
                                   {
            auto read = incoming.Read();
            auto view = read.View();
            for (size_t peer = 0; peer < view.Rows(); ++peer)
            {
                DNDS_check_throw_info(view.RowSize(peer) % 4 == 0, "malformed interpolation owner message");
                for (size_t slot = 0; slot < view.RowSize(peer); slot += 4)
                {
                    const auto parent = cone.Source().GlobalToLocal(view(peer, slot));
                    const auto entity = view(peer, slot + 1), sub = view(peer, slot + 2);
                    DNDS_check_throw_info(parent && *parent < cone.NumOwned() && entity >= 0 && entity < entities.NumGlobal() &&
                                              entities.Owner(entity) == int(peer) && sub >= 0 && size_t(sub) < p2e[size_t(*parent)].size(),
                                          "interpolation owner message outside entity domain");
                    auto &destination = p2e[size_t(*parent)][size_t(sub)];
                    DNDS_check_throw_info(destination == UnInitIndex || destination == entity, "conflicting interpolation owners");
                    destination = entity;
                    if (periodic)
                    {
                        // The generated entity keeps its OWNER's first-parent
                        // frame. A peer's rank-local first parent can have a
                        // different frame, so its pre-ownership XOR is not the
                        // published parent->entity transform. Send/install the
                        // owner's XOR with the entity ID (legacy triplets sent
                        // only the ID and could leave wrong cross-rank flags).
                        const auto bits = view(peer, slot + 3);
                        DNDS_check_throw_info(bits >= 0 && bits <= 255, "invalid interpolation periodic frame bits");
                        parentBits[size_t(*parent)][size_t(sub)] = NodePeriodicBits{uint8_t(bits)};
                    }
                }
            }
            for (index parent = 0; parent < cone.NumOwned(); ++parent)
                for (auto entity : p2e[size_t(parent)])
                    DNDS_check_throw_info(entity != UnInitIndex, "interpolation owner did not resolve an owned parent slot"); });
        NativeRelation parentRelation({cone.Source().Kind(), entityKind}, cone.Source(), entities, BuildRows(p2e, 0, size_t(cone.NumOwned())));
        // Complete appended parent rows from their owners, including entities
        // seen only through ghost parents. An exposed row never carries a
        // fabricated boundary sentinel for a merely unresolved remote entity.
        RelationGhostRoute parentRoute(cone.Source());
        parentRelation = parentRelation.WithGhostRows(parentRoute);
        NativeInterpolation result{entities, std::move(parentRelation), NativeRelation({entityKind, cone.Target().Kind()}, entities, cone.Target(), BuildRows(e2n, 0, e2n.size())), NativeRelation({entityKind, cone.Source().Kind()}, entities, cone.Source(), BuildRows(e2p, 0, e2p.size())), {}, {}, {}, BuildRows(info, 0, info.size())};
        if (periodic)
        {
            result.parent2entityPbi.emplace(BuildRows(parentBits, 0, size_t(cone.NumOwned())));
            const auto pbiSchema = Runtime::ElementSchema::For<NodePeriodicBits>();
            result.parent2entityGhostPbi.emplace(parentRoute.TransferRows(cone.Source(), *result.parent2entityPbi, pbiSchema));
            result.entity2nodePbi.emplace(BuildRows(e2nBits, 0, e2nBits.size()));
        }
        return result;
    }
}

namespace DNDS::Geom
{
    namespace
    {
        const std::vector<std::pair<AdjKind, MeshRelation>> &MeshRelations()
        {
            static const std::vector<std::pair<AdjKind, MeshRelation>> bindings{
                {Adj::Cell2Node, MeshRelation::CellNodes}, {Adj::Bnd2Node, MeshRelation::BoundaryNodes}, {Adj::Bnd2Cell, MeshRelation::BoundaryCells}, {Adj::Cell2Cell, MeshRelation::CellCells}, {Adj::Cell2CellFace, MeshRelation::CellFaceCells}, {Adj::Node2Cell, MeshRelation::NodeCells}, {Adj::Node2Bnd, MeshRelation::NodeBoundaries}, {Adj::Cell2Face, MeshRelation::CellFaces}, {Adj::Face2Node, MeshRelation::FaceNodes}, {Adj::Face2Cell, MeshRelation::FaceCells}, {Adj::Face2Bnd, MeshRelation::FaceBoundaries}, {Adj::Bnd2Face, MeshRelation::BoundaryFaces}, {Adj::Cell2Edge, MeshRelation::CellEdges}, {Adj::Edge2Node, MeshRelation::EdgeNodes}, {Adj::Edge2Cell, MeshRelation::EdgeCells}};
            return bindings;
        }
        MeshRelation QueryRelation(AdjKind kind)
        {
            for (auto binding : MeshRelations())
                if (binding.first == kind)
                    return binding.second;
            DNDS_check_throw_info(false, "no mesh facade relation for requested setup kind");
            return MeshRelation::Count;
        }
    }
    RelationReadAccess::RelationReadAccess(AdjKind kind, std::shared_ptr<MeshReadAccess> meshRead,
                                           EntitySpace source, EntitySpace target)
        : _source(std::move(source)), _target(std::move(target)),
          _sourceLease(_source._versions.Read()), _targetLease(_target._versions.Read()), _kind(kind)
    {
        DNDS_check_throw_info(meshRead, "mesh relation needs a retained scope");
        auto view = meshRead->HostView();
        const auto relation = QueryRelation(kind);
        DNDS_check_throw_info(view.HasRelation(relation), "mesh scope does not contain requested setup relation");
        _view = meshRead->_storage.relations[size_t(relation)];
        _state = view.RelationState(relation);
        _owned = std::make_unique<MeshHold>(std::move(meshRead));
        Validate();
    }
    RelationReadAccess RelationReadAccess::Current(UnstructuredMesh &mesh, AdjKind kind, MeshReadRequirements requirements)
    {
        auto read = std::make_shared<MeshReadAccess>(mesh.AcquireRead(requirements));
        return RelationReadAccess(kind, std::move(read), mesh.Entities(kind.from), mesh.Entities(kind.to));
    }
    void NativeConnectivity::RegisterMesh(UnstructuredMesh &mesh, MeshReadRequirements requirements)
    {
        std::shared_ptr<MeshReadAccess> read;
        ValidateCollectively([&]
                             {
            EntitySpace::CheckCommunicator(mesh.mpi.comm, _mpi.comm);
            read = std::make_shared<MeshReadAccess>(mesh.AcquireRead(requirements));
            DNDS_check_throw_info(_relations.empty(), "register the current mesh into an empty setup registry"); });
        ValidateCollectively([&]
                             {
            auto view = read->HostView();
            for (auto binding : MeshRelations())
                if (view.HasRelation(binding.second))
                    Register(RelationReadAccess(binding.first, read, mesh.Entities(binding.first.from), mesh.Entities(binding.first.to)));
            if (mesh.HasNativeState())
                for (const auto &key : mesh.NativeRelationKeys())
                    if (!_relations.count(key))
                        Register(mesh.ReadNativeRelation(key)); });
    }
    RelationPeriodicBits::RelationPeriodicBits(const RelationReadAccess &relation)
        : _source(relation.Source()), _target(relation.Target()), _kind(relation.Kind())
    {
        const auto *hold = dynamic_cast<const RelationReadAccess::MeshHold *>(relation._owned.get());
        DNDS_check_throw_info(hold, "periodic companion requires a current-mesh relation scope");
        _meshRead = hold->read;
        DNDS_check_throw_info(_meshRead && _meshRead->HostView().IsPeriodic(), "periodic metadata needs a retained periodic mesh scope");
        const auto &storage = _meshRead->_storage;
        if (_kind == Adj::Cell2Node)
            _view = storage.cell2nodePbi;
        else if (_kind == Adj::Bnd2Node)
            _view = storage.bnd2nodePbi;
        else if (_kind == Adj::Face2Node)
            _view = storage.face2nodePbi;
        else if (_kind == Adj::Edge2Node)
            _view = storage.edge2nodePbi;
        else
            DNDS_check_throw_info(false, "requested relation has no node-periodic companion");
        Validate(relation);
    }
}

namespace DNDS::Geom
{
    NativeGhostPublication NativeConnectivity::BuildGhosted(const GhostResult &discovery) const
    {
        NativeGhostPublication result;
        // The registry schedule, not local nonempty discovery, determines MPI
        // participation. Empty owners/requesters execute every route stage.
        std::vector<EntityKind> kinds;
        std::vector<AdjKind> relations;
        for (auto &entry : _spaces)
            kinds.push_back(entry.first);
        for (auto &entry : _relations)
            relations.push_back(entry.first);
        std::sort(kinds.begin(), kinds.end());
        std::sort(relations.begin(), relations.end());
        std::string schedule;
        for (auto kind : relations)
        {
            const auto descriptor = kind.Signature();
            schedule += std::to_string(descriptor.size()) + ":" + descriptor;
        }
        int canonicalSize = _mpi.rank == 0 ? int(schedule.size()) : 0;
        MPI_Bcast(&canonicalSize, 1, MPI_INT, 0, _mpi.comm);
        std::string canonical(size_t(canonicalSize), ' ');
        if (_mpi.rank == 0)
            canonical = schedule;
        MPI_Bcast(canonical.data(), canonicalSize, MPI_CHAR, 0, _mpi.comm);
        ValidateCollectively([&]
                             {
            DNDS_check_throw_info(schedule == canonical, "ghost publication registry differs between ranks");
            for (auto &entry : _relations) entry.second->Validate();
            for (auto &entry : discovery.ghostIndices)
            {
                auto found = _spaces.find(entry.first);
                DNDS_check_throw_info(found != _spaces.end(), "ghost discovery names an unregistered entity family");
                for (auto global : entry.second)
                    DNDS_check_throw_info(found->second.Owner(global) != _mpi.rank, "ghost discovery contains an owned or invalid entity");
            } });
        for (auto kind : kinds)
        {
            const auto &space = _spaces.at(kind);
            std::vector<index> globals;
            auto found = discovery.ghostIndices.find(kind);
            if (found != discovery.ghostIndices.end())
                globals = found->second;
            SortUnique(globals);
            auto ghosts = std::make_shared<OffsetAscendIndexMapping>((*space._global)(_mpi.rank, 0), space.NumOwned(),
                                                                     globals, *space._global, _mpi);
            EntitySpace next(kind, _mpi, space.NumOwned(), index(globals.size()), space._identity, space._stamp,
                             space._global, std::move(ghosts), space._versions);
            next._publishedMaps = true;
            result.spaces.emplace(kind, std::move(next));
            result.routes.emplace(kind, RelationGhostRoute(result.spaces.at(kind)));
        }
        using PayloadKey = std::tuple<EntityKind, EntityKind, MeshAdjState, uintptr_t, size_t, size_t, size_t, uintptr_t, uintptr_t>;
        struct Payload
        {
            int ordinal;
            Runtime::Array<index> owned, ghost;
        };
        std::map<PayloadKey, Payload> payloads;
        ValidateCollectively([&]
                             { DNDS_check_throw_info(relations.size() <= size_t(INT_MAX), "too many ghost relation descriptors"); });
        for (size_t ordinal = 0; ordinal < relations.size(); ++ordinal)
        {
            const auto kind = relations[ordinal];
            const auto &input = *_relations.at(kind);
            const auto &view = input._view.owned;
            const PayloadKey key{kind.from, kind.to, input.State(), reinterpret_cast<uintptr_t>(view.data), view.rows,
                                 view.width, view.stride, reinterpret_cast<uintptr_t>(view.offsets), reinterpret_cast<uintptr_t>(view.sizes)};
            auto cached = payloads.find(key);
            const int candidate = cached != payloads.end() ? cached->second.ordinal : -1;
            int range[]{candidate, -candidate};
            // Every rank must reuse the SAME earlier transfer. Local hits on
            // different descriptors do not prove their remote payloads match.
            MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_INT, MPI_MIN, _mpi.comm);
            if (range[0] >= 0 && range[0] == -range[1])
            {
                ValidateCollectively([&]
                                     {
                    NativeRelation alias(kind, result.spaces.at(kind.from), result.spaces.at(kind.to), cached->second.owned);
                    alias._ghost.emplace(cached->second.ghost);
                    auto checked = alias.Read();
                    result.relations.emplace(kind, std::move(alias)); });
                continue;
            }
            std::optional<Runtime::Array<index>> values;
            ValidateCollectively([&]
                                 {
                const auto *hold = dynamic_cast<const RelationReadAccess::ArrayHold *>(input._owned.get());
                if (input.State() == Adj_PointToGlobal && hold)
                    values = hold->array;
                else
                    values = Runtime::BuildArrayRows<index>(size_t(input.NumOwned()), [&](size_t row) { return input.RowSize(index(row)); },
                        [&](size_t row, auto output)
                        {
                            for (size_t slot = 0; slot < output.size(); ++slot)
                                output[slot] = input.ConvertedIndex(index(row), slot, Adj_PointToGlobal);
                        }); });
            NativeRelation owned(kind, result.spaces.at(kind.from), result.spaces.at(kind.to), std::move(*values));
            auto completed = owned.WithGhostRows(result.routes.at(kind.from));
            payloads.insert_or_assign(key, Payload{int(ordinal), completed.Owned(), *completed.Ghost()});
            result.relations.emplace(kind, std::move(completed));
        }
        return result;
    }
}
