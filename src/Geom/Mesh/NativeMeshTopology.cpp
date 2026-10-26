#include "NativeMeshTopology.hpp"
#include <limits>
#include "NativeMeshAssembler.hpp"
#include "NativeMeshClosure.hpp"
#include "NativeConnectivity.hpp"
#include "NativePeriodicMatch.hpp"
#include "FieldTransfer.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <set>

namespace DNDS::Geom
{
    namespace
    {
        namespace R = Runtime;
        using Array = R::Array<index>;

        Array Rows(const std::vector<std::vector<index>> &rows, std::optional<size_t> width = {})
        {
            R::RowBuilder builder(rows.size(), sizeof(index), width ? R::RowLayoutKind::Table : R::RowLayoutKind::CSR, width.value_or(0));
            for (size_t row = 0; row < rows.size(); ++row)
            {
                builder.ResizeRow(row, rows[row].size());
                auto output = builder.Row<index>(row);
                std::copy(rows[row].begin(), rows[row].end(), output.data);
            }
            return Array::FromSharedStorage(builder.Commit());
        }

        Array Transfer(const R::RowTransferPlan &route, const Array &values, const std::shared_ptr<R::CommunicationContext> &context)
        {
            Array result(route.DestinationLayout(values.Layout()));
            R::TransferOptions options;
            options.mode = R::TransferMode::Transient;
            options.context = context;
            R::PayloadPlan(route, values.Layout(), result.Layout(), R::ElementSchema::Arithmetic<index>())
                .Bind(values, result, R::TransferTransport::Direct, options)
                .Begin()
                .Wait();
            return result;
        }

        Array BitRows(const R::Array<NodePeriodicBits> &values)
        {
            auto read = values.Read();
            std::vector<std::vector<index>> rows(values.Rows());
            for (size_t row = 0; row < rows.size(); ++row)
                for (auto bits : read.View()[row])
                    rows[row].push_back(uint8_t(bits));
            return Rows(rows);
        }

        Array Exchange(const MPIInfo &mpi, const std::vector<std::vector<index>> &messages)
        {
            auto values = Rows(messages);
            std::vector<int> outgoing(size_t(mpi.size)), incoming(size_t(mpi.size));
            for (int peer = 0; peer < mpi.size; ++peer)
                outgoing[size_t(peer)] = !messages[size_t(peer)].empty();
            MPI_Alltoall(outgoing.data(), 1, MPI_INT, incoming.data(), 1, MPI_INT, mpi.comm);
            std::vector<std::vector<size_t>> send(size_t(mpi.size)), receive(size_t(mpi.size));
            for (int peer = 0; peer < mpi.size; ++peer)
            {
                if (outgoing[size_t(peer)])
                    send[size_t(peer)].push_back(size_t(peer));
                if (incoming[size_t(peer)])
                    receive[size_t(peer)].push_back(size_t(peer));
            }
            auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
            auto route = R::RowTransferPlan::Known(context, size_t(mpi.size), size_t(mpi.size), std::move(send), std::move(receive));
            return Transfer(route, values, context);
        }

        std::vector<index> CanonicalRoleKey(Elem::Element element, const NativePeriodicNodes &pattern, bool periodic)
        {
            std::vector<index> best;
            for (uint8_t mask = 0; mask < (periodic ? 8 : 1); ++mask)
            {
                auto image = pattern;
                for (auto &[node, frame] : image)
                {
                    DNDS_check_throw_info(frame < 8, "invalid native topology identity image");
                    frame ^= mask;
                }
                std::sort(image.begin(), image.end());
                std::vector<index> key{element.type, index(image.size())};
                for (const auto &[node, frame] : image)
                {
                    key.push_back(node);
                    key.push_back(frame);
                }
                // Interpolation first compares element vertices, then
                // the complete higher-order nodes. Preserve that role
                // distinction: an endpoint cannot become a midside node.
                const auto count = element.GetNumVertices();
                auto vertices = NativePeriodicNodes(pattern.begin(), pattern.begin() + count);
                for (auto &[node, frame] : vertices)
                    frame ^= mask;
                std::sort(vertices.begin(), vertices.end());
                key.push_back(count);
                for (const auto &[node, frame] : vertices)
                {
                    key.push_back(node);
                    key.push_back(frame);
                }
                // Generated 2-D faces have 1-D subfaces (the Element
                // edge API is for 3-D cells). Preserve which midpoint
                // belongs to each edge, and thus Quad9 center identity.
                std::vector<std::vector<index>> edges;
                if (element.GetDim() == 2)
                    for (t_index slot = 0; slot < element.GetNumFaces(); ++slot)
                    {
                        const auto sub = element.ObtainFace(slot);
                        NativePeriodicNodes occurrences(size_t(sub.GetNumNodes()));
                        element.ExtractFaceNodes(slot, pattern, occurrences);
                        for (auto &[node, frame] : occurrences)
                            frame ^= mask;
                        auto corners = NativePeriodicNodes(occurrences.begin(), occurrences.begin() + sub.GetNumVertices());
                        std::sort(corners.begin(), corners.end());
                        std::sort(occurrences.begin(), occurrences.end());
                        std::vector<index> edge{sub.type, sub.GetNumVertices()};
                        for (const auto &[node, frame] : corners)
                        {
                            edge.push_back(node);
                            edge.push_back(frame);
                        }
                        edge.push_back(index(occurrences.size()));
                        for (const auto &[node, frame] : occurrences)
                        {
                            edge.push_back(node);
                            edge.push_back(frame);
                        }
                        edges.push_back(std::move(edge));
                    }
                std::sort(edges.begin(), edges.end());
                key.push_back(index(edges.size()));
                for (const auto &edge : edges)
                {
                    key.push_back(index(edge.size()));
                    key.insert(key.end(), edge.begin(), edge.end());
                }
                if (best.empty() || key < best)
                    best = std::move(key);
            }
            return best;
        }

        SubEntityQueryPbi Query(R::PairReadView<ElemInfo> types, const RelationReadAccess &cone, bool edges,
                                const RelationPeriodicBits *periodic = nullptr)
        {
            SubEntityQueryPbi query;
            query.numSubEntities = [types, edges](index cell)
            { auto element = Elem::Element{types(size_t(cell), 0).getElemType()}; return edges ? element.GetNumEdges() : element.GetNumFaces(); };
            query.describe = [types, edges](index cell, int slot)
            {
                auto element = Elem::Element{types(size_t(cell), 0).getElemType()};
                auto sub = edges ? element.ObtainEdge(t_index(slot)) : element.ObtainFace(t_index(slot));
                return SubEntityDesc{sub.GetNumVertices(), sub.GetNumNodes(), t_index(sub.type)};
            };
            query.extractNodes = [types, edges](index cell, int slot, const std::function<index(int)> &read, index *output)
            {
                auto element = Elem::Element{types(size_t(cell), 0).getElemType()};
                std::vector<index> nodes(size_t(element.GetNumNodes()));
                for (size_t node = 0; node < nodes.size(); ++node)
                    nodes[node] = read(int(node));
                auto sub = edges ? element.ObtainEdge(t_index(slot)) : element.ObtainFace(t_index(slot));
                std::vector<index> values(size_t(sub.GetNumNodes()));
                if (edges)
                    element.ExtractEdgeNodes(t_index(slot), nodes, values);
                else
                    element.ExtractFaceNodes(t_index(slot), nodes, values);
                std::copy(values.begin(), values.end(), output);
            };
            const auto extract = query.extractNodes;
            const auto describe = query.describe;
            query.extractPbi = [types, edges](index cell, int slot, const std::function<NodePeriodicBits(int)> &read, NodePeriodicBits *output)
            {
                auto element = Elem::Element{types(size_t(cell), 0).getElemType()};
                const auto sub = edges ? element.ObtainEdge(t_index(slot)) : element.ObtainFace(t_index(slot));
                std::vector<NodePeriodicBits> nodes(size_t(element.GetNumNodes())), values(size_t(sub.GetNumNodes()));
                for (size_t node = 0; node < nodes.size(); ++node)
                    nodes[node] = read(int(node));
                if (edges)
                    element.ExtractEdgeNodes(t_index(slot), nodes, values);
                else
                    element.ExtractFaceNodes(t_index(slot), nodes, values);
                std::copy(values.begin(), values.end(), output);
            };
            const auto extractPbi = query.extractPbi;
            query.matchExtra = [&cone, extract, describe, extractPbi, periodic](index parent, int sub, index, index originalParent, int originalSub)
            {
                auto fullNodes = [&](index cell, int slot, bool vertices)
                {
                    const auto desc = describe(cell, slot);
                    std::vector<index> nodes(size_t(desc.nNodes));
                    extract(cell, slot, [&](int node)
                            { return cone.ReadEntry(cell, size_t(node)).global; }, nodes.data());
                    std::vector<NodePeriodicBits> bits(nodes.size());
                    if (periodic)
                        extractPbi(cell, slot, [&](int node)
                                   { return (*periodic)(cell, size_t(node)); }, bits.data());
                    NativePeriodicNodes result;
                    for (size_t node = 0; node < size_t(vertices ? desc.nVertices : desc.nNodes); ++node)
                        result.emplace_back(nodes[node], uint8_t(bits[node]));
                    return result;
                };
                // Different image patterns may share canonical vertices. They
                // are distinct entities, not cracked higher-order geometry.
                if (!NativePeriodicFrame(fullNodes(parent, sub, true), fullNodes(originalParent, originalSub, true)))
                    return false;
                DNDS_check_throw_info(CanonicalRoleKey(Elem::Element{static_cast<Elem::ElemType>(describe(parent, sub).typeTag)}, fullNodes(parent, sub, false), periodic != nullptr) ==
                                          CanonicalRoleKey(Elem::Element{static_cast<Elem::ElemType>(describe(originalParent, originalSub).typeTag)}, fullNodes(originalParent, originalSub, false), periodic != nullptr),
                                      "shared native subentity vertices have inconsistent higher-order nodes or edge roles");
                return true;
            };
            return query;
        }

        void AddIdentity(NativeOwnedMeshData &data, const EntitySpace &space, const std::map<EntityKind, index> &starts)
        {
            const auto found = starts.find(space.Kind());
            const index first = found == starts.end() ? 0 : found->second;
            DNDS_check_throw_info(first >= 0 && space.NumGlobal() <= std::numeric_limits<index>::max() - first,
                                  "generated native persistent IDs exceed high-water range");
            Array ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
            {
                auto write = ids.Write();
                for (index row = 0; row < space.NumOwned(); ++row)
                    write.View()(size_t(row), 0) = first + space.LocalToGlobal(row);
            }
            data.spaces.emplace(space.Kind(), space);
            data.ids.emplace(space.Kind(), std::move(ids));
            data.identity.nextIds[space.Kind()] = first + space.NumGlobal();
        }

        // Detached refresh maps use CURRENT global row numbers. Persistent IDs
        // remain payload; no table is sized or addressed by an ID magnitude.
        struct RefreshMap
        {
            EntitySpace oldSpace, newSpace;
            Array oldToNew, newToOld;
        };

        Array Fetch(const MPIInfo &mpi, const EntitySpace &space, const Array &values,
                    const std::vector<index> &globals)
        {
            std::vector<R::RowRequest> requests;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                requests.reserve(globals.size());
                for (size_t row = 0; row < globals.size(); ++row)
                {
                    DNDS_check_throw_info(globals[row] >= 0 && globals[row] < space.NumGlobal(),
                                          "native topology refresh references an unavailable source row");
                    requests.push_back({space.Owner(globals[row]), space.OwnerLocalRow(globals[row]), row});
                } });
            auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
            R::RowTransferPlan route(context, size_t(space.NumOwned()), globals.size(), requests);
            return Transfer(route, values, context);
        }

        std::vector<std::vector<index>> IdentityKeys(const MPIInfo &mpi, const EntitySpace &nodes,
                                                     const Array &nodeIds, const Array &cone,
                                                     const R::Array<ElemInfo> &information,
                                                     const std::optional<R::Array<NodePeriodicBits>> &periodic)
        {
            std::vector<index> globals;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::set<index> unique;
                auto read = cone.Read();
                for (size_t row = 0; row < cone.Rows(); ++row)
                    for (index global : read.View()[row]) unique.insert(global);
                globals.assign(unique.begin(), unique.end()); });
            auto fetched = Fetch(mpi, nodes, nodeIds, globals);
            std::vector<std::vector<index>> keys;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto ids = fetched.Read(), rows = cone.Read();
                auto info = information.Read();
                std::optional<R::Array<NodePeriodicBits>::ReadLease> bits;
                if (periodic) bits.emplace(periodic->Read());
                std::map<index, index> idByGlobal;
                for (size_t row = 0; row < globals.size(); ++row)
                    idByGlobal.emplace(globals[row], ids.View()(row, 0));
                DNDS_check_throw_info(information.Rows() == cone.Rows(), "native topology identity information extent mismatch");
                keys.resize(cone.Rows());
                for (size_t row = 0; row < keys.size(); ++row)
                {
                    NativePeriodicNodes pattern;
                    for (size_t slot = 0; slot < rows.View().RowSize(row); ++slot)
                        pattern.emplace_back(idByGlobal.at(rows.View()(row, slot)), bits ? uint8_t(bits->View()(row, slot)) : 0);
                    // Same full-node/image equivalence as NativePeriodicFrame:
                    // orientation may change, and a uniform XOR frame is free.
                    keys[row] = CanonicalRoleKey(Elem::Element{info.View()(row, 0).getElemType()}, pattern, bool(periodic));
                } });
            return keys;
        }

        RefreshMap RecoverIdentity(const NativeMeshState &source, const MPIInfo &mpi,
                                   const NativeOwnedMeshData &previous, NativeOwnedMeshData &data,
                                   const EntitySpace &space, MeshRelation coneSlot,
                                   const R::Array<ElemInfo> &information,
                                   const std::optional<R::Array<NodePeriodicBits>> &bits,
                                   std::optional<index> generatedFloor = {})
        {
            const auto kind = space.Kind();
            const auto old = source.Entities(kind);
            const auto &oldInfo = kind == EntityKind::Face ? *previous.faceInfo : *previous.edgeInfo;
            const auto &oldBits = kind == EntityKind::Face ? previous.faceNodesPbi : previous.edgeNodesPbi;
            auto oldKeys = IdentityKeys(mpi, previous.spaces.at(EntityKind::Node), previous.ids.at(EntityKind::Node),
                                        *previous.relations[size_t(coneSlot)], oldInfo, oldBits);
            auto newKeys = IdentityKeys(mpi, previous.spaces.at(EntityKind::Node), previous.ids.at(EntityKind::Node),
                                        *data.relations[size_t(coneSlot)], information, bits);
            std::vector<std::vector<index>> messages(size_t(mpi.size));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto ids = previous.ids.at(kind).Read();
                auto append = [&](const std::vector<std::vector<index>> &keys, const EntitySpace &family, bool isNew)
                {
                    for (size_t row = 0; row < keys.size(); ++row)
                    {
                        uint64_t hash = 1469598103934665603ULL;
                        // Routing uses the full-node part. Vertex roles remain
                        // in the exact key; sharing a hash server is harmless.
                        for (size_t slot = 0; slot < size_t(2 + 2 * keys[row][1]); ++slot)
                            hash = (hash ^ uint64_t(keys[row][slot])) * 1099511628211ULL;
                        auto &message = messages[size_t(hash % uint64_t(mpi.size))];
                        message.insert(message.end(), {int(isNew), index(row), family.LocalToGlobal(index(row)),
                                                       isNew ? UnInitIndex : ids.View()(row, 0), index(keys[row].size())});
                        message.insert(message.end(), keys[row].begin(), keys[row].end());
                    }
                };
                append(oldKeys, old, false);
                append(newKeys, space, true); });
            auto incoming = Exchange(mpi, messages);
            std::vector<std::vector<index>> replies(size_t(mpi.size));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                struct Entry { int rank; index row, global, id; };
                std::map<std::vector<index>, Entry> oldByKey, newByKey;
                auto read = incoming.Read();
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    const auto message = read.View()[size_t(peer)];
                    for (size_t slot = 0; slot < message.size();)
                    {
                        DNDS_check_throw_info(message.size() - slot >= 5, "malformed native topology identity record");
                        const bool isNew = message[slot] == 1;
                        const index count = message[slot + 4];
                        DNDS_check_throw_info((message[slot] == 0 || isNew) && count >= 2 && size_t(count) <= message.size() - slot - 5,
                                              "malformed native topology identity key");
                        std::vector<index> key(message.data() + slot + 5, message.data() + slot + 5 + size_t(count));
                        DNDS_check_throw_info((isNew ? newByKey : oldByKey).emplace(std::move(key), Entry{peer, message[slot + 1], message[slot + 2], message[slot + 3]}).second,
                                              "duplicate canonical key in native topology refresh");
                        slot += 5 + size_t(count);
                    }
                }
                for (const auto &[key, fresh] : newByKey)
                {
                    const auto found = oldByKey.find(key);
                    const auto oldGlobal = found == oldByKey.end() ? UnInitIndex : found->second.global;
                    const auto id = found == oldByKey.end() || generatedFloor ? UnInitIndex : found->second.id;
                    auto &reply = replies[size_t(fresh.rank)];
                    reply.insert(reply.end(), {1, fresh.row, id, oldGlobal});
                }
                for (const auto &[key, prior] : oldByKey)
                {
                    const auto found = newByKey.find(key);
                    auto &reply = replies[size_t(prior.rank)];
                    reply.insert(reply.end(), {0, prior.row, found == newByKey.end() ? UnInitIndex : found->second.global, 0});
                } });
            auto returned = Exchange(mpi, replies);
            std::vector<index> ids(size_t(space.NumOwned()), UnInitIndex), oldGlobals(ids.size(), UnInitIndex), newGlobals(size_t(old.NumOwned()), UnInitIndex);
            index missing = 0;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::vector<bool> seenOld(newGlobals.size()), seenNew(ids.size());
                auto read = returned.Read();
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    const auto message = read.View()[size_t(peer)];
                    DNDS_check_throw_info(message.size() % 4 == 0, "malformed native topology identity reply");
                    for (size_t slot = 0; slot < message.size(); slot += 4)
                    {
                        const auto row = message[slot + 1];
                        auto &seen = message[slot] == 1 ? seenNew : seenOld;
                        DNDS_check_throw_info(row >= 0 && size_t(row) < seen.size() && !seen[size_t(row)], "invalid native topology identity reply row");
                        seen[size_t(row)] = true;
                        if (message[slot] == 1) { ids[size_t(row)] = message[slot + 2]; oldGlobals[size_t(row)] = message[slot + 3]; }
                        else newGlobals[size_t(row)] = message[slot + 2];
                    }
                }
                DNDS_check_throw_info(std::all_of(seenOld.begin(), seenOld.end(), [](bool value){return value;}) &&
                                      std::all_of(seenNew.begin(), seenNew.end(), [](bool value){return value;}), "incomplete native topology identity replies");
                missing = std::count(ids.begin(), ids.end(), UnInitIndex); });
            index prefix = 0, total = 0;
            MPI_Exscan(&missing, &prefix, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
                prefix = 0;
            MPI_Allreduce(&missing, &total, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            const index first = generatedFloor.value_or(source.NextEntityId(kind));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(total <= std::numeric_limits<index>::max() - first, "refreshed native persistent IDs exceed high-water range");
                index next = first + prefix;
                for (auto &id : ids) if (id == UnInitIndex) id = next++;
                std::vector<std::vector<index>> idRows;
                for (auto id : ids) idRows.push_back({id});
                data.ids[kind] = Rows(idRows, 1);
                data.identity.nextIds[kind] = first + total; });
            std::optional<RefreshMap> result;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::vector<std::vector<index>> forward, reverse;
                for (auto global : newGlobals) forward.push_back({global});
                for (auto global : oldGlobals) reverse.push_back({global});
                data.spaces.emplace(kind, space);
                result.emplace(RefreshMap{old, space, Rows(forward, 1), Rows(reverse, 1)}); });
            return std::move(*result);
        }

        Array RefreshSourceRows(const MPIInfo &mpi, const Array &values, const RefreshMap &mapping)
        {
            std::vector<index> globals;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto read = mapping.newToOld.Read();
                for (size_t row = 0; row < mapping.newToOld.Rows(); ++row)
                {
                    const auto global = read.View()(row, 0);
                    DNDS_check_throw_info(global != UnInitIndex, "new topology rows have no inferable named-relation or ancestry payload");
                    globals.push_back(global);
                } });
            return Fetch(mpi, mapping.oldSpace, values, globals);
        }

        Array RefreshTargetRows(const MPIInfo &mpi, const Array &values, const RefreshMap &mapping, bool sortedUnique)
        {
            std::vector<index> globals;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::set<index> unique;
                auto read = values.Read();
                for (size_t row = 0; row < values.Rows(); ++row)
                    for (auto global : read.View()[row]) if (global != UnInitIndex) unique.insert(global);
                globals.assign(unique.begin(), unique.end()); });
            auto fetched = Fetch(mpi, mapping.oldSpace, mapping.oldToNew, globals);
            std::vector<std::vector<index>> rows;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::map<index, index> targets;
                auto read = fetched.Read();
                for (size_t row = 0; row < globals.size(); ++row)
                {
                    DNDS_check_throw_info(read.View()(row, 0) != UnInitIndex, "named relation references a removed topology key");
                    targets.emplace(globals[row], read.View()(row, 0));
                }
                auto original = values.Read();
                rows.resize(values.Rows());
                for (size_t row = 0; row < rows.size(); ++row)
                {
                    for (auto global : original.View()[row]) rows[row].push_back(global == UnInitIndex ? global : targets.at(global));
                    if (sortedUnique)
                    {
                        std::sort(rows[row].begin(), rows[row].end());
                        rows[row].erase(std::unique(rows[row].begin(), rows[row].end()), rows[row].end());
                    }
                } });
            return Rows(rows);
        }

        R::Array<ElemInfo> Information(const R::Array<ElemInfo> &original, const std::vector<t_index> &zones)
        {
            R::Array<ElemInfo> result(R::RowLayout::Dense(original.Rows(), 1));
            auto read = original.Read();
            auto write = result.Write();
            for (size_t row = 0; row < original.Rows(); ++row)
            {
                write.View()(row, 0) = read.View()(row, 0);
                write.View()(row, 0).zone = zones[row];
            }
            return result;
        }
    }

    std::shared_ptr<const NativeMeshState> NativeMeshTopology::Build(const NativeMeshState &source, const MPIInfo &mpi,
                                                                     bool buildFaces, bool buildEdges, int ghostLayers, const std::map<EntityKind, index> &generatedIdStarts)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native topology requires a communicator");
        auto data = NativeMeshAssembler::CaptureOwned(source, mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(data.dimension >= 2 && ghostLayers >= 1, "native topology completion requires dimension two or three and positive closure");
            DNDS_check_throw_info(!buildEdges || data.dimension == 3, "native edge completion currently requires three dimensions"); });
        std::array<index, 2> idStarts{-1, -1}, expectedStarts;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (const auto &[kind,first] : generatedIdStarts)
            {
                DNDS_check_throw_info(first >= 0 && ((kind == EntityKind::Face && buildFaces) || (kind == EntityKind::Edge && buildEdges)),
                                      "generated ID floor requires a requested target family");
                DNDS_check_throw_info(!source.IdHighWater().count(kind) || first >= source.NextEntityId(kind),
                                      "explicit topology ID floor would reuse a previous ID");
                idStarts[kind == EntityKind::Face ? 0 : 1] = first;
            } });
        expectedStarts = idStarts;
        MPI_Bcast(expectedStarts.data(), 2, DNDS_MPI_INDEX, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(idStarts == expectedStarts, "generated native ID floors differ across ranks"); });
        int flags[]{int(buildFaces), int(buildEdges), ghostLayers}, minimum[3], maximum[3];
        MPI_Allreduce(flags, minimum, 3, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(flags, maximum, 3, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::equal(minimum, minimum + 3, maximum), "native topology options differ between ranks");
        std::optional<NativeOwnedMeshData> previousStorage;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { previousStorage.emplace(data); });
        const auto &previous = *previousStorage;
        std::map<EntityKind, RefreshMap> refresh;
        std::set<AdjKind> regenerated;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (auto kind : {EntityKind::Face, EntityKind::Edge})
            {
                if (!(kind == EntityKind::Face ? buildFaces : buildEdges) || !data.spaces.count(kind)) continue;
                data.spaces.erase(kind);
                data.ids.erase(kind);
                const auto slots = kind == EntityKind::Face
                    ? std::vector<MeshRelation>{MeshRelation::CellFaces, MeshRelation::FaceNodes, MeshRelation::FaceCells,
                                                MeshRelation::FaceBoundaries, MeshRelation::BoundaryFaces, MeshRelation::CellFaceCells}
                    : std::vector<MeshRelation>{MeshRelation::CellEdges, MeshRelation::EdgeNodes, MeshRelation::EdgeCells};
                for (auto slot : slots) data.relations[size_t(slot)].reset();
                if (kind == EntityKind::Face)
                {
                    data.faceInfo.reset(); data.cellFacesPbi.reset(); data.faceNodesPbi.reset();
                }
                else
                {
                    data.edgeInfo.reset(); data.cellEdgesPbi.reset(); data.edgeNodesPbi.reset();
                }
            }
            // Reinsert affected named rows after exact source/target remapping.
            for (auto item = data.namedRelations.begin(); item != data.namedRelations.end();)
            {
                const auto path = item->first.Path();
                if ((buildFaces && std::find(path.begin(), path.end(), EntityKind::Face) != path.end()) ||
                    (buildEdges && std::find(path.begin(), path.end(), EntityKind::Edge) != path.end()))
                    item = data.namedRelations.erase(item);
                else ++item;
            } });
        auto closure = BuildNativeCellClosure(source, mpi);
        auto coneRead = closure.cellNodes.Read();
        const auto cells = coneRead.Source(), nodes = coneRead.Target();
        auto metadata = closure.cellInfo.ReadCombined();
        const auto cellTypes = metadata.View();
        std::optional<RelationPeriodicBits> periodic;
        if (closure.cellNodesPbi)
            periodic.emplace(coneRead, closure.cellNodesPbi->owned, closure.cellNodesPbi->ghost);
        auto owner = [mpi](const std::vector<index> &, const std::vector<MPI_int> &ranks, index)
        { return OwnershipDecision{*std::min_element(ranks.begin(), ranks.end()) == mpi.rank, {}}; };

        if (buildFaces)
        {
            auto faces = NativeConnectivity::Interpolate(coneRead, EntityKind::Face, Query(cellTypes, coneRead, false, periodic ? &*periodic : nullptr), owner, periodic ? &*periodic : nullptr);
            auto inverse = NativeConnectivity::Inverse(faces.parent2entity.Read(), mpi);
            auto support = inverse.Read();
            auto occurrences = faces.entity2parent.Read();
            std::vector<std::vector<index>> faceCells(size_t(faces.entities.NumOwned()));
            std::vector<t_index> zones(size_t(faces.entities.NumOwned()));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                for (index face = 0; face < faces.entities.NumOwned(); ++face)
                {
                    auto parents = occurrences.GlobalRow(face), unique = parents;
                    std::sort(unique.begin(), unique.end());
                    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
                    DNDS_check_throw_info(parents.size() >= 1 && parents.size() <= 2 &&
                                              (data.periodic || parents.size() == unique.size()) && unique == support.GlobalRow(face),
                                          "native face has missing, repeated or nonmanifold parent incidences");
                    zones[size_t(face)] = parents.size() == 1 ? BC_ID_NULL : BC_ID_INTERNAL;
                    if (parents.size() == 1) parents.push_back(UnInitIndex);
                    faceCells[size_t(face)] = std::move(parents);
                } });

            // Resolve each independently owned boundary through its containing
            // cell's exact face slots. Candidate cells may be entirely remote.
            std::vector<R::RowRequest> requests;
            Array types(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto write = types.Write();
                for (index cell = 0; cell < cells.NumOwned(); ++cell) write.View()(size_t(cell), 0) = cellTypes(size_t(cell), 0).type;
                auto boundaryCells = data.relations[size_t(MeshRelation::BoundaryCells)]->Read();
                for (index boundary = 0; boundary < data.spaces.at(EntityKind::Bnd).NumOwned(); ++boundary)
                {
                    DNDS_check_throw_info(boundaryCells.View().RowSize(size_t(boundary)) == 2,
                                          "native topology boundary must have two physical slots");
                    const auto cell = boundaryCells.View()(size_t(boundary), 0);
                    requests.push_back({cells.Owner(cell), cells.OwnerLocalRow(cell), size_t(boundary)});
                } });
            auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
            const auto nBoundary = data.spaces.at(EntityKind::Bnd).NumOwned();
            R::RowTransferPlan candidates(context, size_t(cells.NumOwned()), size_t(nBoundary), requests);
            auto candidateFaces = Transfer(candidates, faces.parent2entity.Owned(), context);
            auto candidateNodes = Transfer(candidates, *data.relations[size_t(MeshRelation::CellNodes)], context);
            auto candidateTypes = Transfer(candidates, types, context);
            std::optional<Array> candidateBits, candidateFaceFrames;
            if (data.periodic)
            {
                candidateBits = Transfer(candidates, BitRows(*data.cellNodesPbi), context);
                candidateFaceFrames = Transfer(candidates, BitRows(*faces.parent2entityPbi), context);
            }
            std::vector<std::vector<index>> boundaryFaces(static_cast<size_t>(nBoundary));
            std::vector<std::vector<index>> assignments(size_t(mpi.size));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto foundFaces = candidateFaces.Read(), foundNodes = candidateNodes.Read(), foundTypes = candidateTypes.Read();
                auto boundaryNodes = data.relations[size_t(MeshRelation::BoundaryNodes)]->Read();
                auto boundaryInfo = data.bndInfo->Read();
                std::optional<R::Array<index>::ReadLease> parentBits, parentFaceFrames;
                std::optional<R::Array<NodePeriodicBits>::ReadLease> boundaryBits;
                if (data.periodic)
                {
                    parentBits.emplace(candidateBits->Read());
                    parentFaceFrames.emplace(candidateFaceFrames->Read());
                    boundaryBits.emplace(data.bndNodesPbi->Read());
                }
                for (index boundary = 0; boundary < nBoundary; ++boundary)
                {
                    const auto b = size_t(boundary);
                    const auto raw = foundNodes.View()[b];
                    std::vector<index> parent(raw.begin(), raw.end());
                    const auto rawBoundary = boundaryNodes.View()[b];
                    std::vector<index> expected(rawBoundary.begin(), rawBoundary.end());
                    std::sort(expected.begin(), expected.end());
                    NativePeriodicNodes expectedImages;
                    for (size_t slot = 0; slot < rawBoundary.size(); ++slot)
                        expectedImages.emplace_back(rawBoundary[slot], boundaryBits ? uint8_t(boundaryBits->View()(b, slot)) : 0);
                    Elem::Element element{static_cast<Elem::ElemType>(foundTypes.View()(b, 0))};
                    index match = UnInitIndex;
                    index matchFrame = 0;
                    for (t_index slot = 0; slot < element.GetNumFaces(); ++slot)
                    {
                        const auto face = element.ObtainFace(slot);
                        if (face.type != boundaryInfo.View()(b, 0).type) continue;
                        std::vector<index> actual(size_t(face.GetNumNodes()));
                        element.ExtractFaceNodes(slot, parent, actual);
                        std::vector<index> frameValues(parent.size(), 0), faceFrames(actual.size());
                        if (parentBits)
                            for (size_t i = 0; i < frameValues.size(); ++i) frameValues[i] = parentBits->View()(b, i);
                        element.ExtractFaceNodes(slot, frameValues, faceFrames);
                        NativePeriodicNodes actualImages;
                        for (size_t i = 0; i < actual.size(); ++i) actualImages.emplace_back(actual[i], uint8_t(faceFrames[i]));
                        auto delta = NativePeriodicFrame(std::move(actualImages), expectedImages);
                        if (delta && *delta == 0)
                        {
                            DNDS_check_throw_info(match == UnInitIndex, "boundary matches multiple element faces");
                            match = foundFaces.View()(b, size_t(slot));
                            if (parentFaceFrames) matchFrame = parentFaceFrames->View()(b, size_t(slot));
                        }
                    }
                    DNDS_check_throw_info(match != UnInitIndex && (FaceIDIsExternalBC(boundaryInfo.View()(b, 0).zone) ||
                                                                   (data.periodic && FaceIDIsPeriodic(boundaryInfo.View()(b, 0).zone))),
                                          "boundary does not match a physical element face in its frame");
                    boundaryFaces[b] = {match};
                    auto &message = assignments[size_t(faces.entities.Owner(match))];
                    message.insert(message.end(), {faces.entities.OwnerLocalRow(match), data.spaces.at(EntityKind::Bnd).LocalToGlobal(boundary), boundaryInfo.View()(b, 0).zone,
                                                   data.relations[size_t(MeshRelation::BoundaryCells)]->Read().View()(b, 0), matchFrame});
                } });
            auto received = Exchange(mpi, assignments);
            std::vector<std::vector<index>> faceBoundaries(size_t(faces.entities.NumOwned()), {UnInitIndex});
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                struct Assignment { index boundary, cell, frame; t_index zone; };
                std::vector<std::vector<Assignment>> labels(faceBoundaries.size());
                auto messages = received.Read();
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    DNDS_check_throw_info(messages.View().RowSize(size_t(peer)) % 5 == 0, "malformed native face-boundary assignment");
                    for (size_t slot = 0; slot < messages.View().RowSize(size_t(peer)); slot += 5)
                    {
                        const auto face = messages.View()(size_t(peer), slot), boundary = messages.View()(size_t(peer), slot + 1), zone = messages.View()(size_t(peer), slot + 2);
                        DNDS_check_throw_info(face >= 0 && face < faces.entities.NumOwned() &&
                                                  data.spaces.at(EntityKind::Bnd).Owner(boundary) == peer && zone >= BC_ID_PERIODIC_3_DONOR && zone <= std::numeric_limits<t_index>::max(),
                                              "invalid native face-boundary owner or zone");
                        labels[size_t(face)].push_back({boundary, messages.View()(size_t(peer), slot + 3), messages.View()(size_t(peer), slot + 4), t_index(zone)});
                    }
                }
                for (size_t face = 0; face < labels.size(); ++face)
                {
                    const auto &entries = labels[face];
                    if (entries.empty()) continue;
                    const Assignment *selected = nullptr;
                    if (FaceIDIsPeriodic(entries.front().zone))
                    {
                        DNDS_check_throw_info(entries.size() == 2 && faceCells[face][1] != UnInitIndex &&
                                                  std::abs(int(entries[0].zone) - int(entries[1].zone)) == 3 &&
                                                  FaceIDIsPeriodic(entries[1].zone) && entries[0].frame != entries[1].frame,
                                              "periodic face requires one complementary main/donor boundary pair");
                        for (const auto &entry : entries)
                            if (entry.cell == faceCells[face][0] && entry.frame == 0) selected = &entry;
                        DNDS_check_throw_info(selected, "periodic boundary pair lacks the ordered face reference side");
                    }
                    else
                    {
                        DNDS_check_throw_info(entries.size() == 1 && FaceIDIsExternalBC(entries[0].zone) && faceCells[face][1] == UnInitIndex,
                                              "boundary face is internal or has duplicate boundary assignments");
                        selected = &entries.front();
                    }
                    faceBoundaries[face][0] = selected->boundary;
                    zones[face] = selected->zone;
                } });
            auto physical = Rows(faceCells, 2);
            auto parentFaces = faces.parent2entity.Read();
            std::vector<index> remote;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                std::set<index> unique;
                for (index cell = 0; cell < parentFaces.NumOwned(); ++cell)
                    for (auto face : parentFaces.GlobalRow(cell))
                        if (faces.entities.Owner(face) != mpi.rank) unique.insert(face);
                remote.assign(unique.begin(), unique.end()); });
            auto closure = faces.entities.WithGhosts(std::move(remote));
            NativeRelation physicalRelation(Adj::Face2Cell, closure, cells, physical);
            RelationGhostRoute faceRoute(closure);
            auto physicalComplete = physicalRelation.WithGhostRows(faceRoute);
            auto physicalRead = physicalComplete.Read();
            std::vector<std::vector<index>> neighbors(size_t(cells.NumOwned()));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                for (index cell = 0; cell < cells.NumOwned(); ++cell)
                    for (auto face : parentFaces.GlobalRow(cell))
                    {
                        const auto row = closure.GlobalToLocal(face);
                        DNDS_check_throw_info(row, "owned cell face missing from native local closure");
                        const auto pair = physicalRead.GlobalRow(*row);
                        const auto global = cells.LocalToGlobal(cell);
                        DNDS_check_throw_info(pair[0] == global || pair[1] == global, "native physical face omits its parent cell");
                        neighbors[size_t(cell)].push_back(pair[0] == global ? pair[1] : pair[0]);
                    }
                data.relations[size_t(MeshRelation::CellFaces)] = faces.parent2entity.Owned();
                data.relations[size_t(MeshRelation::FaceNodes)] = faces.entity2node.Owned();
                data.relations[size_t(MeshRelation::FaceCells)] = physical;
                data.relations[size_t(MeshRelation::FaceBoundaries)] = Rows(faceBoundaries, 1);
                data.relations[size_t(MeshRelation::BoundaryFaces)] = Rows(boundaryFaces, 1);
                data.relations[size_t(MeshRelation::CellFaceCells)] = Rows(neighbors);
                data.namedRelations.emplace(inverse.Read().Kind(), inverse.Owned());
                data.faceInfo = Information(faces.entityElemInfo, zones);
                if (data.periodic)
                {
                    data.cellFacesPbi = *faces.parent2entityPbi;
                    data.faceNodesPbi = *faces.entity2nodePbi;
                }
                data.requirements.faces = true;
                data.requirements.cellFaceNeighbors = true;
                data.requirements.allFaceCellsLocal = true; });
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { regenerated.insert(inverse.Read().Kind()); });
            if (source.HasEntities(EntityKind::Face))
            {
                auto recovered = RecoverIdentity(source, mpi, previous, data, faces.entities,
                                                 MeshRelation::FaceNodes, *data.faceInfo, data.faceNodesPbi,
                                                 generatedIdStarts.count(EntityKind::Face) ? std::optional<index>{generatedIdStarts.at(EntityKind::Face)} : std::optional<index>{});
                FieldTransferDetail::LocalStage(mpi, [&]
                                                { refresh.emplace(EntityKind::Face, std::move(recovered)); });
            }
            else
                FieldTransferDetail::LocalStage(mpi, [&]
                                                { AddIdentity(data, faces.entities, generatedIdStarts); });
        }
        if (buildEdges)
        {
            auto edges = NativeConnectivity::Interpolate(coneRead, EntityKind::Edge, Query(cellTypes, coneRead, true, periodic ? &*periodic : nullptr), owner, periodic ? &*periodic : nullptr);
            auto inverse = NativeConnectivity::Inverse(edges.parent2entity.Read(), mpi);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                data.relations[size_t(MeshRelation::CellEdges)] = edges.parent2entity.Owned();
                data.relations[size_t(MeshRelation::EdgeNodes)] = edges.entity2node.Owned();
                data.relations[size_t(MeshRelation::EdgeCells)] = inverse.Owned();
                data.namedRelations.emplace(inverse.Read().Kind(), inverse.Owned());
                data.edgeInfo = Information(edges.entityElemInfo, std::vector<t_index>(size_t(edges.entities.NumOwned()), BC_ID_INTERNAL));
                if (data.periodic)
                {
                    data.cellEdgesPbi = *edges.parent2entityPbi;
                    data.edgeNodesPbi = *edges.entity2nodePbi;
                }
                data.requirements.edges = true; });
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { regenerated.insert(inverse.Read().Kind()); });
            if (source.HasEntities(EntityKind::Edge))
            {
                auto recovered = RecoverIdentity(source, mpi, previous, data, edges.entities,
                                                 MeshRelation::EdgeNodes, *data.edgeInfo, data.edgeNodesPbi,
                                                 generatedIdStarts.count(EntityKind::Edge) ? std::optional<index>{generatedIdStarts.at(EntityKind::Edge)} : std::optional<index>{});
                FieldTransferDetail::LocalStage(mpi, [&]
                                                { refresh.emplace(EntityKind::Edge, std::move(recovered)); });
            }
            else
                FieldTransferDetail::LocalStage(mpi, [&]
                                                { AddIdentity(data, edges.entities, generatedIdStarts); });
        }
        for (const auto &[key, original] : previous.namedRelations)
        {
            if (data.namedRelations.count(key) || regenerated.count(key))
                continue;
            // An explicit fixed-slot alias means the current physical facade
            // relation, whose first-parent orientation may have changed.
            // Regenerate these aliases instead of preserving an obsolete frame.
            bool fixedAlias = false;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                for (const auto &[alias, slot] : std::initializer_list<std::pair<AdjKind, MeshRelation>>{
                    {Adj::Cell2Face, MeshRelation::CellFaces}, {Adj::Face2Node, MeshRelation::FaceNodes},
                    {Adj::Face2Cell, MeshRelation::FaceCells}, {Adj::Face2Bnd, MeshRelation::FaceBoundaries},
                    {Adj::Bnd2Face, MeshRelation::BoundaryFaces}, {Adj::Cell2CellFace, MeshRelation::CellFaceCells},
                    {Adj::Cell2Edge, MeshRelation::CellEdges}, {Adj::Edge2Node, MeshRelation::EdgeNodes},
                    {Adj::Edge2Cell, MeshRelation::EdgeCells}})
                    if (key == alias && data.relations[size_t(slot)])
                    {
                        data.namedRelations.emplace(key, *data.relations[size_t(slot)]);
                        fixedAlias = true;
                        break;
                    } });
            if (fixedAlias)
                continue;
            const auto path = key.Path();
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                for (auto kind : {EntityKind::Face, EntityKind::Edge})
                {
                    if (refresh.count(kind) && key.from != kind && key.to != kind &&
                        std::find(path.begin(), path.end(), kind) != path.end())
                    {
                        auto read = refresh.at(kind).oldToNew.Read();
                        for (size_t row = 0; row < refresh.at(kind).oldToNew.Rows(); ++row)
                            DNDS_check_throw_info(read.View()(row, 0) != UnInitIndex,
                                                  "named relation through a removed topology key cannot be inferred");
                        auto reverse = refresh.at(kind).newToOld.Read();
                        for (size_t row = 0; row < refresh.at(kind).newToOld.Rows(); ++row)
                            DNDS_check_throw_info(reverse.View()(row, 0) != UnInitIndex,
                                                  "named relation through an added topology key cannot be inferred");
                    }
                } });
            Array values = original;
            if (refresh.count(key.from))
                values = RefreshSourceRows(mpi, values, refresh.at(key.from));
            if (refresh.count(key.to))
                values = RefreshTargetRows(mpi, values, refresh.at(key.to), key.order == RelationOrder::SortedUnique);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { data.namedRelations.emplace(key, std::move(values)); });
        }
        if (data.provenance)
            for (auto &[kind, values] : data.provenance->sourceIds)
            {
                if (refresh.count(kind))
                    values = RefreshSourceRows(mpi, values, refresh.at(kind));
            }
        return NativeMeshAssembler::Build(mpi, std::move(data), ghostLayers);
    }
}
