#include "NativeMeshTopology.hpp"
#include "NativeMeshAssembler.hpp"
#include "NativeMeshClosure.hpp"
#include "NativeConnectivity.hpp"
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

        SubEntityQueryPbi Query(R::PairReadView<ElemInfo> types, const RelationReadAccess &cone, bool edges)
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
            query.matchExtra = [&cone, extract, describe](index parent, int sub, index, index originalParent, int originalSub)
            {
                auto fullNodes = [&](index cell, int slot)
                {
                    std::vector<index> nodes(size_t(describe(cell, slot).nNodes));
                    extract(cell, slot, [&](int node)
                            { return cone.ReadEntry(cell, size_t(node)).global; }, nodes.data());
                    std::sort(nodes.begin(), nodes.end());
                    return nodes;
                };
                DNDS_check_throw_info(fullNodes(parent, sub) == fullNodes(originalParent, originalSub),
                                      "shared native subentity vertices have inconsistent higher-order nodes");
                return true;
            };
            return query;
        }

        void AddIdentity(NativeOwnedMeshData &data, const EntitySpace &space)
        {
            // This operation rejects preexisting target families; generated IDs
            // are assigned once from the new family's distributed numbering.
            Array ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
            {
                auto write = ids.Write();
                for (index row = 0; row < space.NumOwned(); ++row)
                    write.View()(size_t(row), 0) = space.LocalToGlobal(row);
            }
            data.spaces.emplace(space.Kind(), space);
            data.ids.emplace(space.Kind(), std::move(ids));
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
                                                                     bool buildFaces, bool buildEdges, int ghostLayers)
    {
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native topology requires a communicator");
        auto data = NativeMeshAssembler::CaptureOwned(source, mpi);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(!data.periodic && ghostLayers >= 1, "native topology completion currently requires nonperiodic input and positive closure");
            DNDS_check_throw_info((!buildFaces || !data.spaces.count(EntityKind::Face)) &&
                                      (!buildEdges || !data.spaces.count(EntityKind::Edge)), "native topology target family already exists");
            DNDS_check_throw_info(!buildEdges || data.dimension == 3, "native edge completion currently requires three dimensions"); });
        int flags[]{int(buildFaces), int(buildEdges), ghostLayers}, minimum[3], maximum[3];
        MPI_Allreduce(flags, minimum, 3, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(flags, maximum, 3, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::equal(minimum, minimum + 3, maximum), "native topology options differ between ranks");
        auto closure = BuildNativeCellClosure(source, mpi);
        auto coneRead = closure.cellNodes.Read();
        const auto cells = coneRead.Source(), nodes = coneRead.Target();
        auto metadata = closure.cellInfo.ReadCombined();
        const auto cellTypes = metadata.View();
        auto owner = [mpi](const std::vector<index> &, const std::vector<MPI_int> &ranks, index)
        { return OwnershipDecision{*std::min_element(ranks.begin(), ranks.end()) == mpi.rank, {}}; };

        if (buildFaces)
        {
            auto faces = NativeConnectivity::Interpolate(coneRead, EntityKind::Face, Query(cellTypes, coneRead, false), owner);
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
                    DNDS_check_throw_info(unique.size() >= 1 && unique.size() <= 2 &&
                                              std::adjacent_find(unique.begin(), unique.end()) == unique.end() && unique == support.GlobalRow(face),
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
                    DNDS_check_throw_info(boundaryCells.View().RowSize(size_t(boundary)) == 2 && boundaryCells.View()(size_t(boundary), 1) == UnInitIndex,
                                          "native topology boundary must have one physical parent");
                    const auto cell = boundaryCells.View()(size_t(boundary), 0);
                    requests.push_back({cells.Owner(cell), cells.OwnerLocalRow(cell), size_t(boundary)});
                } });
            auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
            const auto nBoundary = data.spaces.at(EntityKind::Bnd).NumOwned();
            R::RowTransferPlan candidates(context, size_t(cells.NumOwned()), size_t(nBoundary), requests);
            auto candidateFaces = Transfer(candidates, faces.parent2entity.Owned(), context);
            auto candidateNodes = Transfer(candidates, *data.relations[size_t(MeshRelation::CellNodes)], context);
            auto candidateTypes = Transfer(candidates, types, context);
            std::vector<std::vector<index>> boundaryFaces(static_cast<size_t>(nBoundary));
            std::vector<std::vector<index>> assignments(size_t(mpi.size));
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto foundFaces = candidateFaces.Read(), foundNodes = candidateNodes.Read(), foundTypes = candidateTypes.Read();
                auto boundaryNodes = data.relations[size_t(MeshRelation::BoundaryNodes)]->Read();
                auto boundaryInfo = data.bndInfo->Read();
                for (index boundary = 0; boundary < nBoundary; ++boundary)
                {
                    const auto b = size_t(boundary);
                    const auto raw = foundNodes.View()[b];
                    std::vector<index> parent(raw.begin(), raw.end());
                    const auto rawBoundary = boundaryNodes.View()[b];
                    std::vector<index> expected(rawBoundary.begin(), rawBoundary.end());
                    std::sort(expected.begin(), expected.end());
                    Elem::Element element{static_cast<Elem::ElemType>(foundTypes.View()(b, 0))};
                    index match = UnInitIndex;
                    for (t_index slot = 0; slot < element.GetNumFaces(); ++slot)
                    {
                        const auto face = element.ObtainFace(slot);
                        if (face.type != boundaryInfo.View()(b, 0).type) continue;
                        std::vector<index> actual(size_t(face.GetNumNodes()));
                        element.ExtractFaceNodes(slot, parent, actual);
                        std::sort(actual.begin(), actual.end());
                        if (actual == expected)
                        {
                            DNDS_check_throw_info(match == UnInitIndex, "boundary matches multiple element faces");
                            match = foundFaces.View()(b, size_t(slot));
                        }
                    }
                    DNDS_check_throw_info(match != UnInitIndex && FaceIDIsExternalBC(boundaryInfo.View()(b, 0).zone),
                                          "boundary does not match a physical external element face");
                    boundaryFaces[b] = {match};
                    auto &message = assignments[size_t(faces.entities.Owner(match))];
                    message.insert(message.end(), {faces.entities.OwnerLocalRow(match), data.spaces.at(EntityKind::Bnd).LocalToGlobal(boundary), boundaryInfo.View()(b, 0).zone});
                } });
            auto received = Exchange(mpi, assignments);
            std::vector<std::vector<index>> faceBoundaries(size_t(faces.entities.NumOwned()), {UnInitIndex});
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                auto messages = received.Read();
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    DNDS_check_throw_info(messages.View().RowSize(size_t(peer)) % 3 == 0, "malformed native face-boundary assignment");
                    for (size_t slot = 0; slot < messages.View().RowSize(size_t(peer)); slot += 3)
                    {
                        const auto face = messages.View()(size_t(peer), slot), boundary = messages.View()(size_t(peer), slot + 1), zone = messages.View()(size_t(peer), slot + 2);
                        DNDS_check_throw_info(face >= 0 && face < faces.entities.NumOwned() &&
                                                  data.spaces.at(EntityKind::Bnd).Owner(boundary) == peer && zone > 0 && zone <= std::numeric_limits<t_index>::max(),
                                              "invalid native face-boundary owner or zone");
                        DNDS_check_throw_info(faceCells[size_t(face)][1] == UnInitIndex && faceBoundaries[size_t(face)][0] == UnInitIndex,
                                              "boundary face is internal or has duplicate boundary assignments");
                        faceBoundaries[size_t(face)][0] = boundary;
                        zones[size_t(face)] = t_index(zone);
                    }
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
                AddIdentity(data, faces.entities);
                data.requirements.faces = true;
                data.requirements.cellFaceNeighbors = true;
                data.requirements.allFaceCellsLocal = true; });
        }
        if (buildEdges)
        {
            auto edges = NativeConnectivity::Interpolate(coneRead, EntityKind::Edge, Query(cellTypes, coneRead, true), owner);
            auto inverse = NativeConnectivity::Inverse(edges.parent2entity.Read(), mpi);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                data.relations[size_t(MeshRelation::CellEdges)] = edges.parent2entity.Owned();
                data.relations[size_t(MeshRelation::EdgeNodes)] = edges.entity2node.Owned();
                data.relations[size_t(MeshRelation::EdgeCells)] = inverse.Owned();
                data.namedRelations.emplace(inverse.Read().Kind(), inverse.Owned());
                data.edgeInfo = Information(edges.entityElemInfo, std::vector<t_index>(size_t(edges.entities.NumOwned()), BC_ID_INTERNAL));
                AddIdentity(data, edges.entities);
                data.requirements.edges = true; });
        }
        return NativeMeshAssembler::Build(mpi, std::move(data), ghostLayers);
    }
}
