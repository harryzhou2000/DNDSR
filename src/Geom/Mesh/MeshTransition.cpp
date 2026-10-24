#include "MeshTransition.hpp"
#include "FieldTransfer.hpp"
#include <type_traits>

namespace DNDS::Geom
{
    namespace
    {
        template <class Pair>
        bool Present(const Pair &pair)
        {
            return bool(pair.father) || bool(pair.son);
        }

        template <class Pair, class RowSize, class Value>
        void CloneRows(Pair &destination, index n, const MPIInfo &mpi,
                       const char *name, RowSize &&width, Value &&value)
        {
            destination.InitPair(name, mpi);
            destination.father->Resize(n);
            destination.son->Resize(0);
            for (index i = 0; i < n; ++i)
            {
                if constexpr (Pair::IsCSR())
                    destination.ResizeRow(i, rowsize(width(i)));
                for (rowsize j = 0; j < rowsize(width(i)); ++j)
                    destination(i, j) = value(i, j);
            }
            if constexpr (Pair::IsCSR())
            {
                destination.father->Compress();
                destination.son->Compress();
            }
            destination.TransAttach();
        }

        template <class Pair, class Row>
        void CloneRelation(Pair &destination, index n, const EntitySpace &target, Row &&row,
                           const MPIInfo &mpi, const char *name)
        {
            CloneRows(destination, n, mpi, name, [&](index i)
                      { return row(i).size(); }, [&](index i, rowsize j)
                      {
                auto value = row(i)[j];
                if (value == UnInitIndex)
                    return value;
                return value < 0 ? -1 - value : target.LocalToGlobal(value); });
            destination.idx.markGlobal();
        }

        void CheckInventory(const UnstructuredMesh &mesh, const UnstructuredMeshView &view, const ReorderInput &input,
                            const std::vector<CompanionEntry> &fields)
        {
            DNDS_check_throw_info(view.Dimension() == 2 || view.Dimension() == 3, "transition requires a volume mesh");
            DNDS_check_throw_info(mesh.elevState == Elevation_Untouched && mesh.nNodeO1 < 0,
                                  "transition does not yet preserve elevated original-node prefix semantics");
            if (!mesh.HasNativeState())
                DNDS_check_throw_info(mesh.cell2cellOrig.father && mesh.cell2cellOrig.son &&
                                          mesh.node2nodeOrig.father && mesh.node2nodeOrig.son &&
                                          mesh.bnd2bndOrig.father && mesh.bnd2bndOrig.son &&
                                          mesh.cell2cellOrig.father->Size() == view.NumCell() &&
                                          mesh.node2nodeOrig.father->Size() == view.NumNode() &&
                                          mesh.bnd2bndOrig.father->Size() == view.NumBnd(),
                                      "transition requires complete primary original-ID rows");
            const bool faces = view.HasRelation(MeshRelation::FaceNodes);
            const bool edges = view.HasRelation(MeshRelation::EdgeNodes);
            DNDS_check_throw_info(!view.HasRelation(MeshRelation::CellFaceCells) || faces,
                                  "transition face-neighbor storage requires prepared facial storage");
            DNDS_check_throw_info(!view.HasRelation(MeshRelation::EdgeNodes) || (faces && view.Dimension() == 3),
                                  "transition edge reconstruction requires prepared 3D faces");
            DNDS_check_throw_info(mesh.node2parentNode.empty() && mesh.node2bndNode.empty() && mesh.cell2parentCell.empty(),
                                  "transition does not yet preserve parent/extraction links");
            DNDS_check_throw_info(!Present(mesh.coordsElevDisp), "transition does not yet preserve elevation workspaces");
            for (auto kind : input.destroyKinds)
                DNDS_check_throw_info(kind == EntityKind::Face || kind == EntityKind::Edge, "transition may only destroy/reconstruct faces or edges");
            auto supportedKind = [](EntityKind kind)
            { return kind == EntityKind::Cell || kind == EntityKind::Node || kind == EntityKind::Bnd || kind == EntityKind::Face || kind == EntityKind::Edge; };
            std::unordered_set<EntityKind> explicitKinds;
            for (const auto &map : input.explicitMaps)
            {
                DNDS_check_throw_info(supportedKind(map.kind) && !input.destroyKinds.count(map.kind) &&
                                          explicitKinds.insert(map.kind).second,
                                      "unsupported or duplicate transition entity map");
                index owned = map.kind == EntityKind::Cell ? view.NumCell() : map.kind == EntityKind::Node ? view.NumNode()
                                                                          : map.kind == EntityKind::Bnd    ? view.NumBnd()
                                                                          : map.kind == EntityKind::Face   ? (faces ? view.NumFace() : -1)
                                                                                                           : (edges ? view.NumEdge() : -1);
                DNDS_check_throw_info(owned >= 0 && map.targetRanks.size() == size_t(owned), "transition entity map shape mismatch");
                for (auto rank : map.targetRanks)
                    DNDS_check_throw_info(rank >= 0 && rank < mesh.mpi.size, "transition destination rank out of range");
            }
            auto mappedKinds = explicitKinds;
            for (const auto &follow : input.follows)
            {
                bool hop = follow.follower2leader == Adj::Node2Cell || follow.follower2leader == Adj::Bnd2Cell ||
                           follow.follower2leader == Adj::Cell2Node || follow.follower2leader == Adj::Bnd2Node ||
                           ((follow.follower2leader == Adj::Face2Cell || follow.follower2leader == Adj::Face2Node) &&
                            faces && !input.destroyKinds.count(EntityKind::Face)) ||
                           ((follow.follower2leader == Adj::Edge2Cell || follow.follower2leader == Adj::Edge2Node) &&
                            edges && !input.destroyKinds.count(EntityKind::Edge));
                DNDS_check_throw_info(supportedKind(follow.follower) && supportedKind(follow.leader) &&
                                          !input.destroyKinds.count(follow.follower) && !input.destroyKinds.count(follow.leader) &&
                                          mappedKinds.count(follow.leader) && hop && follow.follower2leader.from == follow.follower &&
                                          follow.follower2leader.to == follow.leader,
                                      "unsupported or unresolvable transition follow placement");
                mappedKinds.insert(follow.follower);
            }
            for (const auto &field : fields)
            {
                DNDS_check_throw_info(field.prepare && field.preflight && field.attachment && field.fn &&
                                          supportedKind(field.kind) && !input.destroyKinds.count(field.kind) &&
                                          (field.kind != EntityKind::Face || faces) && (field.kind != EntityKind::Edge || edges),
                                      "transition requires staged fields on retained entity spaces");
                const auto attached = field.attachment();
                const index owned = field.kind == EntityKind::Cell ? view.NumCell() : field.kind == EntityKind::Node ? view.NumNode()
                                                                                  : field.kind == EntityKind::Bnd    ? view.NumBnd()
                                                                                  : field.kind == EntityKind::Face   ? view.NumFace()
                                                                                                                     : view.NumEdge();
                DNDS_check_throw_info(attached.allocation && attached.rows == size_t(owned) &&
                                          attached.owned == size_t(owned) && attached.ghosts == 0,
                                      "transition requires owned-only source field rows, including unchanged entity families; rebuild field ghosts after publication");
                if (field.requiresEntityBinding)
                {
                    DNDS_check_throw_info(field.validateSourceSpace && field.prepareUnchanged,
                                          "entity-bound field lacks source validation or unchanged staging");
                    field.validateSourceSpace(mesh.Entities(field.kind));
                }
            }
        }

        void AgreeSchedule(const ReorderInput &input, const MPIInfo &mpi)
        {
            std::string schedule;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                for (const auto &map : input.explicitMaps)
                    schedule += "m" + std::to_string(int(map.kind)) + ";";
                for (const auto &follow : input.follows)
                    schedule += "f" + std::to_string(int(follow.follower)) + ":" + std::to_string(int(follow.leader)) +
                                ":" + std::to_string(int(follow.follower2leader.from)) + ":" +
                                std::to_string(int(follow.follower2leader.to)) + ";";
                DNDS_check_throw_info(schedule.size() <= size_t(std::numeric_limits<int>::max()), "transition schedule too large"); });
            int length = int(schedule.size());
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            std::string expected;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { expected = mpi.rank == 0 ? schedule : std::string(size_t(length), '\0'); });
            MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(schedule == expected, "transition reorder schedule differs between ranks"); });
        }

        struct DeferPublication final : PreparedCompanion
        {
            void Publish() noexcept override {}
        };

        bool SameAttachment(const CompanionAttachment &a, const CompanionAttachment &b)
        {
            return a.schema == b.schema && a.association == b.association && a.hasDistribution == b.hasDistribution &&
                   a.domain == b.domain && a.epoch == b.epoch && a.destination == b.destination &&
                   a.allocation == b.allocation && a.valuesVersion == b.valuesVersion && a.shapeVersion == b.shapeVersion &&
                   a.rows == b.rows && a.owned == b.owned && a.ghosts == b.ghosts;
        }
    }

    MeshTransitionHandle::MeshTransitionHandle(std::shared_ptr<UnstructuredMesh> mesh) : _mesh(std::move(mesh))
    {
        DNDS_check_throw_info(_mesh, "transition handle requires a mesh");
    }

    std::shared_ptr<UnstructuredMesh> MeshTransitionHandle::Get() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _mesh;
    }

    PreparedMeshTransition PreparedMeshTransition::Prepare(
        MeshTransitionHandle &handle, const ReorderInput &input, const std::vector<CompanionEntry> &fields,
        const std::function<void(const char *)> &checkpoint, int ghostLayers, TransitionCachePolicy caches, TransitionAuthority authority)
    {
        std::lock_guard<std::mutex> lock(handle._mutex);
        PreparedMeshTransition result(handle);
        result._source = handle._mesh;
        auto &source = *result._source;
        const auto &mpi = source.mpi;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(ghostLayers >= 1, "transition requires at least one explicit ghost layer");
            DNDS_check_throw_info(caches == TransitionCachePolicy::Invalidate || caches == TransitionCachePolicy::RebuildOutput,
                                  "unknown transition cache policy");
            DNDS_check_throw_info(authority == TransitionAuthority::PreserveSource || authority == TransitionAuthority::LegacyWorkspace,
                                  "unknown transition destination authority");
            auto requirements = source.HasNativeState() ? source.CurrentNativeRequirements() : MeshReadRequirements::Primary();
            if (!source.HasNativeState())
            {
                requirements.faces = Present(source.face2node);
                requirements.cellFaceNeighbors = Present(source.cell2cellFace);
                requirements.edges = Present(source.edge2node);
                requirements.wallDistance = Present(source.nodeWallDist);
            }
            result._sourceRead = std::make_unique<MeshReadAccess>(source.AcquireRead(requirements));
            auto view = result._sourceRead->HostView();
            CheckInventory(source, view, input, fields);
            for (auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd})
            {
                auto rows = kind == EntityKind::Cell ? view.NumCell() : kind == EntityKind::Node ? view.NumNode() : view.NumBnd();
                for (index row = 0; row < rows; ++row)
                    (void)source.OriginalEntityId(kind, row);
            }
            result._sourceStamp = source.versions.Stamp();
            for (auto kind : {EntityKind::Cell, EntityKind::Node, EntityKind::Bnd})
                result._sourceSpaces.push_back(source.Entities(kind));
            if (view.HasRelation(MeshRelation::FaceNodes))
                result._sourceSpaces.push_back(source.Entities(EntityKind::Face));
            if (view.HasRelation(MeshRelation::EdgeNodes))
                result._sourceSpaces.push_back(source.Entities(EntityKind::Edge)); });
        AgreeSchedule(input, mpi);
        auto view = result._sourceRead->HostView();
        const bool hasFaces = view.HasRelation(MeshRelation::FaceNodes), hasEdges = view.HasRelation(MeshRelation::EdgeNodes);
        const bool hasAux = view.HasRelation(MeshRelation::CellFaceCells), hasWall = view.HasNodeWallDist();
        const bool nativeDestination = source.HasNativeState() && authority == TransitionAuthority::PreserveSource;

        // Every following optional collective must have one schedule on all ranks.
        int inventory = int(hasFaces) | (int(hasAux) << 1) | (int(hasEdges) << 2) | (int(hasWall) << 3) |
                        (int(caches) << 4) | (int(view.IsPeriodic()) << 5) |
                        (int(input.destroyKinds.count(EntityKind::Face)) << 6) | (view.Dimension() << 7) |
                        (int(input.destroyKinds.count(EntityKind::Edge)) << 9) | (int(nativeDestination) << 10) |
                        (int(source.HasNativeState()) << 11) | (int(authority) << 12);
        int inventoryMin = inventory, inventoryMax = inventory;
        MPI_Allreduce(MPI_IN_PLACE, &inventoryMin, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &inventoryMax, 1, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(inventoryMin == inventoryMax, "transition storage/rebuild inventory differs between ranks");
        int minLayers = ghostLayers, maxLayers = ghostLayers;
        MPI_Allreduce(MPI_IN_PLACE, &minLayers, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &maxLayers, 1, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(minLayers == maxLayers, "transition ghost layer policy differs between ranks");

        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            result._next = std::make_shared<UnstructuredMesh>(mpi, view.Dimension());
            auto &next = *result._next;
            next.mRank = source.mRank;
            next.isPeriodic = view.IsPeriodic();
            next.periodicInfo = view.Periodic();
            next.elevState = source.elevState;
            next.nNodeO1 = source.nNodeO1;
            next.nTotalMoved = source.nTotalMoved;
            next.elevationInfo = source.elevationInfo;
            next.hdf5OutSetting = source.hdf5OutSetting;
            auto &cells = result._sourceSpaces[0];
            auto &nodes = result._sourceSpaces[1];
            auto &boundaries = result._sourceSpaces[2];
            auto one = [](index) { return 1; };
            auto three = [](index) { return 3; };
            CloneRows(next.coords, view.NumNode(), mpi, "transition.coords", three, [&](index i, rowsize j) { return view.Coord(i)[j]; });
            CloneRows(next.cellElemInfo, view.NumCell(), mpi, "transition.cellInfo", one, [&](index i, rowsize) { return view.GetCellInfo(i); });
            CloneRows(next.bndElemInfo, view.NumBnd(), mpi, "transition.bndInfo", one, [&](index i, rowsize) { return view.GetBndInfo(i); });
            CloneRelation(next.cell2node, view.NumCell(), nodes, [&](index i) { return view.CellNodes(i); }, mpi, "transition.cell2node");
            CloneRelation(next.bnd2node, view.NumBnd(), nodes, [&](index i) { return view.BoundaryNodes(i); }, mpi, "transition.bnd2node");
            CloneRows(next.cell2cellOrig, view.NumCell(), mpi, "transition.cellOrig", one,
                      [&](index i, rowsize) { return source.OriginalEntityId(EntityKind::Cell, i); });
            CloneRows(next.node2nodeOrig, view.NumNode(), mpi, "transition.nodeOrig", one,
                      [&](index i, rowsize) { return source.OriginalEntityId(EntityKind::Node, i); });
            CloneRows(next.bnd2bndOrig, view.NumBnd(), mpi, "transition.bndOrig", one,
                      [&](index i, rowsize) { return source.OriginalEntityId(EntityKind::Bnd, i); });
            if (hasWall)
                CloneRows(next.nodeWallDist, view.NumNode(), mpi, "transition.wall", three,
                          [&](index i, rowsize j) { return view.NodeWallDist(i)[j]; });
            if (view.IsPeriodic())
            {
                CloneRows(next.cell2nodePbi, view.NumCell(), mpi, "transition.cellNodePbi",
                          [&](index i) { return view.CellNodePeriodicBits(i).size(); }, [&](index i, rowsize j) { return view.CellNodePeriodicBits(i)[j]; });
                CloneRows(next.bnd2nodePbi, view.NumBnd(), mpi, "transition.bndNodePbi",
                          [&](index i) { return view.BoundaryNodePeriodicBits(i).size(); }, [&](index i, rowsize j) { return view.BoundaryNodePeriodicBits(i)[j]; });
            }
            if (hasFaces && !input.destroyKinds.count(EntityKind::Face))
            {
                auto &faces = result._sourceSpaces[3];
                CloneRelation(next.cell2face, view.NumCell(), faces, [&](index i) { return view.CellFaces(i); }, mpi, "transition.cell2face");
                CloneRelation(next.face2node, view.NumFace(), nodes, [&](index i) { return view.FaceNodes(i); }, mpi, "transition.face2node");
                CloneRelation(next.face2cell, view.NumFace(), cells, [&](index i) { return view.FaceCells(i); }, mpi, "transition.face2cell");
                CloneRelation(next.face2bnd, view.NumFace(), boundaries, [&](index i) { return view.FaceBoundaries(i); }, mpi, "transition.face2bnd");
                CloneRelation(next.bnd2face, view.NumBnd(), faces, [&](index i) { return view.BoundaryFaces(i); }, mpi, "transition.bnd2face");
                CloneRows(next.faceElemInfo, view.NumFace(), mpi, "transition.faceInfo", one, [&](index i, rowsize) { return view.GetFaceInfo(i); });
                if (view.IsPeriodic())
                {
                    CloneRows(next.cell2facePbi, view.NumCell(), mpi, "transition.cellFacePbi",
                              [&](index i) { return view.CellFacePeriodicBits(i).size(); }, [&](index i, rowsize j) { return view.CellFacePeriodicBits(i)[j]; });
                    CloneRows(next.face2nodePbi, view.NumFace(), mpi, "transition.faceNodePbi",
                              [&](index i) { return view.FaceNodePeriodicBits(i).size(); }, [&](index i, rowsize j) { return view.FaceNodePeriodicBits(i)[j]; });
                }
                next.adjFacialState = next.adjC2FState = Adj_PointToGlobal;
            }
            if (hasEdges && !input.destroyKinds.count(EntityKind::Edge))
            {
                const auto &edges = result._sourceSpaces[hasFaces ? 4 : 3];
                CloneRelation(next.cell2edge, view.NumCell(), edges, [&](index i) { return view.CellEdges(i); }, mpi, "transition.cell2edge");
                CloneRelation(next.edge2node, view.NumEdge(), nodes, [&](index i) { return view.EdgeNodes(i); }, mpi, "transition.edge2node");
                CloneRelation(next.edge2cell, view.NumEdge(), cells, [&](index i) { return view.EdgeCells(i); }, mpi, "transition.edge2cell");
                CloneRows(next.edgeElemInfo, view.NumEdge(), mpi, "transition.edgeInfo", one, [&](index i, rowsize) { return view.GetEdgeInfo(i); });
                if (view.IsPeriodic())
                {
                    CloneRows(next.cell2edgePbi, view.NumCell(), mpi, "transition.cellEdgePbi",
                              [&](index i) { return view.CellEdgePeriodicBits(i).size(); }, [&](index i, rowsize j) { return view.CellEdgePeriodicBits(i)[j]; });
                    CloneRows(next.edge2nodePbi, view.NumEdge(), mpi, "transition.edgeNodePbi",
                              [&](index i) { return view.EdgeNodePeriodicBits(i).size(); }, [&](index i, rowsize j) { return view.EdgeNodePeriodicBits(i)[j]; });
                }
                next.adjEdgeState = Adj_PointToGlobal;
            }
            next.adjPrimaryState = Adj_PointToGlobal;
            result._fields.fields.reserve(fields.size());
            result._bindings.reserve(fields.size());
            result._bindingPrepared.assign(fields.size(), false);
            result._attachments = fields;
            for (const auto &field : fields)
                result._attachmentStamps.push_back(field.attachment());
            if (checkpoint)
                checkpoint("cloned"); });

        auto &next = *result._next;
        next.coords.father->createGlobalMapping();
        next.cell2node.father->createGlobalMapping();
        next.bnd2node.father->createGlobalMapping();
        if (next.face2node.father)
            next.face2node.father->createGlobalMapping();
        if (next.edge2node.father)
        {
            next.edge2node.father->createGlobalMapping();
            next.edge2node.trans.pLGlobalMapping = next.edge2node.father->pLGlobalMapping;
        }
        // Fresh row allocations have no inherited source-row mapping. Every
        // relation participating in traversal needs the canonical mapping of
        // its source family, including unchanged owned Face rows.
        auto rowsOf = [](auto &pair, const auto &canonical)
        {
            if (pair.father)
            {
                pair.father->pLGlobalMapping = canonical.father->pLGlobalMapping;
                pair.trans.pLGlobalMapping = pair.father->pLGlobalMapping;
            }
        };
        rowsOf(next.cellElemInfo, next.cell2node);
        rowsOf(next.cell2cellOrig, next.cell2node);
        rowsOf(next.cell2face, next.cell2node);
        rowsOf(next.cell2nodePbi, next.cell2node);
        rowsOf(next.cell2facePbi, next.cell2node);
        rowsOf(next.cell2edge, next.cell2node);
        rowsOf(next.cell2edgePbi, next.cell2node);
        rowsOf(next.bndElemInfo, next.bnd2node);
        rowsOf(next.bnd2bndOrig, next.bnd2node);
        rowsOf(next.bnd2face, next.bnd2node);
        rowsOf(next.bnd2nodePbi, next.bnd2node);
        rowsOf(next.node2nodeOrig, next.coords);
        rowsOf(next.nodeWallDist, next.coords);
        if (next.face2node.father)
        {
            rowsOf(next.face2cell, next.face2node);
            rowsOf(next.face2bnd, next.face2node);
            rowsOf(next.faceElemInfo, next.face2node);
            rowsOf(next.face2nodePbi, next.face2node);
        }
        if (next.edge2node.father)
        {
            rowsOf(next.edge2cell, next.edge2node);
            rowsOf(next.edgeElemInfo, next.edge2node);
            rowsOf(next.edge2nodePbi, next.edge2node);
        }
        next.RecoverNode2CellAndNode2Bnd();
        next.RecoverCell2CellAndBnd2Cell();

        std::vector<CompanionEntry> deferred;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            deferred = fields;
            for (size_t i = 0; i < deferred.size(); ++i)
            {
                auto &field = deferred[i];
                auto prepare = field.prepare;
                bool bound = field.requiresEntityBinding;
                auto kind = field.kind;
                field.requiresEntityBinding = false;
                field.prepare = [prepare, &result, i, bound, kind](const PermutationTransfer &transfer, const MPIInfo &mpi)
                {
                    auto staged = prepare(transfer, mpi);
                    DNDS_check_throw_info(staged, "field adapter returned no prepared value");
                    if (bound)
                    {
                        result._bindings.emplace_back(kind, staged.get());
                        result._bindingPrepared[i] = true;
                    }
                    result._fields.fields.push_back(std::move(staged));
                    return std::make_unique<DeferPublication>();
                };
            } });
        next.ReorderEntitiesWithFields(input, deferred);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (size_t i = 0; i < fields.size(); ++i)
                if (fields[i].requiresEntityBinding && !result._bindingPrepared[i])
                {
                    auto unchanged = fields[i].prepareUnchanged();
                    DNDS_check_throw_info(unchanged, "entity field unchanged preparation returned no value");
                    result._bindings.emplace_back(fields[i].kind, unchanged.get());
                    result._fields.fields.push_back(std::move(unchanged));
                    result._bindingPrepared[i] = true;
                } });
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { if (checkpoint) checkpoint("reordered"); });
        next.RecoverNode2CellAndNode2Bnd();
        next.RecoverCell2CellAndBnd2Cell();
        next.BuildGhostPrimary(ghostLayers);
        next.AdjGlobal2LocalPrimary();
        if (hasWall)
        {
            auto mutation = next.BeginManagedMutation(Runtime::Change::Interface | Runtime::Change::Values);
            next.nodeWallDist.BorrowAndPull(next.coords);
            mutation.Commit();
        }
        if (hasFaces)
        {
            if (input.destroyKinds.count(EntityKind::Face))
                next.InterpolateFace();
            else
            {
                next.BuildGhostFace();
                // Boundary relations and owning face labels were preserved.
                // The initial-interpolation matcher creates a fresh relation
                // and cannot be reapplied to an already local face2bnd pair.
                auto mutation = next.BeginManagedMutation(Runtime::Change::Interface | Runtime::Change::Values);
                next.bnd2faceV.resize(size_t(next.NumBnd()), -1);
                for (index boundary = 0; boundary < next.NumBnd(); ++boundary)
                {
                    auto face = next.bnd2face(boundary, 0);
                    if (face >= 0 && face != UnInitIndex)
                        next.bnd2faceV[size_t(boundary)] = face;
                }
                for (index face = 0; face < next.face2bnd.Size(); ++face)
                {
                    auto boundary = next.face2bnd(face, 0);
                    if (boundary >= 0 && boundary != UnInitIndex)
                        next.face2bndM[face] = boundary;
                }
                mutation.Commit();
            }
            next.AssertOnFaces();
        }
        next.BuildGhostN2CB();
        next.AdjGlobal2LocalN2CB();
        if (hasEdges)
        {
            if (input.destroyKinds.count(EntityKind::Edge))
                next.InterpolateEdge();
            else
            {
                next.BuildGhostEdge();
                next.AdjGlobal2LocalEdge();
            }
        }
        if (hasAux)
        {
            next.BuildCell2CellFace();
            next.AdjGlobal2LocalC2CFace();
        }
        if (caches == TransitionCachePolicy::RebuildOutput)
        {
            next.RecreatePeriodicNodes();
            next.BuildVTKConnectivity();
        }
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto requirements = MeshReadRequirements::Primary();
            requirements.faces = hasFaces;
            requirements.inverse = true;
            requirements.cellFaceNeighbors = hasAux;
            requirements.edges = hasEdges;
            requirements.wallDistance = hasWall;
            auto validate = next.AcquireRead(requirements);
            if (checkpoint)
                checkpoint("rebuilt"); });
        if (nativeDestination)
        {
            auto requirements = MeshReadRequirements::Primary();
            requirements.faces = hasFaces;
            requirements.inverse = true;
            requirements.edges = hasEdges;
            requirements.cellFaceNeighbors = hasAux;
            requirements.wallDistance = hasWall;
            requirements.allFaceCellsLocal = source.CurrentNativeRequirements().allFaceCellsLocal;
            next.PublishNative(requirements);
        }
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            for (const auto &[kind, prepared] : result._bindings)
                prepared->BindEntitySpace(next.Entities(kind)); });
        return result;
    }

    void PreparedMeshTransition::Publish()
    {
        DNDS_check_throw_info(_source && _handle, "invalid moved-from mesh transition");
        std::lock_guard<std::mutex> lock(_handle->_mutex);
        FieldTransferDetail::LocalStage(_source->mpi, [&]
                                        {
            DNDS_check_throw_info(!_published && _next && _handle->_mesh == _source && _sourceStamp.Valid(),
                                  "transition source handle/order changed or transition already published");
            for (const auto &space : _sourceSpaces)
                DNDS_check_throw_info(space.SameOrder(_source->Entities(space.Kind())),
                                      "transition source entity mapping identity changed before publication");
            for (size_t i = 0; i < _attachments.size(); ++i)
                DNDS_check_throw_info(SameAttachment(_attachments[i].attachment(), _attachmentStamps[i]),
                                      "transition field attachment changed before publication"); });
        _fields.Publish();
        _handle->_mesh.swap(_next);
        _published = true;
        _sourceRead.reset();
    }

    std::shared_ptr<UnstructuredMesh> CloneCurrentMeshForSetup(
        UnstructuredMesh &source, int ghostLayers, TransitionCachePolicy caches, BoundaryCachePolicy boundaryCache)
    {
        std::shared_ptr<UnstructuredMesh> metadata;
        FieldTransferDetail::LocalStage(source.mpi, [&]
                                        {
            DNDS_check_throw_info(boundaryCache == BoundaryCachePolicy::Prohibit ||
                                      boundaryCache == BoundaryCachePolicy::DropBoundaryCache, "unknown export boundary cache policy");
            if (boundaryCache == BoundaryCachePolicy::DropBoundaryCache)
            {
                // Copy metadata only; arrays/controller/native authority retain
                // their existing owners. This alias is a read-only source of
                // the subsequent deep export, never externally published.
                metadata = std::make_shared<UnstructuredMesh>(source);
                metadata->node2bndNode.clear();
            }
            else
                metadata = std::shared_ptr<UnstructuredMesh>(&source, [](UnstructuredMesh *) {}); });
        int policies[]{int(boundaryCache), -int(boundaryCache)};
        MPI_Allreduce(MPI_IN_PLACE, policies, 2, MPI_INT, MPI_MIN, source.mpi.comm);
        DNDS_check_throw_info(policies[0] == -policies[1], "export boundary cache policy differs between ranks");
        MeshTransitionHandle handle(std::move(metadata));
        auto prepared = PreparedMeshTransition::Prepare(handle, {}, {}, {}, ghostLayers, caches, TransitionAuthority::LegacyWorkspace);
        prepared.Publish();
        return handle.Get();
    }

    void WriteCurrentPrimaryMeshSnapshot(UnstructuredMesh &source, Serializer::SerializerBaseSSP serializer,
                                         const std::string &name, int ghostLayers, CurrentMeshCheckpointPolicy policy)
    {
        int authority[]{int(source.HasNativeState()), -int(source.HasNativeState())};
        MPI_Allreduce(MPI_IN_PLACE, authority, 2, MPI_INT, MPI_MIN, source.mpi.comm);
        DNDS_check_throw_info(authority[0] == -authority[1], "current checkpoint authority differs between ranks");
        int mode = 0, rebuild = 0;
        std::unique_ptr<MeshReadAccess> retained;
        FieldTransferDetail::LocalStage(source.mpi, [&]
                                        {
            DNDS_check_throw_info(serializer, "current checkpoint requires a serializer");
            DNDS_check_throw_info(policy == CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues ||
                                      policy == CurrentMeshCheckpointPolicy::DropAuxiliary, "unknown primary checkpoint auxiliary policy");
            retained = std::make_unique<MeshReadAccess>(source.AcquireRead());
            auto requirements = source.HasNativeState() ? source.CurrentNativeRequirements() : MeshReadRequirements::Primary();
            if (!source.HasNativeState())
            {
                requirements.faces = Present(source.face2node);
                requirements.edges = Present(source.edge2node);
                requirements.inverse = Present(source.node2cell) || Present(source.node2bnd);
                requirements.cellFaceNeighbors = Present(source.cell2cellFace);
                requirements.wallDistance = Present(source.nodeWallDist);
            }
            DNDS_check_throw_info(!requirements.wallDistance || policy == CurrentMeshCheckpointPolicy::DropAuxiliary,
                                  "primary mesh checkpoint cannot preserve wall values; explicitly choose CurrentMeshCheckpointPolicy::DropAuxiliary");
            // Primary restart rebuilds these relation groups and all output/
            // partition caches. Bit 64 records explicitly discarded wall data.
            rebuild = int(requirements.faces) | (int(requirements.edges) << 1) | (int(requirements.inverse) << 2) |
                      (int(requirements.cellFaceNeighbors) << 3) | 16 | 32 | (int(requirements.wallDistance) << 6);
            mode = int(serializer->IsPerRank());
            DNDS_check_throw_info(mode || source.mpi == serializer->getMPI(), "current checkpoint serializer communicator differs");
            DNDS_check_throw_info(name.size() <= size_t(std::numeric_limits<int>::max()), "current checkpoint path too long"); });
        int modes[]{mode, -mode};
        MPI_Allreduce(MPI_IN_PLACE, modes, 2, MPI_INT, MPI_MIN, source.mpi.comm);
        DNDS_check_throw_info(modes[0] == -modes[1], "current checkpoint serializer mode differs between ranks");
        int metadata[]{int(policy), rebuild, -int(policy), -rebuild};
        MPI_Allreduce(MPI_IN_PLACE, metadata, 4, MPI_INT, MPI_MIN, source.mpi.comm);
        DNDS_check_throw_info(metadata[0] == -metadata[2] && metadata[1] == -metadata[3],
                              "primary checkpoint policy or rebuild inventory differs between ranks");
        int length = int(name.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, source.mpi.comm);
        std::string expected;
        FieldTransferDetail::LocalStage(source.mpi, [&]
                                        { expected = source.mpi.rank == 0 ? name : std::string(size_t(length), '\0'); });
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, source.mpi.comm);
        FieldTransferDetail::LocalStage(source.mpi, [&]
                                        { DNDS_check_throw_info(name == expected, "current checkpoint path differs between ranks"); });
        auto workspace = CloneCurrentMeshForSetup(source, ghostLayers, TransitionCachePolicy::Invalidate);
        workspace->AdjLocal2GlobalPrimary();
        workspace->WriteSerialize(serializer, name);
        auto cwd = serializer->GetCurrentPath();
        serializer->GoToPath(name);
        serializer->WriteString("checkpoint_kind", "DNDSR.CurrentPrimaryMeshSnapshot");
        serializer->WriteInt("checkpoint_schema", 1);
        serializer->WriteInt("rebuild_inventory", rebuild);
        serializer->WriteString("auxiliary_policy", policy == CurrentMeshCheckpointPolicy::DropAuxiliary ? "drop_auxiliary" : "require_no_auxiliary_values");
        serializer->GoToPath(cwd);
    }

    void WriteCurrentMeshSnapshot(UnstructuredMesh &source, Serializer::SerializerBaseSSP serializer,
                                  const std::string &name, int ghostLayers, CurrentMeshCheckpointPolicy policy)
    {
        WriteCurrentPrimaryMeshSnapshot(source, std::move(serializer), name, ghostLayers, policy);
    }
}
