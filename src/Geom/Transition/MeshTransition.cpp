#include "Geom/Transition/MeshTransition.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include <type_traits>

namespace DNDS::Geom
{
    namespace
    {
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

        // Legacy primary-file encoding is an I/O compatibility boundary. Stage
        // one payload at a time; never reconstruct a complete legacy mesh just
        // to serialize the native publication. Ghost rows come from owners,
        // whose incidence rows are complete even at the closure boundary.
        void WritePrimaryPayloads(UnstructuredMesh &source, const UnstructuredMeshView &view,
                                  const Serializer::SerializerBaseSSP &serializer, const std::string &name)
        {
            const auto cwd = serializer->GetCurrentPath();
            struct RestorePath
            {
                Serializer::SerializerBaseSSP serializer;
                std::string path;
                ~RestorePath() { serializer->GoToPath(path); }
            } restore{serializer, cwd};
            serializer->CreatePath(name);
            serializer->GoToPath(name);
            serializer->WriteString("mesh", "UnstructuredMesh");
            serializer->WriteIndex("dim", view.Dimension());
            if (serializer->IsPerRank())
                serializer->WriteIndex("MPIRank", source.mpi.rank);
            serializer->WriteIndex("MPISize", source.mpi.size);
            serializer->WriteInt("isPeriodic", view.IsPeriodic());

            auto write = [&](auto pair, EntityKind kind, const char *field, auto width, auto value)
            {
                const auto space = source.Entities(kind);
                FieldTransferDetail::LocalStage(source.mpi, [&]
                                                {
                    CloneRows(pair, space.NumOwned(), source.mpi, field, width, value);
                    LegacyFieldBridge::BorrowIndexing(pair.trans, space); });
                pair.trans.createMPITypes();
                pair.trans.pullOnce();
                pair.WriteSerialize(serializer, field);
            };
            const auto one = [](index)
            { return rowsize(1); };
            const auto nodes = source.Entities(EntityKind::Node);
            write(tCoordPair{}, EntityKind::Node, "coords", [](index)
                  { return rowsize(3); }, [&](index row, rowsize slot)
                  { return view.Coord(row)[slot]; });
            write(tAdjPair{}, EntityKind::Cell, "cell2node", [&](index row)
                  { return view.CellNodes(row).size(); }, [&](index row, rowsize slot)
                  { return nodes.LocalToGlobal(view.CellNodes(row)[slot]); });
            write(tElemInfoArrayPair{}, EntityKind::Cell, "cellElemInfo", one,
                  [&](index row, rowsize)
                  { return view.GetCellInfo(row); });
            write(tAdjPair{}, EntityKind::Bnd, "bnd2node", [&](index row)
                  { return view.BoundaryNodes(row).size(); }, [&](index row, rowsize slot)
                  { return nodes.LocalToGlobal(view.BoundaryNodes(row)[slot]); });
            write(tElemInfoArrayPair{}, EntityKind::Bnd, "bndElemInfo", one,
                  [&](index row, rowsize)
                  { return view.GetBndInfo(row); });
            if (view.IsPeriodic())
            {
                write(tPbiPair{}, EntityKind::Cell, "cell2nodePbi", [&](index row)
                      { return view.CellNodes(row).size(); }, [&](index row, rowsize slot)
                      { return view.CellNodePeriodicBits(row)[slot]; });
                write(tPbiPair{}, EntityKind::Bnd, "bnd2nodePbi", [&](index row)
                      { return view.BoundaryNodes(row).size(); }, [&](index row, rowsize slot)
                      { return view.BoundaryNodePeriodicBits(row)[slot]; });
                auto periodic = view.Periodic();
                periodic.WriteSerializer(serializer, "periodicInfo");
            }
            for (const auto &[kind, field] : std::array<std::pair<EntityKind, const char *>, 3>{
                     {{EntityKind::Bnd, "bnd2bndOrig"}, {EntityKind::Cell, "cell2cellOrig"}, {EntityKind::Node, "node2nodeOrig"}}})
                write(tAdj1Pair{}, kind, field, one,
                      [&](index row, rowsize)
                      { return source.OriginalEntityId(kind, row); });
        }

        void CheckInventory(const UnstructuredMesh &mesh, const UnstructuredMeshView &view, const ReorderInput &input,
                            const std::vector<CompanionEntry> &fields)
        {
            DNDS_check_throw_info(view.Dimension() >= 1 && view.Dimension() <= 3, "transition requires dimension one, two or three");
            const bool faces = view.HasRelation(MeshRelation::FaceNodes);
            const bool edges = view.HasRelation(MeshRelation::EdgeNodes);
            DNDS_check_throw_info(!view.HasRelation(MeshRelation::CellFaceCells) || faces,
                                  "transition face-neighbor storage requires prepared facial storage");
            DNDS_check_throw_info(!view.HasRelation(MeshRelation::EdgeNodes) || (faces && view.Dimension() == 3),
                                  "transition edge reconstruction requires prepared 3D faces");
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
                DNDS_check_throw_info(supportedKind(follow.follower) && supportedKind(follow.leader) &&
                                          !input.destroyKinds.count(follow.follower) && !input.destroyKinds.count(follow.leader) &&
                                          mappedKinds.count(follow.leader) && follow.follower2leader.from == follow.follower &&
                                          follow.follower2leader.to == follow.leader,
                                      "invalid or unresolvable transition follow placement");
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
                    schedule += "f" + std::to_string(int(follow.follower)) + ":" + std::to_string(int(follow.leader)) + ":" + follow.follower2leader.Signature() + ";";
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
        const std::function<void(const char *)> &checkpoint, int ghostLayers, TransitionCachePolicy caches)
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
            DNDS_check_throw_info(source.HasNativeState(), "detached transition requires native mesh authority");
            const auto requirements = source.CurrentNativeRequirements();
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

        // Every following optional collective must have one schedule on all ranks.
        int inventory = int(hasFaces) | (int(hasAux) << 1) | (int(hasEdges) << 2) | (int(hasWall) << 3) |
                        (int(caches) << 4) | (int(view.IsPeriodic()) << 5) |
                        (int(input.destroyKinds.count(EntityKind::Face)) << 6) | (view.Dimension() << 7) |
                        (int(input.destroyKinds.count(EntityKind::Edge)) << 9);
        int inventoryMin = inventory, inventoryMax = inventory;
        MPI_Allreduce(MPI_IN_PLACE, &inventoryMin, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &inventoryMax, 1, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(inventoryMin == inventoryMax, "transition storage/rebuild inventory differs between ranks");
        int minLayers = ghostLayers, maxLayers = ghostLayers;
        MPI_Allreduce(MPI_IN_PLACE, &minLayers, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &maxLayers, 1, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(minLayers == maxLayers, "transition ghost layer policy differs between ranks");

        {
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                result._fields.fields.reserve(fields.size());
                result._bindings.reserve(fields.size());
                result._bindingPrepared.assign(fields.size(), false);
                result._attachments = fields;
                for (const auto &field : fields)
                    result._attachmentStamps.push_back(field.attachment());
                if (checkpoint) checkpoint("cloned"); });
            auto moved = NativeMeshTransition::Build(*source._nativeState, mpi, input, ghostLayers);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                result._next = UnstructuredMesh::FromNative(mpi, std::move(moved.state));
                result._next->mRank = source.mRank;
                result._next->hdf5OutSetting = source.hdf5OutSetting;
                if (checkpoint) checkpoint("reordered"); });
            ReorderPlan plan;
            plan.transfers = std::move(moved.transfers);
            // Only selected families are active field operations. Local cell
            // permutations also return identity maps; general transitions may
            // omit untouched families. Neither replaces unchanged owned-only
            // fields or advances their epochs.
            for (const auto &map : input.explicitMaps)
                plan.reorderedKinds.insert(map.kind);
            for (const auto &follow : input.follows)
                plan.reorderedKinds.insert(follow.follower);
            if (input.follows.empty() && plan.reorderedKinds.count(EntityKind::Cell))
                for (auto kind : {EntityKind::Node, EntityKind::Bnd})
                    plan.reorderedKinds.insert(kind);
            ReorderRegistry registry;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                registry.companions = fields;
                for (size_t i = 0; i < registry.companions.size(); ++i)
                {
                    auto &entry = registry.companions[i];
                    const bool bind = entry.requiresEntityBinding;
                    const auto kind = entry.kind;
                    auto prepare = entry.prepare;
                    entry.requiresEntityBinding = false;
                    entry.prepare = [prepare, &result, i, bind, kind](const PermutationTransfer &transfer, const MPIInfo &mpi)
                    {
                        auto staged = prepare(transfer, mpi);
                        DNDS_check_throw_info(staged, "native field preparation returned no value");
                        if (bind)
                        {
                            result._bindings.emplace_back(kind, staged.get());
                            result._bindingPrepared[i] = true;
                        }
                        result._fields.fields.push_back(std::move(staged));
                        return std::make_unique<DeferPublication>();
                    };
                } });
            // Validates full field schema/disposition schedule before the first
            // field transfer. Deferred callbacks never publish into live handles.
            auto deferred = plan.prepareCompanions(registry, mpi);
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                for (size_t i = 0; i < fields.size(); ++i)
                    if (fields[i].requiresEntityBinding && !result._bindingPrepared[i])
                    {
                        auto staged = fields[i].prepareUnchanged();
                        DNDS_check_throw_info(staged, "unchanged native field preparation returned no value");
                        result._bindings.emplace_back(fields[i].kind, staged.get());
                        result._fields.fields.push_back(std::move(staged));
                    }
                for (const auto &[kind, staged] : result._bindings)
                    staged->BindEntitySpace(result._next->Entities(kind));
                if (checkpoint) checkpoint("rebuilt"); });
            if (caches == TransitionCachePolicy::RebuildOutput)
            {
                auto output = NativeMeshOutput::Build(*result._next->CurrentNativeState(), mpi);
                FieldTransferDetail::LocalStage(mpi, [&]
                                                { result._output = std::make_shared<NativeMeshOutput>(std::move(output)); });
            }
            return result;
        }
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

    void WriteCurrentPrimaryMeshSnapshot(UnstructuredMesh &source, Serializer::SerializerBaseSSP serializer,
                                         const std::string &name, int ghostLayers, CurrentMeshCheckpointPolicy policy)
    {
        int mode = 0, rebuild = 0;
        std::unique_ptr<MeshReadAccess> retained;
        FieldTransferDetail::LocalStage(source.mpi, [&]
                                        {
            DNDS_check_throw_info(source.HasNativeState(), "current primary checkpoint requires native mesh authority");
            DNDS_check_throw_info(serializer, "current checkpoint requires a serializer");
            DNDS_check_throw_info(policy == CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues ||
                                      policy == CurrentMeshCheckpointPolicy::DropAuxiliary, "unknown primary checkpoint auxiliary policy");
            retained = std::make_unique<MeshReadAccess>(source.AcquireRead());
            const auto requirements = source.CurrentNativeRequirements();
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
        FieldTransferDetail::LocalStage(source.mpi, [&]
                                        { DNDS_check_throw_info(ghostLayers >= 1, "primary checkpoint ghost layers must be positive"); });
        WritePrimaryPayloads(source, retained->HostView(), serializer, name);
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
