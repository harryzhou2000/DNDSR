#include "Mesh.hpp"
#include "FieldTransfer.hpp"
#include "NativeConnectivity.hpp"

namespace DNDS::Geom
{
    namespace
    {
        void CheckGeometryAuthority(const MPIInfo &mpi, bool native)
        {
            int mode = int(native), minimum = 0, maximum = 0;
            MPI_Allreduce(&mode, &minimum, 1, MPI_INT, MPI_MIN, mpi.comm);
            MPI_Allreduce(&mode, &maximum, 1, MPI_INT, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(minimum == maximum, "geometry authority differs between ranks");
        }

        template <class F>
        Runtime::ArrayPair<real> GeometryRows(const UnstructuredMeshView &view, F &&value)
        {
            Runtime::ArrayPair<real> result{
                Runtime::Array<real>(Runtime::RowLayout::Dense(size_t(view.NumNode()), 3)),
                Runtime::Array<real>(Runtime::RowLayout::Dense(size_t(view.NumNodeGhost()), 3))};
            auto owned = result.owned.Write(), ghost = result.ghost.Write();
            for (index node = 0; node < view.NumNodeProc(); ++node)
            {
                const tPoint point = value(node);
                auto output = node < view.NumNode() ? owned.View()[size_t(node)] : ghost.View()[size_t(node - view.NumNode())];
                for (size_t component = 0; component < 3; ++component)
                    output[component] = point[component];
            }
            return result;
        }
    }

    void UnstructuredMesh::PublishNativeGeometry(std::shared_ptr<const NativeMeshState> prepared,
                                                 Runtime::VersionStamp sourceStamp, Runtime::Change changes)
    {
        CheckCollectiveMutationAvailable(true);
        std::shared_ptr<NativeMeshState> next;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            sourceStamp.Validate();
            DNDS_check_throw_info(_nativeState && prepared, "native geometry publication requires source and destination");
            prepared->Validate();
            next = std::shared_ptr<NativeMeshState>(new NativeMeshState(*prepared)); });
        MeshMutationToken mutation(this, versions, changes);
        mutation.Commit();
        // No fallible array construction remains after committing generations.
        next->BindPreparedEpoch(versions);
        _nativeState = std::move(next);
        _nativeSmoothing.reset();
    }

    void UnstructuredMesh::TransformNativeCoords(const std::function<tPoint(const tPoint &)> &transform)
    {
        CheckCollectiveMutationAvailable(true);
        const auto stamp = versions.Stamp();
        std::shared_ptr<const NativeMeshState> prepared;
        {
            auto sourceRead = AcquireRead(_nativeState->Requirements());
            const auto view = sourceRead.HostView();
            std::optional<Runtime::Array<real>> coordinates;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(!view.IsPeriodic(), "arbitrary native coordinate transforms require nonperiodic topology; use ApplyGeometryTransform for periodic rigid transforms");
                coordinates.emplace(Runtime::RowLayout::Dense(size_t(view.NumNode()), 3));
                auto write = coordinates->Write();
                for (index node = 0; node < view.NumNode(); ++node)
                {
                    const auto point = transform(view.Coord(node));
                    DNDS_check_throw_info(point.allFinite(), "native coordinate callback produced nonfinite values");
                    for (size_t component = 0; component < 3; ++component) write.View()(size_t(node), component) = point[component];
                } });
            const auto nodes = Entities(EntityKind::Node);
            RelationGhostRoute route(nodes);
            auto ghost = route.TransferRows(nodes, *coordinates, Runtime::ElementSchema::Arithmetic<real>());
            ghost = Runtime::Array<real>::FromStorage(Runtime::RowLayout::Dense(ghost.Rows(), 3), ghost.Storage());
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { prepared = _nativeState->WithGeometry({std::move(*coordinates), std::move(ghost)}, {}, view.Periodic()); });
        }
        PublishNativeGeometry(std::move(prepared), stamp);
    }

    void UnstructuredMesh::ApplyGeometryTransform(const Eigen::Matrix3d &rotation, real scale)
    {
        CheckCollectiveMutationAvailable(true);
        CheckGeometryAuthority(mpi, HasNativeState());
        std::vector<Runtime::WriteAccess> writes;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(rotation.allFinite() && std::isfinite(scale) && scale > 0 &&
                                                                    (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-10 &&
                                                                    std::abs(rotation.determinant() - 1.0) < 1e-10,
                                                                "geometry transform requires a proper rotation and positive scale"); });
        if (_nativeState)
        {
            // Transforming every owned/ghost row locally is valid only if all
            // ranks use the same affine map. Detect disagreement before writes.
            std::array<real, 10> parameters{};
            std::copy(rotation.data(), rotation.data() + 9, parameters.begin());
            parameters[9] = scale;
            auto expected = parameters;
            MPI_Bcast(expected.data(), int(expected.size()), MPI_DOUBLE, 0, mpi.comm);
            std::shared_ptr<const NativeMeshState> prepared;
            const auto stamp = versions.Stamp();
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(parameters == expected, "native geometry transform differs across ranks");
                const auto view = _nativeState->BorrowHostView();
                auto coordinates = GeometryRows(view, [&](index node) { return (scale * rotation * view.Coord(node)).eval(); });
                std::optional<Runtime::ArrayPair<real>> walls;
                if (view.HasNodeWallDist())
                    walls = GeometryRows(view, [&](index node) { return (scale * rotation * view.NodeWallDist(node)).eval(); });
                auto frame = view.Periodic();
                for (auto &r : frame.rotation) r.map() = (rotation * r.map() * rotation.transpose()).eval();
                for (auto &center : frame.rotationCenter) center.map() = (scale * rotation * center.map()).eval();
                for (auto &translation : frame.translation) translation.map() = (scale * rotation * translation.map()).eval();
                prepared = _nativeState->WithGeometry(std::move(coordinates), std::move(walls), frame); });
            PublishNativeGeometry(std::move(prepared), stamp);
            return;
        }
        try
        {
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
            auto retain = [&](auto &pair)
            {
                if (pair.father) writes.push_back(pair.father->HostWriteData());
                if (pair.son) writes.push_back(pair.son->HostWriteData());
            };
            retain(coords);
            retain(nodeWallDist); });
        }
        catch (...)
        {
            // No payload was written. Release successful local acquisitions
            // outside exception unwinding when another rank rejected setup.
            // A failed write lease would otherwise poison untouched storage.
            writes.clear();
            throw;
        }
        auto mutation = BeginManagedMutation(Runtime::Change::Geometry | Runtime::Change::Values | Runtime::Change::Interface);
        for (index node = 0; node < coords.Size(); ++node)
            coords[node] = (scale * rotation * coords[node]).eval();
        if (nodeWallDist.father && nodeWallDist.son)
            for (index node = 0; node < nodeWallDist.Size(); ++node)
                nodeWallDist[node] = (scale * rotation * nodeWallDist[node]).eval();
        for (auto &r : periodicInfo.rotation)
            r.map() = (rotation * r.map() * rotation.transpose()).eval();
        for (auto &center : periodicInfo.rotationCenter)
            center.map() = (scale * rotation * center.map()).eval();
        for (auto &translation : periodicInfo.translation)
            translation.map() = (scale * rotation * translation.map()).eval();
        mutation.Commit();
    }

    void UnstructuredMesh::RectifyBoundaryCoordinates(std::function<int(t_index)> componentForZone,
                                                      bool volumeBoundary)
    {
        CheckCollectiveMutationAvailable(true);
        CheckGeometryAuthority(mpi, HasNativeState());
        if (_nativeState)
        {
            const auto stamp = versions.Stamp();
            const auto nodes = Entities(EntityKind::Node);
            std::optional<Runtime::Array<uint8_t>> owned, ghost;
            std::shared_ptr<const NativeMeshState> prepared;
            {
                auto sourceRead = AcquireRead(_nativeState->Requirements());
                const auto view = sourceRead.HostView();
                int selectedMode = volumeBoundary ? 1 : 0, expectedMode = selectedMode;
                MPI_Bcast(&expectedMode, 1, MPI_INT, 0, mpi.comm);
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    DNDS_check_throw_info(selectedMode == expectedMode, "native rectification mode differs across ranks");
                    owned.emplace(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 1));
                    ghost.emplace(Runtime::RowLayout::Dense(size_t(nodes.NumGhost()), 1));
                    auto local = owned->Write(), remote = ghost->Write();
                    for (size_t row = 0; row < owned->Rows(); ++row) local.View()(row, 0) = 0;
                    for (size_t row = 0; row < ghost->Rows(); ++row) remote.View()(row, 0) = 0;
                    const auto count = volumeBoundary ? view.NumBnd() : view.NumCell();
                    for (index row = 0; row < count; ++row)
                    {
                        const auto zone = volumeBoundary ? view.GetBndInfo(row).zone : view.GetCellInfo(row).zone;
                        const auto component = componentForZone(zone);
                        DNDS_check_throw_info(component >= 0 && component <= 3, "rectification component must be zero or one through three");
                        if (!component) continue;
                        const auto cone = volumeBoundary ? view.BoundaryNodes(row) : view.CellNodes(row);
                        for (auto node : cone)
                        {
                            DNDS_check_throw_info(node >= 0 && node < nodes.NumLocal(), "native rectification node is unavailable");
                            auto &value = node < nodes.NumOwned() ? local.View()(size_t(node), 0) : remote.View()(size_t(node - nodes.NumOwned()), 0);
                            value |= uint8_t(1u << (component - 1));
                        }
                    } });
                const auto route = nodes.BuildRowTransferPlan();
                Runtime::PayloadPlan reverse(route.Reverse(), ghost->Layout(), owned->Layout(), Runtime::ElementSchema::Arithmetic<uint8_t>());
                reverse.Bind(*ghost, *owned, Runtime::Reducer::Builtin<uint8_t>(Runtime::Reduction::BitOr),
                             Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient})
                    .Begin()
                    .Wait();
                Runtime::PayloadPlan forward(route, owned->Layout(), ghost->Layout(), Runtime::ElementSchema::Arithmetic<uint8_t>());
                forward.Bind(*owned, *ghost, Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient}).Begin().Wait();
                FieldTransferDetail::LocalStage(mpi, [&]
                                                {
                    auto local = owned->Read(), remote = ghost->Read();
                    auto coordinates = GeometryRows(view, [&](index node)
                    {
                        tPoint point = view.Coord(node);
                        const auto mask = node < nodes.NumOwned() ? local.View()(size_t(node), 0) : remote.View()(size_t(node - nodes.NumOwned()), 0);
                        for (size_t component = 0; component < 3; ++component)
                            if (mask & (1u << component)) point[component] = 0;
                        return point;
                    });
                    prepared = _nativeState->WithGeometry(std::move(coordinates), {}, view.Periodic()); });
            }
            PublishNativeGeometry(std::move(prepared), stamp);
            return;
        }
        std::vector<uint8_t> mask;
        // Run fallible callbacks before editing coordinates, with readers
        // preventing callback reentry from modifying the mesh.
        {
            auto read = versions.Read();
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(coords.father && coords.son, "rectification requires coordinates");
                const auto &relation = volumeBoundary ? bnd2node : cell2node;
                DNDS_check_throw_info(relation.idx.isLocal(), "rectification requires local node relations");
                mask.assign(size_t(coords.Size()), 0);
                for (index row = 0; row < relation.father->Size(); ++row)
                {
                    auto zone = volumeBoundary ? bndElemInfo(row, 0).zone : cellElemInfo(row, 0).zone;
                    const auto component = componentForZone(zone);
                    DNDS_check_throw_info(component >= 0 && component <= 3, "rectification component must be zero or one through three");
                    if (!component) continue;
                    for (auto node : relation[row])
                    {
                        DNDS_check_throw_info(node >= 0 && node < coords.Size(), "rectification node is unavailable");
                        mask[size_t(node)] |= uint8_t(1u << (component - 1));
                    }
                } });
            if (volumeBoundary)
            {
                // A boundary owned here may reference a node owned elsewhere.
                // Reduce the selected components onto owners before refreshing
                // coordinate ghosts, otherwise that refresh would undo the edit.
                auto nodes = LegacyEntities(EntityKind::Node);
                auto route = nodes.BuildRowTransferPlan();
                Runtime::Array<uint8_t> owned(Runtime::RowLayout::Dense(size_t(nodes.NumOwned()), 1));
                Runtime::Array<uint8_t> ghost(Runtime::RowLayout::Dense(size_t(nodes.NumGhost()), 1));
                {
                    auto writeOwned = owned.Write();
                    auto writeGhost = ghost.Write();
                    for (size_t i = 0; i < owned.Rows(); ++i)
                        writeOwned.View()(i, 0) = mask[i];
                    for (size_t i = 0; i < ghost.Rows(); ++i)
                        writeGhost.View()(i, 0) = mask[owned.Rows() + i];
                }
                Runtime::PayloadPlan reverse(route.Reverse(), ghost.Layout(), owned.Layout(), Runtime::ElementSchema::Arithmetic<uint8_t>());
                reverse.Bind(ghost, owned, Runtime::Reducer::Builtin<uint8_t>(Runtime::Reduction::BitOr),
                             Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient})
                    .Begin()
                    .Wait();
                auto selected = owned.Read();
                for (size_t i = 0; i < owned.Rows(); ++i)
                    mask[i] = selected.View()(i, 0);
            }
        }
        Runtime::WriteAccess ownedWrite, ghostWrite;
        try
        {
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                ownedWrite = coords.father->HostWriteData();
                ghostWrite = coords.son->HostWriteData(); });
        }
        catch (...)
        {
            ownedWrite = Runtime::WriteAccess{};
            ghostWrite = Runtime::WriteAccess{};
            throw;
        }
        auto mutation = BeginManagedMutation(Runtime::Change::Geometry | Runtime::Change::Values | Runtime::Change::Interface);
        for (index node = 0; node < coords.Size(); ++node)
            for (int component = 0; component < 3; ++component)
                if (mask[size_t(node)] & (1u << component))
                    coords[node](component) = 0;
        if (volumeBoundary)
            coords.trans.pullOnce();
        mutation.Commit();
    }
}
