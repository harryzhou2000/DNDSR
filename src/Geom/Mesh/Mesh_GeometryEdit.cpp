#include "Mesh.hpp"
#include "FieldTransfer.hpp"
#include "NativeConnectivity.hpp"

namespace DNDS::Geom
{
    namespace
    {
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
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                ghost = Runtime::Array<real>::FromStorage(Runtime::RowLayout::Dense(ghost.Rows(), 3), ghost.Storage());
                prepared = _nativeState->WithGeometry({std::move(*coordinates), std::move(ghost)}, {}, view.Periodic()); });
        }
        PublishNativeGeometry(std::move(prepared), stamp);
    }

    void UnstructuredMesh::ApplyGeometryTransform(const Eigen::Matrix3d &rotation, real scale)
    {
        CheckCollectiveMutationAvailable(true);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(HasNativeState(), "geometry transform requires native mesh authority");
                DNDS_check_throw_info(rotation.allFinite() && std::isfinite(scale) && scale > 0 &&
                                                                    (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-10 &&
                                                                    std::abs(rotation.determinant() - 1.0) < 1e-10,
                                                                "geometry transform requires a proper rotation and positive scale"); });
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
    }

    void UnstructuredMesh::ApplyAffineGeometryTransform(const Eigen::Matrix3d &linear, const tPoint &offset)
    {
        CheckCollectiveMutationAvailable(true);
        Eigen::Matrix3d inverse;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            DNDS_check_throw_info(HasNativeState(), "affine geometry transform requires native mesh authority");
            DNDS_check_throw_info(linear.allFinite() && offset.allFinite(), "affine geometry transform must be finite");
            const auto determinant = linear.determinant();
            DNDS_check_throw_info(std::isfinite(determinant) && determinant > 0,
                                  "affine geometry transform must be invertible and orientation preserving");
            inverse = linear.inverse();
            DNDS_check_throw_info(inverse.allFinite(), "affine geometry inverse is not finite"); });
        std::array<real, 12> parameters{};
        std::copy(linear.data(), linear.data() + 9, parameters.begin());
        std::copy(offset.data(), offset.data() + 3, parameters.begin() + 9);
        auto expected = parameters;
        MPI_Bcast(expected.data(), int(expected.size()), MPI_DOUBLE, 0, mpi.comm);
        const auto stamp = versions.Stamp();
        std::shared_ptr<const NativeMeshState> prepared;
        {
            // Keep source geometry, topology and metadata alive through staging;
            // publication occurs only after this operation releases its read.
            std::unique_ptr<MeshReadAccess> sourceRead;
            FieldTransferDetail::LocalStage(mpi, [&]
                                            {
                DNDS_check_throw_info(parameters == expected, "affine geometry transform differs across ranks");
                sourceRead = std::make_unique<MeshReadAccess>(AcquireRead(_nativeState->Requirements()));
                const auto view = sourceRead->HostView();
                auto frame = view.Periodic();
                for (size_t image = 0; image < frame.rotation.size(); ++image)
                {
                    const Eigen::Matrix3d rotation = linear * frame.rotation[image].map() * inverse;
                    DNDS_check_throw_info(rotation.allFinite() &&
                                              (!view.IsPeriodic() ||
                                               ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-10 &&
                                                std::abs(rotation.determinant() - 1.0) < 1e-10)),
                                          "affine geometry transform makes a periodic rotation nonrigid");
                    frame.rotation[image].map() = rotation;
                    frame.rotationCenter[image].map() = (linear * frame.rotationCenter[image].map() + offset).eval();
                    frame.translation[image].map() = (linear * frame.translation[image].map()).eval();
                }
                for (size_t image = 0; image < frame.rotation.size(); ++image)
                    DNDS_check_throw_info(frame.translation[image].map().allFinite() && frame.rotationCenter[image].map().allFinite(),
                                          "affine geometry frame overflowed");
                auto coordinates = GeometryRows(view, [&](index node)
                {
                    const tPoint point = linear * view.Coord(node) + offset;
                    DNDS_check_throw_info(point.allFinite(), "affine geometry coordinate overflowed");
                    return point;
                });
                // General affine maps do not preserve nearest-wall directions or
                // distances. Recompute wall geometry after this publication.
                prepared = _nativeState->WithGeometry(std::move(coordinates), {}, frame); });
        }
        PublishNativeGeometry(std::move(prepared), stamp);
    }

    void UnstructuredMesh::RectifyBoundaryCoordinates(std::function<int(t_index)> componentForZone,
                                                      bool volumeBoundary)
    {
        CheckCollectiveMutationAvailable(true);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(HasNativeState(), "boundary rectification requires native mesh authority"); });
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
    }
}
