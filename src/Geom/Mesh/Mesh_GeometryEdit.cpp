#include "Mesh.hpp"
#include "FieldTransfer.hpp"

namespace DNDS::Geom
{
    void UnstructuredMesh::ApplyGeometryTransform(const Eigen::Matrix3d &rotation, real scale)
    {
        CheckCollectiveMutationAvailable();
        std::vector<Runtime::WriteAccess> writes;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(rotation.allFinite() && std::isfinite(scale) && scale > 0 &&
                                                                    (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() < 1e-10 &&
                                                                    std::abs(rotation.determinant() - 1.0) < 1e-10,
                                                                "geometry transform requires a proper rotation and positive scale"); });
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
        CheckCollectiveMutationAvailable();
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
