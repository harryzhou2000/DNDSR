#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy
{
    /// One frozen, centered face incidence. Multiple periodic images remain
    /// separate contributions even when they address the same matrix column.
    struct JacobianPointRecord
    {
        index other, auxiliary, sample;
        Vector3 normal, location;
        std::array<real, 9> rotation;
        real weight;
        bool changesFrame;
    };
    static_assert(std::is_trivially_copyable_v<JacobianPointRecord>);

    struct JacobianPointValue
    {
        index other, auxiliary, sample;
        Vector3 normal, location;
        Eigen::Matrix3d rotation;
        real weight;
        bool changesFrame;
    };
    inline JacobianPointValue ExpandJacobianPoint(const JacobianPointRecord &point)
    {
        return {point.other, point.auxiliary, point.sample, point.normal,
                point.location, Eigen::Map<const Eigen::Matrix3d>(point.rotation.data()),
                point.weight, point.changesFrame};
    }

    /// Two-level CSR: cell -> block, block -> contributing face incidences.
    /// Pure views; the enclosing evaluator retains the read scope once.
    struct JacobianGeometryView
    {
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows> columns;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> blockOffsets;
        Runtime::ArrayView<const JacobianPointRecord, Runtime::AccessBoundsChecks, Runtime::CSRRows> points;
        DNDS_DEVICE_CALLABLE auto Points(index iCell, index iBlock) const
        {
            DNDS_HD_assert(iBlock >= 0 && iBlock < columns.RowSize(iCell));
            return points[blockOffsets(iCell, 0) + iBlock];
        }
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, columns, blockOffsets, points);
        }
    };

    /** @brief Owned, rank-local geometry of an evaluated sparse Jacobian.
     * @details Setup vectors exist only while publishing. ReadAccess retains
     * all arrays and their layouts; CPU block actions receive only View().
     */
    class JacobianGeometryStorage
    {
        Runtime::Array<index> _columns, _blockOffsets;
        Runtime::Array<JacobianPointRecord> _points;
        JacobianGeometryStorage() = default;
        void Build(FiniteVolumeStencilView, CFV::FiniteVolumeHostView,
                   const std::vector<std::vector<index>> &, bool auxiliaryAreas);

    public:
        bool sameComponentFrames = true;
        JacobianGeometryStorage(FiniteVolumeStencilView stencil,
                                CFV::FiniteVolumeHostView metric,
                                const std::vector<std::vector<index>> &columns);
        /// Setup from centered stencil data alone. auxiliaryFaceArea supplies
        /// the same integrated scalar areas as the metric facade. This permits
        /// device setup to copy only incidence/area metadata to CPU, not every
        /// cell/face quadrature or reconstruction metric payload.
        JacobianGeometryStorage(FiniteVolumeStencilView stencil, const std::vector<std::vector<index>> &columns);
        class ReadAccess
        {
            Runtime::Array<index>::ReadLease _columns, _blockOffsets;
            Runtime::Array<JacobianPointRecord>::ReadLease _points;

        public:
            explicit ReadAccess(const JacobianGeometryStorage &storage, Runtime::ExecutionContext context)
                : _columns(storage._columns.Read(context)), _blockOffsets(storage._blockOffsets.Read(context)), _points(storage._points.Read(context)) {}
            JacobianGeometryView View() const
            {
                return {_columns.View<Runtime::CSRRows>(), _blockOffsets.View<Runtime::FixedRows<1>>(),
                        _points.View<Runtime::CSRRows>()};
            }
        };
        /// Retain through asynchronous use of the borrowed geometry views.
        ReadAccess Read(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const { return ReadAccess(*this, context); }
        /// Synchronous setup placement, independent of the source allocation.
        JacobianGeometryStorage Placed(Runtime::Device device, Runtime::ExecutionContext context) const;
        std::vector<std::vector<index>> CopyColumns() const;
    };
}
