#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    PolynomialSamplingPlan::PolynomialSamplingPlan(VariationalReconstructionPlan reconstruction,
                                                   const R::Array<index> &cells, const R::Array<real> &points, bool cacheDerivatives,
                                                   R::ExecutionContext sourceContext)
        : PolynomialSamplingPlan(reconstruction.Metrics(), reconstruction, cells, points, cacheDerivatives, sourceContext) {}
    PolynomialSamplingPlan::PolynomialSamplingPlan(FiniteVolumePlan metrics, VariationalReconstructionPlan reconstruction,
                                                   const R::Array<index> &cells, const R::Array<real> &points, bool cacheDerivatives,
                                                   R::ExecutionContext sourceContext)
        : _mesh(metrics._data->mesh), _stamp(metrics._data->stamp),
          _cellSpace(metrics.Cells()), _space(metrics.Space()),
          _points(R::RowLayout::Dense(0, 3)), _basis(R::RowLayout::Dense(0, 0)),
          _moments(R::RowLayout::Dense(0, 0)), _cellGeometry(R::RowLayout::Dense(0, 8)), _empty(R::RowLayout::Dense(0, 0)),
          _dimension(metrics.Dimension()), _degree(reconstruction.Settings().maxOrder),
          _modes(reconstruction.NRecDOF()), _cacheDerivatives(cacheDerivatives)
    {
        DNDS_check_throw_info(reconstruction.Space().Same(metrics.Space()) &&
                                  (reconstruction.Device() == R::Device::Host() || sourceContext.device == reconstruction.Device()) &&
                                  (metrics.Device() == R::Device::Host() || sourceContext.device == metrics.Device()),
                              "sampling setup needs matching metric space and source context");
        auto geometry = metrics.AcquireRead(metrics.Device() == R::Device::Host() ? R::ExecutionContext::Host() : sourceContext);
        const auto &sourceMoments = reconstruction.SamplingMoments();
        const auto hostMoments = sourceMoments.GetDevice() == R::Device::Host() ? sourceMoments : sourceMoments.To(R::Device::Host(), sourceContext);
        const auto hostCells = metrics.Device() == R::Device::Host() ? metrics._data->cell : metrics._data->cell.To(R::Device::Host(), sourceContext);
        Validate();
        DNDS_check_throw_info(cells.GetDevice() == R::Device::Host() && points.GetDevice() == R::Device::Host() &&
                                  cells.Layout().Kind() == R::RowLayoutKind::Table && cells.Layout().Width() == 1 &&
                                  points.Layout().Kind() == R::RowLayoutKind::Table && points.Layout().Width() == 3 &&
                                  cells.RowCount() == points.RowCount(),
                              "polynomial sampling input shape/placement mismatch");
        const Math::TaylorBasis polynomial(_dimension, _degree);
        const int cacheOrder = _cacheDerivatives ? reconstruction.Settings().cacheDiffBaseOrder : 0;
        _cachedDerivatives = reconstruction.Settings().cacheDiffBase ? polynomial.DerivativeNumber(cacheOrder) : 0;
        const bool direct = _cachedDerivatives < (_cacheDerivatives ? _modes + 1 : 1);
        _cells = R::Array<index>(R::RowLayout::Dense(cells.RowCount(), 1));
        if (_cachedDerivatives)
            _basis = R::Array<real>(R::RowLayout::Dense(cells.RowCount(), _modes * _cachedDerivatives));
        if (direct)
        {
            _points = R::Array<real>(R::RowLayout::Dense(cells.RowCount(), 3));
            _moments = hostMoments;
            _cellGeometry = hostCells;
        }
        const auto cellRead = cells.Read();
        const auto pointRead = points.Read();
        const auto ids = cellRead.View<R::FixedRows<1>>();
        const auto inputPoints = pointRead.View<R::FixedRows<3>>();
        for (index iPoint = 0; iPoint < ids.RowCount(); ++iPoint)
        {
            const index iCell = ids(iPoint, 0);
            DNDS_check_throw_info(iCell >= 0 && iCell < _cellSpace.NumLocal() &&
                                      std::isfinite(inputPoints(iPoint, 0)) && std::isfinite(inputPoints(iPoint, 1)) && std::isfinite(inputPoints(iPoint, 2)),
                                  "invalid sampling cell or point");
            _needsGhost |= iCell >= _cellSpace.NumOwned();
        }
        auto cellWrite = _cells.Write();
        auto pointWrite = _points.Write(), basisWrite = _basis.Write();
        const auto snapshot = cellWrite.View<R::FixedRows<1>>();
        const auto positions = pointWrite.View<R::FixedRows<3>>();
        const auto cached = basisWrite.View<R::TableRows>();
        const auto momentRead = hostMoments.Read();
        const auto moments = momentRead.View<R::TableRows>();
        const auto cellReadGeometry = hostCells.Read();
        FiniteVolumeHostView geom;
        geom.cells = cellReadGeometry.View<R::FixedRows<8>, false>();
        OMP::ForSetupRows(0, ids.RowCount(), [&](index iPoint)
                          {
            const index iCell = ids(iPoint, 0);
            snapshot(iPoint, 0) = iCell;
            const Geom::tPoint point{inputPoints(iPoint, 0), inputPoints(iPoint, 1), inputPoints(iPoint, 2)};
            if (direct)
                for (int d = 0; d < 3; ++d)
                    positions(iPoint, d) = point[d];
            if (_cachedDerivatives)
            {
                Math::SmallMatrix full;
                polynomial.Reconstruction(full, point, geom.GetCellCent(iCell), geom.cells(iCell, 7), moments[iCell], cacheOrder);
                Eigen::Map<Eigen::MatrixXd>(cached[iPoint].data(), _cachedDerivatives, _modes) = full;
            } });
    }
    void PolynomialSamplingPlan::Validate() const
    {
        DNDS_check_throw_info(_mesh && _stamp.Valid(), "polynomial sampling geometry or numbering changed");
        _cellSpace.Validate();
    }
    void PolynomialSamplingPlan::ValidateField(const FVField &field, index width) const
    {
        Validate();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_space) && index(field.Width()) == width && field.Device() == Device(),
                              "sampling field shape, row space or placement mismatch");
    }
    PolynomialSamplingPlan PolynomialSamplingPlan::Placed(R::Device device, R::ExecutionContext context) const
    {
        Validate();
        if (device == Device())
            return *this;
        auto result = *this;
        result._cells = _cells.To(device, context);
        result._points = _points.To(device, context);
        result._basis = _basis.To(device, context);
        result._moments = _moments.To(device, context);
        result._cellGeometry = _cellGeometry.To(device, context);
        result._empty = _empty.To(device, context);
        result.Validate();
        return result;
    }
}
