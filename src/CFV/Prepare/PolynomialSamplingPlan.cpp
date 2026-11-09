#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    PolynomialSamplingPlan::PolynomialSamplingPlan(VariationalReconstructionPlan reconstruction,
                                                   const R::Array<index> &cells, const R::Array<real> &points, bool cacheDerivatives)
        : _reconstruction(std::move(reconstruction)), _cacheDerivatives(cacheDerivatives)
    {
        auto geometry = _reconstruction.Metrics().AcquireRead();
        Validate();
        DNDS_check_throw_info(cells.GetDevice() == R::Device::Host() && points.GetDevice() == R::Device::Host() &&
                                  cells.Layout().Kind() == R::RowLayoutKind::Table && cells.Layout().Width() == 1 &&
                                  points.Layout().Kind() == R::RowLayoutKind::Table && points.Layout().Width() == 3 &&
                                  cells.RowCount() == points.RowCount(),
                              "polynomial sampling input shape/placement mismatch");
        const int n = _reconstruction.NRecDOF();
        const Math::TaylorBasis polynomial(_reconstruction.Metrics().Dimension(), _reconstruction.Settings().maxOrder);
        _cachedDerivatives = _reconstruction.Settings().cacheDiffBase
                                 ? polynomial.DerivativeNumber(_cacheDerivatives ? _reconstruction.Settings().cacheDiffBaseOrder : 0)
                                 : 0;
        _cells = R::Array<index>(R::RowLayout::Dense(cells.RowCount(), 1));
        _points = R::Array<real>(R::RowLayout::Dense(cells.RowCount(), 3));
        if (_cachedDerivatives)
            _basis = R::Array<real>(R::RowLayout::Dense(cells.RowCount(), n * _cachedDerivatives));
        auto cellRead = cells.Read();
        auto cellWrite = _cells.Write();
        auto pointRead = points.Read(), momentRead = _reconstruction._data->moments.Read();
        auto snapshotWrite = _points.Write();
        std::optional<R::Array<real>::WriteLease> basisWrite;
        if (_cachedDerivatives)
            basisWrite.emplace(_basis.Write());
        const auto ids = cellRead.View<R::FixedRows<1>>();
        const auto snapshot = cellWrite.View<R::FixedRows<1>>();
        const auto point = pointRead.View<R::FixedRows<3>>();
        const auto positions = snapshotWrite.View<R::FixedRows<3>>();
        const auto moment = momentRead.View<R::TableRows>();
        const auto geom = geometry.HostView();
        Math::SmallMatrix full;
        for (index iPoint = 0; iPoint < ids.RowCount(); ++iPoint)
        {
            const index iCell = ids(iPoint, 0);
            const Geom::tPoint p{point(iPoint, 0), point(iPoint, 1), point(iPoint, 2)};
            DNDS_check_throw_info(iCell >= 0 && iCell < _reconstruction.Metrics().NumCellProc() && p.allFinite(), "invalid sampling cell or point");
            snapshot(iPoint, 0) = iCell;
            for (int d = 0; d < 3; ++d)
                positions(iPoint, d) = p[d];
            _needsGhost |= iCell >= _reconstruction.Metrics().NumCell();
            if (basisWrite)
            {
                polynomial.Reconstruction(full, p, geom.GetCellCent(iCell), geom.cells(iCell, 7), moment[iCell],
                                          _cacheDerivatives ? _reconstruction.Settings().cacheDiffBaseOrder : 0);
                Eigen::Map<Eigen::MatrixXd>(basisWrite->View<R::TableRows>()[iPoint].data(), _cachedDerivatives, n) = full;
            }
        }
    }
    void PolynomialSamplingPlan::Evaluate(const FVField &mean, const FVField &coefficients, R::Array<real> &output, int derivative) const
    {
        auto geometry = _reconstruction.Metrics().AcquireRead();
        Validate();
        const index components = mean.Width(), n = _reconstruction.NRecDOF();
        _reconstruction.Metrics().ValidateField(mean, components);
        _reconstruction.Metrics().ValidateField(coefficients, components * n);
        DNDS_check_throw_info(derivative >= 0 && derivative <= (_cacheDerivatives ? n : 0) && output.GetDevice() == R::Device::Host() &&
                                  !output.Layout().IsVariable() && output.RowCount() == Size() &&
                                  output.Layout().Width() == components,
                              "polynomial sample output shape/placement or derivative mismatch");
        auto meanRead = mean.Owned().Read(), recRead = coefficients.Owned().Read();
        std::optional<R::Array<real>::ReadLease> meanGhost, recGhost;
        if (_needsGhost)
            meanGhost.emplace(mean.Ghost().Read()), recGhost.emplace(coefficients.Ghost().Read());
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{meanRead.View<R::UniformTableRows>(), meanGhost ? meanGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> rec{recRead.View<R::UniformTableRows>(), recGhost ? recGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        auto idsRead = _cells.Read();
        std::optional<R::Array<real>::ReadLease> basisRead;
        if (_cachedDerivatives)
            basisRead.emplace(_basis.Read());
        const auto cachedBasis = basisRead ? basisRead->View<R::TableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>{};
        auto pointRead = _points.Read(), momentRead = _reconstruction._data->moments.Read();
        const auto points = pointRead.View<R::FixedRows<3>>();
        const auto moments = momentRead.View<R::TableRows>();
        const auto geom = geometry.HostView();
        const Math::TaylorBasis polynomial(_reconstruction.Metrics().Dimension(), _reconstruction.Settings().maxOrder);
        Math::SmallMatrix scratch;
        const auto ids = idsRead.View<R::FixedRows<1>>();
        auto write = output.Write();
        const auto result = write.View<R::UniformTableRows>();
        for (index iPoint = 0; iPoint < ids.RowCount(); ++iPoint)
        {
            const index iCell = ids(iPoint, 0);
            const auto meanRow = u[iCell], coefficientRow = rec[iCell];
            const auto outputRow = result[iPoint];
            Math::ReconstructionBasisView values;
            if (basisRead && derivative < _cachedDerivatives)
                values = {cachedBasis[iPoint].data(), _cachedDerivatives, int(n), _cachedDerivatives};
            else
            {
                polynomial.Reconstruction(scratch, {points(iPoint, 0), points(iPoint, 1), points(iPoint, 2)},
                                          geom.GetCellCent(iCell), geom.cells(iCell, 7), moments[iCell],
                                          _reconstruction.Settings().maxOrder);
                values = {scratch.data(), int(scratch.rows()), int(n), int(scratch.rows())};
            }
            for (index component = 0; component < components; ++component)
            {
                real value = derivative == 0 ? meanRow[component] : 0;
                for (index mode = 0; mode < n; ++mode)
                    value += values(derivative, mode) * coefficientRow[component * n + mode];
                outputRow[component] = value;
            }
        }
    }
}
