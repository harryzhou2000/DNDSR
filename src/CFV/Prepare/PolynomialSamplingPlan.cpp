#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Plans/ReconstructionData.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    PolynomialSamplingPlan::PolynomialSamplingPlan(VariationalReconstructionPlan reconstruction,
                                                   const R::Array<index> &cells, const R::Array<real> &points)
        : _reconstruction(std::move(reconstruction))
    {
        auto geometry = _reconstruction.Metrics().AcquireRead();
        Validate();
        DNDS_check_throw_info(cells.GetDevice() == R::Device::Host() && points.GetDevice() == R::Device::Host() &&
                                  cells.Layout().Kind() == R::RowLayoutKind::Table && cells.Layout().Width() == 1 &&
                                  points.Layout().Kind() == R::RowLayoutKind::Table && points.Layout().Width() == 3 &&
                                  cells.RowCount() == points.RowCount(),
                              "polynomial sampling input shape/placement mismatch");
        const int n = _reconstruction.NRecDOF();
        _cells = R::Array<index>(R::RowLayout::Dense(cells.RowCount(), 1));
        _basis = R::Array<real>(R::RowLayout::Dense(cells.RowCount(), n * (n + 1)));
        auto cellRead = cells.Read();
        auto cellWrite = _cells.Write();
        auto pointRead = points.Read(), momentRead = _reconstruction._data->moments.Read();
        auto basisWrite = _basis.Write();
        const auto ids = cellRead.View<R::FixedRows<1>>();
        const auto snapshot = cellWrite.View<R::FixedRows<1>>();
        const auto point = pointRead.View<R::FixedRows<3>>();
        const auto basis = basisWrite.View<R::TableRows>();
        const auto moment = momentRead.View<R::TableRows>();
        const auto geom = geometry.HostView();
        const Math::TaylorBasis polynomial(_reconstruction.Metrics().Dimension(), _reconstruction.Settings().maxOrder);
        Math::SmallMatrix full;
        for (index iPoint = 0; iPoint < cells.RowCount(); ++iPoint)
        {
            const index iCell = ids(iPoint, 0);
            const Geom::tPoint p{point(iPoint, 0), point(iPoint, 1), point(iPoint, 2)};
            DNDS_check_throw_info(iCell >= 0 && iCell < _reconstruction.Metrics().NumCellProc() && p.allFinite(), "invalid sampling cell or point");
            snapshot(iPoint, 0) = iCell;
            _needsGhost |= iCell >= _reconstruction.Metrics().NumCell();
            polynomial.Derivatives(full, p, geom.GetCellCent(iCell), geom.cells(iCell, 7));
            Eigen::Map<Eigen::MatrixXd> result(basis[iPoint].data(), n + 1, n);
            result = full.rightCols(n);
            for (int mode = 0; mode < n; ++mode)
                result(0, mode) -= moment(iCell, mode + 1);
        }
    }
    void PolynomialSamplingPlan::Evaluate(const FVField &mean, const FVField &coefficients, R::Array<real> &output, int derivative) const
    {
        auto geometry = _reconstruction.Metrics().AcquireRead();
        Validate();
        const index components = mean.Width(), n = _reconstruction.NRecDOF();
        _reconstruction.Metrics().ValidateField(mean, components);
        _reconstruction.Metrics().ValidateField(coefficients, components * n);
        DNDS_check_throw_info(derivative >= 0 && derivative <= n && output.GetDevice() == R::Device::Host() &&
                                  output.Layout().Kind() == R::RowLayoutKind::Table && output.RowCount() == Size() &&
                                  output.Layout().Width() == components,
                              "polynomial sample output shape/placement or derivative mismatch");
        auto meanRead = mean.Owned().Read(), recRead = coefficients.Owned().Read();
        std::optional<R::Array<real>::ReadLease> meanGhost, recGhost;
        if (_needsGhost)
            meanGhost.emplace(mean.Ghost().Read()), recGhost.emplace(coefficients.Ghost().Read());
        R::PairReadView<real, false, R::TableRows> u{meanRead.View<R::TableRows>(), meanGhost ? meanGhost->View<R::TableRows>() : R::ArrayView<const real, false, R::TableRows>{}};
        R::PairReadView<real, false, R::TableRows> rec{recRead.View<R::TableRows>(), recGhost ? recGhost->View<R::TableRows>() : R::ArrayView<const real, false, R::TableRows>{}};
        auto idsRead = _cells.Read();
        auto basisRead = _basis.Read();
        const auto ids = idsRead.View<R::FixedRows<1>>();
        const auto basis = basisRead.View<R::TableRows>();
        auto write = output.Write();
        const auto result = write.View<R::TableRows>();
        for (index iPoint = 0; iPoint < Size(); ++iPoint)
        {
            const index iCell = ids(iPoint, 0);
            const Eigen::Map<const Eigen::MatrixXd> values(basis[iPoint].data(), n + 1, n);
            for (index component = 0; component < components; ++component)
            {
                real value = derivative == 0 ? u(iCell, component) : 0;
                for (index mode = 0; mode < n; ++mode)
                    value += values(derivative, mode) * rec(iCell, component * n + mode);
                result(iPoint, component) = value;
            }
        }
    }
}
