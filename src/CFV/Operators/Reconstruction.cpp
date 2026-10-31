#include "CFV/Plans/ReconstructionData.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using Matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    using ConstMap = Eigen::Map<const Matrix>;
    using Map = Eigen::Map<Matrix>;
    namespace
    {
        void CheckArray(const R::Array<real> &array, index rows, index width)
        {
            DNDS_check_throw_info(array.GetDevice() == R::Device::Host() && array.IsValid() &&
                                      !array.Layout().IsVariable() && array.RowCount() == rows && index(array.Layout().Width()) == width,
                                  "FV operand has wrong shape, validity or CPU placement");
        }
        void CheckWorkspace(const Detail::ReconstructionData *data, const ReconstructionWorkspace &work)
        {
            DNDS_check_throw_info(work.plan.get() == data && work.components > 0, "reconstruction workspace belongs to another plan");
            data->metrics.ValidateField(work.coefficients, data->n * work.components);
            data->metrics.ValidateField(work.temporary, data->n * work.components);
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(work.coefficients.Owned().Storage(), work.temporary.Owned().Storage()),
                                  "reconstruction input and temporary storage overlap");
            CheckArray(work.scratch, 6, data->n * work.components);
        }
    }
    const FiniteVolumePlan &VariationalReconstructionPlan::Metrics() const { return _data->metrics; }
    const ReconstructionSettings &VariationalReconstructionPlan::Settings() const { return _data->settings; }
    int VariationalReconstructionPlan::NRecDOF() const { return _data->n; }
    ReconstructionInverseDiagnostics VariationalReconstructionPlan::GetInverseDiagnostics() const
    {
        Validate();
        return _data->inverseDiagnostics;
    }
    HardEigen::SVDInverseDiagnostics VariationalReconstructionPlan::GetCellInverseDiagnostics(index iCell) const
    {
        Validate();
        DNDS_check_throw_info(iCell >= 0 && iCell < Metrics().NumCell(), "inverse diagnostic cell is not owned");
        auto read = _data->inverseConditions.Read();
        const auto values = read.View<Runtime::FixedRows<4>>();
        return {_data->n, Eigen::Index(values(iCell, 0)), values(iCell, 3), values(iCell, 1), values(iCell, 2)};
    }
    void VariationalReconstructionPlan::Validate() const { Metrics().Validate(); }
    ReconstructionWorkspace VariationalReconstructionPlan::BuildWorkspace(index components) const
    {
        Validate();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / _data->n, "invalid reconstruction component count");
        return {Metrics().BuildField(_data->n * components), Metrics().BuildField(_data->n * components),
                R::Array<real>(R::RowLayout::Dense(6, _data->n * components)), components, _data};
    }
    ReconstructionBoundaryValues VariationalReconstructionPlan::BuildBoundaryValues(index components) const
    {
        Validate();
        DNDS_check_throw_info(components > 0, "boundary value component count must be positive");
        return {R::Array<real>(R::RowLayout::Dense(NumBoundaryQuadrature(), components)), _data};
    }
    index VariationalReconstructionPlan::NumBoundaryQuadrature() const { return _data->boundaryPoints.RowCount(); }
    BoundaryQuadraturePoint VariationalReconstructionPlan::GetBoundaryQuadrature(index iPoint) const
    {
        Validate();
        auto indices = _data->boundaryPoints.Read();
        const auto entry = indices.View<R::FixedRows<4>, true>()[iPoint];
        auto geometry = Metrics()._data->faceQuadrature.Read();
        const auto values = geometry.View<R::FixedRows<7>>()[entry[3]];
        return {entry[0], entry[1], int(entry[2]), {values[0], values[1], values[2]}, {values[3], values[4], values[5]}};
    }
    FaceCharacteristicTransform VariationalReconstructionPlan::BuildFaceTransform(index first, index count) const
    {
        Validate();
        DNDS_check_throw_info(first >= 0 && count > 0 && count <= 64, "invalid characteristic component range");
        FaceCharacteristicTransform result{first, count,
                                           R::Array<real>(R::RowLayout::Dense(Metrics().NumFaceProc(), count * count)),
                                           R::Array<real>(R::RowLayout::Dense(Metrics().NumFaceProc(), count * count)), _data};
        for (auto *array : {&result.forward, &result.inverse})
        {
            auto write = array->Write();
            auto view = write.View<R::TableRows>();
            for (index iFace = 0; iFace < view.RowCount(); ++iFace)
                Map(view[iFace].data(), count, count).setIdentity();
        }
        FinalizeFaceTransform(result);
        return result;
    }
    void VariationalReconstructionPlan::FinalizeFaceTransform(FaceCharacteristicTransform &transform) const
    {
        Validate();
        transform.forwardVersion = transform.inverseVersion = 0;
        DNDS_check_throw_info(transform.plan == _data && transform.first >= 0 && transform.count > 0 && transform.count <= 64,
                              "characteristic transform belongs to another plan or has an invalid range");
        const index count = transform.count;
        CheckArray(transform.forward, Metrics().NumFaceProc(), count * count);
        CheckArray(transform.inverse, Metrics().NumFaceProc(), count * count);
        auto forward = transform.forward.Read(), inverse = transform.inverse.Read();
        const auto left = forward.View<R::TableRows>(), right = inverse.View<R::TableRows>();
        Matrix product(count, count);
        for (index iFace = 0; iFace < Metrics().NumFaceProc(); ++iFace)
        {
            const ConstMap L(left[iFace].data(), count, count), R(right[iFace].data(), count, count);
            product.noalias() = L * R;
            DNDS_check_throw_info(L.allFinite() && R.allFinite() && product.isApprox(Matrix::Identity(count, count), 1e-10),
                                  "face characteristic matrices must be finite mutual inverses");
        }
        transform.forwardVersion = transform.forward.Version();
        transform.inverseVersion = transform.inverse.Version();
        transform.forwardIdentity = transform.forward.Identity();
        transform.inverseIdentity = transform.inverse.Identity();
    }
    void VariationalReconstructionPlan::DoReconstructionIterTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                                               const ReconstructionBoundaryValues &boundary, ReconstructionRegion region) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        CheckWorkspace(_data.get(), work);
        const index nv = work.components, n = _data->n;
        Metrics().ValidateField(mean, nv);
        DNDS_check_throw_info(boundary.plan == _data, "boundary quadrature values belong to another plan");
        CheckArray(boundary.values, NumBoundaryQuadrature(), nv);
        // All leases and layout projection happen before the numerical loops.
        Metrics().ValidateField(input, n * nv);
        _data->workPartition->Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "invalid reconstruction region");
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input.Owned().Storage(), work.temporary.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(input.Ghost().Storage(), work.temporary.Owned().Storage()),
                              "reconstruction output aliases input");
        auto meanOwned = mean.Owned().Read();
        std::optional<R::Array<real>::ReadLease> meanGhost, recGhost, recOwned;
        std::optional<R::Array<real>::WriteLease> recWrite;
        if (region != ReconstructionRegion::Core)
        {
            meanGhost.emplace(mean.Ghost().Read());
            recGhost.emplace(input.Ghost().Read());
        }
        if (Settings().SORInstead)
            recWrite.emplace(work.coefficients.Owned().Write());
        else
            recOwned.emplace(input.Owned().Read());
        R::ArrayView<const real, false, R::TableRows> current;
        R::ArrayView<real, false, R::TableRows> mutableCurrent;
        if (recWrite)
        {
            mutableCurrent = recWrite->View<R::TableRows>();
            current = {mutableCurrent.data, mutableCurrent.offsets, mutableCurrent.rows,
                       mutableCurrent.width, mutableCurrent.stride, mutableCurrent.sizes};
        }
        else
            current = recOwned->View<R::TableRows>();
        auto nextWrite = work.temporary.Owned().Write();
        auto next = nextWrite.View<R::TableRows>();
        R::PairReadView<real, false, R::TableRows> u{meanOwned.View<R::TableRows>(), meanGhost ? meanGhost->View<R::TableRows>() : R::ArrayView<const real, false, R::TableRows>{}};
        const auto ghost = recGhost ? recGhost->View<R::TableRows>() : R::ArrayView<const real, false, R::TableRows>{};
        auto bRead = boundary.values.Read();
        const auto bValues = bRead.View<R::TableRows>();
        auto oRead = _data->offsets.Read(), nbRead = _data->neighbors.Read(), boRead = _data->boundaryOffsets.Read();
        auto mRead = _data->matrixAInvB.Read(), vRead = _data->vectorAInvB.Read(), injectionRead = _data->boundaryInjection.Read();
        const auto offsets = oRead.View<R::FixedRows<1>>(), neighbors = nbRead.View<R::FixedRows<1>>(), bOffsets = boRead.View<R::FixedRows<1>>();
        const auto matrices = mRead.View<R::TableRows>(), vectors = vRead.View<R::TableRows>(), injection = injectionRead.View<R::TableRows>();
        const real relax = _data->settings.jacobiRelax;
        auto compute = [&](index iCell)
        {
            Map output(next[iCell].data(), n, nv);
            output = ConstMap(current[iCell].data(), n, nv) * (1 - relax);
            for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
            {
                const index other = neighbors(iEntry, 0);
                if (other != UnInitIndex)
                {
                    const real *otherCoefficients = other < Metrics().NumCell() ? current[other].data() : ghost[other - Metrics().NumCell()].data();
                    output.noalias() += relax * ConstMap(matrices[iEntry].data(), n, n) * ConstMap(otherCoefficients, n, nv);
                    for (index variable = 0; variable < nv; ++variable)
                        output.col(variable).noalias() += Eigen::Map<const Eigen::VectorXd>(vectors[iEntry].data(), n) * (relax * (u(other, variable) - u(iCell, variable)));
                }
                else
                    for (index iQ = bOffsets(iEntry, 0); iQ < bOffsets(iEntry + 1, 0); ++iQ)
                        for (index variable = 0; variable < nv; ++variable)
                            output.col(variable).noalias() += Eigen::Map<const Eigen::VectorXd>(injection[iQ].data(), n) * (relax * (bValues(iQ, variable) - u(iCell, variable)));
            }
            if (_data->settings.SORInstead)
                Map(mutableCurrent[iCell].data(), n, nv) = output;
        };
        if (region == ReconstructionRegion::Core || (region == ReconstructionRegion::All && Settings().coreFirst))
            for (auto iCell : _data->workPartition->Core())
                compute(iCell);
        if (region == ReconstructionRegion::HaloDependent || (region == ReconstructionRegion::All && Settings().coreFirst))
            for (auto iCell : _data->workPartition->HaloDependent())
                compute(iCell);
        if (region == ReconstructionRegion::All && !Settings().coreFirst)
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                compute(iCell);
    }
    void VariationalReconstructionPlan::DoReconstructionIter(const FVField &mean, ReconstructionWorkspace &work,
                                                             const ReconstructionBoundaryValues &boundary) const
    {
        DoReconstructionIterTo(mean, work.coefficients, work, boundary);
        if (!Settings().SORInstead)
            CommitReconstruction(work);
    }
    void VariationalReconstructionPlan::CommitReconstruction(ReconstructionWorkspace &work) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        CheckWorkspace(_data.get(), work);
        auto source = work.temporary.Owned().Read();
        auto destination = work.coefficients.Owned().Write();
        const auto src = source.View<R::TableRows>();
        const auto dst = destination.View<R::TableRows>();
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            std::copy(src[iCell].begin(), src[iCell].end(), dst[iCell].begin());
    }
    void VariationalReconstructionPlan::DoLimiterWBAP_3OrderTo(ReconstructionWorkspace &work, int order,
                                                               const FaceCharacteristicTransform *transform, ReconstructionRegion region,
                                                               const R::Array<real> *indicator, real threshold) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        CheckWorkspace(_data.get(), work);
        DNDS_check_throw_info(order >= 1 && order <= Settings().maxOrder, "invalid WBAP polynomial order");
        const index n = NRecDOF(), nv = work.components;
        _data->workPartition->Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "invalid limiter region");
        DNDS_check_throw_info(std::isfinite(threshold), "limiter threshold must be finite");
        std::optional<R::Array<real>::ReadLease> indicatorRead;
        R::ArrayView<const real, false, R::FixedRows<1>> activation;
        if (indicator)
        {
            CheckArray(*indicator, Metrics().NumCell(), 1);
            indicatorRead.emplace(indicator->Read());
            activation = indicatorRead->View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                DNDS_check_throw_info(std::isfinite(activation(iCell, 0)), "limiter indicator must be finite");
        }
        if (transform)
        {
            DNDS_check_throw_info(transform->plan == _data && transform->first >= 0 && transform->count > 0 &&
                                      transform->first <= nv && transform->count <= nv - transform->first,
                                  "invalid face characteristic transform range/plan");
            CheckArray(transform->forward, Metrics().NumFaceProc(), transform->count * transform->count);
            CheckArray(transform->inverse, Metrics().NumFaceProc(), transform->count * transform->count);
            DNDS_check_throw_info(transform->forwardVersion == transform->forward.Version() &&
                                      transform->inverseVersion == transform->inverse.Version() && transform->forwardVersion != 0,
                                  "face characteristic matrices changed; finalize them before limiting");
            DNDS_check_throw_info(transform->forwardIdentity == transform->forward.Identity() && transform->inverseIdentity == transform->inverse.Identity(),
                                  "face characteristic buffers replaced; finalize them before limiting");
        }
        const int begin = order == 1 ? 0 : Math::PolynomialSize(Metrics().Dimension(), order - 1) - 1;
        const int end = Math::PolynomialSize(Metrics().Dimension(), order) - 1, modes = end - begin;
        {
            auto owned = work.coefficients.Owned().Read();
            std::optional<R::Array<real>::ReadLease> ghostRead;
            if (region != ReconstructionRegion::Core)
                ghostRead.emplace(work.coefficients.Ghost().Read());
            R::PairReadView<real, false, R::TableRows> rec{owned.View<R::TableRows>(), ghostRead ? ghostRead->View<R::TableRows>() : R::ArrayView<const real, false, R::TableRows>{}};
            auto outputWrite = work.temporary.Owned().Write(), scratchWrite = work.scratch.Write();
            const auto output = outputWrite.View<R::TableRows>(), scratch = scratchWrite.View<R::TableRows>();
            auto offsetRead = _data->offsets.Read(), neighborRead = _data->neighbors.Read(), faceRead = _data->faces.Read();
            auto secondaryRead = _data->secondary.Read();
            const auto offsets = offsetRead.View<R::FixedRows<1>>(), neighbors = neighborRead.View<R::FixedRows<1>>(), faces = faceRead.View<R::FixedRows<1>>();
            const auto secondary = secondaryRead.View<R::TableRows>();
            std::optional<R::Array<real>::ReadLease> forward, inverse;
            if (transform)
                forward.emplace(transform->forward.Read()), inverse.emplace(transform->inverse.Read());
            const auto forwardView = transform ? forward->View<R::TableRows>() : R::ArrayView<const real, false, R::TableRows>{};
            const auto inverseView = transform ? inverse->View<R::TableRows>() : R::ArrayView<const real, false, R::TableRows>{};
            Map center(scratch[0].data(), modes, nv), other(scratch[1].data(), modes, nv), limited(scratch[2].data(), modes, nv);
            Map up(scratch[3].data(), modes, nv), down(scratch[4].data(), modes, nv), valid(scratch[5].data(), modes, nv);
            const real epsilon = std::pow(verySmallReal, .25);
            auto compute = [&](index iCell)
            {
                Map out(output[iCell].data(), n, nv);
                out = ConstMap(rec.Row(iCell), n, nv);
                if (indicator && activation(iCell, 0) < threshold)
                    return;
                up.setConstant(Settings().WBAP_nStd), down.setConstant(Settings().WBAP_nStd), valid.setOnes();
                for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
                {
                    const index neighbor = neighbors(iEntry, 0);
                    if (neighbor == UnInitIndex)
                        continue;
                    center = out.middleRows(begin, modes);
                    other.noalias() = ConstMap(secondary[iEntry].data(), n, n).middleRows(begin, modes) * ConstMap(rec.Row(neighbor), n, nv);
                    const auto iFace = faces(iEntry, 0);
                    if (transform)
                    {
                        const index start = transform->first, count = transform->count;
                        const ConstMap left(forwardView[iFace].data(), count, count);
                        // A reusable output scratch avoids Eigen alias temporaries.
                        limited.leftCols(count).noalias() = center.middleCols(start, count) * left.transpose();
                        center.middleCols(start, count) = limited.leftCols(count);
                        limited.leftCols(count).noalias() = other.middleCols(start, count) * left.transpose();
                        other.middleCols(start, count) = limited.leftCols(count);
                    }
                    for (index v = 0; v < nv; ++v)
                        for (int mode = 0; mode < modes; ++mode)
                            limited(mode, v) = Math::WBAPBiway(center(mode, v), other(mode, v));
                    if (transform)
                    {
                        const index start = transform->first, count = transform->count;
                        const ConstMap right(inverseView[iFace].data(), count, count);
                        other.leftCols(count).noalias() = limited.middleCols(start, count) * right.transpose();
                        limited.middleCols(start, count) = other.leftCols(count);
                    }
                    for (index v = 0; v < nv; ++v)
                        for (int mode = 0; mode < modes; ++mode)
                        {
                            const real value = limited(mode, v);
                            const real ratio = out(begin + mode, v) / (std::abs(value) + epsilon) * ((value > 0) - (value < 0));
                            up(mode, v) += ratio * ratio * ratio;
                            down(mode, v) += sqr(sqr(ratio));
                            if (value == 0)
                                valid(mode, v) = 0;
                        }
                }
                for (index v = 0; v < nv; ++v)
                    for (int mode = 0; mode < modes; ++mode)
                        out(begin + mode, v) *= valid(mode, v) * up(mode, v) / down(mode, v);
            };
            if (region == ReconstructionRegion::All)
                for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                    compute(iCell);
            else if (region == ReconstructionRegion::Core)
                for (auto iCell : _data->workPartition->Core())
                    compute(iCell);
            else
                for (auto iCell : _data->workPartition->HaloDependent())
                    compute(iCell);
        }
    }
    void VariationalReconstructionPlan::DoLimiterWBAP_3Order(ReconstructionWorkspace &work, int order,
                                                             const FaceCharacteristicTransform *transform) const
    {
        DoLimiterWBAP_3OrderTo(work, order, transform);
        CommitReconstruction(work);
    }
    void VariationalReconstructionPlan::Evaluate(const FVField &mean, const FVField &coefficients,
                                                 const R::Array<index> &cells, const R::Array<real> &points, R::Array<real> &values) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        const index nv = mean.Width(), n = NRecDOF();
        Metrics().ValidateField(mean, nv), Metrics().ValidateField(coefficients, n * nv);
        CheckArray(points, cells.RowCount(), 3), CheckArray(values, cells.RowCount(), nv);
        DNDS_check_throw_info(cells.GetDevice() == R::Device::Host() && !cells.Layout().IsVariable() && cells.Layout().Width() == 1, "sampling cell index shape");
        auto cellRead = cells.Read();
        const auto ids = cellRead.View<R::FixedRows<1>>();
        for (index i = 0; i < cells.RowCount(); ++i)
            DNDS_check_throw_info(ids(i, 0) >= 0 && ids(i, 0) < Metrics().NumCellProc(), "sampling cell outside prepared closure");
        auto mo = mean.Owned().Read(), mg = mean.Ghost().Read(), co = coefficients.Owned().Read(), cg = coefficients.Ghost().Read();
        R::PairReadView<real, false, R::TableRows> u{mo.View<R::TableRows>(), mg.View<R::TableRows>()}, rec{co.View<R::TableRows>(), cg.View<R::TableRows>()};
        auto pRead = points.Read(), geomRead = Metrics()._data->cell.Read(), momentRead = _data->moments.Read();
        auto output = values.Write();
        const auto p = pRead.View<R::FixedRows<3>>();
        const auto geom = geomRead.View<R::FixedRows<8>>();
        const auto moment = momentRead.View<R::TableRows>();
        const auto out = output.View<R::TableRows>();
        Math::SmallMatrix basis;
        for (index i = 0; i < cells.RowCount(); ++i)
        {
            const auto cell = ids(i, 0);
            Math::PolynomialDerivatives(basis, Metrics().Dimension(), Settings().maxOrder, {p(i, 0), p(i, 1), p(i, 2)},
                                        {geom(cell, 1), geom(cell, 2), geom(cell, 3)}, geom(cell, 7));
            for (index v = 0; v < nv; ++v)
            {
                real value = u(cell, v);
                for (index mode = 0; mode < n; ++mode)
                    value += (basis(0, mode + 1) - moment(cell, mode + 1)) * rec(cell, v * n + mode);
                out(i, v) = value;
            }
        }
    }
}
