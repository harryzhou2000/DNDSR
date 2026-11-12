#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Math/PolynomialReconstruction.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::CFV
{
    index GradientPlan::NumBoundaryQuadrature() const
    {
        return index(_data->boundary.size());
    }

    BoundaryQuadraturePoint GradientPlan::GetBoundaryQuadrature(index iPoint) const
    {
        _data->metrics.Validate();
        DNDS_check_throw_info(iPoint >= 0 && iPoint < NumBoundaryQuadrature(), "gradient boundary point out of range");
        return _data->boundary[iPoint];
    }

    Runtime::Array<real> GradientPlan::BuildBoundaryValues(index components) const
    {
        DNDS_check_throw_info(components > 0, "gradient boundary components must be positive");
        return Runtime::Array<real>(Runtime::RowLayout::Dense(NumBoundaryQuadrature(), components));
    }

    GradientFaceCharacteristicTransform GradientPlan::BuildFaceTransform(index first, index count) const
    {
        namespace R = Runtime;
        _data->metrics.Validate();
        DNDS_check_throw_info(first >= 0 && count >= 0 && count <= 64,
                              "invalid gradient characteristic component range");
        if (count == 0)
            return {first, 0, {}, {}, _data->identity};
        GradientFaceCharacteristicTransform result{
            first,
            count,
            R::Array<real>(R::RowLayout::Dense(_data->metrics.NumFaceProc(), count * count)),
            R::Array<real>(R::RowLayout::Dense(_data->metrics.NumFaceProc(), count * count)),
            _data->identity};
        for (auto *array : {&result.forward, &result.inverse})
        {
            auto write = array->Write();
            const auto view = write.View<R::TableRows>();
            for (index iFace = 0; iFace < view.RowCount(); ++iFace)
                Eigen::Map<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>>(view[iFace].data(), count, count).setIdentity();
        }
        FinalizeFaceTransform(result);
        return result;
    }

    void GradientPlan::FinalizeFaceTransform(GradientFaceCharacteristicTransform &transform) const
    {
        namespace R = Runtime;
        _data->metrics.Validate();
        transform.forwardVersion = transform.inverseVersion = 0;
        transform.forwardIdentity = transform.inverseIdentity = 0;
        DNDS_check_throw_info(transform.plan == _data->identity && transform.first >= 0 &&
                                  transform.count >= 0 && transform.count <= 64,
                              "gradient characteristic transform belongs to another plan or has an invalid range");
        const index count = transform.count;
        if (count == 0)
            return;
        const auto valid = [&](const R::Array<real> &array)
        {
            return array.GetDevice() == R::Device::Host() && array.Layout().Kind() == R::RowLayoutKind::Table &&
                   array.RowCount() == _data->metrics.NumFaceProc() && array.Layout().Width() == count * count;
        };
        DNDS_check_throw_info(valid(transform.forward) && valid(transform.inverse),
                              "gradient characteristic matrix shape or placement differs");
        const auto forward = transform.forward.Read(), inverse = transform.inverse.Read();
        const auto left = forward.View<R::TableRows>(), right = inverse.View<R::TableRows>();
        Eigen::MatrixXd product(count, count);
        for (index iFace = 0; iFace < _data->metrics.NumFaceProc(); ++iFace)
        {
            const Eigen::Map<const Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> L(left[iFace].data(), count, count);
            const Eigen::Map<const Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> Rm(right[iFace].data(), count, count);
            product.noalias() = L * Rm;
            DNDS_check_throw_info(L.allFinite() && Rm.allFinite() &&
                                      product.isApprox(Eigen::MatrixXd::Identity(count, count), 1e-10),
                                  "gradient characteristic matrices must be finite mutual inverses");
        }
        transform.forwardVersion = transform.forward.Version();
        transform.inverseVersion = transform.inverse.Version();
        transform.forwardIdentity = transform.forward.Identity();
        transform.inverseIdentity = transform.inverse.Identity();
    }

    void GradientPlan::ConvertToCoefficients(const FVField &gradient, const VariationalReconstructionPlan &reconstruction,
                                             FVField &coefficients) const
    {
        const index dim = _data->metrics.Dimension();
        DNDS_check_throw_info(gradient.Width() % dim == 0 && gradient.Width() > 0 &&
                                  reconstruction.Metrics().Cells().SameOrder(_data->metrics.Cells()),
                              "gradient conversion has wrong component count or mesh order");
        reconstruction.Validate();
        const index components = gradient.Width() / dim, n = reconstruction.NRecDOF();
        _data->metrics.ValidateField(gradient, dim * components);
        _data->metrics.ValidateField(coefficients, n * components);
        auto geometry = _data->metrics.AcquireRead();
        auto input = gradient.Owned().Read();
        auto output = coefficients.Owned().Write();
        const auto values = input.View<Runtime::UniformTableRows>();
        const auto result = output.View<Runtime::UniformTableRows>();
        const auto cells = geometry.HostView().cells;
        for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
        {
            std::fill(result[iCell].begin(), result[iCell].end(), 0);
            for (index component = 0; component < components; ++component)
                for (index d = 0; d < dim; ++d)
                    result(iCell, component * n + d) = values(iCell, component * dim + d) * cells(iCell, 7);
        }
    }

    void GradientPlan::ExtractLinearGradients(const FVField &coefficients, const VariationalReconstructionPlan &reconstruction,
                                              FVField &gradient, bool ghost) const
    {
        const index dim = _data->metrics.Dimension(), modes = reconstruction.NRecDOF();
        DNDS_check_throw_info(gradient.Width() > 0 && gradient.Width() % dim == 0 && modes >= dim &&
                                  reconstruction.Metrics().Cells().SameOrder(_data->metrics.Cells()),
                              "linear gradient extraction has incompatible shape or mesh");
        const index components = gradient.Width() / dim;
        _data->metrics.ValidateField(gradient, dim * components);
        _data->metrics.ValidateField(coefficients, modes * components);
        const auto &source = ghost ? coefficients.Ghost() : coefficients.Owned();
        auto &destination = ghost ? gradient.Ghost() : gradient.Owned();
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(source.Storage(), destination.Storage()), "linear gradient output aliases coefficients");
        const auto geometry = _data->metrics.AcquireRead();
        const auto cells = geometry.HostView().cells;
        const auto read = source.Read();
        auto write = destination.Write();
        const auto input = read.View<Runtime::UniformTableRows>();
        const auto output = write.View<Runtime::UniformTableRows>();
        const index first = ghost ? _data->metrics.NumCell() : 0;
        for (index iRow = 0; iRow < output.RowCount(); ++iRow)
            for (index j = 0; j < components; ++j)
                for (index d = 0; d < dim; ++d)
                    output(iRow, j * dim + d) = input(iRow, j * modes + d) / cells(first + iRow, 7);
    }

    void GradientPlan::Evaluate(const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient,
                                const std::vector<index> &periodicVectorStarts, ReconstructionRegion region) const
    {
        namespace R = Runtime;
        auto geometry = _data->metrics.AcquireRead();
        _data->partition.Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "unknown gradient work region");
        const index components = mean.Width(), dim = _data->metrics.Dimension();
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        _data->metrics.ValidateField(mean, components);
        _data->metrics.ValidateField(gradient, dim * components);
        DNDS_check_throw_info(boundaryValues.GetDevice() == R::Device::Host() && boundaryValues.Layout().Kind() == R::RowLayoutKind::Table &&
                                  boundaryValues.RowCount() == NumBoundaryQuadrature() && boundaryValues.Layout().Width() == components,
                              "gradient boundary value shape or placement differs");
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(gradient.Owned().Storage(), mean.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(gradient.Owned().Storage(), mean.Ghost().Storage()) &&
                                  !Algebra::Detail::Overlaps(gradient.Owned().Storage(), boundaryValues.Storage()),
                              "gradient output overlaps its inputs");
        auto owned = mean.Owned().Read(), boundary = boundaryValues.Read();
        std::optional<R::Array<real>::ReadLease> ghost;
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{owned.View<R::UniformTableRows>(), {}};
        if (region != ReconstructionRegion::Core)
        {
            ghost.emplace(mean.Ghost().Read());
            u.ghost = ghost->View<R::UniformTableRows>();
        }
        auto outWrite = gradient.Owned().Write();
        auto offsets = _data->offsets.Read(), neighbors = _data->neighbors.Read(), boundaryIds = _data->boundaryIds.Read();
        auto weights = _data->weights.Read();
        auto rotationActiveRead = _data->rotationActive.Read();
        auto rotationRead = _data->rotations.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
        const auto offset = offsets.View<R::FixedRows<1>>(), neighbor = neighbors.View<R::FixedRows<1>>(), bnd = boundaryIds.View<R::FixedRows<1>>();
        const auto w = weights.View<R::FixedRows<3>>();
        const auto bv = boundary.View<R::TableRows>();
        const auto output = outWrite.View<R::UniformTableRows>();
        const auto apply = [&](index iCell)
        {
            std::fill(output[iCell].begin(), output[iCell].end(), 0);
            for (index entry = offset(iCell, 0); entry < offset(iCell + 1, 0); ++entry)
            {
                const auto other = neighbor(entry, 0);
                const bool rotated = other != UnInitIndex && _data->hasRotatedFrames &&
                                     rotationActive(entry, 0);
                for (index component = 0; component < components; ++component)
                {
                    real value = other == UnInitIndex ? bv(bnd(entry, 0), component) : u(other, component);
                    if (rotated)
                        for (auto first : periodicVectorStarts)
                            if (component >= first && component < first + 3)
                            {
                                const Eigen::Map<const Geom::tGPoint> rotation(rotations[entry].data());
                                value = 0;
                                for (int d = 0; d < 3; ++d)
                                    value += rotation(component - first, d) * u(other, first + d);
                            }
                    const real difference = value - u(iCell, component);
                    for (index d = 0; d < dim; ++d)
                        output(iCell, component * dim + d) += w(entry, d) * difference;
                }
            }
        };
        if (region == ReconstructionRegion::All)
            for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
                apply(iCell);
        else
            for (auto iCell : region == ReconstructionRegion::Core ? _data->partition.Core() : _data->partition.HaloDependent())
                apply(iCell);
    }

    void GradientPlan::LimitWBAP3(const FVField &input, FVField &output,
                                  const std::vector<index> &periodicVectorStarts,
                                  GradientFaceCharacteristicTransformRef transform,
                                  ReconstructionRegion region,
                                  LimiterIndicatorRef indicator,
                                  real threshold, real centerWeight) const
    {
        namespace R = Runtime;
        _data->metrics.Validate();
        _data->partition.Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core ||
                                  region == ReconstructionRegion::HaloDependent,
                              "unknown gradient limiter work region");
        DNDS_check_throw_info(std::isfinite(threshold) && std::isfinite(centerWeight) && centerWeight > 0,
                              "invalid gradient limiter control");
        const index dim = _data->metrics.Dimension(), width = input.Width();
        DNDS_check_throw_info(width > 0 && width % dim == 0 && output.Width() == width,
                              "gradient limiter field width differs");
        const index components = width / dim;
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        _data->metrics.ValidateField(input, width);
        _data->metrics.ValidateField(output, width);
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Owned().Storage(), input.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(output.Owned().Storage(), input.Ghost().Storage()),
                              "gradient limiter output overlaps its input");

        std::optional<R::Array<real>::ReadLease> indicatorRead;
        R::ArrayView<const real, false, R::FixedRows<1>> activation;
        if (indicator)
        {
            const auto &array = indicator->get();
            DNDS_check_throw_info(array.GetDevice() == R::Device::Host() &&
                                      array.Layout().Kind() == R::RowLayoutKind::Table &&
                                      array.RowCount() == _data->metrics.NumCell() && array.Layout().Width() == 1,
                                  "gradient limiter indicator shape or placement differs");
            indicatorRead.emplace(array.Read());
            activation = indicatorRead->View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
                DNDS_check_throw_info(std::isfinite(activation(iCell, 0)),
                                      "gradient limiter indicator must be finite");
        }

        const auto *selectedTransform = transform && transform->get().count ? &transform->get() : nullptr;
        if (transform && !selectedTransform)
            DNDS_check_throw_info(transform->get().plan == _data->identity && transform->get().first >= 0 &&
                                      transform->get().first <= components,
                                  "disabled gradient characteristic transform belongs to another plan or has an invalid offset");
        if (selectedTransform)
        {
            DNDS_check_throw_info(selectedTransform->plan == _data->identity && selectedTransform->first >= 0 &&
                                      selectedTransform->count > 0 && selectedTransform->first <= components &&
                                      selectedTransform->count <= components - selectedTransform->first,
                                  "invalid gradient characteristic transform range/plan");
            const auto valid = [&](const R::Array<real> &array)
            {
                return array.GetDevice() == R::Device::Host() && array.Layout().Kind() == R::RowLayoutKind::Table &&
                       array.RowCount() == _data->metrics.NumFaceProc() &&
                       array.Layout().Width() == selectedTransform->count * selectedTransform->count;
            };
            DNDS_check_throw_info(valid(selectedTransform->forward) && valid(selectedTransform->inverse),
                                  "gradient characteristic matrix shape or placement differs");
            DNDS_check_throw_info(selectedTransform->forwardVersion == selectedTransform->forward.Version() &&
                                      selectedTransform->inverseVersion == selectedTransform->inverse.Version() &&
                                      selectedTransform->forwardVersion != 0 &&
                                      selectedTransform->forwardIdentity == selectedTransform->forward.Identity() &&
                                      selectedTransform->inverseIdentity == selectedTransform->inverse.Identity(),
                                  "gradient characteristic matrices changed; finalize them before limiting");
        }

        auto owned = input.Owned().Read();
        std::optional<R::Array<real>::ReadLease> ghostRead;
        if (region != ReconstructionRegion::Core)
            ghostRead.emplace(input.Ghost().Read());
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> values{
            owned.View<R::UniformTableRows>(),
            ghostRead ? ghostRead->View<R::UniformTableRows>()
                      : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        auto outputWrite = output.Owned().Write();
        const auto result = outputWrite.View<R::UniformTableRows>();
        auto offsetRead = _data->offsets.Read(), neighborRead = _data->neighbors.Read(), faceRead = _data->faces.Read();
        auto rotationActiveRead = _data->rotationActive.Read();
        auto rotationRead = _data->rotations.Read();
        const auto offsets = offsetRead.View<R::FixedRows<1>>(), neighbors = neighborRead.View<R::FixedRows<1>>(),
                   faces = faceRead.View<R::FixedRows<1>>();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
        std::optional<R::Array<real>::ReadLease> forwardRead, inverseRead;
        if (selectedTransform)
        {
            forwardRead.emplace(selectedTransform->forward.Read());
            inverseRead.emplace(selectedTransform->inverse.Read());
        }
        const auto forward = selectedTransform ? forwardRead->View<R::TableRows>()
                                               : R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>{};
        const auto inverse = selectedTransform ? inverseRead->View<R::TableRows>()
                                               : R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>{};

        std::vector<real> center(static_cast<size_t>(width));
        std::vector<real> other(static_cast<size_t>(width));
        std::vector<real> limited(static_cast<size_t>(width));
        std::vector<real> up(static_cast<size_t>(width));
        std::vector<real> down(static_cast<size_t>(width));
        std::vector<real> valid(static_cast<size_t>(width));
        std::vector<real> transformedCenter(size_t(selectedTransform ? selectedTransform->count : 0));
        std::vector<real> transformedOther(size_t(selectedTransform ? selectedTransform->count : 0));
        const real epsilon = std::pow(verySmallReal, .25);
        const auto vectorStart = [&](index component)
        {
            for (const auto first : periodicVectorStarts)
                if (component >= first && component < first + 3)
                    return first;
            return UnInitIndex;
        };
        const auto compute = [&](index iCell)
        {
            const auto inputRow = values[iCell];
            std::copy(inputRow.begin(), inputRow.end(), result[iCell].begin());
            if (indicator && activation(iCell, 0) < threshold)
                return;
            for (index j = 0; j < width; ++j)
            {
                center[size_t(j)] = inputRow[j];
                up[size_t(j)] = down[size_t(j)] = centerWeight;
                valid[size_t(j)] = 1;
            }
            index lastFace = UnInitIndex;
            for (index entry = offsets(iCell, 0); entry < offsets(iCell + 1, 0); ++entry)
            {
                const index iFace = faces(entry, 0), neighbor = neighbors(entry, 0);
                if (iFace == lastFace)
                    continue;
                lastFace = iFace;
                if (neighbor == UnInitIndex)
                    continue;
                if (!_data->hasRotatedFrames || !rotationActive(entry, 0))
                    std::copy(values[neighbor].begin(), values[neighbor].end(), other.begin());
                else
                {
                    const Eigen::Map<const Geom::tGPoint> rotation(rotations[entry].data());
                    for (index component = 0; component < components; ++component)
                    {
                        const index first = vectorStart(component);
                        for (index d = 0; d < dim; ++d)
                        {
                            real value = 0;
                            if (first == UnInitIndex)
                                for (index k = 0; k < dim; ++k)
                                    value += rotation(d, k) * values(neighbor, component * dim + k);
                            else
                                for (index v = 0; v < 3; ++v)
                                    for (index k = 0; k < dim; ++k)
                                        value += rotation(component - first, v) *
                                                 values(neighbor, (first + v) * dim + k) * rotation(d, k);
                            other[size_t(component * dim + d)] = value;
                        }
                    }
                }
                if (selectedTransform)
                {
                    const index first = selectedTransform->first, count = selectedTransform->count;
                    const Eigen::Map<const Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>>
                        left(forward[iFace].data(), count, count);
                    for (index d = 0; d < dim; ++d)
                        for (index i = 0; i < count; ++i)
                        {
                            transformedCenter[size_t(i)] = 0;
                            transformedOther[size_t(i)] = 0;
                            for (index j = 0; j < count; ++j)
                            {
                                transformedCenter[size_t(i)] += left(i, j) * center[size_t((first + j) * dim + d)];
                                transformedOther[size_t(i)] += left(i, j) * other[size_t((first + j) * dim + d)];
                            }
                            limited[size_t((first + i) * dim + d)] =
                                Math::WBAPBiway(transformedCenter[size_t(i)], transformedOther[size_t(i)]);
                        }
                }
                for (index component = 0; component < components; ++component)
                {
                    const bool transformed = selectedTransform && component >= selectedTransform->first &&
                                             component < selectedTransform->first + selectedTransform->count;
                    if (!transformed)
                        for (index d = 0; d < dim; ++d)
                            limited[size_t(component * dim + d)] =
                                Math::WBAPBiway(center[size_t(component * dim + d)], other[size_t(component * dim + d)]);
                }
                if (selectedTransform)
                {
                    const index first = selectedTransform->first, count = selectedTransform->count;
                    const Eigen::Map<const Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>>
                        right(inverse[iFace].data(), count, count);
                    for (index d = 0; d < dim; ++d)
                    {
                        for (index i = 0; i < count; ++i)
                        {
                            transformedOther[size_t(i)] = 0;
                            for (index j = 0; j < count; ++j)
                                transformedOther[size_t(i)] += right(i, j) * limited[size_t((first + j) * dim + d)];
                        }
                        for (index i = 0; i < count; ++i)
                            limited[size_t((first + i) * dim + d)] = transformedOther[size_t(i)];
                    }
                }
                for (index j = 0; j < width; ++j)
                {
                    const real value = limited[size_t(j)];
                    const real ratio = center[size_t(j)] / (std::abs(value) + epsilon) * ((value > 0) - (value < 0));
                    up[size_t(j)] += ratio * ratio * ratio;
                    down[size_t(j)] += sqr(sqr(ratio));
                    if (value == 0)
                        valid[size_t(j)] = 0;
                }
            }
            for (index j = 0; j < width; ++j)
                result(iCell, j) = center[size_t(j)] * valid[size_t(j)] * up[size_t(j)] / down[size_t(j)];
        };
        if (region == ReconstructionRegion::All)
            for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
                compute(iCell);
        else
            for (auto iCell : region == ReconstructionRegion::Core ? _data->partition.Core()
                                                                   : _data->partition.HaloDependent())
                compute(iCell);
    }
}
