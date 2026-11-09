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
        auto rotationRead = _data->rotations.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto offset = offsets.View<R::FixedRows<1>>(), neighbor = neighbors.View<R::FixedRows<1>>(), bnd = boundaryIds.View<R::FixedRows<1>>();
        const auto w = weights.View<R::FixedRows<3>>();
        const auto bv = boundary.View<R::TableRows>();
        const auto output = outWrite.View<R::UniformTableRows>();
        const auto apply = [&](index iCell)
        {
            std::fill(output[iCell].begin(), output[iCell].end(), 0);
            for (index entry = offset(iCell, 0); entry < offset(iCell + 1, 0); ++entry)
                for (index component = 0; component < components; ++component)
                {
                    real value = neighbor(entry, 0) == UnInitIndex ? bv(bnd(entry, 0), component) : u(neighbor(entry, 0), component);
                    if (neighbor(entry, 0) != UnInitIndex)
                        for (auto first : periodicVectorStarts)
                            if (component >= first && component < first + 3)
                            {
                                const Eigen::Map<const Geom::tGPoint> rotation(rotations[entry].data());
                                value = 0;
                                for (int d = 0; d < 3; ++d)
                                    value += rotation(component - first, d) * u(neighbor(entry, 0), first + d);
                            }
                    const real difference = value - u(iCell, component);
                    for (index d = 0; d < dim; ++d)
                        output(iCell, component * dim + d) += w(entry, d) * difference;
                }
        };
        if (region == ReconstructionRegion::All)
            for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
                apply(iCell);
        else
            for (auto iCell : region == ReconstructionRegion::Core ? _data->partition.Core() : _data->partition.HaloDependent())
                apply(iCell);
    }
}
