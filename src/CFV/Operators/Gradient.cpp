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

    void GradientPlan::Evaluate(const FVField &mean, const Runtime::Array<real> &boundaryValues, FVField &gradient,
                                const std::vector<index> &periodicVectorStarts) const
    {
        namespace R = Runtime;
        auto geometry = _data->metrics.AcquireRead();
        const index components = mean.Width(), dim = _data->metrics.Dimension();
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        _data->metrics.ValidateField(mean, components);
        _data->metrics.ValidateField(gradient, dim * components);
        DNDS_check_throw_info(boundaryValues.GetDevice() == R::Device::Host() && boundaryValues.Layout().Kind() == R::RowLayoutKind::Table &&
                                  boundaryValues.RowCount() == NumBoundaryQuadrature() && boundaryValues.Layout().Width() == components,
                              "gradient boundary value shape or placement differs");
        auto owned = mean.Owned().Read(), ghost = mean.Ghost().Read(), boundary = boundaryValues.Read();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
        auto outWrite = gradient.Owned().Write();
        auto offsets = _data->offsets.Read(), neighbors = _data->neighbors.Read(), boundaryIds = _data->boundaryIds.Read();
        auto weights = _data->weights.Read();
        auto rotationRead = _data->rotations.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto offset = offsets.View<R::FixedRows<1>>(), neighbor = neighbors.View<R::FixedRows<1>>(), bnd = boundaryIds.View<R::FixedRows<1>>();
        const auto w = weights.View<R::FixedRows<3>>();
        const auto bv = boundary.View<R::TableRows>();
        const auto output = outWrite.View<R::UniformTableRows>();
        for (index iCell = 0; iCell < _data->metrics.NumCell(); ++iCell)
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
        }
    }
}
