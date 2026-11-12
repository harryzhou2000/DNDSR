#pragma once

#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Physics/RANSPoint.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy::EquationFVDetail
{
    /// Retains only point-local scratch. Logical block views are resolved
    /// once per operator; neither this helper nor the point loop flattens a
    /// distributed state or assumes adjacent physical flow/species storage.
    struct PointScratch
    {
        std::vector<real> values;
        Eigen::Matrix<real, Eigen::Dynamic, 3, Eigen::RowMajor> gradient;
        explicit PointScratch(index width) : values(size_t(width)), gradient(width, 3) { gradient.setZero(); }
        FlowState Flow(index first) const
        {
            FlowState result{};
            for (int j = 0; j < 5; ++j)
                result[j] = values[first + j];
            return result;
        }
        FlowGradient FlowDerivative(index first) const
        {
            FlowGradient result{};
            for (int j = 0; j < 5; ++j)
                for (int d = 0; d < 3; ++d)
                    result[j][d] = gradient(first + j, d);
            return result;
        }
        Physics::Chemistry::ConstSpeciesBufferView Species(index first, int count) const
        {
            return {count ? values.data() + first : nullptr, count};
        }
    };
    /// Borrowed outputs of one mean-state preparation, consumed while the
    /// evaluator's point properties are valid. No chemistry views escape this
    /// call. The optional mean snapshot is needed only by evaluated matrices.
    /// Caller sizes the buffers and retains exclusive matrix preparation access.
    struct MeanLinearizationOutput
    {
        index flowFirst, speciesFirst;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> pressureDerivative, pressure;
        std::optional<Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>> means;

        NumericalStatus FillRow(index iCell, const PointScratch &point, const PointProperties &properties) const noexcept
        {
            const int species = properties.fractions.nSpecies;
            if (species && (properties.speciesEnthalpy.nSpecies != species || properties.speciesGasConstant.nSpecies != species))
                return NumericalStatus{NumericalError::InvalidInput}.At(iCell);
            const index nv = index(point.values.size());
            for (index j = 0; j < nv; ++j)
            {
                if (!std::isfinite(point.values[j]))
                    return NumericalStatus{NumericalError::InvalidState}.At(iCell);
                if (means)
                    (*means)(iCell, j) = point.values[j];
            }
            const auto flow = point.Flow(flowFirst);
            pressure(iCell, 0) = properties.thermo.pressure;
            for (index column = 0; column < nv; ++column)
            {
                FlowState direction{};
                if (column >= flowFirst && column < flowFirst + 5)
                    direction[column - flowFirst] = 1;
                struct SpeciesDirection
                {
                    index first, column;
                    real operator[](int k) const { return real(column == first + k); }
                } species{speciesFirst, column};
                pressureDerivative(iCell, column) = PreparePrimitiveDerivative(flow, direction, properties, species).pressure;
                if (!std::isfinite(pressureDerivative(iCell, column)))
                    return NumericalStatus{NumericalError::InvalidState}.At(iCell);
            }
            return {};
        }
    };
    template <class Blocks>
    inline void Gather(const Blocks &blocks, index iCell, PointScratch &point, bool clearGradient = true)
    {
        index first = 0;
        for (const auto &block : blocks)
        {
            const auto row = block[iCell];
            for (index j = 0; j < row.size(); ++j)
                point.values[first++] = row[j];
        }
        if (clearGradient)
            point.gradient.setZero();
    }
    inline void RotatePoint(PointScratch &point, const Geom::tGPoint &rotation, SpanView<const index> vectorStarts)
    {
        // Scalar gradients rotate once, vector gradients on both indices.
        for (index j = 0; j < index(point.values.size()); ++j)
            point.gradient.row(j) = (point.gradient.row(j) * rotation.transpose()).eval();
        for (const auto first : vectorStarts)
        {
            const Geom::tPoint value{point.values[first], point.values[first + 1], point.values[first + 2]};
            const Geom::tPoint rotated = rotation * value;
            for (int d = 0; d < 3; ++d)
                point.values[first + d] = rotated[d];
            point.gradient.middleRows(first, 3) = (rotation * point.gradient.middleRows(first, 3)).eval();
        }
    }
    /// cellLength is explicit: source consumers require the prepared mesh
    /// length; transport/diffusion do not use it and may pass zero.
    inline Physics::RANS::Point RANSPoint(const PointScratch &point, index flowFirst, index ransFirst,
                                          const Physics::RANS::Model &model, const PointProperties &properties,
                                          real wallDistance, real cellLength)
    {
        struct Gradient
        {
            const PointScratch &point;
            index first;
            real operator()(int j, int d) const { return point.gradient(first + j, d); }
        };
        const RANSStateView policy{model.Kind(), model.ComponentNumber()};
        return PrepareRANSPoint(policy.components, point.values.data() + flowFirst, point.values.data() + ransFirst,
                                Gradient{point, flowFirst}, Gradient{point, ransFirst}, properties.transport.viscosity,
                                properties.thermo.cp, properties.thermo.cp / properties.thermo.cv, wallDistance, cellLength);
    }
}
