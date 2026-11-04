#pragma once

#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
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
        explicit PointScratch(index width) : values(size_t(width)), gradient(width, 3) {}
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
    inline void Gather(const std::vector<EquationBlockReadView> &blocks, index iCell, PointScratch &point)
    {
        index first = 0;
        for (const auto &block : blocks)
            for (index j = 0; j < block.ComponentNumber(); ++j)
                point.values[first++] = block(iCell, j);
        point.gradient.setZero();
    }
    inline std::vector<EquationBlockReadView> BlockViews(const EquationDofs &state, const EquationDofs::ReadScope &scope)
    {
        std::vector<EquationBlockReadView> result;
        result.reserve(size_t(state.BlockNumber()));
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            result.push_back(scope.Block(iBlock));
        return result;
    }
    inline void RotatePoint(PointScratch &point, const Geom::tGPoint &rotation, const std::vector<index> &vectorStarts)
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
    inline void BoundaryValue(PointScratch &point, index flowFirst, index speciesFirst, int independent,
                              const FiniteVolumeDetail::Sample &sample, EquationPointEvaluator &evaluator,
                              const RigidFrame &frame, real time, bool exterior)
    {
        const Physics::Chemistry::SpeciesBufferView rhoY{independent ? point.values.data() + speciesFirst : nullptr, independent};
        const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]}, location{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
        const auto flow = exterior ? sample.boundary.Exterior(evaluator, point.Flow(flowFirst), rhoY, normal, location, time, &frame)
                                   : sample.boundary.FaceValue(evaluator, point.Flow(flowFirst), rhoY, normal, location, time, &frame);
        for (int j = 0; j < 5; ++j)
            point.values[flowFirst + j] = flow[j];
    }
}
