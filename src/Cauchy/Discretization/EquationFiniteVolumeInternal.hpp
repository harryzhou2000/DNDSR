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
    inline void AdiabaticBoundary(PointScratch &point, index flowFirst, const FiniteVolumeDetail::Sample &sample,
                                  const RigidFrame &frame, bool exterior)
    {
        if (sample.boundary.kind == BoundaryKind::Extrapolate)
            return;
        const auto wall = sample.boundary.WallVelocityAt({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, &frame);
        const auto flow = point.Flow(flowFirst);
        const real rho = flow[0], factor = exterior ? 2 : 1;
        DNDS_check_throw_info(rho > 0, "wall producer requires positive density");
        Vector3 velocity{flow[1] / rho, flow[2] / rho, flow[3] / rho}, target = velocity;
        if (sample.boundary.kind == BoundaryKind::SlipWall)
        {
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += (velocity[d] - wall[d]) * sample.normal[d];
            for (int d = 0; d < 3; ++d)
                target[d] -= factor * speed * sample.normal[d];
        }
        else
            for (int d = 0; d < 3; ++d)
                target[d] += factor * (wall[d] - velocity[d]);
        real energy = flow[4];
        for (int d = 0; d < 3; ++d)
        {
            point.values[flowFirst + 1 + d] = rho * target[d];
            energy += .5 * rho * (target[d] * target[d] - velocity[d] * velocity[d]);
        }
        // Same full internal energy and composition, independent of EOS.
        point.values[flowFirst + 4] = energy;
    }
}
