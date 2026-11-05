#pragma once

#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
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
    /// Explicit inputs for the RANS part of a boundary point. Two-equation
    /// no-slip targets use the cell mean and its mean wall distance, not the
    /// reconstructed point or the face distance. The pointers are borrowed for
    /// this call only. An evaluator with transport reserved is required for
    /// two-equation walls. SA requires neither an extra property call nor mean.
    struct RANSBoundaryInput
    {
        const Physics::RANS::Model *model = nullptr;
        index first = -1;
        const PointScratch *mean = nullptr;
        real meanWallDistance = 0;
    };
    inline void BoundaryValue(PointScratch &point, index flowFirst, index speciesFirst, int independent,
                              const FiniteVolumeDetail::Sample &sample, EquationPointEvaluator &evaluator,
                              const RigidFrame &frame, real time, bool exterior, RANSBoundaryInput rans = {})
    {
        const int components = rans.model ? rans.model->ComponentNumber() : 0;
        DNDS_check_throw_info((rans.first >= 0) == bool(components), "boundary RANS block/model differs");
        const bool twoEquationWall = components == 2 && sample.boundary.kind == BoundaryKind::NoSlipWall;
        DNDS_check_throw_info(!twoEquationWall || (rans.mean && std::isfinite(rans.meanWallDistance) && rans.meanWallDistance > 0),
                              "two-equation wall needs cell mean and positive mean wall distance");
        const Physics::Chemistry::SpeciesBufferView rhoY{independent ? point.values.data() + speciesFirst : nullptr, independent};
        const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]}, location{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
        const auto inside = point.Flow(flowFirst);
        const auto flow = exterior ? sample.boundary.Exterior(evaluator, point.Flow(flowFirst), rhoY, normal, location, time, &frame)
                                   : sample.boundary.FaceValue(evaluator, point.Flow(flowFirst), rhoY, normal, location, time, &frame);
        for (int j = 0; j < 5; ++j)
            point.values[flowFirst + j] = flow[j];
        if (components)
        {
            if (sample.boundary.kind == BoundaryKind::NoSlipWall)
            {
                std::array<real, 2> target{};
                if (twoEquationWall)
                {
                    // Euler uses the wall-temperature state for isothermal
                    // viscosity, and the cell mean for an adiabatic wall.
                    FlowState viscosityFlow = rans.mean->Flow(flowFirst);
                    auto viscositySpecies = rans.mean->Species(speciesFirst, independent);
                    if (sample.boundary.isothermal)
                    {
                        viscosityFlow = flow;
                        viscositySpecies = {rhoY.data, rhoY.nSpecies};
                    }
                    const auto properties = evaluator.Evaluate(viscosityFlow, viscositySpecies, 0, {true});
                    Physics::RANS::Point mean;
                    for (int j = 0; j < 5; ++j)
                        mean.state[j] = rans.mean->values[flowFirst + j];
                    for (int j = 0; j < components; ++j)
                        mean.state[5 + j] = rans.mean->values[rans.first + j];
                    mean.molecularViscosity = properties.transport.viscosity;
                    mean.wallDistance = rans.meanWallDistance;
                    target = RANSWallTarget(*rans.model, mean, inside[0]);
                }
                for (int j = 0; j < components; ++j)
                {
                    // Euler's thermal primitive conversion rescales SA and k
                    // with density. Its prescribed omega/epsilon formula then
                    // reflects the original interior conservative value.
                    real reflected = point.values[rans.first + j];
                    if (sample.boundary.isothermal && (components == 1 || j == 0))
                        reflected *= flow[0] / inside[0];
                    point.values[rans.first + j] = exterior ? 2 * target[j] - reflected : target[j];
                }
            }
            else if (sample.boundary.kind == BoundaryKind::Farfield || sample.boundary.kind == BoundaryKind::Analytic)
            {
                const auto grid = frame.GridVelocity(location);
                real speed = 0;
                for (int d = 0; d < 3; ++d)
                    speed += (inside[1 + d] / inside[0] - grid[d]) * normal[d];
                if (sample.boundary.kind == BoundaryKind::Analytic || speed <= 0)
                    for (int j = 0; j < components; ++j)
                        point.values[rans.first + j] = sample.boundary.prescribedRANS[j];
            }
        }
    }
    /// cellLength is explicit: source consumers require the prepared mesh
    /// length; transport/diffusion do not use it and may pass zero.
    inline Physics::RANS::Point RANSPoint(const PointScratch &point, index flowFirst, index ransFirst,
                                          const Physics::RANS::Model &model, const EquationPointProperties &properties,
                                          real wallDistance, real cellLength)
    {
        struct Gradient
        {
            const PointScratch &point;
            index first;
            real operator()(int j, int d) const { return point.gradient(first + j, d); }
        };
        const RANSStateView policy{model.Kind(), model.ComponentNumber()};
        return PrepareRANSPoint(policy, point.values.data() + flowFirst, point.values.data() + ransFirst,
                                Gradient{point, flowFirst}, Gradient{point, ransFirst}, properties.transport.viscosity,
                                properties.thermo.cp, properties.thermo.cp / properties.thermo.cv, wallDistance, cellLength);
    }
}
