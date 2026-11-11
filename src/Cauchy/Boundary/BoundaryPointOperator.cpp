#include "Cauchy/Boundary/BoundaryPointOperator.hpp"

#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"

#include <cmath>

namespace DNDS::Cauchy
{
    BoundaryPointOperator::BoundaryPointOperator(EquationSystem equation)
        : _equation(std::move(equation))
    {
        for (const auto &block : _equation.Schema().blocks)
        {
            if (block.name == "flow")
                _flowFirst = _components;
            if (block.name == "species")
                _speciesFirst = _components;
            if (block.name == "rans")
                _ransFirst = _components;
            _components += index(block.components.size());
        }
        DNDS_check_throw_info(_flowFirst >= 0 &&
                                  _flowFirst <= _components - 5,
                              "boundary point operator requires flow5");
        if (_equation.HasMixture())
        {
            _independentSpecies = _equation.Mixture().SpeciesNumber() - 1;
            if (_independentSpecies)
                DNDS_check_throw_info(
                    _speciesFirst >= 0 &&
                        _speciesFirst <= _components - _independentSpecies,
                    "mixture boundary point operator lacks its species block");
        }
        if (_equation.HasRANS())
            DNDS_check_throw_info(
                _ransFirst >= 0 &&
                    _ransFirst <=
                        _components - _equation.RANS().ComponentNumber(),
                "RANS boundary point operator lacks its model block");
    }

    void BoundaryPointOperator::Evaluate(
        std::vector<real> &point, const BoundaryCondition &condition,
        index interfaceRow, const Vector3 &normal, const Vector3 &location,
        EquationPointEvaluator &evaluator, const RigidFrame &frame, real time,
        BoundaryPointStateKind stateKind,
        OptionalRef<const std::vector<real>> mean, real meanWallDistance,
        InterfaceBoundaryRows interfaceValues) const
    {
        DNDS_check_throw_info(index(point.size()) == _components,
                              "boundary point scratch width differs");
        if (condition.kind == BoundaryKind::Interface)
        {
            DNDS_check_throw_info(
                interfaceValues.data && interfaceRow >= 0 &&
                    interfaceRow < interfaceValues.RowCount() &&
                    interfaceValues.Width() == _components,
                "interface boundary needs one complete logical equation row");
            for (index j = 0; j < _components; ++j)
                point[size_t(j)] = interfaceValues(interfaceRow, j);
            return;
        }

        const int ransComponents = _equation.HasRANS()
                                       ? _equation.RANS().ComponentNumber()
                                       : 0;
        const bool twoEquationWall =
            ransComponents == 2 && condition.kind == BoundaryKind::NoSlipWall;
        DNDS_check_throw_info(
            !twoEquationWall ||
                (mean && index(mean->size()) == _components &&
                 std::isfinite(meanWallDistance) && meanWallDistance > 0),
            "two-equation wall needs cell mean and positive mean wall distance");

        FlowState inside{};
        for (int j = 0; j < 5; ++j)
            inside[j] = point[size_t(_flowFirst + j)];
        const Physics::Chemistry::SpeciesBufferView rhoY{
            _independentSpecies ? point.data() + _speciesFirst : nullptr,
            _independentSpecies};
        const bool exterior = stateKind == BoundaryPointStateKind::ExteriorState;
        const auto flow = exterior
                              ? condition.Exterior(evaluator, inside, rhoY, normal,
                                                   location, time, &frame)
                              : condition.FaceValue(evaluator, inside, rhoY, normal,
                                                    location, time, &frame);
        for (int j = 0; j < 5; ++j)
            point[size_t(_flowFirst + j)] = flow[j];

        if (!ransComponents)
            return;
        if (condition.kind == BoundaryKind::NoSlipWall)
        {
            std::array<real, 2> target{};
            if (twoEquationWall)
            {
                FlowState viscosityFlow{};
                for (int j = 0; j < 5; ++j)
                    viscosityFlow[j] = (*mean)[size_t(_flowFirst + j)];
                Physics::Chemistry::ConstSpeciesBufferView viscositySpecies{
                    _independentSpecies
                        ? mean->data() + _speciesFirst
                        : nullptr,
                    _independentSpecies};
                if (condition.isothermal)
                {
                    viscosityFlow = flow;
                    viscositySpecies = {rhoY.data, rhoY.nSpecies};
                }
                const auto properties = evaluator.Evaluate(
                    viscosityFlow, viscositySpecies, 0, {true});
                Physics::RANS::Point meanPoint;
                for (int j = 0; j < 5; ++j)
                    meanPoint.state[j] = (*mean)[size_t(_flowFirst + j)];
                for (int j = 0; j < ransComponents; ++j)
                    meanPoint.state[5 + j] = (*mean)[size_t(_ransFirst + j)];
                meanPoint.molecularViscosity = properties.transport.viscosity;
                meanPoint.wallDistance = meanWallDistance;
                target = RANSWallTarget(_equation.RANS(), meanPoint, inside[0]);
            }
            for (int j = 0; j < ransComponents; ++j)
            {
                real reflected = point[size_t(_ransFirst + j)];
                if (condition.isothermal && (ransComponents == 1 || j == 0))
                    reflected *= flow[0] / inside[0];
                point[size_t(_ransFirst + j)] =
                    exterior ? 2 * target[j] - reflected : target[j];
            }
            return;
        }
        if (condition.kind == BoundaryKind::Farfield ||
            condition.kind == BoundaryKind::Analytic)
        {
            const auto grid = frame.GridVelocity(location);
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += (inside[1 + d] / inside[0] - grid[d]) * normal[d];
            if (condition.kind == BoundaryKind::Analytic || speed <= 0)
                for (int j = 0; j < ransComponents; ++j)
                    point[size_t(_ransFirst + j)] = condition.prescribedRANS[j];
        }
    }

    void BoundaryPointOperator::ApproximateExteriorJacobian(
        Eigen::Ref<Eigen::MatrixXd> output,
        const BoundaryCondition &condition, const std::vector<real> &inside,
        const Vector3 &normal, const Vector3 &location,
        const RigidFrame &frame, real time) const
    {
        DNDS_check_throw_info(
            index(inside.size()) == _components &&
                output.rows() == _components && output.cols() == _components,
            "boundary Jacobian state or output width differs");
        output.setZero();
        FlowState flow{};
        for (int j = 0; j < 5; ++j)
            flow[j] = inside[size_t(_flowFirst + j)];
        const auto derivative = condition.ApproximateExteriorJacobian(
            flow, normal, location, time, &frame);
        for (int column = 0; column < 5; ++column)
            for (int row = 0; row < 5; ++row)
                output(_flowFirst + row, _flowFirst + column) =
                    derivative[size_t(row + 5 * column)];
        const bool wall = condition.kind == BoundaryKind::SlipWall ||
                          condition.kind == BoundaryKind::NoSlipWall;
        if (!wall)
            return;
        for (index j = 0; j < _components; ++j)
            if (j < _flowFirst || j >= _flowFirst + 5)
                output(j, j) = 1;
        if (condition.kind == BoundaryKind::NoSlipWall &&
            _equation.HasRANS())
            for (int j = 0; j < _equation.RANS().ComponentNumber(); ++j)
                output(_ransFirst + j, _ransFirst + j) = -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
        if (condition.kind == BoundaryKind::NoSlipWall &&
            _equation.HasRANS() &&
            _equation.RANS().Kind() == Physics::RANS::ModelKind::SA)
            output(_ransFirst, _ransFirst) = 0;
#endif
    }
}
