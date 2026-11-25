#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    void Domain::ConfigureTimeStep(const TimeSettings &time, int power)
    {
        _problem.finiteVolume.Reductions().LocalStage( [&]
                        { time.positivity.Validate(); });
        const auto &control = time.positivity;
        real agreement[]{real(control.enabled), -real(control.enabled), control.relax, -control.relax,
                         control.scale, -control.scale, control.increaseLimit, -control.increaseLimit,
                         real(power), -real(power), real(time.steady), -real(time.steady)};
        _problem.finiteVolume.Reductions().Reduce<Runtime::Reduction::Min>(
            SpanView<real>{agreement, 12});
        for (int j = 0; j < 12; j += 2)
            DNDS_check_throw_info(agreement[j] == -agreement[j + 1], "physical timestep policy differs between domain ranks");
        _timePP = control;
        _timeCompressionPower = power;
    }
    PhysicalStepLimit Domain::LimitPhysicalStep(real requestedDt)
    {
        _problem.finiteVolume.Reductions().LocalStage( [&]
                        { DNDS_check_throw_info(std::isfinite(requestedDt) && requestedDt > 0, "invalid requested physical timestep"); });
        real agreement[]{requestedDt, -requestedDt};
        _problem.finiteVolume.Reductions().Reduce<Runtime::Reduction::Min>(
            SpanView<real>{agreement, 2});
        DNDS_check_throw_info(agreement[0] == -agreement[1], "requested physical timestep differs between domain ranks");
        PhysicalStepLimit result{requestedDt};
        if (_steady || !_timePP.enabled)
            return result;
        if (!_execution.timePPWorkspace)
        {
            std::optional<EquationDofs> residual;
            _problem.finiteVolume.Reductions().LocalStage( [&]
                            { residual.emplace(EquationDofOperations::NewVector(_state.accepted)); });
            auto work = _problem.finiteVolume.BuildMeanWorkspace(_state.accepted, _execution.ChemicalWorkers());
            _execution.timePPResidual = std::move(residual);
            _execution.timePPWorkspace = std::move(work);
            if (_execution.interfaces->active)
                _problem.finiteVolume.SetBoundaryInputs(*_execution.timePPWorkspace, *_execution.interfaces->active);
        }
        _problem.finiteVolume.EvaluateMean(_state.accepted, *_execution.timePPResidual, *_execution.timePPWorkspace,
                                           _execution.Engine(), _execution.ComputeStream(), _state.time);
        _problem.finiteVolume.Reductions().LocalStage( [&]
                        {
            if (!_execution.timeStages)
                _execution.timeStages.emplace(_problem.finiteVolume.Metrics(), _execution.ComputeStream().Context());
            result = _execution.timeStages->LimitPhysicalStep(_execution.Engine(), _execution.ComputeStream(),
                _state.accepted, *_execution.timePPResidual, *_problem.admissibility,
                requestedDt, _timePP.relax, _timePP.scale, _timeCompressionPower); });
        result.minimumAlpha = _problem.finiteVolume.Reductions().Value<Runtime::Reduction::Min>(result.minimumAlpha);
        result.limitedCells = _problem.finiteVolume.Reductions().Value<Runtime::Reduction::Sum>(result.limitedCells);
        if (result.limitedCells)
            result.dt *= result.minimumAlpha;
        // Last accepted dt, never a rejected/prepared proposal, controls growth.
        // Division avoids overflow in an unnecessary previousDt*limit product.
        if (_state.lastAcceptedDt > 0 && result.dt / _state.lastAcceptedDt > _timePP.increaseLimit)
        {
            result.dt = _state.lastAcceptedDt * _timePP.increaseLimit;
            result.growthLimited = true;
        }
        DNDS_check_throw_info(std::isfinite(result.dt) && result.dt > 0 && std::isfinite(_state.time + result.dt) && _state.time + result.dt > _state.time,
                              "physical timestep PP has no positive representable step");
        return result;
    }
}
