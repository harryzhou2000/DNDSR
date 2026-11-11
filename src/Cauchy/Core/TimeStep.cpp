#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    void Domain::ConfigureTimeStep(const TimeSettings &time, int power)
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { time.positivity.Validate(); });
        const auto &control = time.positivity;
        real agreement[]{real(control.enabled), -real(control.enabled), control.relax, -control.relax,
                         control.scale, -control.scale, control.increaseLimit, -control.increaseLimit,
                         real(power), -real(power), real(time.steady), -real(time.steady)};
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{agreement, 12}, mpi.comm);
        for (int j = 0; j < 12; j += 2)
            DNDS_check_throw_info(agreement[j] == -agreement[j + 1], "physical timestep policy differs between domain ranks");
        _timePP = control;
        _timeCompressionPower = power;
    }
    PhysicalStepLimit Domain::LimitPhysicalStep(real requestedDt)
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(std::isfinite(requestedDt) && requestedDt > 0, "invalid requested physical timestep"); });
        real agreement[]{requestedDt, -requestedDt};
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{agreement, 2}, mpi.comm);
        DNDS_check_throw_info(agreement[0] == -agreement[1], "requested physical timestep differs between domain ranks");
        PhysicalStepLimit result{requestedDt};
        if (_steady || !_timePP.enabled)
            return result;
        if (!_timePPWorkspace)
        {
            std::optional<EquationDofs> residual;
            MPI::LocalStage(mpi, [&]
                            { residual.emplace(EquationDofOperations::NewVector(_accepted)); });
            auto work = _fv.BuildMeanWorkspace(_accepted, _execution.ChemicalWorkers());
            _timePPResidual = std::move(residual);
            _timePPWorkspace = std::move(work);
            if (_interfaceInputs)
                _fv.SetBoundaryInputs(*_timePPWorkspace, *_interfaceInputs);
        }
        _fv.EvaluateMean(_accepted, *_timePPResidual, *_timePPWorkspace,
                         _execution.Engine(), _execution.PrimaryStream(), _time);
        result = _fv.LimitPhysicalStep(_accepted, *_timePPResidual,
                                       _admissibility->View().thermal, requestedDt,
                                       _timePP.relax, _timePP.scale, _timeCompressionPower);
        // Last accepted dt, never a rejected/prepared proposal, controls growth.
        // Division avoids overflow in an unnecessary previousDt*limit product.
        if (_lastAcceptedDt > 0 && result.dt / _lastAcceptedDt > _timePP.increaseLimit)
        {
            result.dt = _lastAcceptedDt * _timePP.increaseLimit;
            result.growthLimited = true;
        }
        DNDS_check_throw_info(std::isfinite(result.dt) && result.dt > 0 && std::isfinite(_time + result.dt) && _time + result.dt > _time,
                              "physical timestep PP has no positive representable step");
        return result;
    }
}
