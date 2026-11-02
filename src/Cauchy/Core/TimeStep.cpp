#include "Cauchy/Core/Domain.hpp"
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
        MPI_Allreduce(MPI_IN_PLACE, agreement, 12, MPI_DOUBLE, MPI_MIN, mpi.comm);
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
        MPI_Allreduce(MPI_IN_PLACE, agreement, 2, MPI_DOUBLE, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(agreement[0] == -agreement[1], "requested physical timestep differs between domain ranks");
        PhysicalStepLimit result{requestedDt};
        if (_steady || !_timePP.enabled)
            return result;
        if (!_timePPWorkspace)
        {
            _timePPResidual.emplace(_fv.Metrics().BuildField(Flow().Owned().Layout()));
            _timePPWorkspace = _fv.BuildMeanWorkspace(Flow());
        }
        _fv.EvaluateMean(Flow(), *_timePPResidual, *_timePPWorkspace, _engine, _stream, _time);
        result = _fv.LimitPhysicalStep(Flow(), *_timePPResidual, requestedDt, _timePP.relax, _timePP.scale, _timeCompressionPower);
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
