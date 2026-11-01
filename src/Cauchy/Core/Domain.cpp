#include "Cauchy/Core/Domain.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        R::RowLayout FlowLayout(const DomainSettings &settings)
        {
            if (settings.execution.layout == "soa")
                return R::RowLayout::ColMajor(0, 5);
            if (settings.execution.layout == "aosoa32")
                return R::RowLayout::TiledColMajor(0, 5, 32);
            return R::RowLayout::Dense(0, 5);
        }
        Algebra::DofBundle BuildState(const CFV::FiniteVolumePlan &fv, const DomainSettings &settings)
        {
            auto field = fv.BuildField(FlowLayout(settings));
            Algebra::DofBundle result(field.Space());
            const auto state = EquationSystem::Resolve(settings.equation).Gas().Conservative(settings.initialPrimitive);
            {
                auto write = field.Owned().Write();
                const auto values = write.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < fv.NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        values(iCell, j) = state[j];
            }
            result.Add("flow", std::move(field));
            return result;
        }
        EquationSystem ResolveEquation(const std::string &name, const DomainSettings &settings)
        {
            ResolvedExecutionPlan::Resolve({{name, {0}}, settings});
            return EquationSystem::Resolve(settings.equation);
        }
    }
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings)
        : _name(std::move(name)), _equation(ResolveEquation(_name, settings)), _fv(std::move(mesh), settings, _equation),
          _accepted(BuildState(_fv.Metrics(), settings)), _trial(_accepted.Clone()),
          _residual(_fv.Metrics().BuildField(FlowLayout(settings))), _initial(_fv.Metrics().BuildField(FlowLayout(settings))),
          _workspace(_fv.BuildWorkspace(_trial.Block("flow"))),
          _engine(size_t(settings.execution.hostThreads), settings.execution.checkAccess ? std::make_shared<R::AccessTrackingContext>() : nullptr),
          _stream(_engine.CreateCPUStream())
    {
        _equation.ValidateStateSchema(_accepted);
        DNDS_check_throw_info(_fv.Metrics().Dimension() == settings.mesh.dimension, "Cauchy config/mesh dimension differs");
    }
    void Domain::UpdateStage(real dt, real historyWeight, real stageWeight)
    {
        _fv.Evaluate(_trial.Block("flow"), _residual, *_workspace, _engine, _stream);
        _lastFallbacks += _fv.LastFaceFallbacks(*_workspace);
        auto geometry = _fv.Metrics().AcquireRead();
        const auto metrics = geometry.HostView();
        R::SubmitArrays(_engine, _stream, [dt, historyWeight, stageWeight, metrics](auto, auto initial, auto rhs, auto stage)
                        {
            for (index iCell = 0; iCell < stage.rows; ++iCell)
                for (int j = 0; j < 5; ++j)
                    stage(iCell, j) = historyWeight * initial(iCell, j) + stageWeight * (stage(iCell, j) + dt * rhs(iCell, j) / metrics.GetCellVol(iCell)); }, R::ReadAs<R::UniformTableRows>(_initial.Owned()), R::ReadAs<R::UniformTableRows>(_residual.Owned()), R::ReadWriteAs<R::UniformTableRows>(_trial.Block("flow").Owned()));
        _stream.Record().WaitHost();
        DNDS_check_throw_info(_fv.Admissible(_trial.Block("flow")), "Cauchy SSPRK3 stage is inadmissible; accepted state unchanged");
    }
    void Domain::PrepareStep(real dt)
    {
        _trialReady = false;
        int valid = std::isfinite(dt) && dt > 0 && std::isfinite(_time + dt);
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _fv.Metrics().Communicator());
        DNDS_check_throw_info(valid, "invalid Cauchy physical timestep");
        real min = dt, max = dt;
        MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, _fv.Metrics().Communicator());
        MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, _fv.Metrics().Communicator());
        DNDS_check_throw_info(min == max, "Cauchy timestep differs between domain ranks");
        DNDS_check_throw_info(_fv.Admissible(Flow()), "accepted Cauchy mean is inadmissible");
        // Scheduled reads may coexist with a retained accepted-state reader.
        // Immediate algebra's stronger idle-input contract would reject only
        // the retaining rank and strand peers at the next domain collective.
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            Algebra::Copy(_engine, _stream, _trial.Block("flow"), Flow());
            Algebra::Copy(_engine, _stream, _initial, Flow());
            _stream.Record().WaitHost(); });
        _lastFallbacks = 0;
        _trialEpoch = _epoch;
        _trialEpoch.trial = ++_nextTrial;
        // Shu-Osher SSPRK(3,3); autonomous stationary-grid residual. The future
        // provider contract carries stage times for time-dependent BC/source.
        _trialEpoch.stage = 0;
        UpdateStage(dt, 0, 1);
        _trialEpoch.stage = 1;
        UpdateStage(dt, .75, .25);
        _trialEpoch.stage = 2;
        UpdateStage(dt, 1. / 3, 2. / 3);
        _trialTime = _time + dt;
        _trialReady = true;
    }
    void Domain::CommitStep()
    {
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        { DNDS_check_throw_info(_trialReady, "no prepared Cauchy candidate"); });
        auto epoch = _trialEpoch;
        ++epoch.acceptedStep;
        epoch.stage = 0;
        PublishTrial(_trialTime, epoch);
    }
    void Domain::PublishTrial(real time, StateEpoch epoch)
    {
        // Agree local immediate-access availability before any accepted values
        // change. A retained debug/Python view must not allow a partial commit
        // on only the other ranks. Keep this lease through the actual copy.
        std::optional<R::Array<real>::WriteLease> destination;
        std::optional<R::Array<real>::ReadLease> source;
        const auto comm = _fv.Metrics().Communicator();
        MPIInfo mpi(comm);
        try
        {
            MPI::LocalStage(mpi, [&]
                            { destination.emplace(Flow().Owned().Write()); source.emplace(_trial.Block("flow").Owned().Read()); });
        }
        catch (...)
        {
            // Successful ranks acquired but never used their destination.
            // Cancel before unwinding; a normal exceptional write destructor
            // deliberately poisons possibly partial writes.
            if (destination)
                destination->CancelUnmodified();
            if (source)
                source->CancelUnmodified();
            throw;
        }
        const auto input = source->View<R::UniformTableRows>();
        const auto output = destination->View<R::UniformTableRows>();
        for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
            for (int component = 0; component < 5; ++component)
                output(iCell, component) = input(iCell, component);
        _time = time;
        _epoch = epoch;
        _trialReady = false;
    }
    const CFV::FVField &Domain::Residual()
    {
        _trialReady = false;
        DNDS_check_throw_info(_fv.Admissible(Flow()), "accepted Cauchy mean is inadmissible");
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            Algebra::Copy(_engine, _stream, _trial.Block("flow"), Flow());
            _stream.Record().WaitHost(); });
        _fv.Evaluate(_trial.Block("flow"), _residual, *_workspace, _engine, _stream);
        _lastFallbacks = _fv.LastFaceFallbacks(*_workspace);
        return _residual;
    }
}
