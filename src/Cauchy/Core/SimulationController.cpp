#include "Cauchy/Core/SimulationController.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/DomainController.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/MPI/Failure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::Cauchy
{
    StepResult SimulationController::Advance(real dt,
                                             std::optional<real> acceptedTime)
    {
        auto &_config = _simulation._config;
        auto &_placement = _simulation._placement;
        auto &_domains = _simulation._domains;
        auto &_interfaces = _simulation._interfaces;
        auto &_time = _simulation._time;
        auto &_step = _simulation._step;
        const auto Local = [&](index domain)
        { return _simulation.Local(domain); };
        real limits[2]{dt, -dt};
        int valid = std::isfinite(dt) && dt > 0;
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _placement.Control());
        DNDS_check_throw_info(valid, "invalid coordinated Cauchy timestep");
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{limits, 2}, _placement.Control());
        DNDS_check_throw_info(limits[0] == -limits[1], "simulation ranks requested different timesteps");
        int targetFlags[2]{int(acceptedTime.has_value()), -int(acceptedTime.has_value())};
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<int>{targetFlags, 2}, _placement.Control());
        DNDS_check_throw_info(targetFlags[0] == -targetFlags[1], "simulation ranks disagree on exact accepted-time snapping");
        real acceptedTarget = acceptedTime.value_or(0);
        if (acceptedTime)
        {
            real targets[2]{acceptedTarget, -acceptedTarget};
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<real>{targets, 2}, _placement.Control());
            DNDS_check_throw_info(targets[0] == -targets[1] && std::isfinite(acceptedTarget) && acceptedTarget > _time,
                                  "simulation ranks requested different or invalid exact accepted times");
            const real expected = _time + dt;
            const real tolerance = 8 * std::numeric_limits<real>::epsilon() *
                                   std::max({real(1), std::abs(expected), std::abs(acceptedTarget)});
            DNDS_check_throw_info(std::abs(expected - acceptedTarget) <= tolerance,
                                  "exact accepted time is inconsistent with the requested timestep");
        }
        if (_placement.DomainNumber() != 1 || _interfaces || _placement.Group(0).ranks.size() != size_t(_placement.Size()))
        {
            uint64_t trialIdentifier = 0;
            const auto localPhase = [&](const auto &action, const std::string &name)
            {
                // Numerical rejection is a normal result, already agreed on
                // each domain. Unexpected exceptions abort before this control
                // agreement; they must never masquerade as a rejected step.
                int ready = 1;
                std::string error;
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                    {
                        if constexpr (std::is_same_v<decltype(action(iDomain, *_domains[iDomain])), StepResult>)
                        {
                            auto result = action(iDomain, *_domains[iDomain]);
                            if (!result)
                            {
                                ready = 0;
                                error = std::move(result.reason);
                            }
                        }
                        else
                            action(iDomain, *_domains[iDomain]);
                    }
                ready = MPI::AllreduceValue<MPI::Reduction::Min>(ready, _placement.Control());
                return ready ? StepResult{} : StepResult::Rejected("coordinated " + name + " rejected; accepted states unchanged: " + error);
            };
            const auto ports = [&](bool prepared)
            {
                std::vector<InterfaceDomainInput> inputs(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        inputs[iDomain] = {domain->Controller().PublishInterfaceDonor(prepared)};
                return inputs;
            };
            const auto refresh = [&](real time, int stage, bool prepared, bool acceptedSnapshot = false)
            {
                StateEpoch epoch;
                epoch.acceptedStep = _step + uint64_t(acceptedSnapshot);
                epoch.trial = trialIdentifier;
                epoch.stage = stage;
                epoch.couplingIteration = prepared ? 1 : 0;
                _interfaces->Refresh(ports(prepared), time, epoch);
                localPhase([&](index iDomain, Domain &domain)
                           {
                    if(const auto input=_interfaces->Inputs(iDomain)) domain.Controller().BindInterfaceInputs(*input); }, "interface binding");
            };
            std::vector<std::optional<DomainController::CandidateReservation>> candidates(_domains.size());
            const auto prepareAndCommit = [&]() -> StepResult
            {
                localPhase([&](index, Domain &domain)
                           {
                    auto control = domain.Controller();
                    MPI::LocalStage(MPIInfo(control.Communicator()),[&]
                    {
                        control.RequireAcceptedClock(_time, _step);
                    });
                    control.BeginStep(dt); }, "step preparation");
                if (Local(0))
                    trialIdentifier = _domains[0]->Controller().TrialIdentifier();
                MPI_Bcast(&trialIdentifier, 1, MPI_UINT64_T, _placement.SimulationRank(0, 0), _placement.Control());
                int stages = Local(0) ? _domains[0]->Controller().StageNumber() : 0;
                MPI_Bcast(&stages, 1, MPI_INT, _placement.SimulationRank(0, 0), _placement.Control());
                for (int stage = 0; stage < stages; ++stage)
                {
                    real stageTime = 0;
                    std::vector<real> localTimes(_domains.size(), 0);
                    localPhase([&](index iDomain, Domain &domain)
                               {
                        auto control = domain.Controller();
                        DNDS_check_throw_info(control.StageNumber()==stages,"domain stage counts differ");
                        localTimes[iDomain]=control.PrepareStage(stage); }, "stage preparation");
                    if (Local(0))
                        stageTime = localTimes[0];
                    MPI_Bcast(&stageTime, 1, MPI_DOUBLE, _placement.SimulationRank(0, 0), _placement.Control());
                    localPhase([&](index iDomain, Domain &)
                               { DNDS_check_throw_info(localTimes[iDomain] == stageTime, "domain stage times differ"); }, "stage-time agreement");
                    if (_interfaces)
                    {
                        localPhase([](index, Domain &domain)
                                   { domain.Controller().PrepareInterfaceSampling(); }, "predictor halos");
                        refresh(stageTime, stage, false);
                        localPhase([&](index, Domain &domain)
                                   { domain.Controller().PrepareInterfaceReconstruction(stageTime); }, "interface reconstruction");
                        refresh(stageTime, stage, true);
                    }
                    if (auto result = localPhase([](index, Domain &domain)
                                                 { return domain.Controller().SolveStage(); }, "stage solve");
                        !result)
                        return result;
                    if (auto result = localPhase([](index, Domain &domain)
                                                 { return domain.Controller().CompleteStage(); }, "stage completion");
                        !result)
                        return result;
                }
                if (_interfaces)
                {
                    localPhase([](index, Domain &domain)
                               { domain.Controller().PrepareInterfaceSampling(); }, "accepted-time predictor halos");
                    refresh(_config.Time().steady ? _time : _time + dt, 0, false, true);
                }
                // Only local lease acquisition belongs in this control-stage
                // catch. Domain solves above contain MPI and must not be caught
                // and followed by an unrelated control reduction on local failure.
                try
                {
                    MPI::LocalStage(MPIInfo(_placement.Control()), [&]
                                    {
                    for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                        if (auto &domain = _domains[iDomain])
                            candidates[iDomain].emplace(domain->Controller().ReserveCandidate()); });
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("Cauchy coordinated state reservation failed");
                    candidates.clear();
                    for (auto &domain : _domains)
                        if (domain)
                            domain->DiscardStep();
                    if (_interfaces)
                    {
                        _interfaces->Discard();
                        for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                            if (_domains[iDomain])
                                if (const auto input = _interfaces->Inputs(iDomain))
                                    _domains[iDomain]->Controller().BindInterfaceInputs(*input);
                    }
                    throw;
                }
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                        _domains[iDomain]->Controller().CommitCandidate(*candidates[iDomain]);
                if (_interfaces)
                    _interfaces->Commit();
                return {};
            };
            const auto result = prepareAndCommit();
            if (!result)
            {
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                {
                    candidates[iDomain].reset();
                    if (_domains[iDomain])
                        _domains[iDomain]->DiscardStep();
                }
                if (_interfaces)
                {
                    _interfaces->Discard();
                    // Restore only ready accepted packets. No donor search,
                    // payload transfer or physical recomputation is needed.
                    for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                        if (_domains[iDomain])
                            if (const auto input = _interfaces->Inputs(iDomain))
                                _domains[iDomain]->Controller().BindInterfaceInputs(*input);
                }
                return result;
            }
            if (!_config.Time().steady)
            {
                _time = acceptedTime.value_or(_time + dt);
                for (auto &domain : _domains)
                    if (domain)
                        domain->Controller().SnapAcceptedTime(_time);
            }
            ++_step;
            return {};
        }
        for (auto &domain : _domains)
            if (domain)
            {
                MPI::LocalStage(MPIInfo(domain->Discretization().Metrics().Communicator()), [&]
                                { DNDS_check_throw_info(domain->Time() == _time && domain->Epoch().acceptedStep == _step,
                                                        "domain was advanced outside its Simulation; use the coordinated frontend"); });
                if (auto result = domain->PrepareStep(dt); !result)
                {
                    domain->DiscardStep();
                    return result;
                }
            }
        // Commit is serialized and collectively preflighted inside each domain.
        // External debug writers/aliases must be released before this call.
        for (auto &domain : _domains)
            if (domain)
                domain->CommitStep();
        if (!_config.Time().steady)
        {
            _time = acceptedTime.value_or(_time + dt);
            for (auto &domain : _domains)
                if (domain)
                    domain->Controller().SnapAcceptedTime(_time);
        }
        ++_step;
        return {};
    }
}
