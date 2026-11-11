#include "Cauchy/Core/DomainController.hpp"

namespace DNDS::Cauchy
{
    struct DomainController::CandidateReservation::Data
    {
        Domain *domain = nullptr;
        std::unique_ptr<EquationDofs::WriteScope> destination;
        std::unique_ptr<EquationDofs::ReadScope> source;
        bool committed = false;

        explicit Data(Domain &owner) : domain(&owner) {}
        ~Data()
        {
            if (!committed && destination)
                destination->CancelUnmodified();
        }
    };

    struct DomainController::CheckpointCandidate::Data
    {
        Domain *domain = nullptr;
        std::unique_ptr<IO::DomainCheckpointRead> candidate;
        Data(Domain &owner, std::unique_ptr<IO::DomainCheckpointRead> value)
            : domain(&owner), candidate(std::move(value)) {}
    };

    struct DomainController::InternalCheckpointCandidate::Data
    {
        Domain *domain = nullptr;
        std::unique_ptr<IO::DomainInternalCheckpointRead> candidate;
        Data(Domain &owner, std::unique_ptr<IO::DomainInternalCheckpointRead> value)
            : domain(&owner), candidate(std::move(value)) {}
    };

    DomainController::CandidateReservation::CandidateReservation(
        std::unique_ptr<Data> data)
        : _data(std::move(data)) {}
    DomainController::CandidateReservation::CandidateReservation(CandidateReservation &&) noexcept = default;
    DomainController::CandidateReservation &DomainController::CandidateReservation::operator=(CandidateReservation &&) noexcept = default;
    DomainController::CandidateReservation::~CandidateReservation() = default;
    DomainController::CheckpointCandidate::CheckpointCandidate(
        std::unique_ptr<Data> data)
        : _data(std::move(data)) {}
    DomainController::CheckpointCandidate::CheckpointCandidate(CheckpointCandidate &&) noexcept = default;
    DomainController::CheckpointCandidate &DomainController::CheckpointCandidate::operator=(CheckpointCandidate &&) noexcept = default;
    DomainController::CheckpointCandidate::~CheckpointCandidate() = default;
    real DomainController::CheckpointCandidate::Time() const
    {
        DNDS_check_throw_info(_data && _data->candidate,
                              "empty checkpoint candidate");
        return _data->candidate->time;
    }
    StateEpoch DomainController::CheckpointCandidate::Epoch() const
    {
        DNDS_check_throw_info(_data && _data->candidate,
                              "empty checkpoint candidate");
        return _data->candidate->epoch;
    }
    DomainController::InternalCheckpointCandidate::InternalCheckpointCandidate(
        std::unique_ptr<Data> data)
        : _data(std::move(data)) {}
    DomainController::InternalCheckpointCandidate::InternalCheckpointCandidate(
        InternalCheckpointCandidate &&) noexcept = default;
    DomainController::InternalCheckpointCandidate &
    DomainController::InternalCheckpointCandidate::operator=(
        InternalCheckpointCandidate &&) noexcept = default;
    DomainController::InternalCheckpointCandidate::~InternalCheckpointCandidate() = default;
    StateEpoch DomainController::InternalCheckpointCandidate::TrialEpoch() const
    {
        DNDS_check_throw_info(_data && _data->candidate,
                              "empty internal checkpoint candidate");
        return _data->candidate->trialEpoch;
    }

    DomainController Domain::Controller()
    {
        return DomainController(*this);
    }
    const FiniteVolume &DomainController::Discretization() const
    {
        return _domain->_problem.finiteVolume;
    }
    MPI_Comm DomainController::Communicator() const
    {
        return _domain->_problem.finiteVolume.Metrics().Communicator();
    }
    void DomainController::RequireAcceptedClock(real time, uint64_t acceptedStep) const
    {
        DNDS_check_throw_info(
            _domain->_state.time == time && _domain->_state.epoch.acceptedStep == acceptedStep,
            "domain advanced outside its Simulation");
    }
    void DomainController::BeginStep(real dt)
    {
        _domain->BeginCoordinatedStep(dt);
    }
    bool DomainController::StepIsCoupled() const
    {
        return _domain->CoordinatedStepIsCoupled();
    }
    uint64_t DomainController::TrialIdentifier() const
    {
        return _domain->_state.trialEpoch.trial;
    }
    int DomainController::StageNumber() const
    {
        return _domain->CoordinatedStageNumber();
    }
    real DomainController::PrepareStage(int stage)
    {
        return _domain->PrepareCoordinatedStage(stage);
    }
    StepResult DomainController::SolveStage()
    {
        return _domain->SolveCoordinatedStage();
    }
    StepResult DomainController::CompleteStage()
    {
        return _domain->CompleteCoordinatedStage();
    }
    real DomainController::CoupledStageTime(int coupledStage) const
    {
        return _domain->CoupledStageTime(coupledStage);
    }
    void DomainController::PrepareCoupledInterface(int coupledStage,
                                                   bool reconstructed)
    {
        _domain->PrepareCoupledInterface(coupledStage, reconstructed);
    }
    FiniteVolume::InterfaceDonorPublication
    DomainController::PublishCoupledInterfaceDonor(int coupledStage,
                                                   bool reconstructed)
    {
        return _domain->PublishCoupledInterfaceDonor(coupledStage,
                                                     reconstructed);
    }
    void DomainController::BindCoupledInterfaceInputs(
        int coupledStage, const BoundaryInputs &input)
    {
        _domain->BindCoupledInterfaceInputs(coupledStage, input);
    }
    StepResult DomainController::SolveCoupledStep()
    {
        return _domain->SolveCoordinatedCoupledStep();
    }
    void DomainController::PrepareInterfaceSampling()
    {
        _domain->_problem.finiteVolume.Execute(
            _domain->_state.trial, *_domain->_state.reconstruction, *_domain->_execution.workspace,
            _domain->_execution.Context(), FiniteVolumeEvaluation::Sampling());
    }
    void DomainController::PrepareInterfaceReconstruction(real time)
    {
        _domain->_problem.finiteVolume.Execute(
            _domain->_state.trial, *_domain->_state.reconstruction, *_domain->_execution.workspace,
            _domain->_execution.Context(), FiniteVolumeEvaluation::Reconstruction(time));
    }
    FiniteVolume::InterfaceDonorPublication
    DomainController::PublishInterfaceDonor(bool prepared) const
    {
        return _domain->_problem.finiteVolume.PublishInterfaceDonor(
            _domain->_state.trial, *_domain->_state.reconstruction,
            *_domain->_execution.workspace, prepared);
    }
    void DomainController::BindInterfaceInputs(const BoundaryInputs &input)
    {
        _domain->BindInterfaceInputs(input);
    }
    DomainController::CandidateReservation DomainController::ReserveCandidate()
    {
        DNDS_check_throw_info(
            _domain->_state.trialReady &&
                (!_domain->_state.timeStepper || _domain->_state.timeStepper->HasPendingTrial()),
            "domain has no complete candidate");
        auto data = std::make_unique<CandidateReservation::Data>(*_domain);
        try
        {
            data->destination =
                std::make_unique<EquationDofs::WriteScope>(_domain->_state.accepted.WriteOwned());
            data->source =
                std::make_unique<EquationDofs::ReadScope>(_domain->_state.trial.ReadOwned());
        }
        catch (...)
        {
            if (data->destination)
                data->destination->CancelUnmodified();
            data->destination.reset();
            throw;
        }
        return CandidateReservation(std::move(data));
    }
    void DomainController::CommitCandidate(CandidateReservation &reservation)
    {
        DNDS_check_throw_info(
            reservation._data && reservation._data->domain == _domain &&
                !reservation._data->committed && reservation._data->destination &&
                reservation._data->source,
            "invalid or already committed domain candidate reservation");
        _domain->CommitReserved(*reservation._data->destination,
                                *reservation._data->source);
        reservation._data->committed = true;
    }
    void DomainController::Discard() noexcept
    {
        _domain->DiscardStep();
    }
    void DomainController::SnapAcceptedTime(real time)
    {
        _domain->_state.time = _domain->_state.trialTime = time;
    }
    DomainController::CheckpointCandidate DomainController::PrepareCheckpoint(
        const std::string &path, const Json &resolved)
    {
        return CheckpointCandidate(std::make_unique<CheckpointCandidate::Data>(
            *_domain, _domain->PrepareCheckpointRead(path, resolved)));
    }
    DomainController::InternalCheckpointCandidate
    DomainController::PrepareInternalCheckpoint(const std::string &path,
                                                const Json &resolved)
    {
        return InternalCheckpointCandidate(
            std::make_unique<InternalCheckpointCandidate::Data>(
                *_domain,
                _domain->PrepareInternalCheckpointRead(path, resolved)));
    }
    FiniteVolume::InterfaceDonorPublication
    DomainController::PrepareCheckpointInterfaceDonor(
        CheckpointCandidate &candidate)
    {
        DNDS_check_throw_info(
            candidate._data && candidate._data->domain == _domain &&
                candidate._data->candidate &&
                candidate._data->candidate->equationGuess,
            "interface checkpoint requires a variational reconstruction candidate");
        auto &guess = *candidate._data->candidate->equationGuess;
        std::vector<Runtime::TransferOperation> means, coefficients;
        for (index iField = 0;
             iField < _domain->_state.trial.PhysicalFieldNumber(); ++iField)
        {
            means.push_back(_domain->_problem.finiteVolume.BindHalo(
                                                              _domain->_state.trial.PhysicalField(iField))
                                .Begin());
            coefficients.push_back(_domain->_problem.finiteVolume.BindHalo(
                                                                     guess.PhysicalField(iField))
                                       .Begin());
        }
        for (auto &operation : means)
            operation.Wait();
        for (auto &operation : coefficients)
            operation.Wait();
        return _domain->_problem.finiteVolume.PublishVariationalInterfaceDonor(
            _domain->_state.trial, guess);
    }
    void DomainController::ReserveCheckpoint(CheckpointCandidate &candidate,
                                             bool collectiveChecks)
    {
        DNDS_check_throw_info(candidate._data &&
                                  candidate._data->domain == _domain &&
                                  candidate._data->candidate,
                              "checkpoint candidate belongs to another domain");
        _domain->ReserveCheckpointRead(*candidate._data->candidate,
                                       collectiveChecks);
    }
    void DomainController::CommitCheckpoint(CheckpointCandidate &candidate)
    {
        DNDS_check_throw_info(candidate._data &&
                                  candidate._data->domain == _domain &&
                                  candidate._data->candidate,
                              "checkpoint candidate belongs to another domain");
        _domain->CommitCheckpointRead(*candidate._data->candidate);
    }
    void DomainController::CommitInternalCheckpoint(
        InternalCheckpointCandidate &candidate)
    {
        DNDS_check_throw_info(candidate._data &&
                                  candidate._data->domain == _domain &&
                                  candidate._data->candidate,
                              "internal checkpoint candidate belongs to another domain");
        _domain->CommitInternalCheckpointRead(*candidate._data->candidate);
    }
}
