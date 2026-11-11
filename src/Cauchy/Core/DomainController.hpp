#pragma once

#include "Cauchy/Core/Domain.hpp"
#include "DNDS/Core/OptionalRef.hpp"

namespace DNDS::Cauchy
{
    /// Narrow domain-local phase facade used by Simulation orchestration.
    ///
    /// This object is a non-owning synchronous view. It deliberately exposes
    /// phase operations and opaque transactions instead of EquationDofs,
    /// reconstruction workspaces, execution streams or leases. Simulation can
    /// therefore coordinate every local domain without becoming a second
    /// owner of domain state. All methods preserve Domain's existing MPI and
    /// numerical rejection semantics; the caller still orders every domain at
    /// the simulation level.
    class DomainController
    {
        OptionalRef<Domain> _domain;
        explicit DomainController(Domain &domain) : _domain(domain) {}
        friend class Domain;

    public:
        /// Move-only reservation of one complete trial publication. The token
        /// retains the accepted-state writer and trial-state reader. Dropping
        /// an uncommitted token cancels the untouched writer, so an all-domain
        /// preflight can reserve every destination before publishing any one.
        class CandidateReservation
        {
            struct Data;
            std::unique_ptr<Data> _data;
            explicit CandidateReservation(std::unique_ptr<Data> data);
            friend class DomainController;

        public:
            CandidateReservation(CandidateReservation &&) noexcept;
            CandidateReservation &operator=(CandidateReservation &&) noexcept;
            CandidateReservation(const CandidateReservation &) = delete;
            CandidateReservation &operator=(const CandidateReservation &) = delete;
            ~CandidateReservation();
        };

        /// Detached accepted-checkpoint candidate. File I/O and distributed ID
        /// remapping finish before this object is returned; publication remains
        /// transactional and opaque to Simulation.
        class CheckpointCandidate
        {
            struct Data;
            std::unique_ptr<Data> _data;
            explicit CheckpointCandidate(std::unique_ptr<Data> data);
            friend class DomainController;

        public:
            CheckpointCandidate(CheckpointCandidate &&) noexcept;
            CheckpointCandidate &operator=(CheckpointCandidate &&) noexcept;
            CheckpointCandidate(const CheckpointCandidate &) = delete;
            CheckpointCandidate &operator=(const CheckpointCandidate &) = delete;
            ~CheckpointCandidate();
            real Time() const;
            StateEpoch Epoch() const;
        };

        /// Detached exact-internal cursor restored only after its accepted
        /// checkpoint candidate has been committed.
        class InternalCheckpointCandidate
        {
            struct Data;
            std::unique_ptr<Data> _data;
            explicit InternalCheckpointCandidate(std::unique_ptr<Data> data);
            friend class DomainController;

        public:
            InternalCheckpointCandidate(InternalCheckpointCandidate &&) noexcept;
            InternalCheckpointCandidate &operator=(InternalCheckpointCandidate &&) noexcept;
            InternalCheckpointCandidate(const InternalCheckpointCandidate &) = delete;
            InternalCheckpointCandidate &operator=(const InternalCheckpointCandidate &) = delete;
            ~InternalCheckpointCandidate();
            StateEpoch TrialEpoch() const;
        };

        const FiniteVolume &Discretization() const;
        MPI_Comm Communicator() const;
        /// Local validation only. The surrounding Simulation control stage
        /// decides whether this participates in a collective preflight.
        void RequireAcceptedClock(real time, uint64_t acceptedStep) const;
        void BeginStep(real dt);
        bool StepIsCoupled() const;
        uint64_t TrialIdentifier() const;
        int StageNumber() const;
        real PrepareStage(int stage);
        StepResult SolveStage();
        StepResult CompleteStage();
        real CoupledStageTime(int coupledStage) const;
        void PrepareCoupledInterface(int coupledStage, bool reconstructed);
        FiniteVolume::InterfaceDonorPublication PublishCoupledInterfaceDonor(
            int coupledStage, bool reconstructed);
        void BindCoupledInterfaceInputs(int coupledStage,
                                        const BoundaryInputs &input);
        StepResult SolveCoupledStep();
        /// Prepare predictor halos without reconstructing a new polynomial.
        void PrepareInterfaceSampling();
        /// Prepare the finalized reconstruction at the named physical time.
        void PrepareInterfaceReconstruction(real time);
        FiniteVolume::InterfaceDonorPublication PublishInterfaceDonor(bool prepared) const;
        void BindInterfaceInputs(const BoundaryInputs &input);
        CandidateReservation ReserveCandidate();
        void CommitCandidate(CandidateReservation &reservation);
        void Discard() noexcept;
        /// Exact controller-clock publication after a successful domain commit.
        void SnapAcceptedTime(real time);
        CheckpointCandidate PrepareCheckpoint(const std::string &path,
                                              const Json &resolved);
        InternalCheckpointCandidate PrepareInternalCheckpoint(
            const std::string &path, const Json &resolved);
        /// Complete candidate mean/coefficient halos and publish the detached
        /// variational donor without modifying accepted reconstruction state.
        FiniteVolume::InterfaceDonorPublication PrepareCheckpointInterfaceDonor(
            CheckpointCandidate &candidate);
        void ReserveCheckpoint(CheckpointCandidate &candidate,
                               bool collectiveChecks = true);
        void CommitCheckpoint(CheckpointCandidate &candidate);
        void CommitInternalCheckpoint(InternalCheckpointCandidate &candidate);
    };
}
