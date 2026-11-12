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

        /// Borrow the immutable domain-local FV plan for matching/routing setup.
        const FiniteVolume &Discretization() const;
        /// Domain communicator used by every phase below.
        MPI_Comm Communicator() const;
        /// Local validation only. The surrounding Simulation control stage
        /// decides whether this participates in a collective preflight.
        void RequireAcceptedClock(real time, uint64_t acceptedStep) const;
        /// Create the method-owned trial/history state for one physical step.
        void BeginStep(real dt);
        /// Whether the active time method updates a coupled stage group.
        bool StepIsCoupled() const;
        /// Monotone identity of the currently prepared trial.
        uint64_t TrialIdentifier() const;
        /// Number of serialized implicit stages in the active method.
        int StageNumber() const;
        /// Prepare one serialized stage and return its physical stage time.
        real PrepareStage(int stage);
        /// Execute the prepared serialized stage without publishing accepted state.
        StepResult SolveStage();
        /// Finalize method-owned stage data after a successful solve.
        StepResult CompleteStage();
        /// Physical time of a member of the active coupled stage group.
        real CoupledStageTime(int coupledStage) const;
        /// Prepare one coupled stage's interface sampling/reconstruction state.
        void PrepareCoupledInterface(int coupledStage, bool reconstructed);
        /// Freeze one coupled stage's donor publication.
        FiniteVolume::InterfaceDonorPublication PublishCoupledInterfaceDonor(
            int coupledStage, bool reconstructed);
        /// Bind the matching frozen receiver packet to one coupled stage.
        void BindCoupledInterfaceInputs(int coupledStage,
                                        const BoundaryInputs &input);
        /// Execute the prepared coupled update policy without accepting it.
        StepResult SolveCoupledStep();
        /// Prepare predictor halos without reconstructing a new polynomial.
        void PrepareInterfaceSampling();
        /// Prepare the finalized reconstruction at the named physical time.
        void PrepareInterfaceReconstruction(real time);
        /// Freeze the current ordinary-stage donor publication.
        FiniteVolume::InterfaceDonorPublication PublishInterfaceDonor(bool prepared) const;
        /// Bind an ordinary stage's immutable receiver packet.
        void BindInterfaceInputs(const BoundaryInputs &input);
        /// Reserve accepted-state publication before any domain commits.
        CandidateReservation ReserveCandidate();
        /// Atomically publish this domain's prepared candidate into a reservation.
        void CommitCandidate(CandidateReservation &reservation);
        /// Drop the unaccepted trial and method cursor without throwing.
        void Discard() noexcept;
        /// Exact controller-clock publication after a successful domain commit.
        void SnapAcceptedTime(real time);
        /// Read and validate a detached accepted checkpoint candidate.
        CheckpointCandidate PrepareCheckpoint(const std::string &path,
                                              const Json &resolved);
        /// Read and validate a detached exact-internal checkpoint cursor.
        InternalCheckpointCandidate PrepareInternalCheckpoint(
            const std::string &path, const Json &resolved);
        /// Complete candidate mean/coefficient halos and publish the detached
        /// variational donor without modifying accepted reconstruction state.
        FiniteVolume::InterfaceDonorPublication PrepareCheckpointInterfaceDonor(
            CheckpointCandidate &candidate);
        /// Reserve accepted storage for a validated checkpoint candidate.
        void ReserveCheckpoint(CheckpointCandidate &candidate,
                               bool collectiveChecks = true);
        /// Publish a previously reserved accepted checkpoint candidate.
        void CommitCheckpoint(CheckpointCandidate &candidate);
        /// Publish the exact nonlinear/time-method cursor after accepted state.
        void CommitInternalCheckpoint(InternalCheckpointCandidate &candidate);
    };
}
