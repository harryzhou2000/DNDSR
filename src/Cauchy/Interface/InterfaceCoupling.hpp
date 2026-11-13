#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Interface/InterfaceDiagnostics.hpp"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"

namespace DNDS::Cauchy
{
    /// Optional managed donor publication. Nonmembers leave values empty; all
    /// coupling ranks still enter the same matching schedule. Reconstruction
    /// representation and physical-field grouping remain private to FV and the
    /// coupling adapter instead of leaking into Simulation orchestration.
    struct InterfaceDomainInput
    {
        std::optional<InterfaceDonorPublication> values;
    };

    /// Physical adapter over Geom matching and CFV point interpolation. It owns
    /// ready receiver packets, posed matching/sampling plans and stable scratch.
    /// Every coupling rank calls in the same connection order, including ranks
    /// outside both domains. The simulation's coordinated nonlinear policy
    /// refreshes values; an individual domain never enters it independently.
    /// Optional safety agreement is enabled if any configured domain requests
    /// implicit.collectiveChecks. All-off retains local fail-fast validation,
    /// required routing collectives and numerical communication. The selected
    /// context shares the supplied communicator resource and tag namespace.
    /// Payload mapping follows each receiver's execution.checkAccess policy;
    /// shared checked/unchecked lane pairs preserve mixed-domain settings.
    /// No conservative interface flux is imposed by this point-value coupling.
    class InterfaceCoupling
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        /// Move-only posted value refresh. BeginRefresh freezes donor samples
        /// and posts every connection in canonical order. CompletionEvent is a
        /// backend-neutral dependency covering transfer and receiver mapping;
        /// Complete validates and publishes after that event. Destruction of
        /// an unfinished operation is a solver error, never an implicit MPI wait.
        class RefreshOperation
        {
            struct State;
            std::shared_ptr<State> _state;
            void Finish(bool wait);
            explicit RefreshOperation(std::shared_ptr<State> state)
                : _state(std::move(state)) {}
            friend class InterfaceCoupling;

        public:
            RefreshOperation() = default;
            RefreshOperation(const RefreshOperation &) = delete;
            RefreshOperation &operator=(const RefreshOperation &) = delete;
            RefreshOperation(RefreshOperation &&) noexcept = default;
            RefreshOperation &operator=(RefreshOperation &&) noexcept = delete;
            ~RefreshOperation();
            Runtime::StreamEvent CompletionEvent() const;
            void Complete();
            /// Publish only after CompletionEvent ordered this callback. No
            /// host wait is introduced; early use is rejected.
            void PublishReady();
            /// Packet handles published by this operation, not whichever round
            /// the coordinator currently exposes. Call only after publication;
            /// ordinary packet generation checks still reject later reuse of
            /// its mutable staging storage. Freeze a packet to keep stages.
            std::optional<BoundaryInputs> Inputs(index domain) const;
            /// Stable receiver allocations and round-specific publication slot,
            /// available before completion for declaring queued consumers.
            /// Import CompletionEvent and execute PublishReady before consuming
            /// the publication. This handle never follows a later round.
            std::optional<BoundaryInputs> PendingInputs(index domain) const;
            void Discard();
            bool Finished() const;
        };

    private:
        RefreshOperation BeginRefreshImpl(
            const std::vector<InterfaceDomainInput> &domains, real time,
            StateEpoch epoch, bool valuesOnly);

    public:
        InterfaceCoupling(const InterfaceCoupling &) = delete;
        InterfaceCoupling &operator=(const InterfaceCoupling &) = delete;
        InterfaceCoupling(InterfaceCoupling &&) noexcept = default;
        InterfaceCoupling &operator=(InterfaceCoupling &&) noexcept = default;
        InterfaceCoupling(const SimulationConfig &config, std::shared_ptr<Runtime::CommunicationContext> coupling,
                          const std::vector<std::optional<FiniteVolume>> &localDomains,
                          const std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>> &workers = {});
        /// Whether values are refreshed once per coordinated nonlinear round.
        /// Topology/route still remain frozen for the complete physical stage.
        bool RefreshEveryNonlinearIteration() const;
        RefreshOperation BeginRefresh(
            const std::vector<InterfaceDomainInput> &domains, real time,
            StateEpoch epoch);
        /// Reuse the already frozen stage topology/route and refresh sampled
        /// values only. Geometry, pose, schema or grouping changes are errors.
        RefreshOperation BeginValueRefresh(
            const std::vector<InterfaceDomainInput> &domains, real time,
            StateEpoch epoch);
        /// Rebuild maps only when stage-time poses change. Two refreshes at one
        /// time reuse the exact map, sampling workspace and persistent return
        /// binding. Source inputs are frozen for the complete round before any
        /// receiver packet is published, preserving block-Jacobi semantics.
        void Refresh(const std::vector<InterfaceDomainInput> &domains, real time, StateEpoch epoch);
        std::optional<BoundaryInputs> Inputs(index domain) const;
        /// Collective diagnostic reduction on the coupling communicator.
        /// Does not refresh traces, synchronize kernels or mutate trajectories.
        InterfaceDiagnostics Diagnostics() const;
        /// Retain the currently published packets as accepted traces. The next
        /// round uses separate scratch allocations; rejected trials can restore
        /// accepted packets without copying their payload or changing its epoch.
        void Commit() noexcept;
        void Discard() noexcept;
    };
}
