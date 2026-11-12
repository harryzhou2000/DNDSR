#pragma once

#include "Cauchy/Implicit/UpdatePolicy.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"

namespace DNDS::Cauchy
{
    /// Immutable setup values shared by prepared domain solve records. This is
    /// configuration, not a controller: no iteration or operator is executed.
    struct NonlinearSolveSetup
    {
        ImplicitSettings settings;
        EquationSchema schema;
        int rank = 0;
        std::shared_ptr<const UpdatePolicy> policy;

        NonlinearSolveSetup(const ImplicitSettings &settings,
                            EquationSchema schema, int rank,
                            const std::string &policy,
                            const UpdatePolicyRequirements &requirements);
    };

    /// Move-only domain solve record. Numerical owners are pinned through the
    /// callbacks; this record maintains convergence/CFL/report semantics but
    /// never drives an iteration loop. A concrete UpdatePolicy orders calls.
    class NonlinearSession
    {
        struct Data;
        std::shared_ptr<Data> _data;
        explicit NonlinearSession(std::shared_ptr<Data> data)
            : _data(std::move(data)) {}

    public:
        NonlinearSession() = default;
        NonlinearSession(const NonlinearSession &) = delete;
        NonlinearSession &operator=(const NonlinearSession &) = delete;
        NonlinearSession(NonlinearSession &&) noexcept = default;
        NonlinearSession &operator=(NonlinearSession &&) noexcept = default;
        static NonlinearSession Prepare(const NonlinearSolveSetup &setup,
                                        int stage, NonlinearProblem problem);
        /// Start one coordinated nonlinear round.
        void BeginIteration();
        /// Execute one policy-named correction. Calls are ordered from zero.
        void AdvanceStage(int coupledStage);
        /// Evaluate and globally reduce this domain's stopping residual.
        void Observe();
        /// Publish this round. `coordinatedTerminal` is agreed by the
        /// simulation over all participating domain sessions.
        void FinishIteration(bool coordinatedTerminal);
        [[nodiscard]] bool ObservationBeforeUpdate() const;
        [[nodiscard]] int CoupledStageNumber() const;
        [[nodiscard]] index Iteration() const;
        [[nodiscard]] index NextIteration() const;
        [[nodiscard]] bool LocalTerminal() const;
        [[nodiscard]] bool Complete() const;
        const UpdatePolicy &Policy() const;
        const NonlinearIterationReport &Report() const;
    };

    /**
     * @brief Synchronous borrowed operands for one policy-owned group solve.
     * @details The caller retains sessions until SolveNonlinearGroup returns.
     * Communication retains the existing control communicator without Dup.
     * Interface callbacks post a retained dependency and finish its publication;
     * they are empty for standalone/single-domain solves. Fields and method
     * history stay in their domain owners. None of these helpers loops until
     * convergence: that sequence belongs exclusively to UpdatePolicy::Solve.
     */
    struct NonlinearSolveGroup
    {
        std::vector<std::reference_wrapper<NonlinearSession>> domains;
        std::shared_ptr<Runtime::CommunicationContext> communication;
        /// Common prepared policy, also present on ranks with no local domain.
        /// Standalone callers may resolve it from their one local session.
        std::shared_ptr<const UpdatePolicy> policy;
        /// Simulation-wide diagnostic choice, resolved from all configured
        /// domains, never rank-local membership (which could deadlock checks).
        bool collectiveChecks = false;
        std::function<void(int coupledStage, index iteration)> refreshInterfaces;
        std::function<void()> finishInterfaces;

        /// Begin one round and validate local cursors. Optional cross-rank
        /// diagnostics obey the Cauchy collective-check master switch.
        index BeginIteration();
        /// Agree termination, then publish every domain's iteration state.
        bool FinishIteration();
        void RefreshInterfaces(int coupledStage, index iteration) const;
        void FinishInterfaces() const;

    private:
        index _iteration = 0;
        friend void SolveNonlinearGroup(NonlinearSolveGroup &group);
    };

    /// Validate the prepared policy match and dispatch exactly once. This entry
    /// contains no shared nonlinear or coupled-stage iteration loop.
    void SolveNonlinearGroup(NonlinearSolveGroup &group);
    /// Standalone convenience using exactly the same policy with one domain.
    NonlinearIterationReport SolveNonlinear(NonlinearSession &session);
    NonlinearIterationReport SolveNonlinear(const NonlinearSolveSetup &setup,
                                            int stage, NonlinearProblem problem);
}
