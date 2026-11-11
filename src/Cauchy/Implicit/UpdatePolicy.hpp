#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"

namespace DNDS::Cauchy
{
    /// Method-declared contract checked before an update policy is used. A
    /// policy cannot silently change the number/order of coupled unknowns or
    /// observe a residual at another point in the nonlinear iteration.
    struct UpdatePolicyRequirements
    {
        int coupledStages = 1;
        NonlinearDetail::ResidualObservation observation =
            NonlinearDetail::ResidualObservation::BeforeUpdate;
        bool methodStoppingResidual = false;
    };

    struct UpdatePolicyDescriptor
    {
        std::string name;
        int coupledStages = 1;
        NonlinearDetail::ResidualObservation observation =
            NonlinearDetail::ResidualObservation::BeforeUpdate;
        bool methodStoppingResidual = false;
        int schemaVersion = 1;

        void Validate(const UpdatePolicyRequirements &requirements) const;
        Json Metadata() const;
    };

    /// Solver/operator callbacks for one nonlinear solve. Arithmetic, leases
    /// and MPI remain owned by these adapters; the controller supplies the
    /// immutable settings/schema/rank binding separately.
    struct NonlinearProblem
    {
        NonlinearSolveControl control;
        NonlinearIterationReport initial;
        real reconstructionTolerance = 0;
        std::function<void()> evaluate;
        std::function<void(std::vector<real> &)> measure;
        std::function<std::pair<int, real>()> reconstruction;
        /// Perform exactly one named coupled-stage correction and its
        /// method-owned completion. The policy determines invocation order.
        std::function<void(int, NonlinearIterationReport &)> advanceStage;
    };

    /// Complete setup-time binding consumed by an update policy. Callers make
    /// this only through NonlinearController, so policies cannot substitute
    /// another equation schema or implicit-control snapshot.
    struct BoundNonlinearProblem
    {
        const ImplicitSettings *settings = nullptr;
        const EquationSchema *schema = nullptr;
        int rank = 0, stage = 0;
        NonlinearProblem operations;
        void Validate(const UpdatePolicyDescriptor &descriptor) const;
    };

    class UpdatePolicy
    {
    public:
        virtual ~UpdatePolicy() = default;
        virtual const UpdatePolicyDescriptor &Descriptor() const = 0;
        virtual NonlinearIterationReport Execute(BoundNonlinearProblem problem) const = 0;
    };

    /// Process-local setup registry. Registration is serialized and intended
    /// for startup/tests, never from a numerical loop. Resolution validates the
    /// method contract before returning a prepared stateless policy.
    class UpdatePolicyRegistry
    {
    public:
        using Factory = std::function<std::unique_ptr<UpdatePolicy>()>;
        static void Register(UpdatePolicyDescriptor descriptor, Factory factory);
        static std::unique_ptr<UpdatePolicy> Resolve(
            const std::string &name,
            const UpdatePolicyRequirements &requirements);
        static std::vector<std::string> Names();
    };
}
