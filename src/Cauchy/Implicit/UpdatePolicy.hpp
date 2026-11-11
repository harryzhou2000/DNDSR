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

    /// Bound callbacks for one nonlinear solve. Arithmetic, leases and MPI
    /// remain owned by the finite-volume/nonlinear adapters; policy owns only
    /// update order and residual-observation timing.
    struct UpdatePolicyProblem
    {
        const ImplicitSettings *settings = nullptr;
        const EquationSchema *schema = nullptr;
        int rank = 0, stage = 0;
        NonlinearSolveControl control;
        NonlinearIterationReport initial;
        real reconstructionTolerance = 0;
        std::function<void()> evaluate;
        std::function<void(std::vector<real> &)> measure;
        std::function<std::pair<int, real>()> reconstruction;
        /// Perform exactly one named coupled-stage correction and its
        /// method-owned completion. The policy determines invocation order.
        std::function<void(int, NonlinearIterationReport &)> advanceStage;

        void Validate(const UpdatePolicyDescriptor &descriptor) const;
    };

    class UpdatePolicy
    {
    public:
        virtual ~UpdatePolicy() = default;
        virtual const UpdatePolicyDescriptor &Descriptor() const = 0;
        virtual NonlinearIterationReport Execute(UpdatePolicyProblem problem) const = 0;
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
