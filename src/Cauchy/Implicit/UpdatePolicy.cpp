#include "Cauchy/Implicit/UpdatePolicy.hpp"
#include "Cauchy/Implicit/NonlinearSolve.hpp"

#include <mutex>

namespace DNDS::Cauchy
{
    namespace
    {
        class SingleStageImplicitPolicy final : public UpdatePolicy
        {
            UpdatePolicyDescriptor _descriptor{
                "singleStageImplicit", 1,
                NonlinearDetail::ResidualObservation::BeforeUpdate, false};

        public:
            const UpdatePolicyDescriptor &Descriptor() const override
            {
                return _descriptor;
            }
            void Solve(NonlinearSolveGroup &group) const override
            {
                bool complete = false;
                while (!complete)
                {
                    const index iteration = group.BeginIteration();
                    group.RefreshInterfaces(0, iteration);
                    for (auto reference : group.domains)
                    {
                        auto &domain = reference.get();
                        domain.Observe();
                        if (!domain.LocalTerminal())
                            domain.AdvanceStage(0);
                    }
                    group.FinishInterfaces();
                    complete = group.FinishIteration();
                }
            }
        };

        class DITRSequential2StagePolicy final : public UpdatePolicy
        {
            UpdatePolicyDescriptor _descriptor{
                "ditrSequential2Stage", 2,
                NonlinearDetail::ResidualObservation::AfterUpdate, true};

        public:
            const UpdatePolicyDescriptor &Descriptor() const override
            {
                return _descriptor;
            }
            void Solve(NonlinearSolveGroup &group) const override
            {
                bool complete = false;
                while (!complete)
                {
                    const index iteration = group.BeginIteration();
                    group.RefreshInterfaces(0, iteration);
                    // Correction auxiliary/RHS operators import the packet
                    // event themselves. Keep independent core preparation
                    // ahead of that edge; no policy-wide pre-update wait.
                    for (auto reference : group.domains)
                        reference.get().AdvanceStage(0);
                    group.FinishInterfaces();

                    group.RefreshInterfaces(1, iteration);
                    for (auto reference : group.domains)
                        reference.get().AdvanceStage(1);
                    group.FinishInterfaces();

                    for (auto reference : group.domains)
                        reference.get().Observe();
                    complete = group.FinishIteration();
                }
            }
        };

        struct RegisteredPolicy
        {
            UpdatePolicyDescriptor descriptor;
            UpdatePolicyRegistry::Factory factory;
        };
        struct RegistryData
        {
            std::mutex mutex;
            std::map<std::string, RegisteredPolicy> policies;
            RegistryData()
            {
                policies.emplace(
                    "singleStageImplicit",
                    RegisteredPolicy{
                        SingleStageImplicitPolicy().Descriptor(),
                        []
                        { return std::make_unique<SingleStageImplicitPolicy>(); }});
                policies.emplace(
                    "ditrSequential2Stage",
                    RegisteredPolicy{
                        DITRSequential2StagePolicy().Descriptor(),
                        []
                        { return std::make_unique<DITRSequential2StagePolicy>(); }});
            }
        };
        RegistryData &Registry()
        {
            static RegistryData registry;
            return registry;
        }
    }

    void UpdatePolicyDescriptor::Validate(
        const UpdatePolicyRequirements &requirements) const
    {
        DNDS_check_throw_info(
            schemaVersion == 1 && !name.empty() && coupledStages > 0 &&
                coupledStages == requirements.coupledStages &&
                observation == requirements.observation &&
                methodStoppingResidual == requirements.methodStoppingResidual,
            "nonlinear update policy contradicts the time method's stage or stopping-residual contract");
    }
    Json UpdatePolicyDescriptor::Metadata() const
    {
        return {{"name", name},
                {"schema_version", schemaVersion},
                {"coupled_stages", coupledStages},
                {"residual_observation",
                 observation == NonlinearDetail::ResidualObservation::BeforeUpdate
                     ? "beforeUpdate"
                     : "afterUpdate"},
                {"method_stopping_residual", methodStoppingResidual}};
    }
    void NonlinearProblem::Validate(
        const UpdatePolicyDescriptor &descriptor) const
    {
        DNDS_check_throw_info(
            evaluate && measure && reconstruction && advanceStage &&
                descriptor.coupledStages > 0 &&
                std::isfinite(reconstructionTolerance) &&
                reconstructionTolerance >= 0,
            "incomplete nonlinear update-policy problem");
    }
    void UpdatePolicyRegistry::Register(UpdatePolicyDescriptor descriptor,
                                        Factory factory)
    {
        DNDS_check_throw_info(descriptor.schemaVersion == 1 && !descriptor.name.empty() &&
                                  descriptor.coupledStages > 0 && factory,
                              "invalid nonlinear update-policy registration");
        auto &registry = Registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto name = descriptor.name;
        DNDS_check_throw_info(
            registry.policies.emplace(
                                 name,
                                 RegisteredPolicy{std::move(descriptor),
                                                  std::move(factory)})
                .second,
            "duplicate nonlinear update-policy name");
    }
    std::unique_ptr<UpdatePolicy> UpdatePolicyRegistry::Resolve(
        const std::string &requested,
        const UpdatePolicyRequirements &requirements)
    {
        const std::string name = requested == "auto"
                                     ? (requirements.coupledStages == 1
                                            ? "singleStageImplicit"
                                            : "ditrSequential2Stage")
                                     : requested;
        auto &registry = Registry();
        Factory factory;
        UpdatePolicyDescriptor descriptor;
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            const auto found = registry.policies.find(name);
            DNDS_check_throw_info(found != registry.policies.end(),
                                  "unknown nonlinear update policy: " + name);
            descriptor = found->second.descriptor;
            factory = found->second.factory;
        }
        descriptor.Validate(requirements);
        auto result = factory();
        DNDS_check_throw_info(result &&
                                  result->Descriptor().Metadata() ==
                                      descriptor.Metadata(),
                              "nonlinear update-policy factory returned another descriptor");
        return result;
    }
    std::vector<std::string> UpdatePolicyRegistry::Names()
    {
        auto &registry = Registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        std::vector<std::string> result;
        for (const auto &[name, policy] : registry.policies)
        {
            (void)policy;
            result.push_back(name);
        }
        return result;
    }
}
