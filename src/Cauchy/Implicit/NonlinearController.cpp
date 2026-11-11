#include "Cauchy/Implicit/NonlinearController.hpp"

namespace DNDS::Cauchy
{
    NonlinearController::NonlinearController(
        const ImplicitSettings &settings, EquationSchema schema, int rank,
        const std::string &policy,
        const UpdatePolicyRequirements &requirements)
        : _settings(settings), _schema(std::move(schema)), _rank(rank),
          _policy(UpdatePolicyRegistry::Resolve(policy, requirements))
    {
        DNDS_check_throw_info(_rank >= 0,
                              "nonlinear controller needs a domain rank");
    }

    NonlinearIterationReport NonlinearController::Execute(
        int stage, NonlinearProblem problem) const
    {
        return _policy->Execute(
            {&_settings, &_schema, _rank, stage, std::move(problem)});
    }
}
