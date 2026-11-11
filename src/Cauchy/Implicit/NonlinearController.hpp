#pragma once

#include "Cauchy/Implicit/UpdatePolicy.hpp"

namespace DNDS::Cauchy
{
    /// Prepared nonlinear orchestration boundary shared by ordinary stages and
    /// coupled DITR. It freezes implicit controls, equation schema, rank and the
    /// method-compatible update policy once; each Execute supplies only the
    /// current operator callbacks and live controller state.
    class NonlinearController
    {
        ImplicitSettings _settings;
        EquationSchema _schema;
        int _rank = 0;
        std::unique_ptr<UpdatePolicy> _policy;

    public:
        NonlinearController(const ImplicitSettings &settings,
                            EquationSchema schema, int rank,
                            const std::string &policy,
                            const UpdatePolicyRequirements &requirements);
        NonlinearIterationReport Execute(int stage,
                                         NonlinearProblem problem) const;
        const UpdatePolicyDescriptor &Policy() const
        {
            return _policy->Descriptor();
        }
    };
}
