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
        /// Resolve and retain one method-compatible update policy. Construction
        /// performs no nonlinear evaluation or MPI communication.
        NonlinearController(const ImplicitSettings &settings,
                            EquationSchema schema, int rank,
                            const std::string &policy,
                            const UpdatePolicyRequirements &requirements);
        /// Run the common convergence/CFL/cap loop around the supplied problem.
        /// The problem owns operator calls and their collective ordering.
        NonlinearIterationReport Execute(int stage,
                                         NonlinearProblem problem) const;
        /// Immutable registered policy identity stored in checkpoint metadata.
        const UpdatePolicyDescriptor &Policy() const
        {
            return _policy->Descriptor();
        }
    };
}
