#pragma once

#include "Cauchy/Implicit/LinearProblem.hpp"
#include "DNDS/Algebra/AmgXPreconditioner.hpp"

namespace DNDS::Cauchy
{
    /** @brief Rank-local AmgX adapter for the coupled assembled operator.
     * @details One device, no communicator and no distributed AmgX machinery.
     * Requires a primary int32 pattern and compact row-major blocks with a
     * separate diagonal. For a partition with ghosts, upload uses the owned
     * principal CSR and its geometric weights. Its slot map is reused to
     * gather device blocks on matrix refresh. The distributed operator keeps
     * its original ghost columns. Single-rank native storage
     * instead attaches one allocation shared by native diagonal/off-diagonal
     * aliases, retaining its owner and checking versions across provider calls.
     * The aggregation graph uses one immutable area/distance strength per
     * off-diagonal block. Its pattern and strengths are attached together.
     * Pattern preparation is once per solver; values update when the frozen
     * matrix changes. Provider cycles are fixed, zero-started and linear.
     * Update/Apply drain the supplied engine before entering the provider:
     * its device-wide join must not depend on host MPI progress blocked here.
     * Callers serialize use and keep the matrix unchanged throughout Solve.
     */
    class LocalAMGPreconditioner
    {
        struct State;
        std::shared_ptr<State> _state;

    public:
        explicit LocalAMGPreconditioner(const LinearProblemContext &, Algebra::AmgXSettings settings = {});
        void Update(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const LinearProblemContext &);
        void Apply(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                   const Runtime::Array<real> &rhs, Runtime::Array<real> &output);
        /// Explicit diagnostic join; not part of the production iteration schedule.
        Algebra::AmgXMemorySnapshot MemorySnapshot() const;
    };
}
