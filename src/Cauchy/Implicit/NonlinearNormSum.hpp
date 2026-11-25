#pragma once

#include "Cauchy/Core/NumericalReductions.hpp"

#ifdef DNDS_USE_CUDA
#    include "Cauchy/Implicit/CUDA/NonlinearKernels.hpp"
#endif

namespace DNDS::Cauchy
{
    /// Persistent collective slot for one residual observation. Submission
    /// order is collective. Complete the observation before reusing the slot.
    class NonlinearNormSum
    {
        Runtime::Array<real> _local, _sum;
        Runtime::ExecutionStream _lane, _resultLane;
        std::shared_ptr<const Runtime::GlobalReductionBinding> _binding;
        Runtime::StreamEvent _completion;

    public:
#ifdef DNDS_USE_CUDA
        /// The reduction producer writes directly into this slot. Its compact
        /// status survives subsequent operators on the same compute lane.
        Runtime::Array<CUDA::NonlinearSummary> deviceSummary;
#endif
        NonlinearNormSum(index count, const NumericalReductions &reductions, int ranks, const Runtime::ExecutionStream &compute, const Runtime::ExecutionStream &communication);
        Runtime::Array<real> &Local() { return _local; }
        /// The local producer has already written Local() on the compute lane.
        /// Ordinary MPI consumes staged host memory through controller progress.
        void Submit(Runtime::EagerEngine &, const Runtime::ExecutionStream &producer);
        std::vector<real> Observe() const;
        /// Order a device/worker consumer after the reduction. Ordinary MPI is
        /// progressed and retired on the controller before transferring back.
        const Runtime::Array<real> &Consume(Runtime::EagerEngine &, const Runtime::ExecutionStream &);
    };
}
