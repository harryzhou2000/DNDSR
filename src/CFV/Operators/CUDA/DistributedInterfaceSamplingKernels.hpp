#pragma once
#include "CFV/Detail/DistributedInterfaceSamplingView.hpp"

namespace DNDS::CFV::CUDA
{
    /// Enqueue one thread per donor query. The caller has validated owned-cell
    /// indices at setup and retained all operands through stream completion.
    /// This launcher validates whole-array residency and parameter size once.
    void EvaluateDirectDonors(Runtime::ExecutionContext, Detail::DirectDonorSamplingView);
}
