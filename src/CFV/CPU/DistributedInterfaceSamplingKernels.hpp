#pragma once

#include "CFV/Detail/DistributedInterfaceSamplingView.hpp"

namespace DNDS::CFV::CPU
{
    /// Independent query rows; no allocation, leases or communication in the loop.
    void EvaluateDirectDonors(const Detail::DirectDonorSamplingView &request);
}
