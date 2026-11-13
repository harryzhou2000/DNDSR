#pragma once
#include "Cauchy/Interface/InterfacePacketView.hpp"

namespace DNDS::Cauchy::CUDA
{
    /// One thread per unique receiver query. Combined AoS only; wrapper checks
    /// residency/shape once. Array and metadata ownership stay in the operator.
    void FillInterfacePacket(Runtime::ExecutionContext, InterfacePacketTableView);
}
