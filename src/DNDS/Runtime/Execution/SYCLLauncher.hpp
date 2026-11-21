#pragma once

#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Runtime
{
    /// Create an in-order USM lane in the Runtime context for this device.
    /// Launch callbacks submit immediately. A recorded prefix owns operands
    /// and retirements until completion is observed by the engine.
    std::shared_ptr<StreamLauncher> CreateSYCLLauncher(Device device);
}
