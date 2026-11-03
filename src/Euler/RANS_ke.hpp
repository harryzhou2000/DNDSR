#pragma once

#include "Euler/Euler.hpp"
#include "Physics/RANS/RANSModels.hpp"

/// Preserve the historical public path and namespace, including its Euler
/// declarations. New consumers include the independent Physics header directly.
namespace DNDS::Euler
{
    namespace RANS = DNDS::Physics::RANS;
}
