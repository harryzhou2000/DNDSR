#pragma once

#include "Physics/Chemistry/ChemicalSource.hpp"

/// Source-compatible historical include/namespace. The backend implementation,
/// species views and formatter now have one owner in the shared Physics module.
namespace DNDS::Euler
{
    namespace Chemistry = DNDS::Physics::Chemistry;
}
