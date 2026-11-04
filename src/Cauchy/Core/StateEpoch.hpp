#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Cauchy
{
    /// Logical source identity, independent of physical arrays and transport.
    /// Array/geometry versions must additionally be recorded by packet owners.
    struct StateEpoch
    {
        uint64_t acceptedStep = 0, trial = 0, stage = 0, couplingIteration = 0;
        uint64_t geometry = 0, schema = 0;
        /// Unknown within one nonlinear problem (DITR: intermediate/end).
        /// stage remains the serialized DIRK stage; these are not interchangeable.
        uint64_t coupledStage = 0;
    };
}
