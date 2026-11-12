#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::CFV::CPU
{
    /// Evaluate a face basis using only frozen views and caller-owned scratch.
    /// Cached and cache-off paths preserve the same central Taylor convention.
    Math::ReconstructionBasisView EvaluateFaceBasis(
        const ReconstructionBasisEvaluationView &input,
        index face, int side, index point, int derivativeOrder,
        Math::SmallMatrix &scratch);
}
