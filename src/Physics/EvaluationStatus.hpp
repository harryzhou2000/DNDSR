#pragma once

#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Physics
{
    /// Stable detail codes for numerical property-provider failures.
    enum class EvaluationFailure
    {
        None,
        Request,
        Composition,
        TemperatureInversion,
        Properties,
        ChemicalSource
    };

    DNDS_DEVICE_CALLABLE inline NumericalStatus EvaluationError(NumericalError code, EvaluationFailure detail) noexcept
    {
        return {code, int(detail)};
    }

    /// Only a reported thermodynamic failure permits retry from a valid mean.
    /// Invalid contracts, resource failures and unexpected exceptions cannot
    /// be hidden by reconstruction fallback.
    DNDS_DEVICE_CALLABLE inline bool AllowsMeanFallback(NumericalStatus status) noexcept
    {
        return (status.code == NumericalError::NoConvergence &&
                status.detail == int(EvaluationFailure::TemperatureInversion)) ||
               (status.code == NumericalError::InvalidState &&
                status.detail == int(EvaluationFailure::Properties));
    }
}
