#pragma once

#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <cstdio>

namespace DNDS::Cauchy
{
    /// Controller-side policy, never a kernel helper. A reported numerical
    /// failure cannot publish partial outputs. Production aborts using only
    /// local diagnostics; coordinated tests may observe an exception instead.
    inline void RequireNumericalSuccess(NumericalStatus status, const char *operation)
    {
        if (status)
            return;
        std::fprintf(stderr, "Cauchy %s failed: code=%d detail=%d row=%lld point=%lld\n",
                     operation, int(status.code), status.detail,
                     static_cast<long long>(status.row), static_cast<long long>(status.point));
        MPI::AbortIfSolverFailure("Cauchy numerical kernel failed");
        DNDS_check_throw_info(false, operation);
    }

    /// Unwrap only at the operator/controller boundary, after checking status.
    template <class T>
    T RequireNumericalSuccess(NumericalResult<T> result, const char *operation)
    {
        RequireNumericalSuccess(result.status, operation);
        return std::move(result.value);
    }
}
