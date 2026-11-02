#pragma once

#include "DNDS/MPI/MPI.hpp"
#include <cmath>

namespace DNDS::Cauchy
{
    enum class AlphaReduction
    {
        Cell,
        MinimumLimited,
        MinimumAll
    };
    struct AlphaStatistics
    {
        index limited = 0;
        real minimum = 1;
    };
    /// Collective scalar agreement for Euler RHS-alpha policies. Count only
    /// original alpha<1 rows, including for MinimumAll. Empty owners pass 0/1.
    /// Callers enter on the same live communicator, in the same collective
    /// order on an MPI-permitted thread; summed counts must fit index. Scope agreement and bad
    /// local statistics are checked within the two required reductions.
    inline AlphaStatistics AgreeAlphaStatistics(index limited, real minimum, AlphaReduction scope, MPI_Comm comm)
    {
        const int code = int(scope);
        const bool valid = limited >= 0 && std::isfinite(minimum) && minimum >= 0 && minimum <= 1 &&
                           (limited ? minimum < 1 : minimum == 1) && code >= 0 && code <= 2;
        index counts[2]{limited, valid ? 0 : 1};
        MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT64_T, MPI_SUM, comm);
        DNDS_check_throw_info(!counts[1] && counts[0] >= 0, "invalid distributed positivity statistics");
        real minima[3]{minimum, real(code), -real(code)};
        MPI_Allreduce(MPI_IN_PLACE, minima, 3, MPI_DOUBLE, MPI_MIN, comm);
        DNDS_check_throw_info(minima[1] == -minima[2], "positivity reduction policy differs between ranks");
        return {counts[0], minima[0]};
    }
    /// Pure row selection after successful agreement. No row-mask lookup or
    /// communication; MinimumLimited leaves every originally unmarked row at1.
    DNDS_DEVICE_CALLABLE inline real SelectAlpha(real cell, real minimum, AlphaReduction scope)
    {
        return scope == AlphaReduction::Cell || (scope == AlphaReduction::MinimumLimited && cell == 1) ? cell : minimum;
    }
}
