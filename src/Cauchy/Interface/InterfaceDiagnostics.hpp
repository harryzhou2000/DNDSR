#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Cauchy
{
    /// Global point-coupling diagnostics. Donor changes count query rematches
    /// between successive posed maps, not distinct queries; non-monotone stage
    /// times and rejected trials remain part of this observed work. Map builds
    /// count directed collective preparations once, not once per MPI rank.
    struct InterfaceDiagnostics
    {
        index queryPoints = 0, changedDonors = 0;
        uint64_t mapBuilds = 0;
        real maximumGap = 0;
    };
}
