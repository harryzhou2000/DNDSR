#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace DNDS::Cauchy
{
    /// Canonical shell-safe descriptor for one opted-in boundary integration zone.
    struct BoundaryIntegrationSpec
    {
        int zone = 0;
        std::string rawName;
        std::string canonicalName;
    };

    /// Convert a raw mesh BC name into a shell-safe, cross-platform ASCII token.
    /// Falls back to "zone_<id>" if empty after sanitization.
    std::string CanonicalizeBoundaryName(std::string_view raw, int zone);

    /// Resolve all opted-in (integrateFlux == true) boundaries in deterministic
    /// zone-ID order with case-insensitive collision-free canonical names.
    std::vector<BoundaryIntegrationSpec> ResolveBoundaryIntegrationSpecs(const DomainSettings &settings);

    /// True if any configured boundary zone opts into flux integration.
    bool HasIntegratedBoundaries(const DomainSettings &settings);
}
