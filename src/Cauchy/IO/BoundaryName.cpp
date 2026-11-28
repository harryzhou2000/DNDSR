#include "Cauchy/IO/BoundaryName.hpp"

#include "DNDS/Core/Errors.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <set>

namespace DNDS::Cauchy
{
    namespace
    {
        constexpr size_t MaxCanonicalLength = 64;

        std::string ToLowerAscii(std::string_view text)
        {
            std::string lower;
            lower.reserve(text.size());
            for (const unsigned char ch : text)
                lower.push_back(char(std::tolower(ch)));
            return lower;
        }

        bool IsWindowsReservedName(std::string_view token)
        {
            const auto dot = token.find('.');
            const auto stem = ToLowerAscii(dot == std::string_view::npos ? token : token.substr(0, dot));
            static constexpr std::array<std::string_view, 22> Reserved{
                "con", "prn", "aux", "nul",
                "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
                "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"};
            return std::find(Reserved.begin(), Reserved.end(), stem) != Reserved.end();
        }

        std::string TrimEnds(std::string value)
        {
            const auto trimChar = [](char c)
            { return c == '_' || c == '.' || c == '-'; };
            size_t begin = 0;
            while (begin < value.size() && trimChar(value[begin]))
                ++begin;
            size_t end = value.size();
            while (end > begin && trimChar(value[end - 1]))
                --end;
            return value.substr(begin, end - begin);
        }
    }

    std::string CanonicalizeBoundaryName(std::string_view raw, int zone)
    {
        std::string out;
        out.reserve(std::min(raw.size(), MaxCanonicalLength));
        bool lastUnderscore = false;
        for (const unsigned char ch : raw)
        {
            const bool safe = (ch >= 'A' && ch <= 'Z') ||
                              (ch >= 'a' && ch <= 'z') ||
                              (ch >= '0' && ch <= '9') ||
                              ch == '.' || ch == '-' || ch == '_';
            if (safe)
            {
                if (ch == '_')
                {
                    if (!lastUnderscore)
                        out.push_back('_');
                    lastUnderscore = true;
                }
                else
                {
                    out.push_back(char(ch));
                    lastUnderscore = false;
                }
            }
            else if (!lastUnderscore)
            {
                out.push_back('_');
                lastUnderscore = true;
            }
        }
        out = TrimEnds(std::move(out));
        if (out.size() > MaxCanonicalLength)
            out = TrimEnds(out.substr(0, MaxCanonicalLength));
        if (out.empty())
            return "zone_" + std::to_string(zone);
        if (IsWindowsReservedName(out))
            out = "bc_" + out;
        return out;
    }

    bool HasIntegratedBoundaries(const DomainSettings &settings)
    {
        for (const auto &[zone, condition] : settings.boundaries)
        {
            (void)zone;
            if (condition.integrateFlux)
                return true;
        }
        return false;
    }

    std::vector<BoundaryIntegrationSpec> ResolveBoundaryIntegrationSpecs(const DomainSettings &settings)
    {
        std::vector<int> zones;
        for (const auto &[zoneKey, condition] : settings.boundaries)
            if (condition.integrateFlux)
                zones.push_back(std::stoi(zoneKey));
        std::sort(zones.begin(), zones.end());
        zones.erase(std::unique(zones.begin(), zones.end()), zones.end());

        std::vector<BoundaryIntegrationSpec> specs;
        specs.reserve(zones.size());
        for (const int zone : zones)
        {
            std::string raw;
            for (const auto &[name, id] : settings.mesh.boundaryIds)
                if (id == zone && (raw.empty() || name < raw))
                    raw = name;
            specs.push_back({zone, raw, CanonicalizeBoundaryName(raw, zone)});
        }

        // Count case-insensitive collisions across all opted-in zones so colliding
        // names are deterministically disambiguated with _z<zone>.
        std::map<std::string, int> counts;
        for (const auto &spec : specs)
            ++counts[ToLowerAscii(spec.canonicalName)];

        std::set<std::string> used;
        for (auto &spec : specs)
        {
            std::string base = spec.canonicalName;
            if (counts[ToLowerAscii(base)] > 1)
            {
                const std::string suffix = "_z" + std::to_string(spec.zone);
                if (base.size() + suffix.size() > MaxCanonicalLength)
                    base = TrimEnds(base.substr(0, MaxCanonicalLength - suffix.size()));
                if (base.empty())
                    base = "zone";
                base += suffix;
            }
            std::string candidate = base;
            int attempt = 2;
            while (!used.insert(ToLowerAscii(candidate)).second)
                candidate = base + "_" + std::to_string(attempt++);
            spec.canonicalName = std::move(candidate);
        }
        return specs;
    }
}
