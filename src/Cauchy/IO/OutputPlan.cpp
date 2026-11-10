#include "Cauchy/IO/OutputPlan.hpp"

#include "DNDS/Core/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace DNDS::Cauchy
{
    OutputPlan::OutputPlan(OutputSettings settings, index domains)
        : _settings(std::move(settings)), _domains(domains),
          _visualizationInterval(_settings.visualizationInterval < 0
                                     ? _settings.checkpointInterval
                                     : _settings.visualizationInterval),
          _visualizationTimeInterval(_settings.visualizationTimeInterval < 0
                                         ? _settings.checkpointTimeInterval
                                         : _settings.visualizationTimeInterval)
    {
        DNDS_check_throw_info(_domains > 0 && !_settings.prefix.empty() &&
                                  _settings.checkpointInterval >= 0 && _visualizationInterval >= 0 &&
                                  std::isfinite(_settings.checkpointTimeInterval) && _settings.checkpointTimeInterval >= 0 &&
                                  std::isfinite(_visualizationTimeInterval) && _visualizationTimeInterval >= 0 &&
                                  std::isfinite(_settings.endpointSnapRelativeTolerance) &&
                                  _settings.endpointSnapRelativeTolerance >= 0,
                              "invalid resolved Cauchy output plan");
    }
    std::string OutputPlan::DomainPrefix(index domain) const
    {
        DNDS_check_throw_info(domain >= 0 && domain < _domains, "output domain index is out of range");
        return _settings.prefix + "-D" + std::to_string(domain);
    }
    std::string OutputPlan::HistoryBare(index domain) const { return DomainPrefix(domain) + "-history"; }
    std::string OutputPlan::ResolvedPath() const { return _settings.prefix + "-resolved.json"; }
    std::string OutputPlan::SummaryPath() const { return _settings.prefix + "-summary.json"; }
    std::string OutputPlan::RunManifestPath() const { return _settings.prefix + "-run-manifest.json"; }
    std::string OutputPlan::CheckpointPath(const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(), "checkpoint token is empty");
        return _settings.prefix + (_domains == 1 ? "-D0" : "") + token + ".h5";
    }
    std::string OutputPlan::VisualizationPrefix(index domain, const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(), "visualization token is empty");
        return DomainPrefix(domain) + token;
    }
    std::string OutputPlan::SnapshotManifestPath(const std::string &token) const
    {
        DNDS_check_throw_info(!token.empty(), "snapshot token is empty");
        return _settings.prefix + token + "-manifest.json";
    }
    std::string OutputPlan::StepToken(uint64_t step) { return "-S" + std::to_string(step); }
    std::string OutputPlan::TimeToken(real time)
    {
        DNDS_check_throw_info(std::isfinite(time), "output time is nonfinite");
        std::ostringstream text;
        text.imbue(std::locale::classic());
        text << "-T" << std::setprecision(std::numeric_limits<real>::max_digits10) << time;
        return text.str();
    }
    uint64_t OutputPlan::NextTimeOrdinal(real time, real interval)
    {
        if (!(interval > 0))
            return 0;
        DNDS_check_throw_info(std::isfinite(time) && time >= 0 && std::isfinite(interval),
                              "invalid output-time cursor");
        const auto completed = std::floor(time / interval);
        DNDS_check_throw_info(completed >= 0 && completed < real(std::numeric_limits<uint64_t>::max() - 1),
                              "Cauchy output-time ordinal overflow");
        return uint64_t(completed) + 1;
    }
    real OutputPlan::TimeAt(uint64_t ordinal, real interval)
    {
        DNDS_check_throw_info(std::isfinite(interval) && interval > 0, "invalid output-time interval");
        const real result = real(ordinal) * interval;
        DNDS_check_throw_info(std::isfinite(result), "Cauchy output time overflow");
        return result;
    }
    OutputPlan::StepTarget OutputPlan::SnapStep(real time, real proposedDt, real tEnd,
                                                uint64_t checkpointOrdinal, uint64_t visualizationOrdinal) const
    {
        DNDS_check_throw_info(std::isfinite(time) && std::isfinite(proposedDt) && proposedDt > 0 &&
                                  std::isfinite(tEnd) && tEnd >= 0,
                              "invalid physical time or proposed timestep");
        real target = std::numeric_limits<real>::infinity();
        if (tEnd > 0)
            target = tEnd;
        if (_settings.checkpointTimeInterval > 0)
            target = std::min(target, TimeAt(checkpointOrdinal, _settings.checkpointTimeInterval));
        if (_visualizationTimeInterval > 0)
            target = std::min(target, TimeAt(visualizationOrdinal, _visualizationTimeInterval));
        const real remaining = target - time;
        const real tolerance = _settings.endpointSnapRelativeTolerance * proposedDt;
        if (std::isfinite(target) && remaining > 0 && remaining <= proposedDt + tolerance)
            return {remaining, target};
        DNDS_check_throw_info(time + proposedDt > time,
                              "Cauchy physical clock cannot represent the proposed forward step");
        return {proposedDt, std::nullopt};
    }
}
