#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"

#include <optional>

namespace DNDS::Cauchy
{
    /// Immutable frontend output policy. Numerical operators publish typed
    /// reports; this plan alone resolves cadence inheritance, names and exact
    /// physical-time endpoints.
    class OutputPlan
    {
        OutputSettings _settings;
        index _domains = 0;
        index _visualizationInterval = 0;
        real _visualizationTimeInterval = 0;

    public:
        struct StepTarget
        {
            real dt = 0;
            std::optional<real> exactTime;
        };

        OutputPlan(OutputSettings settings, index domains);
        const OutputSettings &Settings() const { return _settings; }
        index DomainNumber() const { return _domains; }
        index VisualizationInterval() const { return _visualizationInterval; }
        real VisualizationTimeInterval() const { return _visualizationTimeInterval; }
        std::string DomainPrefix(index domain) const;
        std::string HistoryBare(index domain) const;
        std::string ResolvedPath() const;
        std::string SummaryPath() const;
        std::string RunManifestPath() const;
        std::string CheckpointPath(const std::string &token) const;
        std::string VisualizationPrefix(index domain, const std::string &token) const;
        std::string SnapshotManifestPath(const std::string &token) const;
        static std::string StepToken(uint64_t step);
        static std::string TimeToken(real time);
        static uint64_t NextTimeOrdinal(real time, real interval);
        static real TimeAt(uint64_t ordinal, real interval);
        StepTarget SnapStep(real time, real proposedDt, real tEnd,
                            uint64_t checkpointOrdinal, uint64_t visualizationOrdinal) const;
    };
}
