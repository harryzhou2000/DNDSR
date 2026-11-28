#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/IO/CsvLog.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace DNDS::Cauchy
{
    class OutputCoordinator;

    enum class ControlBoundary : int
    {
        Auto = 0,
        Step = 1,
        Iteration = 2
    };

    enum class LiveConfigCategory : int
    {
        Output = 0,
        Steering = 1
    };

    const char *ToString(ControlBoundary boundary) noexcept;
    const char *ToString(LiveConfigCategory category) noexcept;
    std::optional<ControlBoundary> ParseControlBoundary(std::string_view text) noexcept;

    struct LiveConfigRequest
    {
        uint64_t revision = 0;
        ControlBoundary applyAt = ControlBoundary::Step;
        LiveConfigCategory category = LiveConfigCategory::Output;
        Json patch = Json::object(); // Map of canonical JSON pointer -> value
    };

    struct LiveSnapshotRequest
    {
        uint64_t id = 0;
        ControlBoundary applyAt = ControlBoundary::Step;
        bool checkpoint = true;
        bool visualization = true;
    };

    struct ControlUpdateRecord
    {
        uint64_t revision = 0;
        std::string kind;     // "config", "stop", "snapshot"
        std::string category; // "output", "steering"
        std::string applyAt;  // "auto", "step", "iteration"
        std::string state;    // "pending", "applied", "rejected"
        std::string message;
        Json patch = Json::object();
        uint64_t appliedStep = 0;
        index appliedStage = -1;
        index appliedIteration = -1;

        Json ToJson() const;
    };

    /// Process-local command and query API. It owns only host-side copies,
    /// so callers may query or enqueue from a thread outside the solver.
    class RunControl
    {
        friend class OutputCoordinator;
        struct HistoryEntry
        {
            uint64_t seq = 0;
            index domain = 0;
            CsvRow row;
        };

        mutable std::mutex _mutex;
        Json _configuration;
        const int _rank;
        const index _domainCount;
        std::atomic<bool> _stop{false};
        std::atomic<int> _stopBoundary{int(ControlBoundary::Auto)};

        uint64_t _nextRevision = 1;
        uint64_t _appliedConfigRevision = 0;
        std::deque<LiveConfigRequest> _pendingConfigs;
        std::deque<LiveSnapshotRequest> _pendingSnapshots;
        std::deque<ControlUpdateRecord> _updates;

        Json _status = Json::object();
        Json _boundarySpecs = Json::object();
        Json _boundaryLatest = Json::object();

        mutable std::mutex _historyMutex;
        uint64_t _nextHistorySeq = 1;
        std::deque<HistoryEntry> _history;

        void RetainHistory(index domain, const CsvRow &row);
        void RecordUpdateLocked(ControlUpdateRecord record);
        void TransitionUpdateLocked(uint64_t revision, const std::string &kind,
                                    const std::string &state, const std::string &message,
                                    uint64_t step, index stage, index iteration);

    public:
        static constexpr size_t HistoryCapacity = 256;
        static constexpr size_t UpdateHistoryCapacity = 128;

        RunControl(const SimulationConfig &configuration, int rank);

        /// Metadata describing all allow-listed runtime-mutable parameters,
        /// explicitly separated into "output" vs "steering" categories.
        static Json MutableConfigSchema(index domainCount);

        /// Validate and normalize a live config patch (flat JSON pointers or
        /// nested objects). Returns {true, ""} on success and populates
        /// outCategory and outNormalizedPatch (canonical pointer -> value).
        static std::pair<bool, std::string> ValidatePatch(const Json &currentConfig,
                                                          const Json &patch,
                                                          LiveConfigCategory &outCategory,
                                                          Json &outNormalizedPatch);

        /// Apply a validated canonical pointer map to a resolved config JSON.
        static Json ApplyNormalizedPatchToJson(const Json &currentConfig,
                                               const Json &normalizedPatch);

        /// Coalesce repeated stop requests. The solver collectively acts at the
        /// requested boundary. Performs no MPI or I/O.
        void EnqueueStop(ControlBoundary boundary = ControlBoundary::Auto);
        bool StopRequested() const noexcept { return _stop.load(std::memory_order_relaxed); }
        ControlBoundary StopBoundary() const noexcept
        {
            return ControlBoundary(_stopBoundary.load(std::memory_order_relaxed));
        }
        bool StopRequested(ControlBoundary boundary, bool steady) const noexcept;

        /// Validate against current config and enqueue a live config patch.
        std::pair<bool, ControlUpdateRecord> EnqueueConfigPatch(
            const Json &patch, ControlBoundary applyAt = ControlBoundary::Step);

        /// Enqueue an on-demand checkpoint and/or visualization snapshot.
        ControlUpdateRecord EnqueueSnapshot(bool checkpoint, bool visualization,
                                            ControlBoundary applyAt = ControlBoundary::Step);

        /// Solver-thread drains for collective broadcast and application.
        std::vector<LiveConfigRequest> TakePendingConfigPatches(ControlBoundary boundary);
        std::optional<LiveSnapshotRequest> TakePendingSnapshot(ControlBoundary boundary);

        void RecordAppliedConfig(uint64_t revision, const Json &newConfiguration,
                                 uint64_t step, index stage, index iteration);
        void RecordRejectedConfig(uint64_t revision, const std::string &reason,
                                  uint64_t step, index stage, index iteration);
        void RecordAppliedSnapshot(uint64_t id, uint64_t step, index stage, index iteration);
        void RecordStopApplied(uint64_t step, index stage, index iteration,
                               const std::string &boundaryName);

        void UpdateStatus(const Json &status);
        void SetBoundaryMetadata(index domain, const Json &specs);
        void UpdateBoundaryLatest(index domain, int zone, const Json &row);

        int Rank() const noexcept { return _rank; }
        index DomainCount() const noexcept { return _domainCount; }
        Json CurrentConfiguration() const;
        Json CurrentStatus() const;
        Json CurrentBoundaries() const;
        Json RecentUpdates() const;
        Json MutableSchema() const { return MutableConfigSchema(_domainCount); }

        /// Up to limit most recent rows written by domain roots on this rank,
        /// in publication order, with the current retained count and capacity.
        Json CurrentHistory(size_t limit = HistoryCapacity) const;
        Json CurrentHistory(size_t limit, uint64_t afterSeq,
                            std::optional<index> domainFilter) const;
    };
}
