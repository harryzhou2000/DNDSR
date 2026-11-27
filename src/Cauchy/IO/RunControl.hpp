#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/IO/CsvLog.hpp"

#include <atomic>
#include <deque>
#include <mutex>

namespace DNDS::Cauchy
{
    class OutputCoordinator;

    /// Process-local command and query API. It owns only host-side copies,
    /// so callers may query or enqueue from a thread outside the solver.
    class RunControl
    {
        friend class OutputCoordinator;
        struct HistoryEntry
        {
            index domain;
            CsvRow row;
        };
        const Json _configuration;
        const int _rank;
        std::atomic<bool> _stop{false};
        mutable std::mutex _historyMutex;
        std::deque<HistoryEntry> _history;

        void RetainHistory(index domain, const CsvRow &row);

    public:
        static constexpr size_t HistoryCapacity = 256;
        RunControl(const SimulationConfig &configuration, int rank)
            : _configuration(configuration.Root()), _rank(rank) {}

        /// Coalesce repeated requests. The solver collectively acts at its
        /// restartable boundary. This call itself performs no MPI or I/O.
        void EnqueueStop() noexcept { _stop.store(true, std::memory_order_relaxed); }
        bool StopRequested() const noexcept { return _stop.load(std::memory_order_relaxed); }
        Json CurrentConfiguration() const { return _configuration; }
        /// Up to limit most recent rows written by domain roots on this rank,
        /// in publication order, with the current retained count and capacity.
        Json CurrentHistory(size_t limit = HistoryCapacity) const;
    };
}
