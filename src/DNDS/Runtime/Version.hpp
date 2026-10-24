#pragma once

#include "DNDS/Errors.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

namespace DNDS::Runtime
{
    enum class Change : uint32_t
    {
        Topology = 1,
        Geometry = 2,
        Distribution = 4,
        Layout = 8,
        Allocation = 16,
        Interface = 32,
        Values = 64,
        All = 127
    };
    inline Change operator|(Change a, Change b) { return Change(uint32_t(a) | uint32_t(b)); }

    struct VersionState
    {
        std::array<uint64_t, 7> generations{};
        std::mutex mutex;
        size_t readers = 0;
        bool writer = false;
        bool failed = false;
    };

    /// A cache dependency stamp retains identity but does not pin mesh data.
    /// Execution must acquire a ReadLease as well before using mutable storage.
    class VersionStamp
    {
        friend class VersionTracker;
        std::shared_ptr<VersionState> _state;
        std::array<uint64_t, 7> _generations{};
        Change _dependencies = Change::All;

    public:
        bool Valid() const
        {
            if (!_state)
                return false;
            std::lock_guard<std::mutex> lock(_state->mutex);
            if (_state->writer || _state->failed)
                return false;
            for (size_t i = 0; i < _generations.size(); ++i)
                if ((uint32_t(_dependencies) & (1u << i)) && _generations[i] != _state->generations[i])
                    return false;
            return true;
        }
        void Validate() const { DNDS_check_throw_info(Valid(), "stale, mutating or failed version dependency"); }
        uint64_t Generation(Change component) const
        {
            for (size_t i = 0; i < _generations.size(); ++i)
                if (uint32_t(component) == (1u << i))
                    return _generations[i];
            DNDS_check_throw_info(false, "generation query requires one change component");
            return 0;
        }
    };

    class VersionTracker
    {
        std::shared_ptr<VersionState> _state = std::make_shared<VersionState>();

    public:
        class ReadLease
        {
            friend class VersionTracker;
            std::shared_ptr<VersionState> _state;
            explicit ReadLease(std::shared_ptr<VersionState> state) : _state(std::move(state))
            {
                std::lock_guard<std::mutex> lock(_state->mutex);
                DNDS_check_throw_info(!_state->writer && !_state->failed, "cannot read a mutating or failed mesh");
                ++_state->readers;
            }

        public:
            ReadLease(const ReadLease &) = delete;
            ReadLease &operator=(const ReadLease &) = delete;
            ReadLease(ReadLease &&) noexcept = default;
            ReadLease &operator=(ReadLease &&) = delete;
            ~ReadLease()
            {
                if (_state)
                {
                    std::lock_guard<std::mutex> lock(_state->mutex);
                    --_state->readers;
                }
            }
        };

        class Mutation
        {
            friend class VersionTracker;
            std::shared_ptr<VersionState> _state;
            Change _changes;
            explicit Mutation(std::shared_ptr<VersionState> state, Change changes)
                : _state(std::move(state)), _changes(changes)
            {
                DNDS_check_throw_info(uint32_t(changes) && !(uint32_t(changes) & ~uint32_t(Change::All)), "invalid mutation mask");
                std::lock_guard<std::mutex> lock(_state->mutex);
                DNDS_check_throw_info(!_state->writer && !_state->readers && !_state->failed, "mesh mutation requires exclusive valid state");
                _state->writer = true;
            }

            void Finish(bool committed)
            {
                auto state = std::move(_state);
                if (!state)
                    return;
                std::lock_guard<std::mutex> lock(state->mutex);
                for (size_t i = 0; i < state->generations.size(); ++i)
                    if (uint32_t(_changes) & (1u << i))
                        ++state->generations[i];
                state->failed = !committed;
                state->writer = false;
            }

        public:
            Mutation(const Mutation &) = delete;
            Mutation &operator=(const Mutation &) = delete;
            Mutation(Mutation &&) noexcept = default;
            Mutation &operator=(Mutation &&) = delete;
            ~Mutation() { Finish(false); }
            void IncludeChanges(Change changes)
            {
                DNDS_check_throw_info(_state && uint32_t(changes) && !(uint32_t(changes) & ~uint32_t(Change::All)), "invalid nested mutation mask");
                _changes = _changes | changes;
            }
            void Commit()
            {
                DNDS_check_throw(_state);
                Finish(true);
            }
        };

        VersionTracker() = default;
        // Legacy mesh copies may share adjacency allocations. Copies therefore
        // share their mutation controller too; construct fresh trackers only for
        // independently built/deep-copied mesh data.
        VersionTracker(const VersionTracker &) = default;
        VersionTracker &operator=(const VersionTracker &other)
        {
            if (_state != other._state)
            {
                auto mutation = BeginMutation(Change::All);
                mutation.Commit();
                _state = other._state;
            }
            return *this;
        }
        VersionStamp Stamp(Change dependencies = Change::All) const
        {
            DNDS_check_throw_info(uint32_t(dependencies) && !(uint32_t(dependencies) & ~uint32_t(Change::All)), "invalid dependency mask");
            std::lock_guard<std::mutex> lock(_state->mutex);
            DNDS_check_throw_info(!_state->writer && !_state->failed, "cannot snapshot a mutating or failed mesh");
            VersionStamp stamp;
            stamp._state = _state;
            stamp._generations = _state->generations;
            stamp._dependencies = dependencies;
            return stamp;
        }
        ReadLease Read() const { return ReadLease(_state); }
        bool SameController(const VersionTracker &other) const { return _state == other._state; }
        Mutation BeginMutation(Change changes) { return Mutation(_state, changes); }
        bool CanMutate() const
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            return !_state->readers && !_state->writer && !_state->failed;
        }
    };
}
