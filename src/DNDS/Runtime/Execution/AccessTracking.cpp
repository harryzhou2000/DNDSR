#include "DNDS/Runtime/Execution/AccessTracking.hpp"
#include <atomic>
#include <mutex>
#include <sstream>

namespace DNDS::Runtime
{
    namespace
    {
        struct Use
        {
            uint64_t lane = 0, position = 0;
            std::atomic<bool> completed{false};
        };
        using t_Use = std::shared_ptr<Use>;
        struct AllocationUses
        {
            std::weak_ptr<void> lifetime;
            t_Use writer;
            std::map<uint64_t, t_Use> readers;
        };
        struct Registry
        {
            std::mutex mutex;
            std::map<uint64_t, AllocationUses> allocations;
            uint64_t commands = 0;
        };
        Registry &GlobalRegistry()
        {
            static Registry registry;
            return registry;
        }
        void RequireOrder(const AccessOrder &frontier, const t_Use &use, uint64_t allocation)
        {
            if (!use || use->completed.load())
                return;
            const auto found = frontier.prefixes.find(use->lane);
            if (found != frontier.prefixes.end() && found->second >= use->position)
                return;
            std::ostringstream message;
            message << "unordered scheduled access to allocation " << allocation << "; missing predecessor lane "
                    << use->lane << " position " << use->position;
            DNDS_check_throw_info(false, message.str());
        }
    }
    struct AccessTrackingContext::State
    {
        struct Lane
        {
            // Lane issue is serialized just like a native stream. This is not
            // the registry lock: other lanes/allocations can submit concurrently.
            std::recursive_mutex issue;
            bool issuing = false;
            AccessOrder frontier;
        };
        std::mutex mutex;
        std::map<uint64_t, std::shared_ptr<Lane>> lanes;
        std::shared_ptr<Lane> Get(uint64_t id)
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto found = lanes.find(id);
            DNDS_check_throw_info(found != lanes.end(), "unknown checked execution lane");
            return found->second;
        }
    };
    struct AccessTrackingContext::Submission::State
    {
        t_Use use;
    };
    void AccessTrackingContext::Submission::Complete() const noexcept
    {
        if (_state)
            _state->use->completed = true;
    }
    AccessTrackingContext::AccessTrackingContext() : _state(std::make_shared<State>()) {}
    uint64_t AccessTrackingContext::NewLane()
    {
        static std::atomic<uint64_t> next{1};
        const auto id = next.fetch_add(1);
        DNDS_check_throw_info(id, "checked execution lane identity overflow");
        std::lock_guard<std::mutex> lock(_state->mutex);
        auto lane = std::make_shared<State::Lane>();
        lane->frontier.prefixes[id] = 0;
        _state->lanes.emplace(id, std::move(lane));
        return id;
    }
    void AccessTrackingContext::RetireLane(uint64_t id)
    {
        std::lock_guard<std::mutex> lock(_state->mutex);
        _state->lanes.erase(id);
    }
    void AccessTrackingContext::Submit(uint64_t id, const std::vector<ScheduledBufferAccess> &accesses,
                                       const std::function<void(Submission)> &accept, const std::function<void(bool)> &agree)
    {
        auto lane = _state->Get(id);
        std::lock_guard<std::recursive_mutex> issue(lane->issue);
        DNDS_check_throw_info(!lane->issuing, "recursive checked stream submission is unsupported");
        lane->issuing = true;
        struct Finish
        {
            bool &issuing;
            ~Finish() { issuing = false; }
        } finish{lane->issuing};
        auto state = std::make_shared<Submission::State>();
        state->use = std::make_shared<Use>();
        state->use->lane = id;
        state->use->position = lane->frontier.prefixes[id] + 1;
        DNDS_check_throw_info(state->use->position, "checked stream position overflow");
        // Fold aliases in the same command before checking. Whole allocation
        // accounting is deliberate; disjoint components alone are not regions.
        std::map<uint64_t, std::pair<bool, std::weak_ptr<void>>> declared;
        auto &registry = GlobalRegistry();
        std::map<uint64_t, AllocationUses> previous;
        std::exception_ptr invalid;
        try
        {
            for (const auto &access : accesses)
            {
                DNDS_check_throw_info(access.allocation && !access.lifetime.expired(), "checked access needs a live buffer");
                DNDS_check_throw_info(access.mode == AccessMode::Read || access.mode == AccessMode::Write || access.mode == AccessMode::ReadWrite,
                                      "invalid checked access intent");
                auto &entry = declared[access.allocation];
                entry.first |= access.mode != AccessMode::Read;
                entry.second = access.lifetime;
            }
            std::lock_guard<std::mutex> lock(registry.mutex);
            if (!(registry.commands++ % 64))
                for (auto it = registry.allocations.begin(); it != registry.allocations.end();)
                    if (it->second.lifetime.expired())
                        it = registry.allocations.erase(it);
                    else
                        ++it;
            for (const auto &entry : declared)
            {
                const auto found = registry.allocations.find(entry.first);
                if (found == registry.allocations.end())
                    continue;
                auto &uses = found->second;
                if (uses.writer && uses.writer->completed.load())
                    uses.writer.reset();
                for (auto it = uses.readers.begin(); it != uses.readers.end();)
                    if (it->second->completed.load())
                        it = uses.readers.erase(it);
                    else
                        ++it;
                RequireOrder(lane->frontier, uses.writer, entry.first);
                if (entry.second.first)
                    for (const auto &reader : uses.readers)
                        RequireOrder(lane->frontier, reader.second, entry.first);
            }
            for (const auto &entry : declared)
            {
                auto &uses = registry.allocations[entry.first];
                previous.emplace(entry.first, uses);
                uses.lifetime = entry.second.second;
                if (entry.second.first)
                {
                    uses.writer = state->use;
                    uses.readers.clear();
                }
                else
                    uses.readers[id] = state->use;
            }
        }
        catch (...)
        {
            invalid = std::current_exception();
        }
        try
        {
            // Optional provider agreement runs outside the registry lock. MPI
            // adapters use it to reject rank-local ordering mistakes together,
            // before any rank enters payload posting. Native stream kernels
            // do not acquire a collective or any extra wait through this hook.
            if (agree)
                agree(!invalid);
            if (invalid)
                std::rethrow_exception(invalid);
            lane->frontier.prefixes[id] = state->use->position;
            accept(Submission(state));
        }
        catch (...)
        {
            // No other lane can depend on an unaccepted prefix: Record takes
            // this lane's issue lock. Independent concurrent reads are merged
            // back rather than discarded by restoring the previous snapshot.
            std::lock_guard<std::mutex> lock(registry.mutex);
            for (const auto &entry : previous)
            {
                auto &uses = registry.allocations[entry.first];
                if (uses.writer == state->use)
                {
                    uses.writer = entry.second.writer;
                    for (const auto &reader : entry.second.readers)
                        uses.readers.emplace(reader);
                }
                const auto reader = uses.readers.find(id);
                if (reader != uses.readers.end() && reader->second == state->use)
                {
                    uses.readers.erase(reader);
                    const auto old = entry.second.readers.find(id);
                    if (old != entry.second.readers.end())
                        uses.readers.emplace(*old);
                }
            }
            state->use->completed = true;
            throw;
        }
    }
    std::shared_ptr<const AccessOrder> AccessTrackingContext::Record(uint64_t id, const std::function<void()> &record)
    {
        auto lane = _state->Get(id);
        std::lock_guard<std::recursive_mutex> issue(lane->issue);
        DNDS_check_throw_info(!lane->issuing, "cannot export an unaccepted checked stream prefix");
        auto order = std::make_shared<const AccessOrder>(lane->frontier);
        record();
        return order;
    }
    void AccessTrackingContext::WaitFor(uint64_t id, const std::shared_ptr<const AccessOrder> &order, const std::function<void()> &wait)
    {
        auto lane = _state->Get(id);
        std::lock_guard<std::recursive_mutex> issue(lane->issue);
        DNDS_check_throw_info(!lane->issuing, "recursive checked stream wait is unsupported");
        wait();
        if (order)
            for (const auto &prefix : order->prefixes)
                lane->frontier.prefixes[prefix.first] = std::max(lane->frontier.prefixes[prefix.first], prefix.second);
    }
}
