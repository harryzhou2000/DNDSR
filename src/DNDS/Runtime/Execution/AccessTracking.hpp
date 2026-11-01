#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"
#include <functional>
#include <map>
#include <vector>

namespace DNDS::Runtime
{
    struct ScheduledBufferAccess
    {
        uint64_t allocation;
        std::weak_ptr<void> lifetime;
        AccessMode mode;
        /// A declaration does not extend payload lifetime. The operation owns
        /// its operands; retaining a Buffer here could defeat ordered free.
        ScheduledBufferAccess(const Buffer &buffer, AccessMode intent)
            : allocation(buffer.Identity()), lifetime(buffer.AccessLifetime()), mode(intent) {}
    };
    /// Immutable known stream prefixes. Stable lane IDs are process-wide, so an
    /// explicitly imported event can convey dependencies across engine contexts.
    struct AccessOrder
    {
        std::map<uint64_t, uint64_t> prefixes;
    };
    /// Optional host-side scheduled-access proof. It never waits for numerical
    /// completion or inserts a missing dependency. All contexts participate in
    /// one allocation-identity registry to prevent independent alias authority;
    /// stream frontiers are private to their explicit context. Native stream
    /// events remain the execution mechanism on CPU and CUDA alike.
    class AccessTrackingContext
    {
        struct State;
        std::shared_ptr<State> _state;

    public:
        class Submission
        {
            struct State;
            std::shared_ptr<State> _state;
            explicit Submission(std::shared_ptr<State> state) : _state(std::move(state)) {}
            friend class AccessTrackingContext;

        public:
            Submission() = default;
            void Complete() const noexcept;
        };
        AccessTrackingContext();
        uint64_t NewLane();
        /// Discard issue bookkeeping after its last stream handle is released.
        /// Completion records and immutable event prefixes remain valid; lane
        /// IDs are never reused. No further Submit/Record/Wait may use this ID.
        void RetireLane(uint64_t lane);
        /// Check and reserve before backend acceptance. Backend rejection rolls
        /// back access accounting. The launch callback runs without the shared
        /// ledger lock, including CPU-inline numerical execution. A provider's
        /// optional agree callback must coordinate local validity before native
        /// acceptance; rejected agreement rolls back reservations too. Accepted
        /// work retains Submission until actual completion and calls Complete.
        void Submit(uint64_t lane, const std::vector<ScheduledBufferAccess> &accesses,
                    const std::function<void(Submission)> &accept,
                    const std::function<void(bool)> &agree = {});
        /// The native callback must record/wait the corresponding real stream.
        std::shared_ptr<const AccessOrder> Record(uint64_t lane, const std::function<void()> &nativeRecord);
        void WaitFor(uint64_t lane, const std::shared_ptr<const AccessOrder> &order, const std::function<void()> &nativeWait);
    };
}
