#pragma once

#include "DNDS/Errors.hpp"
#include <atomic>
#include <memory>

namespace DNDS::Runtime
{
    /// Operation-level exclusion for mutable workspace/session implementations.
    /// A deliberate internal alias can use the same ticket during nested work;
    /// unrelated callers must acquire independently and reject while busy.
    class SessionAccess
    {
        struct State
        {
            std::atomic<bool> busy{false};
        };
        std::shared_ptr<State> _state = std::make_shared<State>();

    public:
        class Ticket
        {
            friend class SessionAccess;
            std::shared_ptr<State> state;

        public:
            Ticket() = default;
            Ticket(const Ticket &) = delete;
            Ticket &operator=(const Ticket &) = delete;
            ~Ticket()
            {
                if (state)
                    state->busy.store(false, std::memory_order_release);
            }
        };
        using Lease = std::shared_ptr<const Ticket>;
        Lease Acquire(Lease nested = {}) const
        {
            if (nested)
            {
                DNDS_check_throw_info(nested->state == _state, "foreign mutable session ticket");
                return nested;
            }
            auto ticket = std::make_shared<Ticket>();
            bool expected = false;
            DNDS_check_throw_info(_state->busy.compare_exchange_strong(expected, true, std::memory_order_acq_rel),
                                  "mutable session already has an active operation");
            ticket->state = _state;
            return ticket;
        }
    };
}
