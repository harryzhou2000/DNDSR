#pragma once

#include "DNDS/MPI.hpp"
#include <map>
#include <memory>
#include <mutex>

namespace DNDS::Runtime
{
    /// A collectively created namespace shared by short-lived routes/bindings.
    /// Tags are explicit protocol identifiers, matched by the application across
    /// ranks. No rank-local allocation counter determines message matching.
    class CommunicationContext
    {
        struct State
        {
            MPI_Comm comm = MPI_COMM_NULL;
            int tagLimit = 0;
            std::mutex mutex;
            std::map<int, std::weak_ptr<const int>> tags;
            ~State()
            {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized && comm != MPI_COMM_NULL)
                    MPI_Comm_free(&comm);
            }
        };
        std::shared_ptr<State> _state;

    public:
        explicit CommunicationContext(MPI_Comm comm)
            : _state(std::make_shared<State>())
        {
            DNDS_check_throw_info(comm != MPI_COMM_NULL, "communication context needs a communicator");
            DNDS_check_throw_info(MPI_Comm_dup(comm, &_state->comm) == MPI_SUCCESS,
                                  "cannot duplicate communication context");
            int flag = 0;
            int *limit = nullptr;
            DNDS_check_throw_info(MPI_Comm_get_attr(_state->comm, MPI_TAG_UB, &limit, &flag) == MPI_SUCCESS,
                                  "cannot query communication tag limit");
            // World-model predefined attributes need not be present on SELF
            // or split communicators. MPI guarantees tags through 32767 even
            // when the communicator does not carry this environment attribute.
            _state->tagLimit = flag && limit ? *limit : 32767;
        }
        MPI_Comm Communicator() const { return _state->comm; }
        /// Local reservation. The binding collectively checks success before it
        /// posts messages. Retain the returned owner through request retirement.
        std::shared_ptr<const int> Reserve(int tag) const
        {
            DNDS_check_throw_info(tag >= 0 && tag <= _state->tagLimit, "MPI tag out of range");
            std::lock_guard<std::mutex> lock(_state->mutex);
            // Do not accumulate expired IDs from changing mesh epochs.
            for (auto it = _state->tags.begin(); it != _state->tags.end();)
                if (it->second.expired())
                    it = _state->tags.erase(it);
                else
                    ++it;
            DNDS_check_throw_info(_state->tags.find(tag) == _state->tags.end(), "MPI tag already bound in communication context");
            auto owner = std::make_shared<const int>(tag);
            _state->tags.emplace(tag, owner);
            return owner;
        }
    };

    enum class TransferMode
    {
        Persistent,
        Transient
    };

    struct TransferOptions
    {
        TransferMode mode = TransferMode::Persistent;
        /// Null preserves legacy independent-communicator bindings. Supplying
        /// a context avoids communicator duplication for each binding.
        std::shared_ptr<CommunicationContext> context;
        int tag = 0;
    };
}
