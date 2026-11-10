#pragma once

#include "DNDS/MPI/MPI.hpp"
#include "DNDS/Runtime/Communication/CollectiveValidation.hpp"
#include <exception>
#include <utility>
#include <map>
#include <memory>
#include <mutex>

namespace DNDS::Runtime
{
    class TransferProvider;
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
                if (comm != MPI_COMM_NULL && std::uncaught_exceptions())
                    MPI::AbortIfSolverFailure("solver unwinding before communicator release");
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized && comm != MPI_COMM_NULL)
                    MPI_Comm_free(&comm);
            }
        };
        std::shared_ptr<State> _state;
        bool _collectivePreflight = true;
        static void ReadTagLimit(State &state)
        {
            int flag = 0;
            int *limit = nullptr;
            DNDS_check_throw_info(MPI_Comm_get_attr(state.comm, MPI_TAG_UB, &limit, &flag) == MPI_SUCCESS,
                                  "cannot query communication tag limit");
            // Split/SELF communicators need not carry predefined attributes.
            state.tagLimit = flag && limit ? *limit : 32767;
        }
        CommunicationContext(std::shared_ptr<State> state, bool collectivePreflight)
            : _state(std::move(state)), _collectivePreflight(collectivePreflight)
        {
            CommunicationDetail::AgreePolicy(_state->comm, collectivePreflight);
        }

    public:
        /// Explicitly duplicate a raw communicator into an isolated namespace.
        /// Retained for source compatibility; managed consumers should share
        /// the existing context instead. Immutable safety policy is agreed once.
        /// Replacing it requires a coordinated, drained preparation boundary.
        /// Disabled mode retains local checks but aborts on local rejection;
        /// rank-asymmetric exceptions are recoverable only in enabled mode.
        explicit CommunicationContext(MPI_Comm comm, bool collectivePreflight = true)
            : _state(std::make_shared<State>()), _collectivePreflight(collectivePreflight)
        {
            DNDS_check_throw_info(comm != MPI_COMM_NULL, "communication context needs a communicator");
            DNDS_check_throw_info(MPI_Comm_dup(comm, &_state->comm) == MPI_SUCCESS,
                                  "cannot duplicate communication context");
            CommunicationDetail::AgreePolicy(_state->comm, collectivePreflight);
            ReadTagLimit(*_state);
        }
        /// Consume a newly created communicator exactly once, without dup/free.
        /// The input becomes MPI_COMM_NULL. Never adopt WORLD/SELF. All members
        /// call together; downstream copies retain the actual resource owner.
        static std::shared_ptr<CommunicationContext> Adopt(MPI_Comm &comm, bool collectivePreflight = true)
        {
            DNDS_check_throw_info(comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF,
                                  "adoption requires a new, non-predefined communicator");
            auto state = std::make_shared<State>();
            state->comm = std::exchange(comm, MPI_COMM_NULL);
            ReadTagLimit(*state);
            return std::shared_ptr<CommunicationContext>(new CommunicationContext(std::move(state), collectivePreflight));
        }
        static std::shared_ptr<CommunicationContext> Duplicate(MPI_Comm comm, bool collectivePreflight = true)
        {
            return std::make_shared<CommunicationContext>(comm, collectivePreflight);
        }
        /// Collective setup boundary: change policy while sharing communicator
        /// lifetime AND tag reservations. Drain earlier work before changing the
        /// execution policy. This is not an independent message namespace.
        std::shared_ptr<CommunicationContext> WithPreflight(bool collectivePreflight) const
        {
            return std::shared_ptr<CommunicationContext>(new CommunicationContext(_state, collectivePreflight));
        }
        bool SharesResource(const CommunicationContext &other) const { return _state == other._state; }
        MPI_Comm Communicator() const { return _state->comm; }
        bool ChecksCollectivePreflight() const { return _collectivePreflight; }
        void Preflight(bool valid, const char *message) const
        {
            CommunicationDetail::Preflight(valid, _state->comm, _collectivePreflight, message);
        }
        /// Local-only preparation followed by optional agreement. Do not put
        /// collectives inside run; each rank must enter this phase in order.
        /// Preserve the local exception when agreement rejects the phase.
        template <class F>
        void LocalStage(F &&run, const char *message = "communication preparation failed on a rank") const
        {
            CommunicationDetail::LocalStage(_state->comm, _collectivePreflight,
                                            std::forward<F>(run), message);
        }
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
        /// Collective setup reservation over all live rank-local bindings.
        /// Agree an unused tag even when completion retires old bindings at
        /// different times on different ranks. No local counter decides matching.
        std::shared_ptr<const int> ReserveAgreed() const
        {
            int64_t tag = 1;
            for (;;)
            {
                int64_t candidate = tag;
                {
                    std::lock_guard<std::mutex> lock(_state->mutex);
                    while (candidate <= _state->tagLimit)
                    {
                        const auto it = _state->tags.find(int(candidate));
                        if (it == _state->tags.end() || it->second.expired())
                            break;
                        ++candidate;
                    }
                }
                // Agree the common bound in this existing reservation phase,
                // not through an extra reduction at context construction.
                int64_t agreed[]{candidate, -int64_t(_state->tagLimit)};
                MPI_Allreduce(MPI_IN_PLACE, agreed, 2, MPI_INT64_T, MPI_MAX, _state->comm);
                candidate = agreed[0];
                const auto commonLimit = -agreed[1];
                if (candidate > commonLimit)
                {
                    tag = -1;
                    break;
                }
                if (candidate == tag)
                    break;
                tag = candidate;
            }
            std::shared_ptr<const int> result;
            LocalStage([&]
                       {
                DNDS_check_throw_info(tag >= 0 && tag <= _state->tagLimit, "communication tag namespace exhausted");
                result = Reserve(int(tag)); }, "shared communication tag reservation failed");
            return result;
        }
    };

    enum class TransferMode
    {
        Persistent,
        Transient
    };

    /// MPI launch validation, agreed collectively when the binding is built.
    /// AbortOnFailure retains local checks but posts requests without a global
    /// agreement. Any local launch failure aborts MPI; it cannot be recovered.
    enum class TransferPreflight
    {
        Inherit,
        Collective,
        AbortOnFailure
    };

    class CommunicationWorker;
    struct TransferOptions
    {
        TransferMode mode = TransferMode::Persistent;
        /// Null inherits the route owner and agrees an unused channel tag.
        /// An explicit context/tag retains the fixed-tag protocol API.
        std::shared_ptr<CommunicationContext> context;
        int tag = 0;
        /// Explicit non-MPI payload provider; MPI remains setup/control only.
        /// Do not combine with an MPI context/tag. Null preserves all defaults.
        std::shared_ptr<TransferProvider> provider;
        /// Explicit installed-MPI capability, independent of CUDA awareness.
        /// Default rejects Direct endpoints from cudaMallocAsync/VMM pools;
        /// select explicit packing or validate the MPI build before opting in.
        bool allowStreamOrderedDeviceMemory = false;
        /// Resolve once at binding from the selected context when Inherit.
        TransferPreflight preflight = TransferPreflight::Inherit;
        /// Optional CPU Direct/HostPacked execution owner. Null keeps caller
        /// execution. Worker requires MPI_THREAD_MULTIPLE; not stream-native.
        std::shared_ptr<CommunicationWorker> communicationWorker;
        /// Matched protocol name at collective binding setup; not a rank-local
        /// construction counter. Repeated instances receive distinct agreed tags.
        std::string channel = "row-transfer";
        /// Explicit setup-only namespace isolation. Incompatible with context.
        bool isolatedNamespace = false;
    };
}
