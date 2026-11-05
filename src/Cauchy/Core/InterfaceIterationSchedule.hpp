#pragma once

#include "Cauchy/Core/StateEpoch.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <optional>
#include <limits>
#include <cstring>

namespace DNDS::Cauchy
{
    enum class InterfaceRefreshPolicy
    {
        PerStage,
        PerIteration
    };
    struct InterfaceStageContext
    {
        StateEpoch epoch;
        real stageTime = 0;
    };
    /// Transport and packet ownership stay outside the schedule. Begin freezes
    /// sources and posts transfers; complete makes incoming CPU data usable;
    /// discard drains/releases pending operations without publishing a packet.
    /// Capture source versions/owners in these hooks, not only the logical epoch.
    struct InterfaceExchangeHooks
    {
        std::function<void(const InterfaceStageContext &)> begin, complete, discard;
    };
    /// One stage/physical attempt's iteration-start protocol. This facility
    /// supplies C4 hooks, not a physical multi-domain fluid solver. A caller
    /// begins transfers, runs independent mean/BC/core work, completes inputs,
    /// then runs dependent work and closes the iteration. It never changes a
    /// linear operator or refreshes a trace inside a Krylov/preconditioner call.
    ///
    /// Serialize calls and retain communication/packet owners. Use the
    /// coupling context for local schedule validation before publication.
    /// Construction always agrees stage identity/policy. Hooks own their MPI
    /// phases and local preflights; the schedule never catches a hook exception
    /// and attempts another agreement. Unexpected hook errors are fatal to a
    /// distributed run. Disabling per-iteration checks selects caller-ordered eager
    /// execution; callers then coordinate errors before peers enter collectives.
    /// This synchronous CPU protocol does not certify CUDA stream completion.
    /// Construct/call/destroy on an MPI-permitted controller thread and release
    /// the final communication/transfer owners before MPI finalization.
    class InterfaceIterationSchedule
    {
    private:
        enum class State
        {
            Idle,
            Pending,
            Ready,
            Failed
        };
        InterfaceStageContext _stage, _current;
        InterfaceRefreshPolicy _policy;
        InterfaceExchangeHooks _hooks;
        std::shared_ptr<Runtime::CommunicationContext> _coordination;
        bool _checks;
        std::optional<InterfaceStageContext> _published;
        State _state = State::Idle;
        index _lastIteration = -1;
        bool _refresh = false, _discarded = false;
        template <class F>
        void Together(F &&run) const
        {
            if (_checks)
                MPI::LocalStage(MPIInfo(_coordination->Communicator()), std::forward<F>(run));
            else
                run();
        }

    public:
        InterfaceIterationSchedule(std::shared_ptr<Runtime::CommunicationContext> coordination, InterfaceStageContext stage,
                                   InterfaceRefreshPolicy policy, InterfaceExchangeHooks hooks, bool collectiveChecks = true)
            : _stage(stage), _current(stage), _policy(policy), _hooks(std::move(hooks)), _coordination(std::move(coordination)), _checks(collectiveChecks)
        {
            DNDS_check_throw_info(_coordination, "interface schedule needs a coupling context");
            const MPIInfo mpi(_coordination->Communicator());
            MPI::LocalStage(mpi, [&]
                            { DNDS_check_throw_info(std::isfinite(stage.stageTime) && _hooks.begin && _hooks.complete && _hooks.discard &&
                                                        (policy == InterfaceRefreshPolicy::PerStage || policy == InterfaceRefreshPolicy::PerIteration),
                                                    "invalid interface iteration schedule"); });
            _stage.epoch.couplingIteration = 0;
            _current = _stage;
            uint64_t timeBits = 0;
            const real normalizedTime = stage.stageTime == 0 ? 0 : stage.stageTime;
            static_assert(sizeof(timeBits) == sizeof(normalizedTime));
            std::memcpy(&timeBits, &normalizedTime, sizeof(timeBits));
            std::array<uint64_t, 9> key{stage.epoch.acceptedStep, stage.epoch.trial, stage.epoch.stage, stage.epoch.geometry,
                                        stage.epoch.schema, stage.epoch.coupledStage, timeBits, uint64_t(policy), uint64_t(collectiveChecks)};
            auto minimum = key, maximum = key;
            MPI_Allreduce(MPI_IN_PLACE, minimum.data(), int(key.size()), MPI_UINT64_T, MPI_MIN, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, maximum.data(), int(key.size()), MPI_UINT64_T, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(minimum == maximum, "interface stage identity or refresh/check policy differs between participants");
        }
        InterfaceIterationSchedule(const InterfaceIterationSchedule &) = delete;
        InterfaceIterationSchedule &operator=(const InterfaceIterationSchedule &) = delete;
        /// Starts the next consecutive iteration (first is zero). Returns true
        /// if a new packet is being transferred. No packet is exposed yet,
        /// including when PerStage will reuse a previously completed packet.
        bool BeginIteration(index iteration)
        {
            Together([&]
                     { DNDS_check_throw_info(_state == State::Idle && !_discarded && iteration >= 0 &&
                                                 _lastIteration < std::numeric_limits<index>::max() && iteration == _lastIteration + 1,
                                             "interface iteration is out of order or the stage failed"); });
            _current = _stage;
            _current.epoch.couplingIteration = uint64_t(iteration);
            _refresh = _policy == InterfaceRefreshPolicy::PerIteration || !_published;
            _state = State::Pending;
            try
            {
                if (_refresh)
                    _hooks.begin(_current);
            }
            catch (...)
            {
                _state = State::Failed;
                throw;
            }
            return _refresh;
        }
        /// Join required inputs before dependent CPU work. Hooks perform their
        /// own local preflights. An exception marks this schedule failed and
        /// propagates directly; it does not certify failure agreement with peers.
        void CompleteInputs()
        {
            Together([&]
                     { DNDS_check_throw_info(_state == State::Pending, "no pending interface iteration"); });
            try
            {
                if (_refresh)
                    _hooks.complete(_current);
                if (_refresh)
                    _published = _current;
                _state = State::Ready;
            }
            catch (...)
            {
                _state = State::Failed;
                throw;
            }
        }
        /// Copy of the actual packet's source identity. PerStage retains the
        /// first iteration's identity; it never relabels cached data as fresh.
        InterfaceStageContext TraceContext() const
        {
            DNDS_check_throw_info(_state == State::Ready && _published, "interface packet is not ready or its stage failed");
            return *_published;
        }
        void EndIteration()
        {
            Together([&]
                     { DNDS_check_throw_info(_state == State::Ready, "cannot finish an incomplete interface iteration"); });
            _lastIteration = index(_current.epoch.couplingIteration);
            _state = State::Idle;
        }
        /// Explicit cleanup after a failed/rejected attempt. Idempotent; the
        /// stage remains unusable afterwards. No numerical rollback is done.
        /// Destruction issues no implicit Discard; captured transfer owners
        /// retain their ordinary cleanup behavior. Call Discard on failure and
        /// retain/drain transfer owners before releasing buffers/finalizing MPI.
        /// A failed cleanup can be retried; discard hooks must safely drain
        /// remaining work after a previous partial cleanup attempt.
        void Discard()
        {
            if (_discarded)
                return;
            _state = State::Failed;
            _published.reset();
            _hooks.discard(_current);
            _discarded = true;
        }
    };
}
