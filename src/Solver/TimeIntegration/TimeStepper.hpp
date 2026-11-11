#pragma once

#include "DNDS/Core/Defines.hpp"
#include "Solver/TimeIntegration/DITRMethod.hpp"
#include "Solver/TimeIntegration/SerializedMethods.hpp"
#include <functional>
#include <algorithm>
#include <optional>
#include <limits>
#include <type_traits>

namespace DNDS::ODE
{
    /// Conservative semidiscrete time equations on native fields. Spatial
    /// operators supply dU/dt; the stage solver receives
    ///     mass*U - rhsScale*F(U,t) = source.
    /// It owns nonlinear iteration, PP, physical accumulation and MPI policy.
    /// This facade owns transaction/order checks and exactly one method owner.
    /// The method owns its tableau, history, trial scratch and source arithmetic.
    /// The initial CPU profile uses identity accumulation; a future equation
    /// with a different mass map needs an explicit accumulation adapter.
    /// Operations supplies NewVector, CheckCompatible, RequireDistinct, a
    /// binding/version snapshot, and the five pointwise algebra actions. It
    /// owns storage/backend access; this class owns one set of time mathematics.
    template <class State, class Operations>
    class StateTimeStepper
    {
        static_assert(std::is_nothrow_swappable_v<State>, "time state handles must support allocation-free, nonthrowing history swaps");

    public:
        using Field = State;
        using DITR = DITRMethod<State, Operations>;
        using SolveCoupled = std::function<bool(DITR &, Field &)>;
        using Evaluate = std::function<void(Field &, Field &, real, int)>;
        using SolveStage = std::function<bool(Field &, const Field &, real, real, real, int)>;
        using LocalStage = std::function<void(const std::function<void()> &)>;
        /// Immutable stage ticket, not a reference to reusable method scratch.
        /// Resolve it through SolvePreparedStage on the same live stepper.
        class StageEquation
        {
            friend class StateTimeStepper;
            std::shared_ptr<const uint8_t> _session;
            uint64_t _generation;
            StageEquation(std::shared_ptr<const uint8_t> session, uint64_t generation,
                          real m, real scale, real t, int s, bool i)
                : _session(std::move(session)), _generation(generation),
                  mass(m), rhsScale(scale), time(t), stage(s), implicit(i) {}

        public:
            const real mass, rhsScale, time;
            const int stage;
            const bool implicit;
        };
        /// Deep, backend-agnostic payload captured only at a nonlinear
        /// iteration boundary. Runtime handles/leases are deliberately absent;
        /// restore recreates bindings against the caller-provided trial.
        struct ActiveCheckpoint
        {
            int schemaVersion = 1;
            std::string method;
            real pendingDt = 0, startTime = 0;
            bool pendingSteady = false, stagesActive = false, stagePrepared = false;
            int nextStage = 0;
            TimeEquationCoefficients equation;
            bool bdfHistory = false;
            real bdfPreviousDt = 0;
            real ditrTime = 0, ditrDt = 0;
            std::vector<std::string> names;
            std::vector<Field> fields;
        };

    private:
        std::shared_ptr<const uint8_t> _session = std::make_shared<const uint8_t>(0);
        uint64_t _stageGeneration = 0;
        std::string _method;
        std::unique_ptr<BDFMethod<State, Operations>> _bdf;
        std::unique_ptr<DIRKMethod<State, Operations>> _dirk;
        real _pendingDt = 0;
        bool _pending = false, _pendingSteady = false;
        LocalStage _local;
        DITRSettings _ditrSettings;
        std::unique_ptr<DITR> _ditr;
        bool _stagesActive = false, _stagePrepared = false;
        std::optional<TimeEquationCoefficients> _preparedEquation;
        using Binding = typename Operations::Binding;
        std::optional<Binding> _preparedSource, _preparedState;
        int _nextStage = 0;
        real _startTime = 0;
        template <class F>
        void RunLocal(F &&run) const
        {
            if (_local)
                _local(std::forward<F>(run));
            else
                run();
        }
        const Field &Prototype() const
        {
            return _bdf ? _bdf->Source() : _dirk ? _dirk->Source()
                                                 : _ditr->IntermediateState();
        }
        const Field &Source() const
        {
            DNDS_check_throw_info(_bdf || _dirk, "coupled time methods own their own source");
            return _bdf ? _bdf->Source() : _dirk->Source();
        }
        void Check(const Field &field) const { Operations::CheckCompatible(Prototype(), field); }
        void BeginInitial(const Field &initial, Field &trial)
        {
            Check(initial);
            Check(trial);
            Operations::RequireDistinct(initial, trial);
            if (_bdf)
                _bdf->Begin(initial);
            else if (_dirk)
                _dirk->Begin(initial);
        }
        void CheckPrepared(const Field &trial) const
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(_preparedState, "time stage has no reserved state binding");
                _preparedState->Check(trial);
                if (_preparedSource) _preparedSource->Check(Source(), true); });
        }
        static Field Clone(const Field &source)
        {
            auto result = Operations::NewVector(source);
            Operations::Copy(result, source);
            return result;
        }

    public:
        StateTimeStepper(const Field &prototype, std::string method, LocalStage local = {}, DITRSettings ditr = {})
            : _method(std::move(method)), _local(std::move(local)), _ditrSettings(ditr)
        {
            if (_method == "bdf1" || _method == "bdf2")
                _bdf = std::make_unique<BDFMethod<State, Operations>>(prototype, _method == "bdf2" ? 2 : 1);
            else if (_method == "ditr")
                _ditr = std::make_unique<DITR>(prototype, ditr.fraction, ditr.theta, ditr.variant, _local);
            else
                _dirk = std::make_unique<DIRKMethod<State, Operations>>(prototype, _method);
        }
        StateTimeStepper(const StateTimeStepper &) = delete;
        StateTimeStepper &operator=(const StateTimeStepper &) = delete;
        const std::string &Method() const { return _method; }
        /// Avoid a spurious roundoff-sized final step. Times are finite and
        /// nonnegative; relative scaling also preserves tiny physical horizons.
        static bool ReachedTimeTarget(real time, real target)
        {
            // Allow modest accumulated addition error, e.g. 800 IV steps or
            // 2500 cylinder steps, while scaling down for tiny time horizons.
            return time >= target || target - time <= 1024 * std::numeric_limits<real>::epsilon() * std::max(std::abs(time), std::abs(target));
        }
        /// Allocate an independent empty-history workspace with identical method
        /// and local-error policy, e.g. to prepare a restart transaction before
        /// replacing the live accepted history.
        std::unique_ptr<StateTimeStepper> Fresh() const
        {
            return std::make_unique<StateTimeStepper>(Prototype(), _method, _local, _ditrSettings);
        }
        bool HasHistory() const { return _bdf && _bdf->HasHistory(); }
        bool HasPendingTrial() const { return _pending; }
        DITR &CoupledMethod()
        {
            DNDS_check_throw_info(_ditr, "this time method has no coupled problem");
            return *_ditr;
        }
        const DITR &CoupledMethod() const
        {
            DNDS_check_throw_info(_ditr,
                                  "this time method has no coupled problem");
            return *_ditr;
        }
        real PreviousDt() const { return _bdf ? _bdf->PreviousDt() : 0; }
        /// DITR owns a coupled midpoint/end iteration, not a one-pass tableau.
        /// Its external coordination must use that coupled protocol explicitly.
        bool SupportsStageSequence() const { return !_ditr; }
        int StageNumber() const { return _dirk ? _dirk->StageNumber() : 1; }
        const Field &PreviousState() const
        {
            DNDS_check_throw_info(_bdf, "this method has no accepted-step history");
            return _bdf->PreviousState();
        }
        /// Begin a serialized synchronous trial without executing its stages.
        /// All ranks agree controls. Accepted history changes only at Commit.
        /// The controller must CompleteStage before preparing the next one.
        void BeginStages(const Field &initial, Field &trial, real time, real dt, bool steady = false)
        {
            Discard();
            RunLocal([&]
                     {
                DNDS_check_throw_info(SupportsStageSequence(), "DITR requires coupled iteration coordination, not a one-pass stage sequence");
                DNDS_check_throw_info(std::isfinite(time) && (steady ? _method == "bdf1" :
                    std::isfinite(dt) && dt > 0 && std::isfinite(time + dt) && time + dt > time), "invalid stage-sequence controls");
                BeginInitial(initial, trial); });
            _pendingSteady = steady;
            _pendingDt = steady ? 0 : dt;
            _startTime = time;
            _nextStage = 0;
            _stagesActive = true;
        }
        StageEquation PrepareStage(Field &trial)
        {
            DNDS_check_throw_info(_stagesActive && !_stagePrepared && _nextStage < StageNumber(), "no next time stage, or previous stage is unfinished");
            DNDS_check_throw_info(_stageGeneration != std::numeric_limits<uint64_t>::max(), "time stage generation exhausted");
            const int stage = _nextStage;
            const bool implicit = !_dirk || _dirk->Implicit(stage);
            TimeEquationCoefficients coefficients;
            RunLocal([&]
                     {
                Check(trial);
                coefficients = _bdf ? _bdf->Prepare(trial, _startTime, _pendingDt, _pendingSteady)
                                    : _dirk->Prepare(trial, stage, _startTime, _pendingDt);
                _preparedState.emplace(trial);
                if (implicit) _preparedSource.emplace(Source()); else _preparedSource.reset(); });
            _stagePrepared = true;
            _preparedEquation = coefficients;
            return StageEquation(_session, ++_stageGeneration, coefficients.inverseDt, coefficients.rhsScale, coefficients.time, stage, implicit);
        }
        /// Synchronous source borrow, validated before and after the solver.
        /// The callback may update trial values, but must not retain references,
        /// replace bindings, or re-enter this stepper. Method scratch never
        /// escapes through a ticket, and no source payload is copied.
        bool SolvePreparedStage(const StageEquation &equation, Field &trial, const SolveStage &solve)
        {
            RunLocal([&]
                     { DNDS_check_throw_info(_stagesActive && _stagePrepared && equation.implicit && solve &&
                                                 equation._session == _session && equation._generation == _stageGeneration && equation.stage == _nextStage,
                                             "expired or foreign time stage ticket"); });
            CheckPrepared(trial);
            const bool accepted = solve(trial, Source(), equation.mass, equation.rhsScale, equation.time, equation.stage);
            CheckPrepared(trial);
            return accepted;
        }
        /// Record the solved/current stage rate before the next stage can use
        /// it. BDF/steady need no rate callback. This posts no hidden solve or
        /// exchange. The caller supplies a coherent fixed stage boundary trace.
        void CompleteStage(Field &trial, const Evaluate &evaluate = {})
        {
            DNDS_check_throw_info(_stagesActive && _stagePrepared, "no prepared time stage to complete");
            CheckPrepared(trial);
            if (_dirk)
            {
                DNDS_check_throw_info(evaluate, "DIRK stage completion needs its rate evaluation");
                const Binding input(trial), output(_dirk->Rate(_nextStage));
                evaluate(trial, _dirk->Rate(_nextStage), _dirk->StageTime(_nextStage, _startTime, _pendingDt), _nextStage);
                RunLocal([&]
                         { input.Check(trial, true); output.Check(_dirk->Rate(_nextStage)); });
            }
            _preparedSource.reset();
            _preparedState.reset();
            _preparedEquation.reset();
            _stagePrepared = false;
            if (++_nextStage == StageNumber())
            {
                _pending = true;
                _stagesActive = false;
            }
        }
        /// Begin a coupled DITR trial without entering its nonlinear update.
        /// This split lets an external simulation controller construct distinct
        /// frozen midpoint/endpoint interface packets before either correction.
        void BeginCoupled(const Field &initial, Field &trial, real time, real dt,
                          const Evaluate &evaluate)
        {
            Discard();
            RunLocal([&]
                     {
                DNDS_check_throw_info(
                    _ditr && evaluate && std::isfinite(time) &&
                        std::isfinite(dt) && dt > 0 && std::isfinite(time + dt) &&
                        time + dt > time,
                    "invalid coupled time trial controls");
                BeginInitial(initial, trial); });
            _pendingDt = dt;
            _startTime = time;
            _ditr->Begin(initial, trial, time, dt, evaluate);
        }
        /// Execute the already-begun coupled update. False leaves method-owned
        /// midpoint/rate scratch active only until the caller discards the trial.
        bool RunCoupled(Field &trial, const SolveCoupled &coupled)
        {
            RunLocal([&]
                     { DNDS_check_throw_info(_ditr && coupled && !_pending,
                                             "no begun DITR trial, or trial already solved"); });
            _pending = coupled(*_ditr, trial);
            return _pending;
        }
        /// Preparation is serialized and synchronous. All participating ranks
        /// must call it with agreed controls and callbacks that coordinate local
        /// failures before communication. Callbacks may inspect dynamic state.
        /// Initial and trial must be distinct; only trial/scratch changes here.
        /// False/exception leaves accepted history unchanged. A successful trial
        /// still needs Commit *after* its external accepted-state publication.
        bool Prepare(const Field &initial, Field &trial, real time, real dt, const Evaluate &evaluate, const SolveStage &solve, const SolveCoupled &coupled = {})
        {
            Discard();
            RunLocal([&]
                     { DNDS_check_throw_info(evaluate && (_ditr ? bool(coupled) : bool(solve)) && std::isfinite(time) && std::isfinite(dt) && dt > 0 && std::isfinite(time + dt) && time + dt > time, "invalid time trial controls"); });
            _pendingDt = dt;
            if (_ditr)
            {
                BeginCoupled(initial, trial, time, dt, evaluate);
                return RunCoupled(trial, coupled);
            }
            BeginStages(initial, trial, time, dt);
            for (int stage = 0; stage < StageNumber(); ++stage)
            {
                const auto equation = PrepareStage(trial);
                if (equation.implicit && !SolvePreparedStage(equation, trial, solve))
                {
                    CheckPrepared(trial);
                    Discard();
                    return false;
                }
                CompleteStage(trial, evaluate);
            }
            return _pending; // All supported DIRK tableaus are stiffly accurate.
        }
        /// BDF1 infinite-dt limit: zero physical mass/history source and frozen
        /// time. No infinity enters arithmetic and no rate callback is needed.
        /// The same stage solver owns PTC/PP/convergence; a failed stage cannot
        /// be committed. Caller serializes/co-ordinates callbacks as in Prepare.
        bool PrepareSteady(const Field &initial, Field &trial, real time, const SolveStage &solve)
        {
            DNDS_check_throw_info(solve, "steady stage needs its solver");
            BeginStages(initial, trial, time, 0, true);
            const auto equation = PrepareStage(trial);
            if (!SolvePreparedStage(equation, trial, solve))
            {
                CheckPrepared(trial);
                Discard();
                return false;
            }
            CompleteStage(trial);
            return _pending;
        }
        /// Only call once the matching physical trial was published. Pure
        /// handle swaps allocate nothing and cannot fail midway through history.
        void Commit()
        {
            DNDS_check_throw_info(_pending, "no complete time trial to commit");
            if (_ditr)
            {
                _pending = false;
                _ditr->Discard();
                return;
            }
            if (_bdf)
                _bdf->Commit(_pendingDt, _pendingSteady);
            _pending = _pendingSteady = false;
        }
        void Discard() noexcept
        {
            _pending = _pendingSteady = _stagesActive = _stagePrepared = false;
            if (_ditr)
                _ditr->Discard();
            _preparedSource.reset();
            _preparedState.reset();
            _preparedEquation.reset();
        }
        /// After an externally replaced state or incompatible restart, take a
        /// BDF1 startup step rather than silently consuming stale history.
        void ResetHistory() noexcept
        {
            Discard();
            if (_bdf)
                _bdf->ResetHistory();
        }
        void RestoreHistory(const Field &previous, real dt)
        {
            Check(previous);
            DNDS_check_throw_info(_bdf, "this time method cannot restore BDF history");
            Discard();
            _bdf->RestoreHistory(previous, dt);
        }

        ActiveCheckpoint CaptureActive(const Field &trial) const
        {
            ActiveCheckpoint checkpoint;
            checkpoint.method = _method;
            checkpoint.pendingDt = _pendingDt;
            checkpoint.startTime = _startTime;
            checkpoint.pendingSteady = _pendingSteady;
            const auto append = [&](const std::string &name, const Field &field)
            {
                checkpoint.names.push_back(name);
                checkpoint.fields.push_back(Clone(field));
            };
            RunLocal([&]
                     {
                Check(trial);
                append("trial", trial);
                if (_ditr)
                {
                    const auto control = _ditr->CheckpointControl();
                    checkpoint.ditrTime = control.time;
                    checkpoint.ditrDt = control.dt;
                    append("ditr_initial", _ditr->InitialState());
                    append("ditr_middle", _ditr->IntermediateState());
                    append("ditr_initial_rate", _ditr->InitialRate());
                    append("ditr_middle_rate", _ditr->IntermediateRate());
                    append("ditr_end_rate", _ditr->EndpointRate());
                    append("ditr_res_other", _ditr->ResidualOther());
                    return;
                }
                DNDS_check_throw_info(_stagesActive && _stagePrepared && _preparedEquation,
                                      "serialized time checkpoint needs an active prepared stage");
                checkpoint.stagesActive = _stagesActive;
                checkpoint.stagePrepared = _stagePrepared;
                checkpoint.nextStage = _nextStage;
                checkpoint.equation = *_preparedEquation;
                if (_bdf)
                {
                    checkpoint.bdfHistory = _bdf->HasHistory();
                    checkpoint.bdfPreviousDt = _bdf->PreviousDt();
                    append("time_initial", _bdf->InitialState());
                    append("time_source", _bdf->PreparedSource());
                    append("bdf_previous", _bdf->PreviousStorage());
                }
                else
                {
                    append("time_initial", _dirk->InitialState());
                    append("time_source", _dirk->Source());
                    for (index stage = 0; stage < index(_dirk->Rates().size()); ++stage)
                        append("dirk_rate_" + std::to_string(stage), _dirk->Rates()[size_t(stage)]);
                } });
            return checkpoint;
        }

        void RestoreActive(const ActiveCheckpoint &checkpoint, Field &trial)
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(checkpoint.schemaVersion == 1 && checkpoint.method == _method &&
                                          checkpoint.names.size() == checkpoint.fields.size(),
                                      "time checkpoint method or payload schema differs");
                const auto payload = [&](const std::string &name) -> const Field &
                {
                    const auto found = std::find(checkpoint.names.begin(), checkpoint.names.end(), name);
                    DNDS_check_throw_info(found != checkpoint.names.end(), "time checkpoint payload is missing: " + name);
                    return checkpoint.fields[size_t(found - checkpoint.names.begin())];
                };
                Operations::Copy(trial, payload("trial"));
                Discard();
                _pendingDt = checkpoint.pendingDt;
                _startTime = checkpoint.startTime;
                _pendingSteady = checkpoint.pendingSteady;
                if (_ditr)
                {
                    _ditr->RestoreActive(trial, {checkpoint.ditrTime, checkpoint.ditrDt},
                                         payload("ditr_initial"), payload("ditr_middle"),
                                         payload("ditr_initial_rate"), payload("ditr_middle_rate"),
                                         payload("ditr_end_rate"), payload("ditr_res_other"));
                    return;
                }
                DNDS_check_throw_info(checkpoint.stagesActive && checkpoint.stagePrepared &&
                                          checkpoint.nextStage >= 0 && checkpoint.nextStage < StageNumber(),
                                      "serialized time checkpoint has no prepared stage");
                checkpoint.equation.Validate();
                if (_bdf)
                    _bdf->RestoreActive(payload("time_initial"), payload("bdf_previous"), payload("time_source"),
                                        checkpoint.bdfHistory, checkpoint.bdfPreviousDt);
                else
                {
                    std::vector<std::reference_wrapper<const Field>> rates;
                    for (index stage = 0; stage < index(_dirk->Rates().size()); ++stage)
                        rates.push_back(std::cref(payload("dirk_rate_" + std::to_string(stage))));
                    _dirk->RestoreActive(payload("time_initial"), payload("time_source"), rates);
                }
                _stagesActive = true;
                _stagePrepared = true;
                _nextStage = checkpoint.nextStage;
                _preparedEquation = checkpoint.equation;
                _preparedState.emplace(trial);
                _preparedSource.emplace(Source());
                _session = std::make_shared<const uint8_t>(0);
                ++_stageGeneration; });
        }

        StageEquation RestoredStageEquation() const
        {
            DNDS_check_throw_info(!_ditr && _stagesActive && _stagePrepared && _preparedEquation,
                                  "time stepper has no restored serialized stage");
            return StageEquation(_session, _stageGeneration, _preparedEquation->inverseDt,
                                 _preparedEquation->rhsScale, _preparedEquation->time,
                                 _nextStage, true);
        }
        bool ResumeCoupled(Field &trial, const SolveCoupled &coupled)
        {
            DNDS_check_throw_info(_ditr && coupled && !_pending, "time stepper has no restorable coupled solve");
            _pending = coupled(*_ditr, trial);
            return _pending;
        }
    };
}
