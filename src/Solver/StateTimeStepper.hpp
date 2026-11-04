#pragma once

#include "DNDS/Core/Defines.hpp"
#include "Solver/StateDITR.hpp"
#include <functional>
#include <optional>
#include <limits>
#include <type_traits>

namespace DNDS::ODE
{
    /// Conservative semidiscrete time equations on native fields. Spatial
    /// operators supply dU/dt; the stage solver receives
    ///     mass*U - rhsScale*F(U,t) = source.
    /// It owns nonlinear iteration, PP, physical accumulation and MPI policy.
    /// This class owns only tableaus, accepted BDF history and trial scratch.
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
        using DITR = StateDITR<State, Operations>;
        using Evaluate = std::function<void(Field &, Field &, real, int)>;
        using SolveStage = std::function<bool(Field &, const Field &, real, real, real, int)>;
        using LocalStage = std::function<void(const std::function<void()> &)>;
        /// Borrowed equation for one prepared stage. Source is null for an
        /// explicit stage; otherwise it remains valid until completion/discard.
        /// A controller may exchange boundary
        /// inputs between PrepareStage and solving this equation on each domain.
        struct StageEquation
        {
            const Field *source;
            real mass, rhsScale, time;
            int stage;
            bool implicit;
        };

    private:
        std::string _method;
        std::vector<std::vector<real>> _a;
        std::vector<real> _c;
        std::vector<Field> _rates;
        Field _source, _previous, _pendingInitial;
        real _previousDt = 0, _pendingDt = 0;
        bool _history = false, _pending = false, _pendingSteady = false;
        LocalStage _local;
        DITRSettings _ditrSettings;
        typename DITR::StageFactory _stageFactory;
        std::unique_ptr<DITR> _ditr;
        bool _rebuildDITR = false;
        bool _stagesActive = false, _stagePrepared = false;
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
        static Field NewVector(const Field &prototype)
        {
            return Operations::NewVector(prototype);
        }
        void Check(const Field &field) const
        {
            Operations::CheckCompatible(_source, field);
        }
        void BeginInitial(const Field &initial, Field &trial)
        {
            Check(initial);
            Check(trial);
            Operations::RequireDistinct(initial, trial);
            Operations::Copy(_pendingInitial, initial);
        }
        void CheckPrepared(const Field &trial) const
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(_preparedState, "time stage has no reserved state binding");
                _preparedState->Check(trial);
                if (_preparedSource) _preparedSource->Check(_source, true); });
        }
        void BuildTableau()
        {
            if (_method == "bdf1" || _method == "bdf2" || _method == "ditr")
                return;
            if (_method == "trapz")
            {
                _a = {{0, 0}, {.5, .5}};
                _c = {0, 1};
            }
            else if (_method == "esdirk2")
            {
                const real gamma = 1 - std::sqrt(2.) * .5, b2 = (1 - 2 * gamma) / (4 * gamma);
                _a = {{0, 0, 0}, {gamma, gamma, 0}, {1 - b2 - gamma, b2, gamma}};
                _c = {0, 2 * gamma, 1};
            }
            else if (_method == "esdirk3")
            {
                const real gamma = real(1767732205903) / real(4055673282236);
                _a = {{0, 0, 0, 0}, {gamma, gamma, 0, 0}, {real(2746238789719) / real(10658868560708), real(-640167445237) / real(6845629431997), gamma, 0}, {real(1471266399579) / real(7840856788654), real(-4482444167858) / real(7529755066697), real(11266239266428) / real(11593286722821), gamma}};
                _c = {0, 2 * gamma, _a[2][0] + _a[2][1] + gamma, 1};
            }
            else if (_method == "esdirk4")
            {
                // Preserve Euler's six-stage ESDIRK4 coefficients verbatim,
                // including its published decimal precision and stage times.
                _a = {{0, 0, 0, 0, 0, 0}, {.25, .25, 0, 0, 0, 0}, {.137776, -.055776, .25, 0, 0, 0}, {.1446368660269822, -.2239319076133447, .4492950415863626, .25, 0, 0}, {.09825878328356477, -.5915442428196704, .8101210205756877, .283164405707806, .25, 0}, {.1579162951616714, 0, .1867589405240008, .6805652953093346, -.2752405309950067, .25}};
                _c = {0, .5, .332, .62, .849999966747388, 1};
            }
            else
                DNDS_check_throw_info(false, "unavailable native time method: " + _method);
        }

    public:
        StateTimeStepper(const Field &prototype, std::string method, LocalStage local = {}, DITRSettings ditr = {}, typename DITR::StageFactory stageFactory = {})
            : _method(std::move(method)), _source(NewVector(prototype)), _previous(NewVector(prototype)), _pendingInitial(NewVector(prototype)), _local(std::move(local)),
              _ditrSettings(ditr), _stageFactory(std::move(stageFactory))
        {
            BuildTableau();
            _rates.reserve(_a.size());
            for (index i = 0; i < index(_a.size()); ++i)
                _rates.push_back(NewVector(prototype));
            if (_method == "ditr")
                _ditr = std::make_unique<DITR>(_source, ditr, _stageFactory, _local);
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
            return std::make_unique<StateTimeStepper>(_source, _method, _local, _ditrSettings, _stageFactory);
        }
        bool HasHistory() const { return _history; }
        bool HasPendingTrial() const { return _pending; }
        const DITRIterationReport *LastDITRReport() const { return _ditr ? &_ditr->LastReport() : nullptr; }
        real PreviousDt() const { return _previousDt; }
        /// DITR owns a coupled midpoint/end iteration, not a one-pass tableau.
        /// Its external coordination must use that coupled protocol explicitly.
        bool SupportsStageSequence() const { return !_ditr; }
        int StageNumber() const { return _a.empty() ? 1 : int(_a.size()); }
        const Field &PreviousState() const
        {
            DNDS_check_throw_info(_history, "no accepted BDF history");
            return _previous;
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
            const int stage = _nextStage;
            const bool implicit = _a.empty() || _a[stage][stage] != 0;
            real mass = _pendingSteady ? 0 : 1 / _pendingDt, scale = 1;
            const real time = _pendingSteady ? _startTime : _startTime + (_a.empty() ? 1 : _c[stage]) * _pendingDt;
            RunLocal([&]
                     {
                Check(trial);
                Operations::Copy(trial, _pendingInitial);
                if (_pendingSteady)
                    Operations::Fill(_source, 0.);
                else if (_a.empty())
                {
                    const real ratio = _pendingDt / (_history ? _previousDt : _pendingDt);
                    if (_method == "bdf2" && _history)
                    {
                        DNDS_check_throw_info(std::isfinite(ratio) && ratio > 0 && std::isfinite(ratio * ratio), "invalid BDF step ratio");
                        const real denominator = 1 + 2 * ratio;
                        Operations::LinearCombination(_source, (1 + ratio) * (1 + ratio) / (denominator * _pendingDt), _pendingInitial,
                                                   -ratio * ratio / (denominator * _pendingDt), _previous);
                        scale = (1 + ratio) / denominator;
                    }
                    else
                    {
                        Operations::Copy(_source, _pendingInitial);
                        Operations::Scale(_source, mass);
                    }
                }
                else if (implicit)
                {
                    Operations::Copy(_source, _pendingInitial);
                    Operations::Scale(_source, mass);
                    for (int previous = 0; previous < stage; ++previous)
                        if (_a[stage][previous] != 0)
                            Operations::Axpy(_source, _a[stage][previous], _rates[previous]);
                    scale = _a[stage][stage];
                }
                _preparedState.emplace(trial);
                if (implicit) _preparedSource.emplace(_source); else _preparedSource.reset(); });
            _stagePrepared = true;
            return {implicit ? &_source : nullptr, implicit ? mass : 0, implicit ? scale : 0, time, stage, implicit};
        }
        /// Record the solved/current stage rate before the next stage can use
        /// it. BDF/steady need no rate callback. This posts no hidden solve or
        /// exchange. The caller supplies a coherent fixed stage boundary trace.
        void CompleteStage(Field &trial, const Evaluate &evaluate = {})
        {
            DNDS_check_throw_info(_stagesActive && _stagePrepared, "no prepared time stage to complete");
            CheckPrepared(trial);
            if (!_a.empty())
            {
                DNDS_check_throw_info(evaluate, "DIRK stage completion needs its rate evaluation");
                const Binding input(trial), output(_rates[_nextStage]);
                evaluate(trial, _rates[_nextStage], _startTime + _c[_nextStage] * _pendingDt, _nextStage);
                RunLocal([&]
                         { input.Check(trial, true); output.Check(_rates[_nextStage]); });
            }
            _preparedSource.reset();
            _preparedState.reset();
            _stagePrepared = false;
            if (++_nextStage == StageNumber())
            {
                _pending = true;
                _stagesActive = false;
            }
        }
        /// Preparation is serialized and synchronous. All participating ranks
        /// must call it with agreed controls and callbacks that coordinate local
        /// failures before communication. Callbacks may inspect dynamic state.
        /// Initial and trial must be distinct; only trial/scratch changes here.
        /// False/exception leaves accepted history unchanged. A successful trial
        /// still needs Commit *after* its external accepted-state publication.
        bool Prepare(const Field &initial, Field &trial, real time, real dt, const Evaluate &evaluate, const SolveStage &solve, const typename DITR::Converged &coupled = {})
        {
            Discard();
            RunLocal([&]
                     { DNDS_check_throw_info(evaluate && solve && std::isfinite(time) && std::isfinite(dt) && dt > 0 && std::isfinite(time + dt) && time + dt > time, "invalid time trial controls"); });
            _pendingDt = dt;
            if (_ditr)
            {
                RunLocal([&]
                         { BeginInitial(initial, trial); });
                if (_rebuildDITR)
                {
                    RunLocal([&]
                             { _ditr = _ditr->Fresh(); });
                    _rebuildDITR = false;
                }
                try
                {
                    _pending = _ditr->Prepare(initial, trial, time, dt, evaluate, solve, coupled);
                }
                catch (...)
                {
                    _rebuildDITR = true;
                    throw;
                }
                return _pending;
            }
            BeginStages(initial, trial, time, dt);
            for (int stage = 0; stage < StageNumber(); ++stage)
            {
                const auto equation = PrepareStage(trial);
                if (equation.implicit && !solve(trial, *equation.source, equation.mass, equation.rhsScale, equation.time, equation.stage))
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
            if (!solve(trial, *equation.source, equation.mass, equation.rhsScale, equation.time, equation.stage))
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
            if (_pendingSteady)
            {
                _history = _pending = _pendingSteady = false;
                _previousDt = 0; // Infinity is not finite BDF history.
                return;
            }
            std::swap(_previous, _pendingInitial);
            _previousDt = _pendingDt;
            _history = true;
            _pending = false;
        }
        void Discard() noexcept
        {
            _pending = _pendingSteady = _stagesActive = _stagePrepared = false;
            _preparedSource.reset();
            _preparedState.reset();
        }
        /// After an externally replaced state or incompatible restart, take a
        /// BDF1 startup step rather than silently consuming stale history.
        void ResetHistory() noexcept
        {
            Discard();
            _history = false;
            _previousDt = 0;
        }
        void RestoreHistory(const Field &previous, real dt)
        {
            Check(previous);
            DNDS_check_throw_info(std::isfinite(dt) && dt > 0, "invalid restored BDF step");
            Operations::Copy(_previous, previous);
            _previousDt = dt;
            _history = true;
            Discard();
        }
    };
}
