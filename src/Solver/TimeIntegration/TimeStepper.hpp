#pragma once

#include "DNDS/Core/Defines.hpp"
#include "Solver/TimeIntegration/DITRMethod.hpp"
#include "Solver/TimeIntegration/SerializedMethods.hpp"
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
        std::unique_ptr<BDFMethod<State, Operations>> _bdf;
        std::unique_ptr<DIRKMethod<State, Operations>> _dirk;
        real _pendingDt = 0;
        bool _pending = false, _pendingSteady = false;
        LocalStage _local;
        DITRSettings _ditrSettings;
        std::unique_ptr<DITR> _ditr;
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
            return {implicit ? &Source() : nullptr, coefficients.inverseDt, coefficients.rhsScale, coefficients.time, stage, implicit};
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
        bool Prepare(const Field &initial, Field &trial, real time, real dt, const Evaluate &evaluate, const SolveStage &solve, const SolveCoupled &coupled = {})
        {
            Discard();
            RunLocal([&]
                     { DNDS_check_throw_info(evaluate && (_ditr ? bool(coupled) : bool(solve)) && std::isfinite(time) && std::isfinite(dt) && dt > 0 && std::isfinite(time + dt) && time + dt > time, "invalid time trial controls"); });
            _pendingDt = dt;
            if (_ditr)
            {
                RunLocal([&]
                         { BeginInitial(initial, trial); });
                _ditr->Begin(initial, trial, time, dt, evaluate);
                _pending = coupled(*_ditr, trial);
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
            _bdf->RestoreHistory(previous, dt);
            Discard();
        }
    };
}
