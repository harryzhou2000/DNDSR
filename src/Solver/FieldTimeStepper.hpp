#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"
#include <functional>
#include <optional>

namespace DNDS::ODE
{
    /// Conservative semidiscrete time equations on native fields. Spatial
    /// operators supply dU/dt; the stage solver receives
    ///     mass*U - rhsScale*F(U,t) = source.
    /// It owns nonlinear iteration, PP, physical accumulation and MPI policy.
    /// This class owns only tableaus, accepted BDF history and trial scratch.
    /// The initial CPU profile uses identity accumulation; a future equation
    /// with a different mass map needs an explicit accumulation adapter.
    class FieldTimeStepper
    {
    public:
        using Field = Algebra::DofField<real>;
        using Evaluate = std::function<void(Field &, Field &, real, int)>;
        using SolveStage = std::function<bool(Field &, const Field &, real, real, real, int)>;

    private:
        std::string _method;
        std::vector<std::vector<real>> _a;
        std::vector<real> _c;
        std::vector<Field> _rates;
        Field _source, _previous, _pendingInitial;
        real _previousDt = 0, _pendingDt = 0;
        bool _history = false, _pending = false;
        static Field NewVector(const Field &prototype)
        {
            prototype.Validate();
            DNDS_check_throw_info(prototype.Device() == Runtime::Device::Host(), "FieldTimeStepper CPU profile cannot execute device fields");
            return Field(prototype.Space(), Runtime::Array<real>(prototype.Owned().Layout()), Runtime::Array<real>(prototype.Ghost().Layout()));
        }
        void Check(const Field &field) const
        {
            field.Validate();
            DNDS_check_throw_info(field.Space().Same(_source.Space()) && field.Width() == _source.Width() && field.Device() == _source.Device(),
                                  "time field space, components or backend differs");
        }
        void BuildTableau()
        {
            if (_method == "bdf1" || _method == "bdf2")
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
        FieldTimeStepper(const Field &prototype, std::string method)
            : _method(std::move(method)), _source(NewVector(prototype)), _previous(NewVector(prototype)), _pendingInitial(NewVector(prototype))
        {
            BuildTableau();
            _rates.reserve(_a.size());
            for (index i = 0; i < index(_a.size()); ++i)
                _rates.push_back(NewVector(prototype));
        }
        FieldTimeStepper(const FieldTimeStepper &) = delete;
        FieldTimeStepper &operator=(const FieldTimeStepper &) = delete;
        const std::string &Method() const { return _method; }
        bool HasHistory() const { return _history; }
        real PreviousDt() const { return _previousDt; }
        const Field &PreviousState() const
        {
            DNDS_check_throw_info(_history, "no accepted BDF history");
            return _previous;
        }
        /// Preparation is serialized and synchronous. All participating ranks
        /// must call it with agreed controls and callbacks that coordinate local
        /// failures before communication. Callbacks may inspect dynamic state.
        /// Initial and trial must be distinct; only trial/scratch changes here.
        /// False/exception leaves accepted history unchanged. A successful trial
        /// still needs Commit *after* its external accepted-state publication.
        bool Prepare(const Field &initial, Field &trial, real time, real dt, const Evaluate &evaluate, const SolveStage &solve)
        {
            _pending = false;
            Check(initial);
            Check(trial);
            DNDS_check_throw_info(evaluate && solve && std::isfinite(time) && std::isfinite(dt) && dt > 0 && std::isfinite(time + dt), "invalid time trial controls");
            for (const auto *input : {&initial.Owned(), &initial.Ghost()})
                for (const auto *output : {&trial.Owned(), &trial.Ghost()})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "time initial/trial fields alias");
            Algebra::Copy(_pendingInitial, initial);
            _pendingDt = dt;
            if (_a.empty())
            {
                const real ratio = dt / (_history ? _previousDt : dt);
                if (_method == "bdf2" && _history)
                {
                    // Euler's normalized variable-step BDF2: physical mass
                    // stays 1/dt and RHS scale carries (1+r)/(1+2r).
                    DNDS_check_throw_info(std::isfinite(ratio) && ratio > 0 && std::isfinite(ratio * ratio), "invalid BDF step ratio");
                    const real denominator = 1 + 2 * ratio;
                    Algebra::LinearCombination(_source, (1 + ratio) * (1 + ratio) / (denominator * dt), initial,
                                               -ratio * ratio / (denominator * dt), _previous);
                    Algebra::Copy(trial, initial);
                    _pending = solve(trial, _source, 1 / dt, (1 + ratio) / denominator, time + dt, 0);
                }
                else
                {
                    Algebra::Copy(_source, initial);
                    Algebra::Scale(_source, 1 / dt);
                    Algebra::Copy(trial, initial);
                    _pending = solve(trial, _source, 1 / dt, 1, time + dt, 0);
                }
                return _pending;
            }
            for (int stage = 0; stage < int(_a.size()); ++stage)
            {
                Algebra::Copy(trial, initial);
                if (_a[stage][stage] != 0)
                {
                    Algebra::Copy(_source, initial);
                    Algebra::Scale(_source, 1 / dt);
                    for (int previous = 0; previous < stage; ++previous)
                        if (_a[stage][previous] != 0)
                            Algebra::Axpy(_source, _a[stage][previous], _rates[previous]);
                    if (!solve(trial, _source, 1 / dt, _a[stage][stage], time + _c[stage] * dt, stage))
                        return false;
                }
                evaluate(trial, _rates[stage], time + _c[stage] * dt, stage);
            }
            _pending = true;
            return true; // All supported DIRK tableaus are stiffly accurate.
        }
        /// Only call once the matching physical trial was published. Pure
        /// handle swaps allocate nothing and cannot fail midway through history.
        void Commit()
        {
            DNDS_check_throw_info(_pending, "no complete time trial to commit");
            std::swap(_previous, _pendingInitial);
            _previousDt = _pendingDt;
            _history = true;
            _pending = false;
        }
        void Discard() noexcept { _pending = false; }
        /// After an externally replaced state or incompatible restart, take a
        /// BDF1 startup step rather than silently consuming stale history.
        void ResetHistory() noexcept
        {
            _history = _pending = false;
            _previousDt = 0;
        }
        void RestoreHistory(const Field &previous, real dt)
        {
            Check(previous);
            DNDS_check_throw_info(std::isfinite(dt) && dt > 0, "invalid restored BDF step");
            Algebra::Copy(_previous, previous);
            _previousDt = dt;
            _history = true;
            _pending = false;
        }
    };
}
