#pragma once

#include "Solver/TimeIntegration/BDFCoefficients.hpp"
#include "Solver/TimeIntegration/DIRKTableau.hpp"
#include <type_traits>

namespace DNDS::ODE
{
    /// BDF-owned history and source arithmetic. The transaction facade validates
    /// stage ordering/bindings and commits only after accepted-state publication.
    /// No RHS cache, DIRK tableau, MPI or nonlinear iteration is owned here.
    template <class State, class Operations>
    class BDFMethod
    {
        static_assert(std::is_nothrow_swappable_v<State>);
        int _order;
        State _initial, _previous, _source;
        real _previousDt = 0;
        bool _history = false;

    public:
        BDFMethod(const State &prototype, int order)
            : _order(order), _initial(Operations::NewVector(prototype)),
              _previous(Operations::NewVector(prototype)), _source(Operations::NewVector(prototype))
        {
            DNDS_check_throw_info(order == 1 || order == 2, "unsupported BDF order");
        }
        const State &Source() const { return _source; }
        void Begin(const State &initial) { Operations::Copy(_initial, initial); }
        TimeEquationCoefficients Prepare(State &trial, real time, real dt, bool steady)
        {
            Operations::Copy(trial, _initial);
            const BDFCoefficients coefficients(_order, time, dt, _history ? _previousDt : 0, steady);
            coefficients.template AssembleResOther<Operations>(_source, _initial, &_previous);
            return coefficients.equation;
        }
        bool HasHistory() const { return _history; }
        real PreviousDt() const { return _previousDt; }
        const State &PreviousState() const
        {
            DNDS_check_throw_info(_history, "no accepted BDF history");
            return _previous;
        }
        void Commit(real dt, bool steady) noexcept
        {
            if (steady)
                ResetHistory();
            else
            {
                std::swap(_previous, _initial);
                _previousDt = dt;
                _history = true;
            }
        }
        void ResetHistory() noexcept
        {
            _history = false;
            _previousDt = 0;
        }
        void RestoreHistory(const State &previous, real dt)
        {
            DNDS_check_throw_info(std::isfinite(dt) && dt > 0, "invalid restored BDF step");
            Operations::Copy(_previous, previous);
            _previousDt = dt;
            _history = true;
        }
    };

    /// DIRK-owned initial state, resOther and completed-stage RHS cache. No
    /// accepted-step history is allocated. Stage ordering and rate completion
    /// are enforced by the transaction facade before exposing the next source.
    template <class State, class Operations>
    class DIRKMethod
    {
        DIRKTableau _tableau;
        State _initial, _source;
        std::vector<State> _rates;

    public:
        DIRKMethod(const State &prototype, const std::string &method)
            : _tableau(method), _initial(Operations::NewVector(prototype)), _source(Operations::NewVector(prototype))
        {
            _rates.reserve(_tableau.a.size());
            for (index stage = 0; stage < index(_tableau.a.size()); ++stage)
                _rates.push_back(Operations::NewVector(prototype));
        }
        int StageNumber() const { return int(_tableau.a.size()); }
        bool Implicit(int stage) const { return _tableau.a.at(stage).at(stage) != 0; }
        real StageTime(int stage, real time, real dt) const { return time + _tableau.c.at(stage) * dt; }
        const State &Source() const { return _source; }
        State &Rate(int stage) { return _rates.at(stage); }
        void Begin(const State &initial) { Operations::Copy(_initial, initial); }
        TimeEquationCoefficients Prepare(State &trial, int stage, real time, real dt)
        {
            DNDS_check_throw_info(std::isfinite(time) && std::isfinite(dt) && dt > 0 &&
                                      std::isfinite(1 / dt) && std::isfinite(time + dt) && time + dt > time,
                                  "invalid DIRK timestep");
            const bool implicit = Implicit(stage); // Validate stage before writing.
            Operations::Copy(trial, _initial);
            const real inverseDt = 1 / dt;
            if (implicit)
            {
                Operations::Copy(_source, _initial);
                Operations::Scale(_source, inverseDt);
                for (int previous = 0; previous < stage; ++previous)
                    if (_tableau.a[stage][previous] != 0)
                        Operations::Axpy(_source, _tableau.a[stage][previous], _rates[previous]);
            }
            return {implicit ? inverseDt : 0, _tableau.a[stage][stage], StageTime(stage, time, dt)};
        }
    };
}
