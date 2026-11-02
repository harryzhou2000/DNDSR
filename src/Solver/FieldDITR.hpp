#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"
#include <functional>
#include <optional>
#include <array>

namespace DNDS::ODE
{
    struct DITRSettings
    {
        real fraction = .55, theta = .9146;
        index iterations = 100;
        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(fraction) && fraction > 0 && fraction < 1 &&
                                      std::isfinite(theta) && theta > 0 && iterations > 0,
                                  "invalid DITR fraction, transformed-equation weight or iteration cap");
        }
    };
    struct DITRCoefficients
    {
        std::array<real, 4> interpolation;
        std::array<real, 3> integration;
        explicit DITRCoefficients(real a)
        {
            DNDS_check_throw_info(std::isfinite(a) && a > 0 && a < 1, "DITR fraction must be between zero and one");
            // Euler's active U2R2/mask0 Hermite interpolation and integration.
            interpolation = {1 - 3 * a * a + 2 * a * a * a, 3 * a * a - 2 * a * a * a,
                             a - 2 * a * a + a * a * a, -a * a + a * a * a};
            integration = {.5 - 1 / (6 * a), -1 / (6 * a * (a - 1)), .5 + 1 / (6 * (a - 1))};
            for (real value : integration)
                DNDS_check_throw_info(std::isfinite(value), "nonfinite DITR integration coefficient");
        }
    };
    struct DITRIterationReport
    {
        bool converged = false;
        index iterations = 0;
    };
    /// Coupled U2R2 DITR physical-time equations on native CPU fields.
    /// Stage operators supply F=dU/dt and solve mass*U-scale*F=source.
    /// The intermediate stage has its own factory-created operator/workspace;
    /// silently using an endpoint's persistent binding for it is forbidden.
    /// The factory runs lazily at Prepare, outside local-only construction.
    ///
    /// Preparation changes trial/scratch only. Caller serializes calls, agrees
    /// settings/order on all ranks, coordinates callback errors, and supplies
    /// a collective convergence decision for BOTH original H and I equations.
    /// Callbacks complete synchronously, keep provided storage, leave owned
    /// operator inputs and solve sources unchanged, and may refresh ghosts.
    /// False is a rejected/capped trial and is reusable. A callback exception
    /// may leave partial scratch; rebuild this object before retry. This class
    /// owns no accepted-state publication or time/history commit.
    class FieldDITR
    {
    public:
        using Field = Algebra::DofField<real>;
        using Evaluate = std::function<void(Field &, Field &, real, int)>;
        using SolveStage = std::function<bool(Field &, const Field &, real, real, real, int)>;
        using LocalStage = std::function<void(const std::function<void()> &)>;
        using Converged = std::function<bool(const Field &, const Field &, index)>;
        struct StageOperator
        {
            Evaluate evaluate;
            SolveStage solve;
            std::function<void()> reset; // e.g. cold midpoint polynomial per trial
        };
        using StageFactory = std::function<StageOperator(Field &)>;

    private:
        DITRSettings _settings;
        DITRCoefficients _coefficient;
        LocalStage _local;
        StageFactory _factory;
        std::optional<StageOperator> _middle;
        Field _prototype, _mid, _initialRate, _midRate, _endRate, _source, _hermite, _integral;
        Runtime::ArrayStorage::Shared _midOwnedPin, _midGhostPin;
        DITRIterationReport _report;
        bool _failed = false;
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
            prototype.Validate();
            DNDS_check_throw_info(prototype.Device() == Runtime::Device::Host(), "DITR CPU field profile cannot execute device fields");
            return Field(prototype.Space(), Runtime::Array<real>(prototype.Owned().Layout()), Runtime::Array<real>(prototype.Ghost().Layout()));
        }
        void Check(const Field &field) const
        {
            field.Validate();
            DNDS_check_throw_info(field.Space().Same(_mid.Space()) && field.Width() == _mid.Width() && field.Device() == _mid.Device(), "DITR field space, width or backend differs");
        }
        struct Binding
        {
            Runtime::ArrayStorage::Shared owned, ghost;
            explicit Binding(const Field &field) : owned(field.Owned().SharedStorage()), ghost(field.Ghost().SharedStorage()) {}
            void Check(const Field &field) const
            {
                field.Validate();
                DNDS_check_throw_info(field.Owned().SharedStorage() == owned && field.Ghost().SharedStorage() == ghost, "DITR callback rebound a field");
            }
        };
        void EvaluateChecked(const Evaluate &evaluate, Field &state, Field &value, real time, int stage)
        {
            const Binding input(state), output(value);
            const auto version = state.Owned().Version();
            evaluate(state, value, time, stage);
            RunLocal([&]
                     {
                input.Check(state); output.Check(value);
                DNDS_check_throw_info(state.Owned().Version() == version, "DITR rate operator modified owned input"); });
        }
        bool SolveChecked(const SolveStage &solve, Field &state, real scale, real time, int stage, real dt)
        {
            const Binding input(_source), output(state);
            const auto version = _source.Owned().Version();
            const bool result = solve(state, _source, 1 / dt, scale, time, stage);
            RunLocal([&]
                     {
                input.Check(_source); output.Check(state);
                DNDS_check_throw_info(_source.Owned().Version() == version, "DITR stage solver modified its source"); });
            return result;
        }

    public:
        FieldDITR(const Field &prototype, DITRSettings settings, StageFactory factory, LocalStage local = {})
            : _settings(settings), _coefficient(settings.fraction), _local(std::move(local)), _factory(std::move(factory)),
              _prototype(prototype), _mid(NewVector(prototype)), _initialRate(NewVector(prototype)), _midRate(NewVector(prototype)), _endRate(NewVector(prototype)),
              _source(NewVector(prototype)), _hermite(NewVector(prototype)), _integral(NewVector(prototype)),
              _midOwnedPin(_mid.Owned().SharedStorage()), _midGhostPin(_mid.Ghost().SharedStorage())
        {
            settings.Validate();
            DNDS_check_throw_info(_factory, "DITR requires an explicit independent intermediate-stage factory");
        }
        FieldDITR(const FieldDITR &) = delete;
        FieldDITR &operator=(const FieldDITR &) = delete;
        const DITRIterationReport &LastReport() const { return _report; }
        /// Independent scratch and lazily recreated stage bindings, even when
        /// a failed callback left the old intermediate allocation unusable.
        std::unique_ptr<FieldDITR> Fresh() const { return std::make_unique<FieldDITR>(_prototype, _settings, _factory, _local); }
        bool Prepare(const Field &initial, Field &trial, real time, real dt, const Evaluate &evaluate, const SolveStage &solve, const Converged &converged)
        {
            _report = {};
            RunLocal([&]
                     {
                DNDS_check_throw_info(!_failed, "DITR callback failed; rebuild its workspace before retry");
                Check(initial); Check(trial);
                DNDS_check_throw_info(evaluate && solve && converged && std::isfinite(time) && std::isfinite(dt) && dt > 0 && std::isfinite(1 / dt) && std::isfinite(time + dt), "invalid DITR trial controls");
                for (const auto *input : {&initial.Owned(), &initial.Ghost()})
                    for (const auto *output : {&trial.Owned(), &trial.Ghost()})
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "DITR initial/trial allocations alias");
                Algebra::Copy(trial, initial); Algebra::Copy(_mid, initial); });
            try
            {
                const auto midVersion = _mid.Owned().Version();
                if (!_middle)
                    _middle = _factory(_mid);
                RunLocal([&]
                         { DNDS_check_throw_info(_middle->evaluate && _middle->solve, "DITR intermediate factory returned incomplete operators"); });
                if (_middle->reset)
                    _middle->reset();
                RunLocal([&]
                         { DNDS_check_throw_info(_mid.Owned().SharedStorage() == _midOwnedPin && _mid.Ghost().SharedStorage() == _midGhostPin && _mid.Owned().Version() == midVersion,
                                                 "DITR intermediate factory/reset changed its state binding or owned values"); });
                EvaluateChecked(evaluate, trial, _initialRate, time, 0);
                const auto &c = _coefficient.interpolation;
                const auto &w = _coefficient.integration;
                for (index iteration = 0; iteration <= _settings.iterations; ++iteration)
                {
                    EvaluateChecked(evaluate, trial, _endRate, time + dt, 0);
                    EvaluateChecked(_middle->evaluate, _mid, _midRate, time + _settings.fraction * dt, 1);
                    RunLocal([&]
                             {
                        Algebra::LinearCombination(_hermite, c[0] / dt, initial, c[1] / dt, trial);
                        Algebra::Axpy(_hermite, -1 / dt, _mid); Algebra::Axpy(_hermite, c[2], _initialRate); Algebra::Axpy(_hermite, c[3], _endRate);
                        Algebra::LinearCombination(_integral, 1 / dt, initial, -1 / dt, trial);
                        Algebra::Axpy(_integral, w[0], _initialRate); Algebra::Axpy(_integral, w[1], _midRate); Algebra::Axpy(_integral, w[2], _endRate); });
                    _report.iterations = iteration;
                    if (converged(_hermite, _integral, iteration))
                    {
                        _report.converged = true;
                        return true;
                    }
                    if (iteration == _settings.iterations)
                        return false;
                    RunLocal([&]
                             {
                        // Midpoint residual H+theta*I, followed by endpoint I.
                        Algebra::LinearCombination(_source, (c[0] + _settings.theta) / dt, initial, (c[1] - _settings.theta) / dt, trial);
                        Algebra::Axpy(_source, c[2] + _settings.theta * w[0], _initialRate);
                        Algebra::Axpy(_source, c[3] + _settings.theta * w[2], _endRate); });
                    if (!SolveChecked(_middle->solve, _mid, _settings.theta * w[1], time + _settings.fraction * dt, 1, dt))
                        return false;
                    EvaluateChecked(_middle->evaluate, _mid, _midRate, time + _settings.fraction * dt, 1);
                    RunLocal([&]
                             {
                        Algebra::Copy(_source, initial); Algebra::Scale(_source, 1 / dt);
                        Algebra::Axpy(_source, w[0], _initialRate); Algebra::Axpy(_source, w[1], _midRate); });
                    if (!SolveChecked(solve, trial, w[2], time + dt, 0, dt))
                        return false;
                }
            }
            catch (...)
            {
                _failed = true;
                throw;
            }
            return false;
        }
    };
}
