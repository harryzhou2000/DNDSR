#pragma once

#include "DNDS/Algebra/ComponentSelection.hpp"
#include <functional>
#include <cstring>
#include <memory>
#include <optional>

namespace DNDS::Multigrid
{
    template <class TState>
    struct StateFASLevel
    {
        /// Shallow retained state owner. Storage is distinct between levels;
        /// callbacks may bind its allocation before constructing the hierarchy.
        TState state;
        /// F(U), including physical mass when present. Own communication here;
        /// input ghosts may be refreshed but owned inputs remain unchanged.
        std::function<void(TState &, TState &)> apply;
        /// Relax F(U)=source, at most the requested number of corrections.
        /// Return actual corrections; throw on an unusable numerical result.
        std::function<index(TState &, const TState &, index)> smooth;
        /// Optional physical update policy (e.g. flow PP). Default is U+=delta.
        std::function<void(TState &, const TState &)> update;
        /// Required for a proper subset cycle. Solve only the selected
        /// equations with all remaining owned components frozen. Selection is
        /// normalized, nonempty and identical across levels. This is a
        /// principal-system smoother, not a full solve followed by masking.
        std::function<index(TState &, const TState &, index, const std::vector<index> &)> smoothSelected;
        /// Optional operator with a common linear term removed analytically.
        /// Used only by an identity-state edge declaring matching cancellation.
        /// The full apply/smoother still describes the actual level equation.
        std::function<void(TState &, TState &)> applyForIdentityTau;
    };
    template <class TState>
    struct StateFASTransfer
    {
        /// State and residual restriction are different operators in general.
        std::function<void(const TState &, TState &)> restrictState, restrictResidual;
        std::function<void(const TState &, TState &)> prolongCorrection;
        /// Caller proves restriction is identity in physical state and the
        /// removed linear term is identical on both levels. Both level tau
        /// operators must exist. This avoids subtracting large common U/dt.
        bool cancelIdentityLinearTerm = false;
    };
    struct FASCycleSettings
    {
        index preSmooth = 1, postSmooth = 1, coarseSmooth = 8;
        /// Empty selects the full state. Otherwise select distinct component
        /// indices, common to all levels; restriction still transfers the full
        /// background state and residual for nonlinear tau construction.
        std::vector<index> updateComponents;
    };
    struct FASCycleReport
    {
        index cycles = 0, operatorEvaluations = 0;
        std::vector<index> corrections;
    };
    /// Full Approximation Scheme V-cycle over explicit levels and transfers.
    /// The hierarchy does not own mesh construction, equation physics, stopping
    /// criteria, time history or communicator policy. Operations supplies owned
    /// algebra, allocation/layout checks, binding/version checks and selected
    /// component access; it must reject unsupported backends. New mesh/coarsening families
    /// provide level operators and three transfer maps, not a new recursion.
    ///
    /// Callbacks complete synchronously and coordinate distributed errors before
    /// communication. All ranks use the same hierarchy/cycle order. LocalStage
    /// optionally coordinates local algebra failures. Caller serializes access,
    /// retains callback resources and releases conflicting views before Apply.
    /// The Operations contract is NewVector, Binding::Check, CheckCompatible,
    /// RequireDistinct, ComponentNumber, Copy/LinearCombination/Axpy, CheckFrozen,
    /// ProjectCorrection and AddSelected. It performs no hidden communication.
    /// Write provided allocations without rebinding outputs. A failed cycle may
    /// leave partial level states; restore the trial and rebuild before retry.
    template <class TState, class Operations>
    class StateFASCycle
    {
    public:
        using Field = TState;
        using LevelDefinition = StateFASLevel<Field>;
        using TransferDefinition = StateFASTransfer<Field>;
        using LocalStage = std::function<void(const std::function<void()> &)>;

    private:
        struct Level
        {
            LevelDefinition operation;
            Field source, value, residual, reference, correction;
            std::optional<Field> frozen;
            typename Operations::Binding binding;
            explicit Level(LevelDefinition input)
                : operation(std::move(input)), source(Operations::NewVector(operation.state)), value(Operations::NewVector(operation.state)),
                  residual(Operations::NewVector(operation.state)), reference(Operations::NewVector(operation.state)), correction(Operations::NewVector(operation.state)),
                  binding(operation.state) {}
        };
        std::vector<std::unique_ptr<Level>> _levels;
        std::vector<TransferDefinition> _transfers;
        FASCycleSettings _settings;
        LocalStage _local;
        bool _failed = false;
        bool Selected() const { return !_settings.updateComponents.empty(); }
        void SaveFrozen(Level &level)
        {
            if (Selected())
                Operations::Copy(*level.frozen, level.operation.state);
        }
        void CheckFrozen(const Level &level) const
        {
            if (!Selected())
                return;
            Operations::CheckFrozen(level.operation.state, *level.frozen, _settings.updateComponents);
        }
        void ProjectCorrection(Field &correction) const
        {
            if (Selected())
                Operations::ProjectCorrection(correction, _settings.updateComponents);
        }
        template <class F>
        void RunLocal(F &&run) const
        {
            if (_local)
                _local(std::forward<F>(run));
            else
                run();
        }
        void Relax(index level, index corrections, FASCycleReport &report)
        {
            if (!corrections)
                return;
            auto &data = *_levels[level];
            if (Selected())
                RunLocal([&]
                         { SaveFrozen(data); });
            const index actual = Selected() ? data.operation.smoothSelected(data.operation.state, data.source, corrections, _settings.updateComponents)
                                            : data.operation.smooth(data.operation.state, data.source, corrections);
            RunLocal([&]
                     {
                DNDS_check_throw_info(actual >= 0 && actual <= corrections, "FAS smoother correction count is invalid");
                CheckBinding(data); CheckFrozen(data); });
            report.corrections[level] += actual;
        }
        static void CheckBinding(const Level &level)
        {
            level.binding.Check(level.operation.state);
        }
        void ApplyOperator(Level &level, FASCycleReport &report, bool identityTau = false)
        {
            std::optional<typename Operations::Binding> input, output;
            RunLocal([&]
                     {
                CheckBinding(level);
                input.emplace(level.operation.state); output.emplace(level.value); });
            const auto &apply = identityTau ? level.operation.applyForIdentityTau : level.operation.apply;
            apply(level.operation.state, level.value);
            RunLocal([&]
                     {
                input->Check(level.operation.state, true); output->Check(level.value); });
            ++report.operatorEvaluations;
        }
        void Cycle(index level, FASCycleReport &report)
        {
            auto &fine = *_levels[level];
            if (level + 1 == index(_levels.size()))
            {
                Relax(level, _settings.coarseSmooth, report);
                return;
            }
            Relax(level, _settings.preSmooth, report);
            auto &transfer = _transfers[level];
            ApplyOperator(fine, report, transfer.cancelIdentityLinearTerm);
            if (!transfer.cancelIdentityLinearTerm)
                RunLocal([&]
                         { Operations::LinearCombination(fine.residual, real(1), fine.source, real(-1), fine.value); });
            auto &coarse = *_levels[level + 1];
            transfer.restrictState(fine.operation.state, coarse.operation.state);
            std::optional<typename Operations::Binding> sourceBinding, differenceBinding;
            RunLocal([&]
                     { CheckBinding(coarse); });
            RunLocal([&]
                     {
                Operations::Copy(coarse.reference, coarse.operation.state);
                sourceBinding.emplace(coarse.source);
                if (transfer.cancelIdentityLinearTerm) differenceBinding.emplace(coarse.residual); });
            ApplyOperator(coarse, report, transfer.cancelIdentityLinearTerm);
            if (transfer.cancelIdentityLinearTerm)
            {
                transfer.restrictResidual(fine.value, coarse.residual);
                RunLocal([&]
                         {
                    differenceBinding->Check(coarse.residual);
                    Operations::Axpy(coarse.residual, real(-1), coarse.value); });
                transfer.restrictResidual(fine.source, coarse.source);
                RunLocal([&]
                         {
                    sourceBinding->Check(coarse.source);
                    // Form the small spatial difference first, then add the
                    // history source once. Two separate additions to a large
                    // resOther can lose a representable net tau correction.
                    Operations::Axpy(coarse.source, real(-1), coarse.residual); });
            }
            else
            {
                transfer.restrictResidual(fine.residual, coarse.source);
                RunLocal([&]
                         {
                    sourceBinding->Check(coarse.source);
                    Operations::Axpy(coarse.source, real(1), coarse.value); });
            }
            // F_c(U_c)=F_c(R U_f)+R(b_f-F_f(U_f)). The entrance reference
            // belongs to the incoming edge and is never overwritten by a
            // recursive child cycle, including in a hierarchy of 3+ levels.
            Cycle(level + 1, report);
            std::optional<typename Operations::Binding> correctionBinding;
            RunLocal([&]
                     {
                Operations::LinearCombination(coarse.correction, real(1), coarse.operation.state, real(-1), coarse.reference);
                correctionBinding.emplace(fine.correction); });
            transfer.prolongCorrection(coarse.correction, fine.correction);
            RunLocal([&]
                     {
                correctionBinding->Check(fine.correction);
                ProjectCorrection(fine.correction); SaveFrozen(fine); });
            if (fine.operation.update)
                fine.operation.update(fine.operation.state, fine.correction);
            else
                RunLocal([&]
                         { Operations::AddSelected(fine.operation.state, fine.correction, _settings.updateComponents); });

            RunLocal([&]
                     { CheckBinding(fine); CheckFrozen(fine); });
            Relax(level, _settings.postSmooth, report);
        }

    public:
        StateFASCycle(std::vector<LevelDefinition> levels, std::vector<TransferDefinition> transfers, FASCycleSettings settings = {}, LocalStage local = {})
            : _transfers(std::move(transfers)), _settings(settings), _local(std::move(local))
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(!levels.empty() && _transfers.size() + 1 == levels.size() && settings.preSmooth >= 0 && settings.postSmooth >= 0 && settings.coarseSmooth > 0,
                                      "invalid FAS hierarchy or sweep controls");
                const index width = Operations::ComponentNumber(levels.front().state);
                _settings.updateComponents = Algebra::ComponentSelection(width, settings.updateComponents).Indices();
                _levels.reserve(levels.size());
                for (auto &level : levels)
                {
                    DNDS_check_throw_info(level.apply && level.smooth, "FAS level is missing an operator or smoother");
                    DNDS_check_throw_info(!Selected() || (Operations::ComponentNumber(level.state) == width && level.smoothSelected),
                                          "selected FAS needs equal level widths and principal-system smoothers");
                    Operations::CheckCompatible(level.state, level.state);
                    for (const auto &existing : _levels)
                        Operations::RequireDistinct(level.state, existing->operation.state);
                    _levels.push_back(std::make_unique<Level>(std::move(level)));
                    if (Selected())
                        _levels.back()->frozen.emplace(Operations::NewVector(_levels.back()->operation.state));
                }
                for (const auto &transfer : _transfers)
                    DNDS_check_throw_info(transfer.restrictState && transfer.restrictResidual && transfer.prolongCorrection, "FAS edge is missing a transfer"); });
            RunLocal([&]
                     {
                for (index edge = 0; edge < index(_transfers.size()); ++edge)
                    DNDS_check_throw_info(!_transfers[edge].cancelIdentityLinearTerm ||
                                              (_levels[edge]->operation.applyForIdentityTau && _levels[edge + 1]->operation.applyForIdentityTau),
                                          "identity FAS cancellation requires both reduced operators"); });
        }
        StateFASCycle(const StateFASCycle &) = delete;
        StateFASCycle &operator=(const StateFASCycle &) = delete;
        const Field &State(index level = 0) const { return _levels.at(level)->operation.state; }
        /// Apply fixed V-cycles in place to the bound fine state. This reports
        /// work, not convergence; the physical caller checks its original
        /// unmodified finest equation. Coarse forcing stays internal.
        FASCycleReport Apply(const Field &source, index cycles = 1)
        {
            RunLocal([&]
                     {
                Operations::CheckCompatible(State(), source);
                DNDS_check_throw_info(!_failed, "FAS callback failed; rebuild the hierarchy before reuse");
                DNDS_check_throw_info(cycles > 0, "invalid FAS cycle count");
                for (const auto &level : _levels)
                {
                    CheckBinding(*level);
                    Operations::RequireDistinct(source, level->operation.state);
                }
                Operations::Copy(_levels.front()->source, source); });
            FASCycleReport report;
            report.corrections.assign(_levels.size(), 0);
            try
            {
                for (index cycle = 0; cycle < cycles; ++cycle)
                {
                    Cycle(0, report);
                    ++report.cycles;
                }
            }
            catch (...)
            {
                _failed = true;
                throw;
            }
            return report;
        }
    };
}
