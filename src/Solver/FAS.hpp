#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Algebra/ComponentSelection.hpp"
#include <functional>
#include <cstring>

namespace DNDS::Multigrid
{
    using Field = Algebra::DofField<real>;
    struct FASLevel
    {
        /// Shallow retained state owner. Storage is distinct between levels;
        /// callbacks may bind its allocation before constructing the hierarchy.
        Field state;
        /// F(U), including physical mass when present. Own communication here;
        /// input ghosts may be refreshed but owned inputs remain unchanged.
        std::function<void(Field &, Field &)> apply;
        /// Relax F(U)=source, at most the requested number of corrections.
        /// Return actual corrections; throw on an unusable numerical result.
        std::function<index(Field &, const Field &, index)> smooth;
        /// Optional physical update policy (e.g. flow PP). Default is U+=delta.
        std::function<void(Field &, const Field &)> update;
        /// Required for a proper subset cycle. Solve only the selected
        /// equations with all remaining owned components frozen. Selection is
        /// normalized, nonempty and identical across levels. This is a
        /// principal-system smoother, not a full solve followed by masking.
        std::function<index(Field &, const Field &, index, const std::vector<index> &)> smoothSelected;
    };
    struct FASTransfer
    {
        /// State and residual restriction are different operators in general.
        std::function<void(const Field &, Field &)> restrictState, restrictResidual;
        std::function<void(const Field &, Field &)> prolongCorrection;
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
    /// criteria, time history or communicator policy. Initial vector algebra is
    /// CPU-only; reject device fields explicitly. New mesh/coarsening families
    /// provide level operators and three transfer maps, not a new recursion.
    ///
    /// Callbacks complete synchronously and coordinate distributed errors before
    /// communication. All ranks use the same hierarchy/cycle order. LocalStage
    /// optionally coordinates local algebra failures. Caller serializes access,
    /// retains callback resources and releases conflicting views before Apply.
    /// Write provided allocations without rebinding outputs. A failed cycle may
    /// leave partial level states; restore the trial and rebuild before retry.
    class FASCycle
    {
    public:
        using LocalStage = std::function<void(const std::function<void()> &)>;

    private:
        struct Level
        {
            FASLevel operation;
            Field source, value, residual, reference, correction;
            std::optional<Field> frozen;
            Runtime::ArrayStorage::Shared ownedStorage, ghostStorage;
            static Field NewVector(const Field &prototype)
            {
                prototype.Validate();
                DNDS_check_throw_info(prototype.Device() == Runtime::Device::Host(), "FAS CPU vector profile cannot execute device fields");
                return Field(prototype.Space(), Runtime::Array<real>(prototype.Owned().Layout()), Runtime::Array<real>(prototype.Ghost().Layout()));
            }
            explicit Level(FASLevel input)
                : operation(std::move(input)), source(NewVector(operation.state)), value(NewVector(operation.state)),
                  residual(NewVector(operation.state)), reference(NewVector(operation.state)), correction(NewVector(operation.state)),
                  ownedStorage(operation.state.Owned().SharedStorage()), ghostStorage(operation.state.Ghost().SharedStorage()) {}
        };
        std::vector<std::unique_ptr<Level>> _levels;
        std::vector<FASTransfer> _transfers;
        FASCycleSettings _settings;
        LocalStage _local;
        bool _failed = false;
        bool Selected() const { return !_settings.updateComponents.empty(); }
        bool Includes(index component) const
        {
            return !Selected() || std::binary_search(_settings.updateComponents.begin(), _settings.updateComponents.end(), component);
        }
        void SaveFrozen(Level &level)
        {
            if (Selected())
                Algebra::Copy(*level.frozen, level.operation.state);
        }
        void CheckFrozen(const Level &level) const
        {
            if (!Selected())
                return;
            const auto state = level.operation.state.Owned().Read(), reference = level.frozen->Owned().Read();
            const auto u = state.View<Runtime::UniformTableRows>(), saved = reference.View<Runtime::UniformTableRows>();
            for (index iRow = 0; iRow < u.RowCount(); ++iRow)
                for (index j = 0; j < index(level.operation.state.Width()); ++j)
                    if (!Includes(j))
                        DNDS_check_throw_info(std::memcmp(&u(iRow, j), &saved(iRow, j), sizeof(real)) == 0,
                                              "FAS callback changed a frozen component");
        }
        void ProjectCorrection(Field &correction) const
        {
            if (!Selected())
                return;
            auto write = correction.Owned().Write();
            const auto values = write.View<Runtime::UniformTableRows>();
            for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                for (index j = 0; j < index(correction.Width()); ++j)
                    if (!Includes(j))
                        values(iRow, j) = 0;
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
            level.operation.state.Validate();
            DNDS_check_throw_info(level.operation.state.Owned().SharedStorage() == level.ownedStorage && level.operation.state.Ghost().SharedStorage() == level.ghostStorage,
                                  "FAS bound state was rebound");
        }
        void ApplyOperator(Level &level, FASCycleReport &report)
        {
            RunLocal([&]
                     { CheckBinding(level); });
            const auto version = level.operation.state.Owned().Version();
            const auto valueStorage = level.value.Owned().SharedStorage(), ghostStorage = level.value.Ghost().SharedStorage();
            level.operation.apply(level.operation.state, level.value);
            RunLocal([&]
                     {
                CheckBinding(level); level.value.Validate();
                DNDS_check_throw_info(level.operation.state.Owned().Version() == version && level.value.Owned().SharedStorage() == valueStorage &&
                                          level.value.Ghost().SharedStorage() == ghostStorage, "FAS operator changed owned input or rebound its output"); });
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
            ApplyOperator(fine, report);
            RunLocal([&]
                     { Algebra::LinearCombination(fine.residual, real(1), fine.source, real(-1), fine.value); });
            auto &coarse = *_levels[level + 1];
            auto &transfer = _transfers[level];
            transfer.restrictState(fine.operation.state, coarse.operation.state);
            RunLocal([&]
                     { CheckBinding(coarse); });
            RunLocal([&]
                     { Algebra::Copy(coarse.reference, coarse.operation.state); });
            ApplyOperator(coarse, report);
            const auto sourceStorage = coarse.source.Owned().SharedStorage(), sourceGhost = coarse.source.Ghost().SharedStorage();
            transfer.restrictResidual(fine.residual, coarse.source);
            RunLocal([&]
                     {
                coarse.source.Validate();
                DNDS_check_throw_info(coarse.source.Owned().SharedStorage() == sourceStorage && coarse.source.Ghost().SharedStorage() == sourceGhost,
                                      "FAS residual restriction rebound its output"); });
            // F_c(U_c)=F_c(R U_f)+R(b_f-F_f(U_f)). The entrance reference
            // belongs to the incoming edge and is never overwritten by a
            // recursive child cycle, including in a hierarchy of 3+ levels.
            RunLocal([&]
                     { Algebra::Axpy(coarse.source, real(1), coarse.value); });
            Cycle(level + 1, report);
            RunLocal([&]
                     { Algebra::LinearCombination(coarse.correction, real(1), coarse.operation.state, real(-1), coarse.reference); });
            const auto correctionStorage = fine.correction.Owned().SharedStorage(), correctionGhost = fine.correction.Ghost().SharedStorage();
            transfer.prolongCorrection(coarse.correction, fine.correction);
            RunLocal([&]
                     {
                fine.correction.Validate();
                DNDS_check_throw_info(fine.correction.Owned().SharedStorage() == correctionStorage && fine.correction.Ghost().SharedStorage() == correctionGhost,
                                      "FAS prolongation rebound its output");
                ProjectCorrection(fine.correction); SaveFrozen(fine); });
            if (fine.operation.update)
                fine.operation.update(fine.operation.state, fine.correction);
            else
                RunLocal([&]
                         {
                    if (!Selected()) { Algebra::Axpy(fine.operation.state, real(1), fine.correction); return; }
                    const auto read = fine.correction.Owned().Read(); auto write = fine.operation.state.Owned().Write();
                    const auto delta = read.View<Runtime::UniformTableRows>(); const auto state = write.View<Runtime::UniformTableRows>();
                    for (index iRow = 0; iRow < state.RowCount(); ++iRow)
                        for (auto j : _settings.updateComponents) state(iRow, j) += delta(iRow, j); });
            RunLocal([&]
                     { CheckBinding(fine); CheckFrozen(fine); });
            Relax(level, _settings.postSmooth, report);
        }

    public:
        FASCycle(std::vector<FASLevel> levels, std::vector<FASTransfer> transfers, FASCycleSettings settings = {}, LocalStage local = {})
            : _transfers(std::move(transfers)), _settings(settings), _local(std::move(local))
        {
            RunLocal([&]
                     {
                DNDS_check_throw_info(!levels.empty() && _transfers.size() + 1 == levels.size() && settings.preSmooth >= 0 && settings.postSmooth >= 0 && settings.coarseSmooth > 0,
                                      "invalid FAS hierarchy or sweep controls");
                const index width = levels.front().state.Width();
                _settings.updateComponents = Algebra::ComponentSelection(width, settings.updateComponents).Indices();
                _levels.reserve(levels.size());
                for (auto &level : levels)
                {
                    DNDS_check_throw_info(level.apply && level.smooth, "FAS level is missing an operator or smoother");
                    DNDS_check_throw_info(!Selected() || (index(level.state.Width()) == width && level.smoothSelected),
                                          "selected FAS needs equal level widths and principal-system smoothers");
                    for (const auto &existing : _levels)
                        for (const auto *source : {&level.state.Owned(), &level.state.Ghost()})
                            for (const auto *target : {&existing->operation.state.Owned(), &existing->operation.state.Ghost()})
                                DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), target->Storage()), "FAS levels share mutable state storage");
                    _levels.push_back(std::make_unique<Level>(std::move(level)));
                    if (Selected())
                        _levels.back()->frozen.emplace(Level::NewVector(_levels.back()->operation.state));
                }
                for (const auto &transfer : _transfers)
                    DNDS_check_throw_info(transfer.restrictState && transfer.restrictResidual && transfer.prolongCorrection, "FAS edge is missing a transfer"); });
        }
        FASCycle(const FASCycle &) = delete;
        FASCycle &operator=(const FASCycle &) = delete;
        const Field &State(index level = 0) const { return _levels.at(level)->operation.state; }
        /// Apply fixed V-cycles in place to the bound fine state. This reports
        /// work, not convergence; the physical caller checks its original
        /// unmodified finest equation. Coarse forcing stays internal.
        FASCycleReport Apply(const Field &source, index cycles = 1)
        {
            RunLocal([&]
                     {
                source.Validate();
                DNDS_check_throw_info(!_failed, "FAS callback failed; rebuild the hierarchy before reuse");
                DNDS_check_throw_info(cycles > 0 && source.Space().Same(State().Space()) && source.Width() == State().Width() && source.Device() == State().Device(), "invalid FAS source or cycle count");
                for (const auto &level : _levels)
                {
                    CheckBinding(*level);
                    const auto &state = level->operation.state;
                    for (const auto *input : {&source.Owned(), &source.Ghost()})
                        for (const auto *output : {&state.Owned(), &state.Ghost()})
                            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "FAS RHS aliases mutable state");
                }
                Algebra::Copy(_levels.front()->source, source); });
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
