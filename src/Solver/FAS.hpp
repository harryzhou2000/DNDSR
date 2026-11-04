#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"
#include "Solver/StateFAS.hpp"

namespace DNDS::Multigrid
{
    using Field = Algebra::DofField<real>;

    /// CPU DOF-field adapter for the storage-independent FAS recursion. Every
    /// numerical access is owned-only; halo readiness belongs to level kernels.
    /// Other physical state types provide the same small Operations contract.
    struct FieldFASOperations
    {
        struct Binding
        {
            Field snapshot;
            uint64_t version;
            std::array<uint64_t, 2> shape;
            explicit Binding(const Field &state)
                : snapshot(state), version(state.Owned().Version()), shape{state.Owned().ShapeVersion(), state.Ghost().ShapeVersion()} {}
            void Check(const Field &state, bool unchangedOwned = false) const
            {
                CheckCompatible(snapshot, state);
                DNDS_check_throw_info(state.Owned().SharedStorage() == snapshot.Owned().SharedStorage() &&
                                          state.Ghost().SharedStorage() == snapshot.Ghost().SharedStorage() &&
                                          state.Owned().ShapeVersion() == shape[0] && state.Ghost().ShapeVersion() == shape[1],
                                      "FAS callback rebound or reshaped a field");
                DNDS_check_throw_info(!unchangedOwned || state.Owned().Version() == version,
                                      "FAS operator changed its owned input");
            }
        };
        static void CheckCompatible(const Field &a, const Field &b)
        {
            Algebra::Detail::Pair(a, b);
            DNDS_check_throw_info(a.Device() == Runtime::Device::Host(), "FAS CPU vector profile cannot execute device fields");
        }
        static Field NewVector(const Field &state)
        {
            CheckCompatible(state, state);
            return Field(state.Space(), Runtime::Array<real>(state.Owned().Layout()), Runtime::Array<real>(state.Ghost().Layout()));
        }
        static index ComponentNumber(const Field &state) { return index(state.Width()); }
        static void RequireDistinct(const Field &a, const Field &b)
        {
            for (const auto *input : {&a.Owned(), &a.Ghost()})
                for (const auto *output : {&b.Owned(), &b.Ghost()})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()),
                                          "FAS state/source allocations overlap");
        }
        static void Copy(Field &out, const Field &a) { Algebra::Copy(out, a); }
        static void LinearCombination(Field &out, real a, const Field &x, real b, const Field &y)
        {
            Algebra::LinearCombination(out, a, x, b, y);
        }
        static void Axpy(Field &out, real scale, const Field &a) { Algebra::Axpy(out, scale, a); }
        static void CheckFrozen(const Field &state, const Field &reference, const std::vector<index> &components)
        {
            const Algebra::ComponentSelection selection(ComponentNumber(state), components);
            const auto read = state.Owned().Read(), savedRead = reference.Owned().Read();
            const auto values = read.View<Runtime::UniformTableRows>(), saved = savedRead.View<Runtime::UniformTableRows>();
            for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                for (index j = 0; j < ComponentNumber(state); ++j)
                    if (!selection.Includes(j))
                        DNDS_check_throw_info(std::memcmp(&values(iRow, j), &saved(iRow, j), sizeof(real)) == 0,
                                              "FAS callback changed a frozen component");
        }
        static void ProjectCorrection(Field &correction, const std::vector<index> &components)
        {
            const Algebra::ComponentSelection selection(ComponentNumber(correction), components);
            auto write = correction.Owned().Write();
            const auto values = write.View<Runtime::UniformTableRows>();
            for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                for (index j = 0; j < ComponentNumber(correction); ++j)
                    if (!selection.Includes(j))
                        values(iRow, j) = 0;
        }
        static void AddSelected(Field &state, const Field &correction, const std::vector<index> &components)
        {
            if (components.empty())
            {
                Axpy(state, 1, correction);
                return;
            }
            const auto read = correction.Owned().Read();
            auto write = state.Owned().Write();
            const auto delta = read.View<Runtime::UniformTableRows>();
            const auto values = write.View<Runtime::UniformTableRows>();
            for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                for (const auto j : components)
                    values(iRow, j) += delta(iRow, j);
        }
    };

    /// Historical field API; shares the single recursion with logical states.
    using FASLevel = StateFASLevel<Field>;
    using FASTransfer = StateFASTransfer<Field>;
    using FASCycle = StateFASCycle<Field, FieldFASOperations>;
}
