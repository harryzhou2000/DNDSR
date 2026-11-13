#pragma once

#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"

namespace DNDS::Cauchy::Detail
{
    /** @brief Per-solve algebra adapter bound to the common compute lane.
     * @details Only weak runtime handles are retained. Numerical operations
     * reserve their Array operands and retain them in the engine. No thread
     * local/global dispatch context, device synchronization, or MPI is hidden
     * here. GMRES products and operator callbacks supply observation boundaries.
     */
    template <class Operations>
    class LinearSolverAlgebra
    {
        Runtime::ExecutionStream _stream;
        using State = typename Operations::State;
        template <class F>
        void Submit(F &&run) const
        {
            auto engine = _stream.LockEngine();
            run(engine, _stream);
        }

    public:
        LinearSolverAlgebra() = default;
        explicit LinearSolverAlgebra(Runtime::ExecutionStream stream) : _stream(std::move(stream)) {}
        static State NewVector(const State &prototype) { return Operations::NewVector(prototype); }
        static void CheckCompatible(const State &a, const State &b) { Operations::CheckCompatible(a, b); }
        static void RequireDistinct(const State &a, const State &b) { Operations::RequireDistinct(a, b); }
        void Copy(State &out, const State &x) const
        {
            if (Operations::SameStorage(out, x))
                return;
            Submit([&](auto &engine, const auto &stream)
                   {
                if constexpr (std::is_same_v<State, EquationDofs>) EquationDofOperations::Copy(engine, stream, out, x);
                else Algebra::Copy(engine, stream, out, x); });
        }
        void CopyGuess(State &out, const State &x) const { Copy(out, x); }
        void Fill(State &out, real value) const
        {
            Submit([&](auto &engine, const auto &stream)
                   {
                if constexpr (std::is_same_v<State, EquationDofs>) EquationDofOperations::Fill(engine, stream, out, value);
                else Algebra::Fill(engine, stream, out, value); });
        }
        void Scale(State &out, real value) const
        {
            Submit([&](auto &engine, const auto &stream)
                   {
                if constexpr (std::is_same_v<State, EquationDofs>) EquationDofOperations::Scale(engine, stream, out, value);
                else Algebra::Scale(engine, stream, out, value); });
        }
        void Axpy(State &out, real value, const State &x) const
        {
            Submit([&](auto &engine, const auto &stream)
                   {
                if constexpr (std::is_same_v<State, EquationDofs>) EquationDofOperations::Axpy(engine, stream, out, value, x);
                else Algebra::Axpy(engine, stream, out, value, x); });
        }
        void LinearCombination(State &out, real a, const State &x, real b, const State &y) const
        {
            Submit([&](auto &engine, const auto &stream)
                   {
                if constexpr (std::is_same_v<State, EquationDofs>) EquationDofOperations::LinearCombination(engine, stream, out, a, x, b, y);
                else Algebra::LinearCombination(engine, stream, out, a, x, b, y); });
        }
    };
}
