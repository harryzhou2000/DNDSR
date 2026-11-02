#pragma once

#include "Solver/Direct.hpp"
#include "Solver/BlockInverse.hpp"
#include "DNDS/Algebra/BlockMatrixStorage.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Math/HardEigen.hpp"
#include <Eigen/LU>
#include <atomic>
#include <cmath>
#include <optional>
#include <utility>

namespace DNDS::Direct
{
    namespace Detail
    {
        template <class Scalar>
        struct NativeFactorView : LocalLUBase<NativeFactorView<Scalar>, Eigen::MatrixXd, std::vector<Eigen::VectorXd>>
        {
            using Base = LocalLUBase<NativeFactorView<Scalar>, Eigen::MatrixXd, std::vector<Eigen::VectorXd>>;
            Algebra::BlockMatrixView<Scalar> values;
            std::atomic<index> *fallbacks;
            NativeFactorView(ssp<SerialSymLUStructure> symbolic, Algebra::BlockMatrixView<Scalar> blocks, bool decomposed, std::atomic<index> *count = nullptr)
                : Base(std::move(symbolic)), values(blocks), fallbacks(count) { this->isDecomposed = decomposed; }
            auto GetDiag(index iCell) { return values(iCell, 0); }
            auto GetLower(index iCell, index j) { return values(iCell, 1 + j); }
            auto GetUpper(index iCell, index j) { return values(iCell, 1 + index(this->symLU->lowerTriStructure[iCell].size()) + j); }
            Eigen::MatrixXd InvertDiag(const Eigen::MatrixXd &matrix)
            {
                // Same diagonal scaling and guarded partial-pivot LU/SVD policy
                // as the active Euler local block inverse, without its storage.
                auto result = InvertDiagonalBlock(matrix);
                if (result.usedSVD && fallbacks)
                    fallbacks->fetch_add(1, std::memory_order_relaxed);
                return std::move(result.value);
            }
        };
    }

    /// Rank-local block factors on native Array storage. Symbolic preparation is
    /// shared and immutable after construction. Matrix writes invalidate factors
    /// by generation; native views/leases are acquired once outside arithmetic.
    /// No Euler storage class or legacy ArrayDOF is used.
    class NativeBlockLU
    {
        ssp<SerialSymLUStructure> _symbolic;
        Algebra::BlockMatrixStorage _matrix, _factors;
        uint64_t _layoutGeneration;
        uint64_t _symbolicGeneration = 0, _matrixVersion = 0, _factorVersion = 0;
        bool _valid = false;
        index _fallbacks = 0;
        bool CurrentVersions() const
        {
            return _valid && _symbolicGeneration == _symbolic->generation && _layoutGeneration == _symbolic->generation &&
                   _matrixVersion == _matrix.Values().Version() && _factorVersion == _factors.Values().Version();
        }
        static std::vector<index> Counts(const ssp<SerialSymLUStructure> &symbolic)
        {
            DNDS_check_throw_info(symbolic && symbolic->hasSymbolicPattern, "native blocks need a published symbolic pattern");
            const auto &parts = symbolic->localPartStarts;
            DNDS_check_throw_info(parts.size() >= 2 && parts.front() == 0 && parts.back() == symbolic->Num() && std::is_sorted(parts.begin(), parts.end()),
                                  "invalid native factor local parts");
            if (parts.size() > 2)
                for (index part = 0; part + 1 < index(parts.size()); ++part)
                    for (index iCell = parts[part]; iCell < parts[part + 1]; ++iCell)
                        for (const auto *row : {&symbolic->lowerTriStructure[iCell], &symbolic->upperTriStructure[iCell]})
                            for (auto other : *row)
                                DNDS_check_throw_info(other >= parts[part] && other < parts[part + 1],
                                                      "native factor local parts have cross-part dependencies; supply induced part graphs");
            std::vector<index> result(symbolic->Num());
            for (index iCell = 0; iCell < symbolic->Num(); ++iCell)
                result[iCell] = 1 + index(symbolic->lowerTriStructure[iCell].size()) + index(symbolic->upperTriStructure[iCell].size());
            return result;
        }

    public:
        /// One local triangular solve. Ranges follow factor positions and
        /// advance consecutively within each independent symbolic local part.
        /// The scope owns numeric leases; no MPI is performed between scans.
        /// Partial forward work is not a published final result. Finish is
        /// required; abandonment invalidates started, unfinished output writes.
        class TriangularSolve
        {
            struct State
            {
                ssp<SerialSymLUStructure> symbolic;
                uint64_t generation;
                index blockSize;
                Algebra::BlockMatrixStorage::ReadAccess matrix, factors;
                Runtime::Array<real> rhs, forward, result;
                Runtime::Array<real>::ReadLease input;
                std::optional<Runtime::Array<real>::WriteLease> middle, output;
                std::vector<index> lowerNext, upperNext;
                bool started = false, lowerCalled = false, upperCalled = false;
                State(const NativeBlockLU &owner, const Runtime::Array<real> &b, Runtime::Array<real> &y, Runtime::Array<real> &x)
                    : symbolic(owner._symbolic), generation(symbolic->generation), blockSize(owner._matrix.BlockSize()),
                      matrix(owner._matrix.Read()), factors(owner._factors.Read()), rhs(b), forward(y), result(x), input(rhs.Read())
                {
                    DNDS_check_throw_info(owner.CurrentVersions(), "native factorization is absent or stale");
                    const auto &parts = symbolic->localPartStarts;
                    DNDS_check_throw_info(parts.size() >= 2 && parts.front() == 0 && parts.back() == symbolic->Num() && std::is_sorted(parts.begin(), parts.end()),
                                          "invalid native triangular local parts");
                    lowerNext.assign(parts.begin(), parts.end() - 1);
                    upperNext.assign(parts.begin() + 1, parts.end());
                    middle.emplace(forward.Write());
                    try
                    {
                        output.emplace(result.Write());
                    }
                    catch (...)
                    {
                        middle->CancelUnmodified();
                        middle.reset();
                        throw;
                    }
                }
                ~State()
                {
                    for (auto *write : {&middle, &output})
                        if (*write)
                        {
                            if (started)
                                (*write)->Fail();
                            else
                                (*write)->CancelUnmodified();
                        }
                }
                void Check(index begin, index end, bool backward) const
                {
                    DNDS_check_throw_info(symbolic->hasSymbolicPattern && generation == symbolic->generation &&
                                              begin >= 0 && end >= begin && end <= symbolic->Num(),
                                          "stale symbolic data or invalid triangular solve range");
                    DNDS_check_throw_info(!backward || lowerCalled, "upper solve needs a prior lower phase");
                    const auto &parts = symbolic->localPartStarts;
                    for (index part = 0; part < index(lowerNext.size()); ++part)
                    {
                        const index first = std::max(begin, parts[part]), last = std::min(end, parts[part + 1]);
                        if (first >= last)
                            continue;
                        if (backward)
                            DNDS_check_throw_info(lowerNext[part] == parts[part + 1] && last == upperNext[part],
                                                  "upper solve needs complete forward part and descending ranges");
                        else
                            DNDS_check_throw_info(first == lowerNext[part], "lower solve ranges must advance consecutively within each local part");
                    }
                }
                void Advance(index begin, index end, bool backward)
                {
                    const auto &parts = symbolic->localPartStarts;
                    for (index part = 0; part < index(lowerNext.size()); ++part)
                    {
                        const index first = std::max(begin, parts[part]), last = std::min(end, parts[part + 1]);
                        if (first < last)
                        {
                            if (backward)
                                upperNext[part] = first;
                            else
                                lowerNext[part] = last;
                        }
                    }
                }
                bool Complete() const
                {
                    if (!lowerCalled || !upperCalled || generation != symbolic->generation)
                        return false;
                    const auto &parts = symbolic->localPartStarts;
                    for (index part = 0; part < index(lowerNext.size()); ++part)
                        if (lowerNext[part] != parts[part + 1] || upperNext[part] != parts[part])
                            return false;
                    return true;
                }
            };
            std::unique_ptr<State> _state;
            explicit TriangularSolve(std::unique_ptr<State> state) : _state(std::move(state)) {}
            friend class NativeBlockLU;

        public:
            TriangularSolve(const TriangularSolve &) = delete;
            TriangularSolve &operator=(const TriangularSolve &) = delete;
            TriangularSolve(TriangularSolve &&) noexcept = default;
            TriangularSolve &operator=(TriangularSolve &&) noexcept = default;
            /// No per-block readiness checks. Validate range dependencies once
            /// here, then use the generic CRTP numerical scan unchanged.
            void Lower(index begin, index end)
            {
                DNDS_check_throw_info(_state, "native triangular solve is finished or moved");
                auto &state = *_state;
                state.Check(begin, end, false);
                const auto b = Algebra::MakeEigenRowMatrices(state.input.View<Runtime::UniformTableRows>(), state.blockSize, 1);
                auto y = Algebra::MakeEigenRowMatrices(state.middle->View<Runtime::UniformTableRows>(), state.blockSize, 1);
                Detail::NativeFactorView<const real> factor(state.symbolic, state.factors.View(), true);
                state.started = true;
                factor.SolveLowerRange(b, y, begin, end);
                state.Advance(begin, end, false);
                state.lowerCalled = true;
            }
            void Upper(index begin, index end)
            {
                DNDS_check_throw_info(_state, "native triangular solve is finished or moved");
                auto &state = *_state;
                state.Check(begin, end, true);
                const auto y = Algebra::MakeEigenRowMatrices(state.middle->View<Runtime::UniformTableRows>(), state.blockSize, 1);
                auto x = Algebra::MakeEigenRowMatrices(state.output->View<Runtime::UniformTableRows>(), state.blockSize, 1);
                Detail::NativeFactorView<const real> factor(state.symbolic, state.factors.View(), true);
                state.started = true;
                factor.SolveUpperRange(y, x, begin, end);
                state.Advance(begin, end, true);
                state.upperCalled = true;
            }
            void Finish()
            {
                DNDS_check_throw_info(_state && _state->Complete(), "cannot publish an incomplete native triangular solve");
                _state->middle.reset();
                _state->output.reset();
                _state.reset();
            }
        };
        NativeBlockLU(ssp<SerialSymLUStructure> symbolic, index blockSize)
            : _symbolic(std::move(symbolic)), _matrix(Counts(_symbolic), blockSize), _factors(Counts(_symbolic), blockSize),
              _layoutGeneration(_symbolic->generation) {}
        Algebra::BlockMatrixStorage &Matrix() { return _matrix; }
        const Algebra::BlockMatrixStorage &Matrix() const { return _matrix; }
        const SerialSymLUStructure &Symbolic() const { return *_symbolic; }
        index LastInverseFallbacks() const { return _fallbacks; }
        bool FactorsCurrent() const
        {
            try
            {
                auto matrix = _matrix.Read();
                auto factors = _factors.Read();
                return CurrentVersions();
            }
            catch (const std::exception &)
            {
                return false;
            }
        }
        /// Numeric ranges are optional consecutive pivot ends. Empty uses the
        /// legacy full V1 algorithm; caller fills all matrix values first.
        void Factorize(const std::vector<index> &pivotEnds = {})
        {
            _valid = false;
            _fallbacks = 0;
            std::atomic<index> fallbacks{0};
            DNDS_check_throw_info(_layoutGeneration == _symbolic->generation, "native block layout belongs to another symbolic generation");
            DNDS_check_throw_info(pivotEnds.empty() || (pivotEnds.front() >= 0 && pivotEnds.back() == _symbolic->Num() &&
                                                        std::is_sorted(pivotEnds.begin(), pivotEnds.end())),
                                  "invalid native factor pivot ranges");
            {
                auto input = _matrix.Read();
                auto output = _factors.Write();
                const auto source = input.View();
                const auto target = output.View();
                for (index iCell = 0; iCell < _matrix.RowNumber(); ++iCell)
                    for (index j = 0; j < source.BlockNumber(iCell); ++j)
                        target(iCell, j) = source(iCell, j);
                Detail::NativeFactorView<real> factor(_symbolic, target, false, &fallbacks);
                if (pivotEnds.empty() || !_symbolic->Num())
                    factor.InPlaceDecompose();
                else
                {
                    factor.BeginNumericFactorization();
                    index begin = 0;
                    for (auto end : pivotEnds)
                    {
                        if (end > begin)
                            factor.DecomposeRange(begin, end);
                        begin = end;
                    }
                }
                _matrixVersion = _matrix.Values().Version();
            }
            _factorVersion = _factors.Values().Version();
            _fallbacks = fallbacks.load(std::memory_order_relaxed);
            _symbolicGeneration = _symbolic->generation;
            _valid = true;
        }
        /// Begin a staged solve with fully host-visible RHS. Retains allocations
        /// and symbolic metadata after original owners expire. Callers serialize
        /// symbolic changes; a changed published generation rejects later phases.
        TriangularSolve BeginSolve(const Runtime::Array<real> &rhs, Runtime::Array<real> &forward, Runtime::Array<real> &result) const
        {
            for (const auto *array : {&rhs, &std::as_const(forward), &std::as_const(result)})
                DNDS_check_throw_info(array->GetDevice() == Runtime::Device::Host() && array->RowCount() == _matrix.RowNumber() &&
                                          !array->Layout().IsVariable() && array->Layout().Width() == _matrix.BlockSize(),
                                      "native triangular vector shape/backend differs");
            const auto overlap = [](const auto &a, const auto &b)
            { return Algebra::Detail::Overlaps(a.Storage(), b.Storage()); };
            DNDS_check_throw_info(!overlap(rhs, forward) && !overlap(rhs, result) && !overlap(forward, result) &&
                                      !overlap(forward, _factors.Values()) && !overlap(result, _factors.Values()) &&
                                      !overlap(forward, _matrix.Values()) && !overlap(result, _matrix.Values()),
                                  "native triangular output aliases an input");
            return TriangularSolve(std::make_unique<TriangularSolve::State>(*this, rhs, forward, result));
        }
        /// Full local solve with distinct RHS, forward scratch and result.
        /// All may be AoS/SoA/AoSoA uniform DOF tables; factors remain local.
        void Solve(const Runtime::Array<real> &rhs, Runtime::Array<real> &forward, Runtime::Array<real> &result) const
        {
            auto solve = BeginSolve(rhs, forward, result);
            solve.Lower(0, _symbolic->Num());
            solve.Upper(0, _symbolic->Num());
            solve.Finish();
        }
    };
}
