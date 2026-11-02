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
            std::vector<index> result(symbolic->Num());
            for (index iCell = 0; iCell < symbolic->Num(); ++iCell)
                result[iCell] = 1 + index(symbolic->lowerTriStructure[iCell].size()) + index(symbolic->upperTriStructure[iCell].size());
            return result;
        }

    public:
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
        /// Full local solve with distinct RHS, forward scratch and result.
        /// All may be AoS/SoA/AoSoA uniform DOF tables; factors remain local.
        void Solve(const Runtime::Array<real> &rhs, Runtime::Array<real> &forward, Runtime::Array<real> &result) const
        {
            auto matrixGuard = _matrix.Read();
            auto factors = _factors.Read();
            DNDS_check_throw_info(CurrentVersions(), "native factorization is absent or stale");
            for (const auto *array : {&rhs, &std::as_const(forward), &std::as_const(result)})
                DNDS_check_throw_info(array->GetDevice() == Runtime::Device::Host() && array->RowCount() == _matrix.RowNumber() &&
                                          !array->Layout().IsVariable() && array->Layout().Width() == _matrix.BlockSize(),
                                      "native triangular vector shape/backend differs");
            const auto overlap = [](const auto &a, const auto &b)
            { return Algebra::Detail::Overlaps(a.Storage(), b.Storage()); };
            DNDS_check_throw_info(!overlap(rhs, forward) && !overlap(rhs, result) && !overlap(forward, result) &&
                                      !overlap(forward, _factors.Values()) && !overlap(result, _factors.Values()),
                                  "native triangular output aliases an input");
            auto input = rhs.Read();
            auto middle = forward.Write();
            std::optional<Runtime::Array<real>::WriteLease> output;
            try
            {
                output.emplace(result.Write());
            }
            catch (...)
            {
                middle.CancelUnmodified();
                throw;
            }
            const auto b = Algebra::MakeEigenRowMatrices(input.View<Runtime::UniformTableRows>(), _matrix.BlockSize(), 1);
            auto y = Algebra::MakeEigenRowMatrices(middle.View<Runtime::UniformTableRows>(), _matrix.BlockSize(), 1);
            auto x = Algebra::MakeEigenRowMatrices(output->View<Runtime::UniformTableRows>(), _matrix.BlockSize(), 1);
            Detail::NativeFactorView<const real> factor(_symbolic, factors.View(), true);
            factor.SolveLowerRange(b, y, 0, _symbolic->Num());
            factor.SolveUpperRange(y, x, 0, _symbolic->Num());
        }
    };
}
