#pragma once

#include "DNDS/Math/HardEigen.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include <Eigen/LU>
#include <cmath>

namespace DNDS::Direct
{
    template <class Matrix>
    struct DiagonalBlockInverseResult
    {
        Matrix value;
        bool usedSVD = false;
    };
    using DiagonalBlockInverse = DiagonalBlockInverseResult<Eigen::MatrixXd>;
    /// Active Euler local-block policy, independent of matrix/Array storage:
    /// diagonal row scaling, guarded partial-pivot LU, then filtered SVD.
    /// Preserve fixed matrix dimensions supplied by a kernel specialization;
    /// dynamic callers retain the same dynamic result. No size dispatch here.
    template <class Derived>
    auto TryInvertDiagonalBlock(const Eigen::MatrixBase<Derived> &matrix) noexcept
        -> NumericalResult<DiagonalBlockInverseResult<typename Derived::PlainObject>>
    {
        using Matrix = typename Derived::PlainObject;
        using Vector = Eigen::Matrix<real, Derived::RowsAtCompileTime, 1>;
        if (!(matrix.rows() > 0 && matrix.rows() == matrix.cols() && matrix.allFinite() &&
              matrix.diagonal().array().abs().minCoeff() > 0))
            return NumericalStatus{NumericalError::InvalidInput, 1};
        const Vector scale = matrix.diagonal().array().inverse();
        Matrix normalized = scale.asDiagonal() * matrix;
        DiagonalBlockInverseResult<Matrix> result;
        Eigen::PartialPivLU<Matrix> lu(normalized);
        const real conditionReciprocal = lu.rcond();
        bool usable = std::isfinite(conditionReciprocal) && conditionReciprocal >= 1e-7;
        if (usable)
        {
            result.value = lu.solve(Matrix::Identity(matrix.rows(), matrix.cols()));
            usable = result.value.allFinite();
        }
        if (!usable)
        {
            result.usedSVD = true;
            // HardEigen exposes dynamic matrices. Conversion belongs only to
            // the exceptional SVD path, not every healthy fixed-size pivot.
            const Eigen::MatrixXd dynamicNormalized = normalized;
            Eigen::MatrixXd inverse;
            HardEigen::EigenLeastSquareInverse_Filtered(dynamicNormalized, inverse, 0, 0);
            result.value = inverse;
        }
        if (!result.value.allFinite())
            return NumericalStatus{NumericalError::InvalidState, 2};
        result.value = (result.value * scale.asDiagonal()).eval();
        return result;
    }

    /// Checked library adapter. CPU numerical kernels use TryInvertDiagonalBlock
    /// and propagate its status; only the operator boundary chooses error policy.
    template <class Derived>
    auto InvertDiagonalBlock(const Eigen::MatrixBase<Derived> &matrix)
    {
        auto result = TryInvertDiagonalBlock(matrix);
        DNDS_check_throw_info(result.status, "invalid or nonfinite native block inverse");
        return std::move(result.value);
    }
}
