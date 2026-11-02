#pragma once

#include "DNDS/Math/HardEigen.hpp"
#include <Eigen/LU>
#include <cmath>

namespace DNDS::Direct
{
    struct DiagonalBlockInverse
    {
        Eigen::MatrixXd value;
        bool usedSVD = false;
    };
    /// Active Euler local-block policy, independent of matrix/Array storage:
    /// diagonal row scaling, guarded partial-pivot LU, then filtered SVD.
    inline DiagonalBlockInverse InvertDiagonalBlock(const Eigen::MatrixXd &matrix)
    {
        DNDS_check_throw_info(matrix.rows() > 0 && matrix.rows() == matrix.cols() && matrix.allFinite() &&
                                  matrix.diagonal().array().abs().minCoeff() > 0,
                              "invalid or zero diagonal in native block factorization");
        const Eigen::VectorXd scale = matrix.diagonal().array().inverse();
        Eigen::MatrixXd normalized = scale.asDiagonal() * matrix;
        DiagonalBlockInverse result;
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(normalized);
        if (std::isfinite(lu.rcond()) && lu.rcond() >= 1e-7)
            result.value = lu.solve(Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()));
        if (!result.value.size() || !result.value.allFinite())
        {
            result.usedSVD = true;
            HardEigen::EigenLeastSquareInverse_Filtered(normalized, result.value, 0, 0);
        }
        DNDS_check_throw_info(result.value.allFinite(), "native block inverse is nonfinite");
        result.value = (result.value * scale.asDiagonal()).eval();
        return result;
    }
}
