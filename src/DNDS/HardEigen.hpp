#pragma once
/// @file HardEigen.hpp
/// @brief Robust linear-algebra primitives for small / moderately-sized
/// Eigen matrices that are more numerically careful than Eigen's default paths.
///
/// These are used on the "hot" side of reconstruction / limiter code where
/// matrices may be nearly rank-deficient and a plain `A.inverse()` or
/// `A.jacobiSvd().solve(b)` would introduce large errors.

#include "Defines.hpp"
#include <limits>

namespace DNDS::HardEigen
{
    /// Setup diagnostics from the same SVD used to form the inverse.
    /// Conditions describe the raw and retained spectra respectively; rank
    /// zero and an exactly singular raw matrix report infinity.
    struct SVDInverseDiagnostics
    {
        Eigen::Index modes = 0, rank = 0;
        real relativeCutoff = 0, rawCondition = 0, retainedCondition = 0;
    };

    /// Standard truncated SVD with explicit relative/absolute cutoffs.
    /// Equality is discarded; zero tolerances retain every strictly positive
    /// singular mode. Unlike the legacy zero-setting API, this has no implicit
    /// Eigen tolerance. Bounded Eigen matrix types keep small setup scratch
    /// bounded too. The coefficient metric is the input matrix's own metric.
    template <class Matrix>
    SVDInverseDiagnostics EigenSVDInverse(const Matrix &A, Matrix &inverse, real relativeCutoff, real absoluteCutoff = 0)
    {
        DNDS_check_throw_info(A.rows() > 0 && A.cols() > 0 && A.allFinite() &&
                                  std::isfinite(relativeCutoff) && relativeCutoff >= 0 &&
                                  std::isfinite(absoluteCutoff) && absoluteCutoff >= 0,
                              "invalid SVD inverse matrix or cutoff");
        Eigen::JacobiSVD<Matrix> factor(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        DNDS_check_throw_info(factor.info() == Eigen::Success, "local SVD did not converge");
        const auto &singular = factor.singularValues();
        const real largest = singular[0], smallest = singular[singular.size() - 1];
        const real cutoff = largest > 0 ? std::max(relativeCutoff, absoluteCutoff / largest) : relativeCutoff;
        // Eigen versions differ at threshold equality. Enforce our strict
        // predicate explicitly, comparing relative scales without multiplying
        // large values or losing a positive mode when relativeCutoff is zero.
        auto reciprocals = singular.eval();
        Eigen::Index rank = 0;
        for (Eigen::Index iMode = 0; iMode < singular.size(); ++iMode)
            if (singular[iMode] > absoluteCutoff &&
                (relativeCutoff == 0 || singular[iMode] / largest > relativeCutoff))
            {
                reciprocals[iMode] = 1 / singular[iMode];
                ++rank;
            }
            else
                reciprocals[iMode] = 0;
        inverse.noalias() = factor.matrixV() * reciprocals.asDiagonal() * factor.matrixU().transpose();
        DNDS_check_throw_info(inverse.allFinite(), "SVD inverse is not representable at the selected cutoff");
        SVDInverseDiagnostics diagnostics;
        diagnostics.modes = singular.size();
        diagnostics.rank = rank;
        diagnostics.relativeCutoff = cutoff;
        diagnostics.rawCondition = smallest > 0 ? largest / smallest : std::numeric_limits<real>::infinity();
        diagnostics.retainedCondition = diagnostics.rank ? largest / singular[diagnostics.rank - 1]
                                                         : std::numeric_limits<real>::infinity();
        return diagnostics;
    }

    /// @brief Moore-Penrose pseudoinverse via SVD, dropping singular values below
    /// `svdTol` (relative to the largest). Returns the largest / smallest ratio
    /// (condition number).
    real EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI, real svdTol = 0,
                                 SVDInverseDiagnostics *diagnostics = nullptr);

    /**
     * @brief Pseudoinverse with a choice of singular-value filter.
     *
     * @param A       Input matrix.
     * @param AI      Output pseudoinverse.
     * @param svdTol  Filter tolerance (relative to the largest singular value).
     * @param mode    Filter selection:
     *                - `0`: standard "lsqminnorm" -- drop *smallest* singular values
     *                  below the tolerance (stabilises rank-deficient systems);
     *                - `1`: drop *largest* singular values above `1/svdTol` relative
     *                  to the smallest (inverse filtering, rarely needed).
     * @return Legacy regularized raw condition estimate, before filtering.
     * Optional diagnostics report unregularized raw/retained conditions.
     */
    real EigenLeastSquareInverse_Filtered(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI, real svdTol = 0, int mode = 0,
                                          SVDInverseDiagnostics *diagnostics = nullptr);

    /// @brief Least-squares solve `A * AIB ~= B` via a rank-revealing QR-style
    /// decomposition; returns the computed rank of `A`.
    Eigen::Index EigenLeastSquareSolve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &AIB);

    /// @brief Analytic eigen-decomposition of a 3x3 real symmetric matrix.
    /// Returns the eigenvector matrix (columns = eigenvectors, scaled so that
    /// diagonal == eigenvalues).
    Eigen::Matrix3d Eigen3x3RealSymEigenDecomposition(const Eigen::Matrix3d &A);
    /// @brief Analytic 2x2 analogue of @ref Eigen3x3RealSymEigenDecomposition.
    Eigen::Matrix2d Eigen2x2RealSymEigenDecomposition(const Eigen::Matrix2d &A);

    /// @brief Condition number of a 3x3 SPD matrix from its eigenvalues.
    real Eigen3x3RealSymEigenDecompositionGetCond(const Eigen::Matrix3d &A);
    /// @brief Like @ref Eigen3x3RealSymEigenDecompositionGetCond but returns
    /// `lambda0 / lambda1` only (ignores the smallest eigenvalue).
    real Eigen3x3RealSymEigenDecompositionGetCond01(const Eigen::Matrix3d &A);
    /// @brief 2x2 analogue of @ref Eigen3x3RealSymEigenDecompositionGetCond.
    real Eigen2x2RealSymEigenDecompositionGetCond(const Eigen::Matrix2d &A);

    /// @brief Eigen-decomposition with eigenvector columns normalised to unit length.
    Eigen::Matrix3d Eigen3x3RealSymEigenDecompositionNormalized(const Eigen::Matrix3d &A);
    /// @brief 2x2 analogue of @ref Eigen3x3RealSymEigenDecompositionNormalized.
    Eigen::Matrix2d Eigen2x2RealSymEigenDecompositionNormalized(const Eigen::Matrix2d &A);
}
