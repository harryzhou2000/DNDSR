#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/HardEigen.hpp"

TEST_CASE("SVD cutoff retains constrained modes and reports the actual rank")
{
    Eigen::MatrixXd A = Eigen::Vector3d(1, 1e-8, 1e-14).asDiagonal();
    Eigen::MatrixXd inverse;
    DNDS::HardEigen::SVDInverseDiagnostics diagnostics;
    DNDS::HardEigen::EigenLeastSquareInverse_Filtered(A, inverse, 1e-12, 0, &diagnostics);
    CHECK(inverse.isApprox(Eigen::Vector3d(1, 1e8, 0).asDiagonal().toDenseMatrix(), 1e-15));
    CHECK(diagnostics.rank == 2);
    CHECK(diagnostics.modes == 3);
    CHECK(diagnostics.rawCondition == doctest::Approx(1e14));
    CHECK(diagnostics.retainedCondition == doctest::Approx(1e8));

    // Equality is excluded by the explicit relative filter.
    DNDS::HardEigen::EigenLeastSquareInverse_Filtered(A, inverse, 1e-8, 0, &diagnostics);
    CHECK(diagnostics.rank == 1);
    CHECK(inverse(1, 1) == 0);

    // Orthogonal rotations must preserve the retained spectral subspace.
    const Eigen::Matrix3d Q = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    A = Q * Eigen::Vector3d(1, 1e-8, 1e-14).asDiagonal() * Q.transpose();
    DNDS::HardEigen::EigenLeastSquareInverse_Filtered(A, inverse, 1e-12, 0, &diagnostics);
    CHECK(diagnostics.rank == 2);
    CHECK(inverse.isApprox(Q * Eigen::Vector3d(1, 1e8, 0).asDiagonal() * Q.transpose(), 1e-7));
}

TEST_CASE("SVD diagnostics match automatic Eigen cutoff and rank-zero matrices")
{
    Eigen::MatrixXd A = Eigen::Vector3d(2, 1, 0).asDiagonal(), inverse;
    DNDS::HardEigen::SVDInverseDiagnostics diagnostics;
    DNDS::HardEigen::EigenLeastSquareInverse(A, inverse, 0, &diagnostics);
    CHECK(diagnostics.rank == 2);
    CHECK(std::isinf(diagnostics.rawCondition));
    CHECK(diagnostics.retainedCondition == doctest::Approx(2));
    CHECK(inverse.isApprox(Eigen::Vector3d(0.5, 1, 0).asDiagonal().toDenseMatrix()));
    A.setZero();
    DNDS::HardEigen::EigenLeastSquareInverse(A, inverse, 0, &diagnostics);
    CHECK(diagnostics.rank == 0);
    CHECK(std::isinf(diagnostics.retainedCondition));
    CHECK(inverse.isZero());
    DNDS::HardEigen::EigenLeastSquareInverse_Filtered(A, inverse, 1e-12, 0, &diagnostics);
    CHECK(diagnostics.rank == 0);
    CHECK(inverse.isZero());
}

TEST_CASE("Explicit SVD cutoffs are scale relative with optional absolute threshold")
{
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 20, 20>;
    Matrix A = Eigen::Vector3d(1, 1e-8, 1e-14).asDiagonal(), inverse;
    auto diagnostics = DNDS::HardEigen::EigenSVDInverse(A, inverse, 1e-12);
    CHECK(diagnostics.rank == 2);
    CHECK(inverse(0, 0) == doctest::Approx(1));
    CHECK(inverse(1, 1) == doctest::Approx(1e8));
    CHECK(inverse(2, 2) == 0);
    A *= 1e-100;
    diagnostics = DNDS::HardEigen::EigenSVDInverse(A, inverse, 1e-12);
    CHECK(diagnostics.rank == 2);
    CHECK(inverse(1, 1) == doctest::Approx(1e108));
    diagnostics = DNDS::HardEigen::EigenSVDInverse(A, inverse, 1e-12, 1e-105);
    CHECK(diagnostics.rank == 1);
    diagnostics = DNDS::HardEigen::EigenSVDInverse(A, inverse, 1);
    CHECK(diagnostics.rank == 0);
    CHECK(inverse.isZero());
    CHECK_THROWS(DNDS::HardEigen::EigenSVDInverse(A, inverse, -1));
    A(0, 0) = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(DNDS::HardEigen::EigenSVDInverse(A, inverse, 1e-12));
}
