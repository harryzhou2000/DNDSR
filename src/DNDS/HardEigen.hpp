#pragma once
#include "Defines.hpp"

namespace DNDS::HardEigen
{

    real EigenLeastSquareInverse(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI);
    real EigenLeastSquareInverse_Filtered(const Eigen::MatrixXd &A, Eigen::MatrixXd &AI);

    Eigen::Index EigenLeastSquareSolve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &AIB);


    Eigen::Matrix3d Eigen3x3RealSymEigenDecomposition(const Eigen::Matrix3d &A);
    Eigen::Matrix2d Eigen2x2RealSymEigenDecomposition(const Eigen::Matrix2d &A);

    real Eigen3x3RealSymEigenDecompositionGetCond(const Eigen::Matrix3d &A);
    real Eigen3x3RealSymEigenDecompositionGetCond01(const Eigen::Matrix3d &A);
    real Eigen2x2RealSymEigenDecompositionGetCond(const Eigen::Matrix2d &A);
    
    Eigen::Matrix3d Eigen3x3RealSymEigenDecompositionNormalized(const Eigen::Matrix3d &A);
    Eigen::Matrix2d Eigen2x2RealSymEigenDecompositionNormalized(const Eigen::Matrix2d &A);
}
