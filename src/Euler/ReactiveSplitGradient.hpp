#pragma once

#include "DNDS/Defines.hpp"
#include <Eigen/QR>

namespace DNDS::Euler
{
    /**
     * @brief Squared norm of the least-squares normalized thermochemical gradient.
     * @param normal Sum of neighbor separation outer products, in length squared.
     * @param moments Sum of separation times normalized T/species increments.
     * @details The minimum-norm solve handles a rank-deficient one-dimensional
     * cell chain embedded in two or three dimensions. On a uniform chain this
     * averages the signed left/right normalized slopes before taking their
     * squared norm. No physical threshold, spatial expansion or communication
     * is introduced. The result has units of inverse length squared.
     */
    inline real ReactiveSplitCenteredGradientSquared(
        const Eigen::Matrix<real, 3, 3> &normal,
        const Eigen::Matrix<real, 3, Eigen::Dynamic> &moments)
    {
        return normal.completeOrthogonalDecomposition().solve(moments).squaredNorm();
    }
}
