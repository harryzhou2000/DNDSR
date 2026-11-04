#pragma once

#include "DNDS/Core/Defines.hpp"
#include <Eigen/Core>
#include <cmath>

namespace DNDS::Physics::Transport
{
    /// Euler's QCR correction to an already assembled total Newtonian stress.
    /// diffVelocity(direction,component) is the primitive velocity derivative;
    /// viscosityRatio is mu_t/(mu_total+verySmallReal). The equivalent tensor
    /// correction is -0.3*ratio*(O*stress-stress*O), with
    /// O=(diffVelocity.transpose()-diffVelocity)/(||diffVelocity||_F+eps).
    /// Preserve Euler's explicit symmetric evaluation and coefficient ordering.
    /// No closure selection, EOS, allocation, repair or synchronization occurs.
    /// Caller supplies distinct local stress/gradient matrices in one frame.
    template <int dim, class TStress, class TVelocityDerivative>
    DNDS_DEVICE_CALLABLE void ApplyQCRStress(TStress &stress, const TVelocityDerivative &diffVelocity, real viscosityRatio)
    {
        static_assert(dim == 2 || dim == 3, "QCR needs two or three velocity dimensions");
        const real norm = std::sqrt((diffVelocity.array() * diffVelocity.array()).sum());
        const Eigen::Matrix<real, dim, dim> rotation = (diffVelocity.transpose() - diffVelocity) / (norm + verySmallReal);
        Eigen::Matrix<real, dim, dim> correction;
        correction.setZero();
        correction.diagonal() = (stress.array() * rotation.array()).rowwise().sum();
        correction(0, 1) = rotation(0, 1) * (stress(1, 1) - stress(0, 0));
        if constexpr (dim == 3)
        {
            correction(0, 1) += rotation(1, 2) * stress(0, 2) + rotation(0, 2) * stress(1, 2);
            correction(0, 2) = rotation(0, 2) * (stress(2, 2) - stress(0, 0)) + rotation(0, 1) * stress(2, 1) + rotation(2, 1) * stress(0, 1);
            correction(1, 2) = rotation(1, 2) * (stress(2, 2) - stress(1, 1)) + rotation(1, 0) * stress(2, 0) + rotation(2, 0) * stress(1, 0);
        }
        const Eigen::Matrix<real, dim, dim> symmetric = correction + correction.transpose();
        stress -= 0.3 * viscosityRatio * symmetric;
    }
}
