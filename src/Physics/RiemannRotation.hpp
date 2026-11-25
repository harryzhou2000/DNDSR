#pragma once

#include "DNDS/Math/ComputeScalar.hpp"

namespace DNDS::Physics
{
    /// Skip directional splitting when the cell-mean velocity jump is too
    /// small to normalize. The absolute floor also covers stagnant states.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline bool RiemannRotationSmallJump(Scalar jump, Scalar speedSum) noexcept
    {
        return jump <= Scalar(1000 * smallReal) * speedSum || jump <= Scalar(1e-98);
    }

    /// Blend the face-center normal toward the consistently oriented velocity
    /// jump, linearly in log10 of the relative jump over one decade.
    /// Call after applying RiemannRotationSmallJump.
    template <int Dim, class Vector, class Scalar>
    DNDS_DEVICE_CALLABLE inline Vector RiemannVelocityJumpAxis(
        const Vector &velocityJump, Scalar jump, Scalar speedSum, const Vector &faceNormal) noexcept
    {
        const Scalar relativeJump = jump / (speedSum + Scalar(verySmallReal));
        if (relativeJump <= Scalar(1e-3))
            return faceNormal;

        Scalar projection = 0;
        for (int d = 0; d < Dim; ++d)
            projection += velocityJump[d] * faceNormal[d];
        const Scalar orientation = projection < Scalar(0) ? Scalar(-1) : Scalar(1);
        Vector axis = faceNormal;
        for (int d = 0; d < Dim; ++d)
            axis[d] = orientation * velocityJump[d] / jump;
        if (relativeJump >= Scalar(1e-2))
            return axis;

        const Scalar weight = Scalar(std::log10(static_cast<real>(relativeJump)) + 3);
        Scalar normSquared = 0;
        for (int d = 0; d < Dim; ++d)
        {
            axis[d] = (Scalar(1) - weight) * faceNormal[d] + weight * axis[d];
            normSquared += axis[d] * axis[d];
        }
        const Scalar norm = Math::ScalarSqrt(normSquared);
        for (int d = 0; d < Dim; ++d)
            axis[d] /= norm;
        return axis;
    }
}
