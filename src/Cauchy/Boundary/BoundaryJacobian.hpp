#pragma once

#include "Cauchy/Boundary/BoundaryKind.hpp"
#include "Cauchy/Physics/FlowState.hpp"

namespace DNDS::Cauchy
{
    /** @brief Approximate exterior-state derivative for the flow block.
     * @details This is Euler's homogeneous wall reflection (linMode=1), not
     * the derivative of the thermal wall law or of a prescribed/open state.
     * Slip reflection is taken about the supplied wall normal speed; no-slip
     * reflection uses the complete wall velocity. Energy changes preserve
     * internal energy under that reflection. The result is column-major.
     * The caller resolves conditional boundary kinds and the wall velocity in
     * the mesh frame. No EOS, pose lookup, allocation or error handling occurs
     * here. Non-wall boundaries return zero, including extrapolation: this is
     * the established approximate implicit policy, not an exact derivative.
     */
    DNDS_DEVICE_CALLABLE inline std::array<real, 25> ApproximateFlowExteriorJacobian(
        BoundaryKind kind, const Vector3 &normal, const Vector3 &wallVelocity) noexcept
    {
        std::array<real, 25> result{};
        if (kind != BoundaryKind::SlipWall && kind != BoundaryKind::NoSlipWall)
            return result;
        for (int j = 0; j < 5; ++j)
            result[j * 5 + j] = 1;
        if (kind == BoundaryKind::SlipWall)
        {
            real wallNormal = 0;
            for (int d = 0; d < 3; ++d)
                wallNormal += wallVelocity[d] * normal[d];
            result[4] += 2 * wallNormal * wallNormal;
            for (int d = 0; d < 3; ++d)
            {
                result[1 + d] += 2 * wallNormal * normal[d];
                result[(1 + d) * 5 + 4] -= 2 * wallNormal * normal[d];
            }
            for (int column = 0; column < 3; ++column)
                for (int row = 0; row < 3; ++row)
                    result[(1 + column) * 5 + 1 + row] -= 2 * normal[row] * normal[column];
            return result;
        }
        // m' = 2*rho*v_wall-m; E' = E+2*rho*|v_wall|^2-2*v_wall.m.
        for (int d = 0; d < 3; ++d)
        {
            result[1 + d] = 2 * wallVelocity[d];
            result[(1 + d) * 5 + 1 + d] = -1;
            result[4] += 2 * wallVelocity[d] * wallVelocity[d];
            result[(1 + d) * 5 + 4] = -2 * wallVelocity[d];
        }
        return result;
    }
}
