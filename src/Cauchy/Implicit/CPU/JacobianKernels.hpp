#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::Cauchy::CPU
{
    using FrozenFluxRow6 = Runtime::RowSpan<const real, Runtime::AccessBoundsChecks, 6, 6, 1>;
    using FrozenFluxMatrix6 = Runtime::RowSpan<real, Runtime::AccessBoundsChecks, 36, 36, 1>;
    using FrozenFluxDirection6 = Runtime::RowSpan<const real, Runtime::AccessBoundsChecks, 6, 6>;
    using FrozenFluxProduct6 = Runtime::RowSpan<real, Runtime::AccessBoundsChecks, 6, 6, 1>;

    /// CPU leaf for an ordered flow5 plus one scalar in an identity component
    /// frame. All EOS information is explicit, including the scalar pressure
    /// derivative. Add to an existing column-major six-by-six block, retaining
    /// every coupling and the distinct flow/scalar normal-speed expressions.
    /// Caller validates the shape/selection/frame once and retains all input
    /// and output leases. Views borrow contiguous components; no packing,
    /// allocation, geometry, source term or communication occurs here.
    void AddFrozenFluxBlock6(FrozenFluxRow6 mean, FrozenFluxRow6 pressureDerivative, real pressure,
                             const Vector3 &normal, const Vector3 &gridVelocity,
                             real factor, real lambda, FrozenFluxMatrix6 output);

    /// Directional counterpart for the same complete six-component state.
    /// The direction keeps its actual component stride; no row packing or
    /// dense matrix is constructed. Adds to caller-owned contiguous scratch.
    /// Every EOS pressure derivative, including the scalar column, is retained.
    // Inline at the row-action boundary so fixed spans do not cross an
    // out-of-line ABI as repeated three-word descriptors for every product.
    inline void AddFrozenFluxProduct6(FrozenFluxRow6 mean, FrozenFluxRow6 pressureDerivative, real pressure,
                                      const Vector3 &normal, const Vector3 &gridVelocity,
                                      real factor, real lambda, FrozenFluxDirection6 direction, FrozenFluxProduct6 output)
    {
        FlowState flow{}, delta{};
        real dp = 0;
        for (index j = 0; j < 6; ++j)
            dp += pressureDerivative[j] * direction[j];
        for (int j = 0; j < 5; ++j)
        {
            flow[j] = mean[j];
            delta[j] = direction[j];
        }
        const auto derivative = PrepareCompressibleFluxDerivative(flow, delta, pressure, dp, normal, gridVelocity);
        real speed = 0, deltaSpeed = 0;
        for (int d = 0; d < 3; ++d)
        {
            const real velocity = flow[1 + d] / flow[0];
            speed += (velocity - gridVelocity[d]) * normal[d];
            deltaSpeed += (delta[1 + d] - velocity * delta[0]) / flow[0] * normal[d];
        }
        for (int j = 0; j < 5; ++j)
            output[j] += factor * (derivative[j] - lambda * delta[j]);
        output[5] += factor * (PrepareAdvectedScalarFluxDerivative(mean[5], direction[5], speed, deltaSpeed) - lambda * direction[5]);
    }
}
