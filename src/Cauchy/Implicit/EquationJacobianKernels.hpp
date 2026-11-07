#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::Cauchy::Detail
{
    using FrozenFluxRow6 = Runtime::RowSpan<const real, Runtime::AccessBoundsChecks, 6, 6, 1>;
    using FrozenFluxMatrix6 = Runtime::RowSpan<real, Runtime::AccessBoundsChecks, 36, 36, 1>;

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
}
