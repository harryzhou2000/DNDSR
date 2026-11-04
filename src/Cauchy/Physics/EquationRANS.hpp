#pragma once

#include "Physics/RANS/Model.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"

namespace DNDS::Cauchy
{
    struct RANSStateSettings;
    /// Resolve a complete named conservative RANS row at setup, validating
    /// extent, model-specific admissibility and unknown names. A non-RANS
    /// equation accepts only an empty map. No allocation or name lookup is
    /// needed by the subsequent cell/point kernels.
    std::array<real, 2> ResolveRANSValues(const EquationSystem &equation,
                                          const std::map<std::string, real> &values);
    /// Plain scalar policy, separate from the flow5 thermal admissibility rule.
    /// Resolve outside a loop and apply to the logical RANS row, not a physical
    /// allocation's tail. Rows may be strided; no pointers or Array owners are
    /// retained. All values are conservative and in common code units.
    struct RANSStateView
    {
        Physics::RANS::ModelKind kind = Physics::RANS::ModelKind::None;
        int components = 0;
        real saUpperLimit = 1e5, omegaLowerLimit = 0;

        /// Euler's model-specific mean admissibility, before property kernels.
        template <class Row>
        bool Admissible(const Row &values) const
        {
            for (int j = 0; j < components; ++j)
            {
                if (!std::isfinite(values[j]))
                    return false;
#ifdef USE_NS_SA_ALLOW_NEGATIVE_MEAN
                if (kind == Physics::RANS::ModelKind::SA)
                    continue;
#endif
                if (values[j] < 0)
                    return false;
            }
            return true;
        }

        /// Euler reconstruction fallback is component-wise: an invalid RANS
        /// quadrature value reverts to its cell mean, without flattening flow
        /// or other RANS components. SA follows its existing compile-time flag.
        template <class Mean, class Row>
        void RepairReconstruction(const Mean &mean, Row &&values) const
        {
            for (int j = 0; j < components; ++j)
            {
#ifndef USE_NS_SA_NUT_REDUCED_ORDER
                if (kind == Physics::RANS::ModelKind::SA)
                    continue;
#endif
                if (values[j] < 0)
                    values[j] = mean[j];
            }
        }

        /// Apply after the common flow5 increment compression. Preserve Euler's
        /// exponential scalar repair and SA-only 1e-6 floor, not a new common
        /// line search. Precondition: the old mean is admissible and finite.
        template <class Mean, class Delta>
        void RepairIncrement(const Mean &mean, Delta &&delta) const
        {
            for (int j = 0; j < components; ++j)
            {
#ifdef USE_NS_SA_ALLOW_NEGATIVE_MEAN
                if (kind == Physics::RANS::ModelKind::SA)
                    continue;
#endif
                if (mean[j] + delta[j] < 0)
                {
                    real next = mean[j] * std::exp(delta[j] / (mean[j] + 1e-6));
                    if (kind == Physics::RANS::ModelKind::SA)
                        next = std::max(real(1e-6), next);
                    delta[j] = next - mean[j];
                }
            }
        }

        /// Apply after adding the repaired increment. Omega's floor references
        /// the prescribed far-field conservative value, not the local density.
        template <class Row>
        void RepairMean(real density, Row &&values) const
        {
            if (kind == Physics::RANS::ModelKind::SA)
                values[0] = std::min(values[0], saUpperLimit * density);
            else if (kind == Physics::RANS::ModelKind::KOmegaWilcox || kind == Physics::RANS::ModelKind::KOmegaSST)
                values[1] = std::max(values[1], omegaLowerLimit);
        }
    };

    /// Setup validation for the plain policy. farfield contains the resolved
    /// conservative RANS values; no name lookup occurs in numerical kernels.
    RANSStateView PrepareRANSStatePolicy(const EquationSystem &equation,
                                         const std::array<real, 2> &farfield,
                                         real saUpperLimit = 1e5, real lowerFactor = .01);
    /// Omega's numerical floor references an explicit prescribed state; it
    /// never guesses from the initial condition or one of several boundaries.
    RANSStateView ResolveRANSStatePolicy(const EquationSystem &equation, const RANSStateSettings &settings);

    /// Gather into the shared bounded point scratch. Logical flow/RANS row
    /// views preserve AoS/SoA/AoSoA strides. Derivative accessors are indexed
    /// (component,direction), as in Cauchy; shared RANS stores the transpose.
    /// Caller supplies molecular properties, scalar wall distance and cell
    /// length explicitly. This performs no EOS evaluation, clipping or leasing.
    template <class Flow, class Scalars, class FlowDerivative, class ScalarDerivative>
    Physics::RANS::Point PrepareRANSPoint(const RANSStateView &policy, const Flow &flow, const Scalars &scalars,
                                          const FlowDerivative &flowDerivative, const ScalarDerivative &scalarDerivative,
                                          real molecularViscosity, real cp, real gamma, real wallDistance, real cellLength)
    {
        Physics::RANS::Point point;
        for (int j = 0; j < 5; ++j)
        {
            point.state[j] = flow[j];
            for (int d = 0; d < 3; ++d)
                point.gradient(d, j) = flowDerivative(j, d);
        }
        for (int j = 0; j < policy.components; ++j)
        {
            point.state[5 + j] = scalars[j];
            for (int d = 0; d < 3; ++d)
                point.gradient(d, 5 + j) = scalarDerivative(j, d);
        }
        point.molecularViscosity = molecularViscosity;
        point.cp = cp;
        point.gamma = gamma;
        point.wallDistance = wallDistance;
        point.cellLength = cellLength;
        return point;
    }

    /// Euler no-slip RANS target. The caller selects viscosity at the prescribed
    /// wall temperature for an isothermal wall, otherwise at the cell mean.
    /// meanWallDistance is the arithmetic mean of cell quadrature samples.
    /// A physical face receives target; a Riemann exterior receives 2*target-U.
    /// SA/k targets are zero. Epsilon uses cell-mean k and reconstructed interior
    /// density, including for isothermal walls (not the prescribed wall density).
    std::array<real, 2> RANSWallTarget(const Physics::RANS::Model &model,
                                       const Physics::RANS::Point &mean, real interiorDensity);

    /// Euler two-equation isotropic Reynolds stress, evaluated at the
    /// arithmetic conservative face state. This is separate from EOS pressure
    /// and Riemann acoustics. Energy work uses absolute normal velocity in the
    /// mesh-bound axes, including for ALE; grid velocity is not subtracted.
    DNDS_DEVICE_CALLABLE inline FlowState TurbulentNormalFlux(const FlowState &state, real rhoK, const Vector3 &normal)
    {
        FlowState flux{};
        const real pressure = (2. / 3.) * rhoK;
        for (int d = 0; d < 3; ++d)
        {
            flux[1 + d] = pressure * normal[d];
            flux[4] += flux[1 + d] * state[1 + d] / state[0];
        }
        return flux;
    }
    /// Directional conservative derivative of TurbulentNormalFlux. Callers
    /// map the scalar direction through the logical RANS block; it is not
    /// necessarily component five in physical storage. No gamma/EOS closure.
    DNDS_DEVICE_CALLABLE inline FlowState TurbulentNormalFluxDerivative(const FlowState &state, const FlowState &direction,
                                                                        real rhoK, real deltaRhoK, const Vector3 &normal)
    {
        FlowState flux{};
        for (int d = 0; d < 3; ++d)
        {
            const real velocity = state[1 + d] / state[0];
            flux[1 + d] = (2. / 3.) * deltaRhoK * normal[d];
            flux[4] += (2. / 3.) * normal[d] * (deltaRhoK * velocity + rhoK * (direction[1 + d] - velocity * direction[0]) / state[0]);
        }
        return flux;
    }
}
