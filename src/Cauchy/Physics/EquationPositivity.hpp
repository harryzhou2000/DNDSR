#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "Physics/Chemistry/SpeciesAdmissibility.hpp"

namespace DNDS::Cauchy
{
    namespace EquationPositivityDetail
    {
        template <class Values>
        struct PointRow
        {
            const Values &values;
            index point;
            DNDS_DEVICE_CALLABLE real operator[](index j) const { return values(point, j); }
        };
        template <class Base, class Delta>
        struct SegmentRow
        {
            const Base &base;
            const Delta &delta;
            real fraction;
            DNDS_DEVICE_CALLABLE real operator[](index j) const { return base[j] + fraction * delta[j]; }
        };
        template <class Base, class Higher>
        struct DifferenceRow
        {
            const Base &base;
            const Higher &higher;
            real scale;
            DNDS_DEVICE_CALLABLE real operator[](index j) const { return scale * (higher[j] - base[j]); }
        };
    }

    /// Frozen numerical view. Energy references are prepared e_k(T) values,
    /// not EOS callbacks. Caller retains the policy owner and declared input
    /// scopes. Rows supply operator[] and may span several physical arrays.
    /// This view contains no ownership, Array, worker, MPI or allocation.
    struct EquationThermalView
    {
        real densityFloor;
        std::array<real, 2> floors, densityCoefficients;
        std::array<const real *, 2> speciesEnergies{};
        index components, flowFirst, speciesFirst, independent;

        /// Flow and transported species share reconstruction compression.
        /// RANS reconstruction uses separate scalar rules. This is not the
        /// final increment policy: Euler scales the whole increment there.
        /// Indices are logical equation components, independent of storage layout.
        DNDS_DEVICE_CALLABLE bool CompressesReconstructionComponent(index j) const
        {
            return (j >= flowFirst && j < flowFirst + 5) ||
                   (j >= speciesFirst && j < speciesFirst + independent);
        }

        template <class Row>
        DNDS_DEVICE_CALLABLE FlowState Flow(const Row &values) const
        {
            FlowState result;
            for (int j = 0; j < 5; ++j)
                result[j] = values[flowFirst + j];
            return result;
        }
        /// Raw is affine in transported conservative values. Repaired mirrors
        /// the property producer's clipped/normalized temporary composition;
        /// it never changes the row or the conservative derivatives.
        template <class Row>
        DNDS_DEVICE_CALLABLE real Reference(const Row &values, int constraint, bool repaired = false) const
        {
            const real rho = values[flowFirst];
            const auto *energy = speciesEnergies[constraint];
            if (!energy)
                return rho * densityCoefficients[constraint];
            if (!repaired)
            {
                real value = rho * energy[independent];
                for (index k = 0; k < independent; ++k)
                    value += values[speciesFirst + k] * (energy[k] - energy[independent]);
                return value;
            }
            real sum = 0;
            for (index k = 0; k < independent; ++k)
                sum += std::max(real(0), values[speciesFirst + k] / rho);
            const real dependent = std::max(real(0), 1 - sum), total = sum + dependent;
            real value = 0;
            for (index k = 0; k < independent; ++k)
                value += std::max(real(0), values[speciesFirst + k] / rho) / total * energy[k];
            value += dependent / total * energy[independent];
            return rho * value;
        }
        template <class Row>
        DNDS_DEVICE_CALLABLE real Sensible(const Row &values, int constraint, bool repaired = false) const
        {
            return PositivityDetail::InternalEnergy(Flow(values)) - Reference(values, constraint, repaired);
        }
        /// Reference/mean checks also require the raw species simplex. Final
        /// quadrature checks may omit that test, matching Euler's thermal
        /// postcheck while preserving the property-repaired energy constraint.
        template <class Row>
        DNDS_DEVICE_CALLABLE bool Admissible(const Row &values, bool requireSpecies = true) const
        {
            for (index j = 0; j < components; ++j)
                if (!std::isfinite(values[j]))
                    return false;
            if (!(values[flowFirst] >= densityFloor))
                return false;
            if (requireSpecies)
            {
                real sum = 0;
                for (index k = 0; k < independent; ++k)
                {
                    if (values[speciesFirst + k] < 0)
                        return false;
                    sum += values[speciesFirst + k];
                }
                if (sum > values[flowFirst])
                    return false;
            }
            for (int c = 0; c < 2; ++c)
                if (!(Sensible(values, c) >= floors[c]) || !(Sensible(values, c, true) >= floors[c]))
                    return false;
            return true;
        }
        /// Euler density relaxation and scheme-one energy compression.
        /// Returns alpha for U+alpha*dU: density first, then both relaxed
        /// energy bounds, followed by alpha^(compression*power)*(1-1e-5)
        /// when limited. The caller scales the complete increment/solve RHS,
        /// or uses alpha to restrict its scalar dt/dTau. RANS positivity is
        /// not part of the factor calculation. Species affect the thermal
        /// references and must move with flow along this conservative line.
        /// The old state must be admissible; a nonfinite result must fail.
        /// Scalar repair and final combined-state validation remain required.
        template <class Row, class Increment>
        DNDS_DEVICE_CALLABLE real IncrementFraction(const Row &state, const Increment &increment, real relax, int compression, int power) const
        {
            using namespace EquationPositivityDetail;
            const real rho = state[flowFirst], deltaRho = increment[flowFirst];
            const real relaxedDensity = densityFloor * relax + rho * (1 - relax);
            const real density = deltaRho < 0 ? std::min(real(1), (rho - relaxedDensity) / (-deltaRho - smallReal * deltaRho)) : 1;
            const SegmentRow<Row, Increment> next{state, increment, density};
            auto delta = Flow(increment);
            for (auto &value : delta)
                value *= density;
            real energy = 1;
            for (int c = 0; c < 2; ++c)
            {
                const real oldEnergy = Sensible(state, c, true), newEnergy = Sensible(next, c, true);
                if (!std::isfinite(oldEnergy) || !std::isfinite(newEnergy))
                    return std::numeric_limits<real>::quiet_NaN();
                const real floor = newEnergy < oldEnergy ? floors[c] + (oldEnergy - floors[c]) * (1 - relax) : floors[c];
                if (newEnergy < floor)
                {
                    const real value = InternalEnergyCompression(Flow(state), delta, floor, Reference(state, c, true), Reference(next, c, true));
                    if (!std::isfinite(value))
                        return value;
                    energy = std::min(energy, value);
                }
            }
            const real fraction = density * energy;
            return fraction < 1 ? std::pow(fraction, compression * power) * (1 - 1e-5) : 1;
        }
        /// Euler final density/energy increment repair (.99 safety, .9 pull-up).
        /// Returns the whole-increment multiplier, or NaN if pull-up fails.
        /// Callers must reject a nonfinite result, scale all components, then
        /// perform the separate RANS/species repairs and validate the combined
        /// state. This function neither mutates a row nor repairs species.
        template <class Row, class Increment>
        DNDS_DEVICE_CALLABLE real FixFraction(const Row &state, const Increment &increment) const
        {
            using namespace EquationPositivityDetail;
            real fraction = 1;
            if (state[flowFirst] + increment[flowFirst] <= densityFloor)
                fraction = (densityFloor - state[flowFirst]) / (increment[flowFirst] - verySmallReal);
            for (int c = 0; c < 2; ++c)
            {
                SegmentRow<Row, Increment> next{state, increment, fraction};
                if (!std::isfinite(Sensible(next, c, true)))
                    return std::numeric_limits<real>::quiet_NaN();
                if (Sensible(next, c, true) <= floors[c])
                {
                    auto delta = Flow(increment);
                    for (auto &value : delta)
                        value *= fraction;
                    const real alpha = InternalEnergyCompression(Flow(state), delta, floors[c], Reference(state, c, true), Reference(next, c, true));
                    if (!std::isfinite(alpha))
                        return alpha;
                    fraction *= alpha * .99;
                    for (int iteration = 0; iteration < 1000; ++iteration)
                    {
                        next.fraction = fraction;
                        if (Sensible(next, c, true) >= floors[c])
                            break;
                        fraction *= .9;
                    }
                    next.fraction = fraction;
                    if (!(Sensible(next, c, true) >= floors[c]))
                        return std::numeric_limits<real>::quiet_NaN();
                }
            }
            const SegmentRow<Row, Increment> final{state, increment, fraction};
            for (int c = 0; c < 2; ++c)
                if (!(Sensible(final, c, true) >= floors[c]))
                    return std::numeric_limits<real>::quiet_NaN();
            return fraction;
        }
        /// Euler flow/species quadrature beta. Apply only to components selected
        /// by CompressesReconstructionComponent(), relative to the supplied
        /// admissible base. This differs from whole-vector increment repair.
        /// densityLimit is an optional SA reduced-order ratio in [0,1], merged
        /// into the density stage before the energy test and beta power.
        /// Raw affine references determine the
        /// concave energy ratio; a failed property-repaired postcheck falls
        /// back to the validated reference, as in the legacy species path.
        template <class Row, class BaseValues, class HigherValues>
        DNDS_DEVICE_CALLABLE real ReconstructionFraction(const Row &mean, const BaseValues &baseValues, const HigherValues &higherValues,
                                                         index begin, index end, int power, real cutoff, real densityLimit = 1) const
        {
            using namespace EquationPositivityDetail;
            bool good = true;
            for (index q = begin; q < end; ++q)
                good = good && Admissible(PointRow<HigherValues>{higherValues, q});
            if (good && densityLimit == 1)
                return 1;
            real density = densityLimit;
            for (index q = begin; q < end; ++q)
            {
                const real base = baseValues(q, flowFirst), delta = higherValues(q, flowFirst) - base;
                if (delta < 0)
                    density = std::min(density, (base - densityFloor) / (-delta + verySmallReal));
            }
            real energy = 1;
            for (int c = 0; c < 2; ++c)
            {
                if (Sensible(mean, c) <= 2 * floors[c])
                {
                    energy = 0;
                    continue;
                }
                for (index q = begin; q < end; ++q)
                {
                    const PointRow<BaseValues> base{baseValues, q};
                    const PointRow<HigherValues> high{higherValues, q};
                    const DifferenceRow<decltype(base), decltype(high)> delta{base, high, density};
                    const SegmentRow<decltype(base), decltype(delta)> next{base, delta, 1};
                    if (!std::isfinite(Sensible(next, c)))
                        return std::numeric_limits<real>::quiet_NaN();
                    if (Sensible(next, c) < 2 * floors[c])
                    {
                        const real value = InternalEnergyCompression(Flow(base), Flow(delta), floors[c], Reference(base, c), Reference(next, c));
                        if (!std::isfinite(value))
                            return value;
                        energy = std::min(energy, value);
                    }
                    if (speciesEnergies[c] && Sensible(next, c, true) < floors[c])
                        energy = 0;
                }
            }
            const real fraction = density * energy;
            const real compressed = fraction < 1 ? fraction * std::pow(fraction, power) * (1 - 1e-5) : 1;
            return compressed < cutoff ? 0 : compressed;
        }
    };

    /// CPU preparation owner for two affine bounds: e(Tmin,Y) and the model's
    /// reference-energy margin. For a perfect gas these reduce to the existing
    /// temperature and pressure energy constraints. Mixtures use the preserved
    /// backend reference temperature and an explicit sensible-energy margin.
    /// View pointers remain valid until owner mutation/move/destruction. A CUDA
    /// consumer must upload the coefficient buffers and freeze device pointers;
    /// passing this host owner's pointers to a device is not supported.
    class EquationThermalPolicy
    {
        std::array<std::vector<real>, 2> _energies;
        EquationThermalView _view{};

    public:
        EquationThermalPolicy(const EquationSystem &equation, real densityFloor, real temperatureFloor,
                              real sensibleEnergyFloor, real pressureFloor, Physics::Chemistry::ChemicalWorkerPool::Lease worker = {});
        EquationThermalView View() const
        {
            auto result = _view;
            for (int c = 0; c < 2; ++c)
                result.speciesEnergies[c] = _energies[c].empty() ? nullptr : _energies[c].data();
            return result;
        }
    };
}
