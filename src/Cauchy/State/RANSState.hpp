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
        DNDS_DEVICE_CALLABLE bool Admissible(const Row &values) const
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

        /// Apply after Euler's whole-vector thermal increment compression.
        /// Preserve Euler's
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
}
