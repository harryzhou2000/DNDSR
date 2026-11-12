#pragma once

#include "Cauchy/Physics/PointProperties.hpp"
#include "Physics/Transport/NewtonianFourier.hpp"
#include "Physics/Transport/MixtureAveragedDiffusion.hpp"
#include <utility>

namespace DNDS::Cauchy
{
    /// Private CPU-worker/tile scratch, allocated once for a fixed material.
    /// Never share concurrently or alias the point evaluator's property/rate
    /// buffers. No resize/allocation occurs inside the normal-flux kernel.
    struct ViscousFluxScratch
    {
        std::vector<real> fractions, normalMassFlux;
        explicit ViscousFluxScratch(int species)
        {
            DNDS_check_throw_info(species >= 0, "transport scratch species count is negative");
            fractions.resize(size_t(species));
            normalMassFlux.resize(size_t(species));
        }
    };
    struct ViscousFluxOptions
    {
        bool adiabatic = false, impermeable = false;
        /// Already prepared model contributions, in code units. QCR and RANS
        /// closures remain separate physical producers; no closure is selected
        /// by merely supplying these effective scalar transport contributions.
        real turbulentViscosity = 0, turbulentConductivity = 0, turbulentSpeciesDiffusivity = 0;
        bool useQCR = false;
    };
    /// Positive viscous normal flux: total flux is convective minus viscous.
    /// Ready point must match the conservative state and retain its worker.
    /// Mixtures need requested transport, diffusivities and full enthalpies.
    /// rawRhoY is the transported Ns-1 composition, not the repaired property Y.
    /// Gradient has (species,direction) indexing and output supports [species];
    /// strided views are sufficient. Every independent output is replaced.
    ///
    /// Caller validates access, extent, finite effective coefficients and output/
    /// scratch nonaliasing outside the point loop. All inputs use code units
    /// in one Cartesian frame. No EOS refresh, interpolation, clipping, MPI,
    /// integration, persistent property cache or point-loop allocation. Extent
    /// assertions enforce setup contracts; this kernel never throws to select
    /// a numerical fallback. The scratch constructor remains checked setup.
    template <class TSpecies, class TSpeciesGradient, class TSpeciesFlux>
    FlowState PrepareViscousFlux(const FlowState &state, const FlowGradient &gradient, const PointProperties &point,
                                 const TSpecies &rawRhoY, const TSpeciesGradient &speciesGradient,
                                 const Vector3 &normal, TSpeciesFlux &&speciesFlux, ViscousFluxScratch &scratch,
                                 ViscousFluxOptions options = {})
    {
        namespace T = Physics::Thermodynamics;
        const int species = point.fractions.nSpecies;
        DNDS_assert_info(scratch.fractions.size() == size_t(species) && scratch.normalMassFlux.size() == size_t(species), "transport scratch belongs to another species count");
        DNDS_assert_info(!species || options.impermeable || (point.diffusivity.nSpecies == species && point.speciesEnthalpy.nSpecies == species),
                         "mixture diffusion needs full diffusivities and enthalpies");
        const auto velocity = T::ConservativeVelocity<3>(state);
        std::array<std::array<real, 3>, 3> gradientVelocity;
        Vector3 gradientTemperature{};
        const real viscosity = point.transport.viscosity + options.turbulentViscosity;
        const real conductivity = options.adiabatic ? 0 : point.transport.conductivity + options.turbulentConductivity;
        struct SpeciesDirection
        {
            const TSpeciesGradient &values;
            int direction;
            real operator[](int k) const { return values(k, direction); }
        };
        for (int d = 0; d < 3; ++d)
        {
            for (int i = 0; i < 3; ++i)
                gradientVelocity[i][d] = T::ConservativeSpecificDerivative(state[0], velocity[i], gradient[0][d], gradient[1 + i][d]);
            if (conductivity != 0)
            {
                FlowState direction;
                for (int j = 0; j < 5; ++j)
                    direction[j] = gradient[j][d];
                gradientTemperature[d] = PreparePrimitiveDerivative(state, direction, point, SpeciesDirection{speciesGradient, d}).rhoVelocityTemperature[4];
            }
        }
        struct VelocityGradient
        {
            const std::array<std::array<real, 3>, 3> &values;
            real operator()(int i, int j) const { return values[i][j]; }
        };
        const auto flux = Physics::Transport::NewtonianFourierNormalFlux(velocity, VelocityGradient{gradientVelocity}, gradientTemperature, normal, viscosity, conductivity,
                                                                         options.useQCR, options.turbulentViscosity / (viscosity + verySmallReal));
        FlowState result{};
        for (int i = 0; i < 3; ++i)
            result[1 + i] = flux.traction[i];
        result[4] = flux.energy;
        for (int k = 0; k < species - 1; ++k)
            speciesFlux[k] = 0;
        if (!species || options.impermeable)
            return result;
        real sumFractions = 0, sumGradient = 0;
        for (int k = 0; k < species - 1; ++k)
        {
            const real fraction = rawRhoY[k] / state[0];
            scratch.fractions[k] = fraction;
            real normalGradient = 0;
            for (int d = 0; d < 3; ++d)
                normalGradient += T::ConservativeSpecificDerivative(state[0], fraction, gradient[0][d], speciesGradient(k, d)) * normal[d];
            scratch.normalMassFlux[k] = -state[0] * (point.diffusivity[k] + options.turbulentSpeciesDiffusivity) * normalGradient;
            sumFractions += fraction;
            sumGradient += normalGradient;
        }
        scratch.fractions[species - 1] = 1 - sumFractions;
        scratch.normalMassFlux[species - 1] = state[0] * (point.diffusivity[species - 1] + options.turbulentSpeciesDiffusivity) * sumGradient;
        Physics::Transport::AddCorrectedMixtureAveragedNormalFlux(state[0], 1 / state[0], scratch.fractions,
                                                                  scratch.normalMassFlux, point.speciesEnthalpy, species, std::forward<TSpeciesFlux>(speciesFlux), result[4]);
        return result;
    }
}
