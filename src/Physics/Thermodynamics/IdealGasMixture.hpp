#pragma once

#include "Physics/ReferenceScales.hpp"
#include "Physics/Chemistry/ChemicalWorker.hpp"

namespace DNDS::Physics::Thermodynamics
{
    /// Optional point outputs, resolved once when borrowing a worker for a loop.
    struct MixturePropertyRequest
    {
        bool transport = false, diffusivity = false, speciesEnthalpy = false;
    };

    /// Code-scaled thermodynamics and optional molecular transport. Species
    /// views borrow worker scratch and expire at the next point evaluation or
    /// worker release. Full species enthalpies retain the chemical reference.
    struct MixturePoint
    {
        Properties thermo;
        Transport::Properties transport;
        Chemistry::ConstSpeciesBufferView fractions, diffusivity, speciesEnthalpy;
    };

    /// Immutable ideal-gas-mixture metadata, independent of any solver, mesh,
    /// conservative-vector ordering or physical Array layout. Copies share this
    /// metadata; mutable Cantera states belong to explicit worker leases.
    /// All public point inputs/outputs use code units. Three velocity components
    /// and subtraction of kinetic energy remain the caller's flow convention.
    class IdealGasMixture
    {
        struct State;
        std::shared_ptr<const State> _state;

    public:
        /// Build from the already-loaded backend, with matching U0 and rho0.
        /// Clones share its material identity. Construction/cloning must not run
        /// concurrently with evaluation of that same backend instance.
        IdealGasMixture(const Chemistry::ChemicalSource &prototype, ReferenceScales scales);
        int SpeciesNumber() const;
        const std::vector<std::string> &SpeciesNames() const;
        const ReferenceScales &Scales() const;
        /// Bookkeeping reference temperature, not a universal EOS validity bound.
        real BaseTemperature() const;
        std::shared_ptr<const void> ModelIdentity() const;

        /// Linear volumetric internal energy at the backend's reference floor.
        /// Uses raw transported rhoY (including the dependent species), without
        /// repairing composition. This preserves Euler's base-energy/PP algebra.
        real BaseInternalEnergy(real rho, Chemistry::ConstSpeciesBufferView rhoY) const;

        class PointEvaluator
        {
            friend class IdealGasMixture;
            std::shared_ptr<const State> _state;
            Chemistry::ChemicalWorkerPool::Lease _worker;
            MixturePropertyRequest _request;
            PointEvaluator(std::shared_ptr<const State> state, Chemistry::ChemicalWorkerPool::Lease worker,
                           MixturePropertyRequest request);
            void PrepareComposition(real rho, Chemistry::ConstSpeciesBufferView rhoY);
            MixturePoint Evaluate(real rho, real temperature);

        public:
            PointEvaluator(const PointEvaluator &) = delete;
            PointEvaluator &operator=(const PointEvaluator &) = delete;
            PointEvaluator(PointEvaluator &&) noexcept = default;
            PointEvaluator &operator=(PointEvaluator &&) noexcept = default;

            /// Stable Ns-1 input scratch for gathering strided logical rows.
            /// Retained by this evaluator; evaluation never rewrites rhoY.
            Chemistry::SpeciesBufferView SpeciesInput() const;

            /// Full thermochemical specific internal energy, not an energy
            /// shifted by BaseInternalEnergy. A positive temperature guess is
            /// forwarded to the UV inversion; stale guesses are permitted and
            /// never substituted for the solved current temperature.
            MixturePoint FromRhoE(real rho, real internalEnergy, Chemistry::ConstSpeciesBufferView rhoY,
                                  real temperatureGuess = 0, real uvTolerance = 1e-12);
            MixturePoint FromRhoT(real rho, real temperature, Chemistry::ConstSpeciesBufferView rhoY);
            MixturePoint FromRhoP(real rho, real pressure, Chemistry::ConstSpeciesBufferView rhoY);
        };

        /// Consumes a lease for one loop/tile. Checks material identity once;
        /// the evaluator retains both metadata and worker allocation lifetime.
        /// Caller serializes its use. No OpenMP thread-ID dispatch per point.
        PointEvaluator Borrow(Chemistry::ChemicalWorkerPool::Lease worker,
                              MixturePropertyRequest request = {}) const;
    };
}
