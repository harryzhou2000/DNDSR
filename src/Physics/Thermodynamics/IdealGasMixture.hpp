#pragma once

#include "Physics/ReferenceScales.hpp"
#include "Physics/Chemistry/ChemicalWorker.hpp"
#include "Physics/Thermodynamics/TotalCondition.hpp"
#include <array>

namespace DNDS::Physics::Chemistry
{
    struct ReactorStepSettings;
}

namespace DNDS::Physics::Thermodynamics
{
    /// Optional point outputs, resolved once when borrowing a worker for a loop.
    struct MixturePropertyRequest
    {
        bool transport = false, diffusivity = false, speciesEnthalpy = false;
        /// Applied by FromConservative only. A Jacobian also requests rates.
        bool chemicalSource = false, sourceJacobian = false;
    };

    /// Code-scaled thermodynamics and optional molecular transport. Species
    /// views borrow worker scratch and expire at the next point evaluation or
    /// worker release. Full species enthalpies retain the chemical reference.
    struct MixturePoint
    {
        Properties thermo;
        Transport::Properties transport;
        Chemistry::ConstSpeciesBufferView fractions, diffusivity, speciesEnthalpy, speciesGasConstant;
        /// Ns-1 independent mass production rates in code mass/volume/time.
        /// No energy source: total energy includes chemical formation energy.
        Chemistry::ConstSpeciesBufferView chemicalSource;
        /// Positive dS/dU, Ns-1 rows by (flow5 + Ns-1) columns, in code units.
        /// Columns have this explicit point convention, independent of the
        /// caller's physical allocation and logical equation ordering.
        Chemistry::ConstJacobianBufferView sourceJacobian;
    };
    /// Fixed-density/internal-energy source result in code units. The species
    /// view borrows this evaluator's input scratch until its next use/release.
    struct ConstantVolumeResult
    {
        Properties thermo;
        Chemistry::ConstSpeciesBufferView rhoY;
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
        /// Persistent loaded-model description frozen at material construction.
        const std::string &ModelDescription() const;

        /// Linear volumetric internal energy at the backend's reference floor.
        /// Uses raw transported rhoY (including the dependent species), without
        /// repairing composition. This preserves Euler's base-energy/PP algebra.
        real BaseInternalEnergy(real rho, Chemistry::ConstSpeciesBufferView rhoY) const;
        /// Immutable code-scaled coefficients of BaseInternalEnergy. Retain
        /// the material while borrowing this reference; no worker is needed.
        const std::vector<real> &BaseSpecificInternalEnergies() const;
        /// Code-scaled ideal-gas species constants; immutable material data.
        const std::vector<real> &SpeciesGasConstants() const;
        /// Setup-only ideal-gas coefficients e_k(T), in code energy/mass.
        /// Consumes one compatible worker; returned values own their storage.
        /// Their raw composition-weighted sum is affine in rho and rhoY and
        /// supplies an exact temperature-floor energy bound without gamma.
        std::vector<real> SpecificInternalEnergies(real temperature, Chemistry::ChemicalWorkerPool::Lease worker) const;

        class PointEvaluator
        {
            friend class IdealGasMixture;
            std::shared_ptr<const State> _state;
            Chemistry::ChemicalWorkerPool::Lease _worker;
            MixturePropertyRequest _request;
            std::vector<real> _sourceJacobian;
            PointEvaluator(std::shared_ptr<const State> state, Chemistry::ChemicalWorkerPool::Lease worker,
                           MixturePropertyRequest request);
            void PrepareComposition(real rho, Chemistry::ConstSpeciesBufferView rhoY);
            MixturePoint Evaluate(real rho, real temperature, MixturePropertyRequest request);
            MixturePoint CompleteSources(MixturePoint point, real rho, real rhoE,
                                         const std::array<real, 3> &momentum, MixturePropertyRequest request);
            MixturePoint FromRhoEImpl(real rho, real internalEnergy, Chemistry::ConstSpeciesBufferView rhoY,
                                      real temperatureGuess, real uvTolerance, MixturePropertyRequest request);

        public:
            PointEvaluator(const PointEvaluator &) = delete;
            PointEvaluator &operator=(const PointEvaluator &) = delete;
            PointEvaluator(PointEvaluator &&) noexcept = default;
            PointEvaluator &operator=(PointEvaluator &&) noexcept = default;

            /// Stable Ns-1 input scratch for gathering strided logical rows.
            /// Retained by this evaluator; evaluation never rewrites rhoY.
            Chemistry::SpeciesBufferView SpeciesInput() const;
            /// Frozen-composition total-to-static closure using this worker.
            /// independentFractions contains Ns-1 mass fractions, not rhoY.
            NumericalResult<TotalConditionStaticState> TryTotalToStatic(
                real pressure, real temperature, real speedSquared,
                Chemistry::ConstSpeciesBufferView independentFractions) noexcept;

            /// Numerical entry: all backend exceptions are intercepted and
            /// returned as status. Only UV nonconvergence permits mean fallback.
            /// On failure output must not be consumed; worker scratch may have
            /// changed. Successful species views keep the usual worker lifetime.
            NumericalStatus TryFromConservative(real rho, real rhoE, const std::array<real, 3> &momentum,
                                                Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess,
                                                MixturePropertyRequest request, MixturePoint &output) noexcept;

            /// Full thermochemical specific internal energy, not an energy
            /// shifted by BaseInternalEnergy. A positive temperature guess is
            /// forwarded to the UV inversion; stale guesses are permitted and
            /// never substituted for the solved current temperature.
            MixturePoint FromRhoE(real rho, real internalEnergy, Chemistry::ConstSpeciesBufferView rhoY,
                                  real temperatureGuess = 0, real uvTolerance = 1e-12);
            MixturePoint FromRhoT(real rho, real temperature, Chemistry::ConstSpeciesBufferView rhoY);
            MixturePoint FromRhoP(real rho, real pressure, Chemistry::ConstSpeciesBufferView rhoY);
            /// Existing Euler constant-volume CVODE source step. Density and
            /// full specific internal energy remain fixed; only independent
            /// rhoY changes. Units convert to SI here, never in solver kernels.
            /// Euler's clipping/remainder rule is followed, then UV is solved
            /// again at the unchanged energy. A CVODE temperature is only that
            /// final solve's guess. Zero duration/scale preserves raw rhoY.
            /// Nonthrowing adapter for the external reactor and its final UV
            /// solve. Failure does not publish a result or request a fallback.
            NumericalResult<ConstantVolumeResult> TryAdvanceConstVolume(real rho, real internalEnergy,
                                                                        Chemistry::ConstSpeciesBufferView rhoY, real dt, const Chemistry::ReactorStepSettings &controls,
                                                                        real sourceScale = 1, real temperatureGuess = 0) noexcept;
            ConstantVolumeResult AdvanceConstVolume(real rho, real internalEnergy, Chemistry::ConstSpeciesBufferView rhoY,
                                                    real dt, const Chemistry::ReactorStepSettings &controls,
                                                    real sourceScale = 1, real temperatureGuess = 0);
            /// Full thermochemical conservative point, in code units. Momentum
            /// always has three Cartesian components. Optional rates/Jacobian
            /// use Euler's repaired temporary composition and source-temperature
            /// floor, leaving transported values unchanged. The derivative is
            /// the existing implicit chemistry approximation at that point,
            /// not a derivative through clipping/repair. Returned views expire
            /// at the next point evaluation or release of this worker.
            MixturePoint FromConservative(real rho, real rhoE, const std::array<real, 3> &momentum,
                                          Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess = 0);
            /// Select a subset of the outputs reserved at construction. A face
            /// acoustic query need not perform transport or chemistry, while a
            /// source point need not perform transport. No default-policy
            /// mutation, extra worker acquisition or point-time allocation.
            MixturePoint FromConservative(real rho, real rhoE, const std::array<real, 3> &momentum,
                                          Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess,
                                          MixturePropertyRequest request);
        };

        /// Consumes a lease for one loop/tile. Checks material identity once;
        /// the evaluator retains both metadata and worker allocation lifetime.
        /// Caller serializes its use. No OpenMP thread-ID dispatch per point.
        PointEvaluator Borrow(Chemistry::ChemicalWorkerPool::Lease worker,
                              MixturePropertyRequest request = {}) const;
    };
}
