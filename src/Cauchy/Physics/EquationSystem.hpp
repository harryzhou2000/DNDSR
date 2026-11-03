#pragma once

#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/Positivity.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Physics/EquationSchema.hpp"
#include "DNDS/Algebra/DofBundle.hpp"
#include "Physics/Thermodynamics/IdealGasMixture.hpp"
#include <algorithm>
#include <optional>

namespace DNDS::Cauchy
{
    struct EquationSettings;
    class EquationDofs;
    class EquationPointEvaluator;
    /// Coarse dispatch key. CPU branching and specialized device kernels may
    /// share this interface; a missing backend never falls back implicitly.
    enum class EquationKernelProfile
    {
        CompressiblePerfectGasCPU,
        CompressibleIdealMixtureCPU
    };
    /// Exact local property provenance. Perfect-gas parameters compare by value;
    /// a loaded mixture also carries its immutable in-process material identity.
    /// This is not a persistent restart fingerprint or permission to read data.
    struct EquationPropertyKey
    {
        std::shared_ptr<const void> material;
        std::array<real, 5> parameters{};
        bool operator==(const EquationPropertyKey &other) const { return material == other.material && parameters == other.parameters; }
        bool operator!=(const EquationPropertyKey &other) const { return !(*this == other); }
    };
    /// Value owner for optional model-prepared data. Input() creates borrowed
    /// pointers after this owner reaches its final location; copies/moves never
    /// leave embedded pointers referring to an old stack object.
    struct PreparedRiemannData
    {
        Riemann::AcousticState leftMean, rightMean;
        Riemann::RoeData roe;
        Riemann::WaveBounds outerWaves;
        Riemann::AUSMData ausm;
        Riemann::Requirements required;
        Riemann::FluxInput Input(Riemann::AcousticState left, Riemann::AcousticState right,
                                 Vector3 normal, Vector3 gridVelocity = {}) const
        {
            return {left, right, normal, gridVelocity, required.meanStates ? &leftMean : nullptr, required.meanStates ? &rightMean : nullptr,
                    required.roe ? &roe : nullptr, required.outerWaveBounds ? &outerWaves : nullptr, required.ausmFaceSpeed ? &ausm : nullptr};
        }
    };
    struct FlowCharacteristics
    {
        std::array<real, 25> forward, inverse;
        Vector3 velocity{};
        real soundSpeed = 0;
        bool componentwiseFallback = false;
    };
    /// Coarse physical-model owner. Schema and accumulation choices belong here;
    /// mesh/DOF allocations, time history and reconstruction belong to Domain.
    /// Resolve once outside numerical loops, then pass plain model parameters to
    /// the selected kernel family. New equation profiles may add models/blocks
    /// and non-conservative physical mass operators without changing DOF algebra.
    class EquationSystem
    {
        EquationSchema _schema;
        PerfectGas _gas;
        std::optional<Physics::Thermodynamics::IdealGasMixture> _mixture;
        bool _viscous;
        explicit EquationSystem(PerfectGas gas, bool viscous);

    public:
        static EquationSystem Resolve(const EquationSettings &settings);
        /// Explicit point-property/model construction. The production frontend
        /// still rejects missing spatial/time profiles until they are connected.
        /// Material metadata is shared; mutable chemistry workers are external.
        static EquationSystem FromIdealGasMixture(Physics::Thermodynamics::IdealGasMixture material, bool viscous);
        const EquationSchema &Schema() const { return _schema; }
        const PerfectGas &Gas() const
        {
            DNDS_check_throw_info(!_mixture, "perfect-gas-only operation needs the mixture property/profile adapter");
            return _gas;
        }
        bool HasMixture() const { return _mixture.has_value(); }
        const Physics::Thermodynamics::IdealGasMixture &Mixture() const;
        EquationPropertyKey PropertyKey() const;
        /// Reserve/validate one private worker outside a loop. Perfect gas
        /// requires no chemistry context. A mixture never falls back to gamma.
        /// Include EquationProperties.hpp to retain/use the returned evaluator.
        EquationPointEvaluator MakePointEvaluator(Physics::Chemistry::ChemicalWorkerPool::Lease worker = {},
                                                  Physics::Thermodynamics::MixturePropertyRequest request = {true, true, false}) const;
        bool Viscous() const { return _viscous; }
        EquationKernelProfile ResolveKernelProfile(Runtime::Device device) const;
        /// Model-owned thermodynamic chain rule. Numerical face penalties stay
        /// in Discretization; no EOS or energy calculation enters Riemann.
        FlowGradient PreparePrimitiveGradient(const FlowState &state, const FlowGradient &gradient) const
        {
            return Gas().PrimitiveGradient(state, gradient);
        }
        ThermalBounds PrepareThermalBounds(real densityFloor, real temperatureFloor, real pressureFloor) const
        {
            const auto &gas = Gas();
            return {densityFloor, gas.gasConstant * temperatureFloor / (gas.gamma - 1), pressureFloor / (gas.gamma - 1)};
        }
        /// Thermodynamic producer for the initial acoustic flux profile. EOS
        /// dispatch belongs here, never inside the numerical Riemann method.
        Riemann::AcousticState PrepareFluxState(const FlowState &state) const
        {
            const auto &gas = Gas();
            const real pressure = gas.Pressure(state);
            return {{state, pressure}, std::sqrt(gas.gamma * pressure / state[0])};
        }
        /// Model-owned physical flux derivative for approximate implicit
        /// operators. Numerical dissipation and time/mass scaling are separate.
        std::array<real, 25> PrepareFluxJacobian(const FlowState &state, const Vector3 &normal, const Vector3 &gridVelocity = {}) const
        {
            auto jacobian = Gas().FluxJacobian(state, normal);
            real normalSpeed = 0;
            for (int d = 0; d < 3; ++d)
                normalSpeed += gridVelocity[d] * normal[d];
            for (int j = 0; j < 5; ++j)
                jacobian[j + 5 * j] -= normalSpeed;
            return jacobian;
        }
        /// Maximum molecular momentum/thermal diffusivity for the initial
        /// Newtonian/Fourier profile. Other transport models supply this bound.
        real DiffusionScale(const FlowState &state) const
        {
            const auto &gas = Gas();
            return _viscous ? gas.viscosity / state[0] * std::max(real(4. / 3), gas.gamma / gas.prandtl) : 0;
        }
        /// Perfect-gas initial provider. General EOS must supply its own valid
        /// linearization, shock bounds and AUSM scale; gamma substitution inside
        /// the numerical Riemann algorithm is prohibited.
        PreparedRiemannData PrepareRiemann(const Riemann::RiemannFlux &flux,
                                           const Riemann::AcousticState &left, const Riemann::AcousticState &right,
                                           const FlowState &leftMean, const FlowState &rightMean,
                                           const Vector3 &normal, const Vector3 &gridVelocity = {}) const;
        PreparedRiemannData PrepareRiemann(const Riemann::RiemannFlux &flux,
                                           const Riemann::AcousticState &left, const Riemann::AcousticState &right,
                                           const Riemann::AcousticState &leftMean, const Riemann::AcousticState &rightMean,
                                           const Vector3 &normal, const Vector3 &gridVelocity = {}) const;
        /// Cell-mean characteristic basis for reconstruction limiting; EOS
        /// eigenstructure belongs to the equation provider, never CFV policy.
        FlowCharacteristics PrepareCharacteristics(const FlowState &left, const FlowState &right, const Vector3 &normal) const;
        /// Compatibility check of physical block names/widths only. New
        /// frontend calls use EquationDofs for exact component semantics.
        void ValidateStateSchema(const Algebra::DofBundle &state) const;
        void ValidateStateSchema(const EquationDofs &state) const;
    };
}
