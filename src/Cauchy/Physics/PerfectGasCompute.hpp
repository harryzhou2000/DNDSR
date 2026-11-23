#pragma once

#include "Cauchy/Physics/RiemannPreparation.hpp"
#include "Cauchy/Physics/PerfectGasViscousFlux.hpp"
#include "Cauchy/Riemann/RiemannCompute.hpp"

namespace DNDS::Cauchy
{
    namespace PerfectGasComputeDetail
    {
        template <class Scalar, class From, std::size_t N>
        DNDS_DEVICE_CALLABLE inline std::array<Scalar, N> Convert(const std::array<From, N> &input) noexcept
        {
            std::array<Scalar, N> result;
            for (int j = 0; j < int(N); ++j)
                result[j] = DNDS::Math::ScalarCast<Scalar>(input[j]);
            return result;
        }
        template <class Scalar>
        DNDS_DEVICE_CALLABLE inline NumericalResult<Riemann::FluxResult> Finish(const Riemann::FluxResultT<Scalar> &input) noexcept
        {
            for (int j = 0; j < 5; ++j)
                if (!DNDS::Math::ScalarFinite(input.flux[j]))
                    return Physics::EvaluationError(NumericalError::InvalidState, Physics::EvaluationFailure::Properties);
            if (!DNDS::Math::ScalarFinite(input.acousticMinus) || !DNDS::Math::ScalarFinite(input.contact) || !DNDS::Math::ScalarFinite(input.acousticPlus))
                return Physics::EvaluationError(NumericalError::InvalidState, Physics::EvaluationFailure::Properties);
            return Riemann::ComputeDetail::Convert(input);
        }
        template <class Scalar>
        DNDS_DEVICE_CALLABLE inline NumericalResult<Riemann::AcousticStateT<Scalar>> Acoustic(
            const PerfectGasT<Scalar> &gas, const FlowStateT<Scalar> &state) noexcept
        {
            const auto point = TryPerfectGasPoint(gas, state, false);
            if (!point)
                return point.status;
            return Riemann::AcousticStateT<Scalar>{{state, point.value.thermo.pressure}, point.value.thermo.soundSpeed};
        }
    }

    /// Direction-dependent mean waves and face acoustic data are prepared
    /// independently for each selected solver. All RHS kernel mappings use
    /// this dispatch, including reduced dissipation precision.
    template <class DissipationScalar = real, class Scalar>
    DNDS_DEVICE_CALLABLE inline Riemann::FluxResultT<Scalar> EvaluatePerfectGasRiemann(
        const PerfectGasT<Scalar> &gas, Riemann::Solver solver,
        const Riemann::AcousticStateT<Scalar> &left, const Riemann::AcousticStateT<Scalar> &right,
        const Riemann::AcousticStateT<Scalar> &leftMean, const Riemann::AcousticStateT<Scalar> &rightMean,
        const Vector3T<Scalar> &normal, const Vector3T<Scalar> &grid, const Vector3T<Scalar> &faceNormal,
        const Riemann::FluxSettingsT<Scalar> &settings) noexcept
    {
        return Riemann::Math::EvaluateDirections(solver, settings.rotation, leftMean.flow.conservative,
                                                 rightMean.flow.conservative, normal, faceNormal,
                                                 [&](Riemann::Solver kind, const Vector3T<Scalar> &direction)
                                                 {
                                                     const auto prepared = PreparePerfectGasRiemann(gas, kind, left, right, leftMean, rightMean, direction, grid);
                                                     const auto input = prepared.Input(left, right, direction, grid);
                                                     if constexpr (std::is_same_v<Scalar, real> && !std::is_same_v<DissipationScalar, real>)
                                                         return Riemann::EvaluateRoeDissipationCompute<DissipationScalar>(kind, input, settings);
                                                     else
                                                         return Riemann::Math::Evaluate(kind, input, settings);
                                                 });
    }

    /** @brief Full local perfect-gas thermodynamics and convective flux in Scalar.
     * @details Inputs and result are FP64; pressure, acoustics, Roe/AUSM
     * preparation and the numerical flux are all evaluated after narrowing.
     * This differs from EvaluateRiemannCompute, which only narrows already
     * prepared thermodynamics. Mean thermodynamics are recomputed as needed,
     * never taken from an FP64 auxiliary cache. No state repair is performed
     * when the selected compute type cannot represent an admissible input.
     * The FP64 specialization passes directly through the canonical helpers.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline NumericalResult<Riemann::FluxResult> TryPerfectGasRiemannCompute(
        const PerfectGas &gas, Riemann::Solver solver, const FlowState &left, const FlowState &right,
        const FlowState &leftMean, const FlowState &rightMean, const Vector3 &normal,
        const Vector3 &gridVelocity = {}, const Riemann::FluxSettings &settings = {},
        const Vector3 *faceNormal = nullptr) noexcept
    {
        const auto typedGas = CastPerfectGas<Scalar>(gas);
        const auto typedLeft = PerfectGasComputeDetail::Acoustic(typedGas, PerfectGasComputeDetail::Convert<Scalar>(left));
        if (!typedLeft)
            return typedLeft.status;
        const auto typedRight = PerfectGasComputeDetail::Acoustic(typedGas, PerfectGasComputeDetail::Convert<Scalar>(right));
        if (!typedRight)
            return typedRight.status;
        auto typedLeftMean = typedLeft.value, typedRightMean = typedRight.value;
        if (Riemann::RequiredInputs(solver).meanStates || settings.rotation.Enabled())
        {
            const auto a = PerfectGasComputeDetail::Acoustic(typedGas, PerfectGasComputeDetail::Convert<Scalar>(leftMean));
            if (!a)
                return a.status;
            const auto b = PerfectGasComputeDetail::Acoustic(typedGas, PerfectGasComputeDetail::Convert<Scalar>(rightMean));
            if (!b)
                return b.status;
            typedLeftMean = a.value;
            typedRightMean = b.value;
        }
        const auto typedNormal = PerfectGasComputeDetail::Convert<Scalar>(normal);
        const auto typedGrid = PerfectGasComputeDetail::Convert<Scalar>(gridVelocity);
        const auto typedFaceNormal = faceNormal ? PerfectGasComputeDetail::Convert<Scalar>(*faceNormal) : typedNormal;
        return PerfectGasComputeDetail::Finish(EvaluatePerfectGasRiemann(
            typedGas, solver, typedLeft.value, typedRight.value, typedLeftMean, typedRightMean, typedNormal, typedGrid, typedFaceNormal,
            Riemann::ComputeDetail::Convert<Scalar>(settings)));
    }

    /** @brief Full flow5 molecular diffusion in a selected kernel-local scalar.
     * @details Conversion precedes face averaging, jump correction, conservative
     * chain rule, viscosity and Newtonian/Fourier flux evaluation. Persistent
     * input/output storage remains FP64. An optional prescribed face state is
     * a local value, never an unmanaged nullable borrow. Unsupported RANS, QCR
     * and primitive-gradient policies must be rejected by host dispatch; this
     * interface intentionally has no switches suggesting those are implemented.
     * The returned flux is positive viscous flux, to subtract from convection.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowState> TryPerfectGasPolynomialViscousCompute(
        const PerfectGas &gas, const FlowState &left, const FlowState &right,
        const FlowGradient &leftGradient, const FlowGradient &rightGradient,
        const Vector3 &normal, real inverseDistance, bool adiabatic,
        bool hasFaceState = false, const FlowState &faceState = {}) noexcept
    {
        FlowGradientT<Scalar> typedLeftGradient, typedRightGradient;
        for (int j = 0; j < 5; ++j)
        {
            typedLeftGradient[j] = PerfectGasComputeDetail::Convert<Scalar>(leftGradient[j]);
            typedRightGradient[j] = PerfectGasComputeDetail::Convert<Scalar>(rightGradient[j]);
        }
        const auto result = PreparePerfectGasPolynomialViscousFlux(
            CastPerfectGas<Scalar>(gas), PerfectGasComputeDetail::Convert<Scalar>(left), PerfectGasComputeDetail::Convert<Scalar>(right),
            typedLeftGradient, typedRightGradient, PerfectGasComputeDetail::Convert<Scalar>(normal), Scalar(inverseDistance), adiabatic,
            hasFaceState, PerfectGasComputeDetail::Convert<Scalar>(faceState));
        if (!result)
            return result.status;
        for (int j = 0; j < 5; ++j)
            if (!DNDS::Math::ScalarFinite(result.value[j]))
                return Physics::EvaluationError(NumericalError::InvalidState, Physics::EvaluationFailure::Properties);
        return PerfectGasComputeDetail::Convert<real>(result.value);
    }
}
