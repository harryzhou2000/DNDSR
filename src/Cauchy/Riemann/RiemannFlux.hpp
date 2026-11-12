#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "DNDS/Core/Errors.hpp"
#include <cmath>
#include <string>
#include <limits>

namespace DNDS::Cauchy::Riemann
{
    /// EOS-independent physical input. Pressure is supplied by the equation
    /// layer; conservative energy retains that layer's energy convention.
    struct EulerState
    {
        FlowState conservative;
        real pressure;
    };
    /// Rusanov additionally requires the equation's physical sound speed.
    /// Further algorithms add their own typed enthalpy/derivative/average data;
    /// absent thermodynamics must never be guessed from conservative energy.
    struct AcousticState
    {
        EulerState flow;
        real soundSpeed;
    };
    struct Requirements
    {
        bool pressure, soundSpeed, enthalpy;
        bool meanStates = false, roe = false, outerWaveBounds = false, ausmFaceSpeed = false;
    };
    enum class Solver
    {
        Rusanov,
        Roe,
        Roe_M1,
        Roe_M2,
        Roe_M3,
        Roe_M4,
        Roe_M5,
        Roe_M6,
        Roe_M7,
        Roe_M8,
        Roe_M9,
        HLLC,
        HLLEP,
        HLLEP_V1,
        AUSMPWPlus,
        AUSMPlusUp,
        AUSMPlusUp2
    };

    /// Equation-prepared Roe linearization. contactProjection is a row of its
    /// inverse eigenbasis, mapping conservative jumps onto the density/contact
    /// wave. contactEnergy and enthalpy are the energy entries of its right
    /// basis. This describes the supplied model; no gamma or EOS recovery is
    /// required in Roe or HLLEP. Extra pressure-neutral energy can be supplied
    /// explicitly for multi-energy/species profiles with the corresponding mode.
    struct RoeData
    {
        Vector3 velocity;
        real soundSpeed, enthalpy, contactEnergy;
        FlowState contactProjection;
        real contactEnergyJump = 0;
    };
    struct WaveBounds
    {
        real left, right;
    };
    struct AUSMData
    {
        real faceSoundSpeed;
    };
    /// Common modular flux interface. Optional data pointers are required only
    /// by RequiredInputs(); their storage is caller-owned through Evaluate.
    /// Face and mean states are separate, preserving legacy implicit estimates.
    struct FluxInput
    {
        AcousticState left, right;
        Vector3 normal, gridVelocity{};
        const AcousticState *leftMean = nullptr, *rightMean = nullptr;
        const RoeData *roe = nullptr;
        const WaveBounds *outerWaves = nullptr;
        const AUSMData *ausm = nullptr;
    };
    struct FluxResult
    {
        FlowState flux;
        real acousticMinus, contact, acousticPlus;
    };
    struct FluxSettings
    {
        real dLambda = 0, fixScale = 1, incrementScale = 1;
        real referenceMach = .1, pressureStencilMin = std::numeric_limits<real>::max();
    };

    /// Unit normal and explicit grid velocity (zero by default); only
    /// kinematics and supplied pressure are used. No EOS, heat capacity or
    /// total-enthalpy recovery occurs here.
    DNDS_DEVICE_CALLABLE inline FlowState PhysicalFlux(const EulerState &input, const Vector3 &normal, const Vector3 &gridVelocity = {})
    {
        const auto &state = input.conservative;
        real speed = 0;
        for (int d = 0; d < 3; ++d)
            speed += (state[1 + d] / state[0] - gridVelocity[d]) * normal[d];
        FlowState flux{state[0] * speed, 0, 0, 0, (state[4] + input.pressure) * speed};
        for (int d = 0; d < 3; ++d)
        {
            flux[1 + d] = state[1 + d] * speed + input.pressure * normal[d];
            flux[4] += input.pressure * gridVelocity[d] * normal[d];
        }
        return flux;
    }
    DNDS_DEVICE_CALLABLE inline real SignalSpeed(const AcousticState &input, const Vector3 &normal, const Vector3 &gridVelocity = {})
    {
        real speed = 0;
        for (int d = 0; d < 3; ++d)
            speed += (input.flow.conservative[1 + d] / input.flow.conservative[0] - gridVelocity[d]) * normal[d];
        return std::abs(speed) + input.soundSpeed;
    }
    /// Face contribution to the legacy H-correction stencil. The sound speeds
    /// are equation-provided. This scalar is invariant under swapping sides
    /// and reversing the normal, and under a common constant grid velocity.
    DNDS_DEVICE_CALLABLE inline real TransverseSignalJump(const AcousticState &left, const AcousticState &right, const Vector3 &normal)
    {
        real jump = 0;
        for (int d = 0; d < 3; ++d)
            jump += (right.flow.conservative[1 + d] / right.flow.conservative[0] -
                     left.flow.conservative[1 + d] / left.flow.conservative[0]) *
                    normal[d];
        // max(|du+da|, |du-da|, |da|) = |du|+|da|.
        return std::abs(jump) + std::abs(right.soundSpeed - left.soundSpeed);
    }
    DNDS_DEVICE_CALLABLE inline FlowState Rusanov(const AcousticState &left, const AcousticState &right, const Vector3 &normal, const Vector3 &gridVelocity = {})
    {
        const auto a = PhysicalFlux(left.flow, normal, gridVelocity), b = PhysicalFlux(right.flow, normal, gridVelocity);
        const real speedLeft = SignalSpeed(left, normal, gridVelocity), speedRight = SignalSpeed(right, normal, gridVelocity);
        const real alpha = speedLeft > speedRight ? speedLeft : speedRight;
        FlowState result;
        for (int j = 0; j < 5; ++j)
            result[j] = .5 * (a[j] + b[j] - alpha * (right.flow.conservative[j] - left.flow.conservative[j]));
        return result;
    }
    /// Coarse selection resolved before face loops. Required inputs are explicit
    /// and kernel-ready. Caller validates state positivity, finite model data
    /// and unit/frame-consistent normals at the equation/operator boundary.
    /// This is a compressible-Euler flux profile, not a universal AC-NS flux.
    class RiemannFlux
    {
        Solver _solver;
        explicit RiemannFlux(Solver solver) : _solver(solver) {}

    public:
        static RiemannFlux Resolve(const std::string &name);
        static bool Available(const std::string &name);
        Solver Kind() const { return _solver; }
        Requirements RequiredInputs() const;
        /// These optional spatial inputs are prepared by the FV caller, never
        /// by the pointwise flux. A manual scalar remains valid for unit tests
        /// and one-dimensional users.
        bool UsesTransverseSignal() const;
        bool UsesPressureStencil() const;
        /// Validate optional data once in an outer preparation/operator boundary.
        /// Evaluate assumes that contract; unsupported/missing data never falls
        /// back to another numerical algorithm.
        void Validate(const FluxInput &input, const FluxSettings &settings = {}) const;
        FluxResult Evaluate(const FluxInput &input, const FluxSettings &settings = {}) const noexcept;
    };
}
