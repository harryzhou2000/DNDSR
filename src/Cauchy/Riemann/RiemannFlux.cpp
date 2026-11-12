#include "Cauchy/Riemann/RiemannFlux.hpp"
#include <algorithm>

namespace DNDS::Cauchy::Riemann
{
    namespace
    {
        const std::pair<const char *, Solver> names[]{
            {"rusanov", Solver::Rusanov},
            {"Roe", Solver::Roe},
            {"Roe_M1", Solver::Roe_M1},
            {"Roe_M2", Solver::Roe_M2},
            {"Roe_M3", Solver::Roe_M3},
            {"Roe_M4", Solver::Roe_M4},
            {"Roe_M5", Solver::Roe_M5},
            {"Roe_M6", Solver::Roe_M6},
            {"Roe_M7", Solver::Roe_M7},
            {"Roe_M8", Solver::Roe_M8},
            {"Roe_M9", Solver::Roe_M9},
            {"HLLC", Solver::HLLC},
            {"HLLEP", Solver::HLLEP},
            {"HLLEP_V1", Solver::HLLEP_V1},
            {"AUSMPWPlus", Solver::AUSMPWPlus},
            {"AUSMPlusUp", Solver::AUSMPlusUp},
            {"AUSMPlusUp2", Solver::AUSMPlusUp2}};
        real Dot(const Vector3 &a, const Vector3 &b)
        {
            real result = 0;
            for (int d = 0; d < 3; ++d)
                result += a[d] * b[d];
            return result;
        }
        Vector3 Velocity(const AcousticState &state)
        {
            Vector3 result;
            for (int d = 0; d < 3; ++d)
                result[d] = state.flow.conservative[1 + d] / state.flow.conservative[0];
            return result;
        }
        Vector3 Relative(Vector3 velocity, const Vector3 &grid)
        {
            for (int d = 0; d < 3; ++d)
                velocity[d] -= grid[d];
            return velocity;
        }
        real NormalSpeed(const AcousticState &state, const FluxInput &input) { return Dot(Relative(Velocity(state), input.gridVelocity), input.normal); }
        const AcousticState &Mean(const FluxInput &input, bool left) { return left ? (input.leftMean ? *input.leftMean : input.left) : (input.rightMean ? *input.rightMean : input.right); }
        bool AUSM(Solver kind) { return kind >= Solver::AUSMPWPlus; }
        bool Roe(Solver kind) { return kind >= Solver::Roe && kind <= Solver::Roe_M9; }
        void EntropyFix(int scheme, real aL, real aR, real a, real uL, real uR, real u, real velocityMagnitude, const FluxSettings &settings, FluxResult &result)
        {
            constexpr real hartenYee = .05, lowDissipation = .2, hCorrection = .25;
            const real scaleHY = hartenYee * settings.fixScale, scaleLD = lowDissipation * settings.fixScale, scaleH = hCorrection * settings.fixScale;
            const real scale = settings.incrementScale;
            auto set = [&](real speed, real acoustic)
            {
                result.acousticMinus = std::abs(speed - acoustic);
                result.contact = std::abs(speed);
                result.acousticPlus = std::abs(speed + acoustic);
            };
            if (scheme == 0)
            {
                const auto limit = [&](real value, real lambda, real left, real right)
                {
                    const real threshold = std::max({real(0), lambda - left, right - lambda}) * scaleHY;
                    return value < threshold ? .5 * (value * value / threshold + threshold) : value;
                };
                result.acousticMinus = limit(result.acousticMinus, u - a, uL - aL, uR - aR);
                result.contact = limit(result.contact * scale, u, uL, uR);
                result.acousticPlus = limit(result.acousticPlus, u + a, uL + aL, uR + aR);
            }
            else if (scheme == 1)
            {
                const real vL = signTol(uL, aL * smallReal) * std::max(std::abs(uL) * scale, aL * scaleLD);
                const real vR = signTol(uR, aR * smallReal) * std::max(std::abs(uR) * scale, aR * scaleLD);
                result.acousticMinus = std::max(std::abs(vL - aL), std::abs(vR - aR));
                result.contact = std::max(std::abs(vL), std::abs(vR));
                result.acousticPlus = std::max(std::abs(vL + aL), std::abs(vR + aR));
            }
            else if (scheme == 3)
            {
                const real star = std::min(std::abs(u) / scaleLD, a);
                result.acousticMinus = std::abs(u * scale - star);
                result.acousticPlus = std::abs(u * scale + star);
            }
            else if (scheme == 4)
                set(signTol(u, a * smallReal) * std::max(a * scaleLD, std::abs(u) * scale), a);
            else if (scheme == 5)
            {
                const real vL = uL * scale, vR = uR * scale;
                const real acousticL = std::min(aL, std::abs(vL) / scaleLD), acousticR = std::min(aR, std::abs(vR) / scaleLD);
                result.acousticMinus = std::max(std::abs(vL - acousticL), std::abs(vR - acousticR));
                result.contact = std::max(std::abs(vL), std::abs(vR));
                result.acousticPlus = std::max(std::abs(vL + acousticL), std::abs(vR + acousticR));
            }
            else
            {
                const real acoustic = scheme == 9 ? std::max(aL, aR) : a;
                set(u * scale, acoustic);
                if (scheme != 7)
                {
                    result.acousticMinus = std::max(result.acousticMinus, settings.dLambda * scaleH);
                    result.contact = std::max(result.contact, settings.dLambda * scaleH);
                    result.acousticPlus = std::max(result.acousticPlus, settings.dLambda * scaleH);
                }
                if (scheme != 6)
                {
                    const real threshold = scaleHY * (velocityMagnitude + acoustic);
                    auto smooth = [&](real &value)
                    {
                        if (value < threshold)
                            value = (value * value + threshold * threshold) / (2 * threshold);
                    };
                    smooth(result.acousticMinus);
                    smooth(result.contact);
                    smooth(result.acousticPlus);
                }
            }
        }
        struct Decomposition
        {
            real minus, contact, plus;
            Vector3 shear;
        };
        Decomposition Decompose(const FluxInput &input)
        {
            const auto &roe = *input.roe;
            FlowState jump;
            for (int j = 0; j < 5; ++j)
                jump[j] = input.right.flow.conservative[j] - input.left.flow.conservative[j];
            // Pressure-neutral mode data comes from the equation adapter. This
            // numerical decomposition never reconstructs thermodynamics.
            jump[4] -= roe.contactEnergyJump;
            Decomposition result{};
            for (int j = 0; j < 5; ++j)
                result.contact += roe.contactProjection[j] * jump[j];
            real momentumNormal = 0;
            for (int d = 0; d < 3; ++d)
            {
                momentumNormal += jump[1 + d] * input.normal[d];
                result.shear[d] = jump[1 + d] - jump[0] * roe.velocity[d];
            }
            const real shearNormal = Dot(result.shear, input.normal), un = Dot(roe.velocity, input.normal);
            for (int d = 0; d < 3; ++d)
                result.shear[d] -= shearNormal * input.normal[d];
            result.minus = (jump[0] * (un + roe.soundSpeed) - momentumNormal - roe.soundSpeed * result.contact) / (2 * roe.soundSpeed);
            result.plus = jump[0] - result.minus - result.contact;
            return result;
        }
        FlowState Dissipation(const FluxInput &input, const Decomposition &waves, const FluxResult &magnitudes)
        {
            const auto &roe = *input.roe;
            const real minus = waves.minus * magnitudes.acousticMinus, contact = waves.contact * magnitudes.contact, plus = waves.plus * magnitudes.acousticPlus;
            const real un = Dot(roe.velocity, input.normal);
            FlowState result{minus + contact + plus, 0, 0, 0,
                             (roe.enthalpy - un * roe.soundSpeed) * minus + roe.contactEnergy * contact + (roe.enthalpy + un * roe.soundSpeed) * plus +
                                 Dot(waves.shear, roe.velocity) * magnitudes.contact + roe.contactEnergyJump * magnitudes.contact};
            for (int d = 0; d < 3; ++d)
                result[1 + d] = (roe.velocity[d] - roe.soundSpeed * input.normal[d]) * minus + roe.velocity[d] * contact +
                                (roe.velocity[d] + roe.soundSpeed * input.normal[d]) * plus + waves.shear[d] * magnitudes.contact;
            return result;
        }
        real MachSplit(real mach, bool positive, bool fourth)
        {
            const real s = positive ? 1 : -1;
            if (std::abs(mach) >= 1)
                return .5 * (mach + s * std::abs(mach));
            return s * .25 * (mach + s) * (mach + s) + (fourth ? s * .125 * sqr(mach * mach - 1) : 0);
        }
        real PressureSplit(real mach, bool positive, real alpha)
        {
            const real s = positive ? 1 : -1;
            if (std::abs(mach) >= 1)
                return .5 * (1 + s * (mach > 0 ? 1 : -1));
            return .25 * sqr(mach + s) * (2 - s * mach) + s * alpha * mach * sqr(mach * mach - 1);
        }
        FlowState AUSMFlux(Solver kind, const FluxInput &input, const FluxSettings &settings)
        {
            const auto &left = input.left.flow, &right = input.right.flow;
            const auto vL = Relative(Velocity(input.left), input.gridVelocity), vR = Relative(Velocity(input.right), input.gridVelocity);
            const real unL = Dot(vL, input.normal), unR = Dot(vR, input.normal), a = input.ausm->faceSoundSpeed;
            const real ml = unL / a, mr = unR / a, pL = left.pressure, pR = right.pressure;
            real cL, cR, pressure;
            if (kind == Solver::AUSMPWPlus)
            {
                const real mp = MachSplit(ml, true, false), mm = MachSplit(mr, false, false);
                pressure = PressureSplit(ml, true, 0) * pL + PressureSplit(mr, false, 0) * pR;
                const real ratio = std::min(pL / pR, pR / pL), weight = 1 - ratio * ratio * ratio;
                const real stencil = sqr(std::min(real(1), settings.pressureStencilMin / std::min(pL, pR)));
                const real fL = pressure > 0 ? (pL / pressure - 1) * stencil : 0, fR = pressure > 0 ? (pR / pressure - 1) * stencil : 0;
                if (mp + mm >= 0)
                {
                    cL = a * (mp + mm * ((1 - weight) * (1 + fR) - fL));
                    cR = a * mm * weight * (1 + fR);
                }
                else
                {
                    cL = a * mp * weight * (1 + fL);
                    cR = a * (mm + mp * ((1 - weight) * (1 + fL) - fR));
                }
            }
            else
            {
                const real meanMach = .5 * (ml * ml + mr * mr), m0 = std::sqrt(std::min(real(1), std::max(meanMach, sqr(settings.referenceMach)))), fa = m0 * (2 - m0);
                const real density = .5 * (left.conservative[0] + right.conservative[0]);
                const real massMach = MachSplit(ml, true, true) + MachSplit(mr, false, true) - .25 / fa * std::max(1 - meanMach, real(0)) * (pR - pL) / (density * a * a);
                cL = massMach >= 0 ? a * massMach : 0;
                cR = massMach < 0 ? a * massMach : 0;
                if (kind == Solver::AUSMPlusUp)
                {
                    const real alpha = 3. / 16 * (-4 + 5 * fa * fa), pp = PressureSplit(ml, true, alpha), pm = PressureSplit(mr, false, alpha);
                    pressure = pp * pL + pm * pR - .75 * pp * pm * (left.conservative[0] + right.conservative[0]) * fa * a * (unR - unL);
                }
                else
                {
                    const real pp = PressureSplit(ml, true, 0), pm = PressureSplit(mr, false, 0), speed = std::sqrt(.5 * (Dot(vL, vL) + Dot(vR, vR)));
                    pressure = .5 * (pL + pR) + .5 * (pp - pm) * (pL - pR) + speed * (pp + pm - 1) * density * a;
                }
            }
            FlowState result;
            for (int j = 0; j < 5; ++j)
                result[j] = cL * left.conservative[j] + cR * right.conservative[j];
            for (int d = 0; d < 3; ++d)
                result[1 + d] += pressure * input.normal[d];
            result[4] += cL * pL + cR * pR + pressure * Dot(input.gridVelocity, input.normal);
            return result;
        }
    }
    bool RiemannFlux::Available(const std::string &name)
    {
        for (const auto &entry : names)
            if (name == entry.first)
                return true;
        return false;
    }
    RiemannFlux RiemannFlux::Resolve(const std::string &name)
    {
        for (const auto &entry : names)
            if (name == entry.first)
                return RiemannFlux(entry.second);
        DNDS_check_throw_info(false, "unavailable Cauchy Riemann solver " + name);
        return RiemannFlux(Solver::Rusanov);
    }
    Requirements RiemannFlux::RequiredInputs() const
    {
        const bool linearization = (Roe(_solver) && _solver != Solver::Roe_M2) || _solver == Solver::HLLEP || _solver == Solver::HLLEP_V1 || _solver == Solver::HLLC;
        return {true, true, linearization, _solver != Solver::Rusanov, linearization,
                _solver == Solver::HLLC, AUSM(_solver)};
    }
    bool RiemannFlux::UsesTransverseSignal() const
    {
        return _solver == Solver::Roe_M6 || _solver == Solver::Roe_M8 || _solver == Solver::Roe_M9 ||
               _solver == Solver::HLLEP || _solver == Solver::HLLEP_V1;
    }
    bool RiemannFlux::UsesPressureStencil() const
    {
        return _solver == Solver::AUSMPWPlus;
    }
    void RiemannFlux::Validate(const FluxInput &input, const FluxSettings &settings) const
    {
        const auto requirements = RequiredInputs();
        DNDS_check_throw_info(!requirements.meanStates || (input.leftMean && input.rightMean), "Riemann flux requires explicit mean-state inputs");
        DNDS_check_throw_info(!requirements.roe || input.roe, "Riemann flux requires an equation-prepared Roe linearization");
        DNDS_check_throw_info(!requirements.outerWaveBounds || input.outerWaves, "HLLC requires equation-prepared outer wave bounds");
        DNDS_check_throw_info(!requirements.ausmFaceSpeed || input.ausm, "AUSM requires an equation-prepared face acoustic scale");
        DNDS_check_throw_info((_solver != Solver::Roe_M3 && _solver != Solver::Roe_M5) || settings.fixScale > 0,
                              "low-dissipation Roe flux requires a positive fixScale divisor");
        DNDS_check_throw_info(std::isfinite(settings.fixScale) && settings.fixScale >= 0 && std::isfinite(settings.incrementScale) && settings.incrementScale >= 0 &&
                                  std::isfinite(settings.dLambda) && settings.dLambda >= 0 && std::isfinite(settings.referenceMach) && settings.referenceMach > 0 &&
                                  std::isfinite(settings.pressureStencilMin) && settings.pressureStencilMin > 0,
                              "invalid Riemann numerical controls");
        for (const auto *state : {&input.left, &input.right, input.leftMean, input.rightMean})
            if (state)
            {
                DNDS_check_throw_info(state->flow.conservative[0] > 0 && std::isfinite(state->flow.pressure) && state->flow.pressure > 0 && state->soundSpeed > 0 && std::isfinite(state->soundSpeed), "invalid prepared Riemann state");
                for (real value : state->flow.conservative)
                    DNDS_check_throw_info(std::isfinite(value), "nonfinite Riemann conservative state");
            }
        DNDS_check_throw_info(std::abs(Dot(input.normal, input.normal) - 1) < 1e-12, "Riemann normal must have unit length");
        for (real value : input.gridVelocity)
            DNDS_check_throw_info(std::isfinite(value), "nonfinite Riemann grid velocity");
        if (input.roe)
        {
            DNDS_check_throw_info(input.roe->soundSpeed > 0 && std::isfinite(input.roe->soundSpeed) && std::isfinite(input.roe->enthalpy) && std::isfinite(input.roe->contactEnergy) && std::isfinite(input.roe->contactEnergyJump), "invalid Roe linearization data");
            for (real value : input.roe->contactProjection)
                DNDS_check_throw_info(std::isfinite(value), "nonfinite Roe contact projection");
            for (real value : input.roe->velocity)
                DNDS_check_throw_info(std::isfinite(value), "nonfinite Roe velocity");
        }
        if (input.outerWaves)
            DNDS_check_throw_info(std::isfinite(input.outerWaves->left) && std::isfinite(input.outerWaves->right) &&
                                      (input.outerWaves->left >= 0 || input.outerWaves->right <= 0 || input.outerWaves->left < input.outerWaves->right),
                                  "invalid transonic HLLC outer wave bounds");
        if (input.ausm)
            DNDS_check_throw_info(input.ausm->faceSoundSpeed > 0 && std::isfinite(input.ausm->faceSoundSpeed), "invalid AUSM acoustic input");
    }
    FluxResult RiemannFlux::Evaluate(const FluxInput &input, const FluxSettings &settings) const noexcept
    {
        const auto &left = input.left.flow.conservative, &right = input.right.flow.conservative;
        const auto fluxLeft = PhysicalFlux(input.left.flow, input.normal, input.gridVelocity), fluxRight = PhysicalFlux(input.right.flow, input.normal, input.gridVelocity);
        const auto &meanL = Mean(input, true), &meanR = Mean(input, false);
        const real uL = NormalSpeed(meanL, input), uR = NormalSpeed(meanR, input);
        FluxResult result{};
        if (_solver == Solver::Rusanov || _solver == Solver::Roe_M2)
        {
            const real speed = std::max(std::abs(uL) + meanL.soundSpeed, std::abs(uR) + meanR.soundSpeed);
            result.acousticMinus = result.contact = result.acousticPlus = speed;
            for (int j = 0; j < 5; ++j)
                result.flux[j] = .5 * (fluxLeft[j] + fluxRight[j] - speed * (right[j] - left[j]));
            return result;
        }
        if (AUSM(_solver))
        {
            result.flux = AUSMFlux(_solver, input, settings);
            const real u = .5 * (uL + uR), a = std::max(meanL.soundSpeed, meanR.soundSpeed);
            result.acousticMinus = std::abs(u - a);
            result.contact = std::abs(u);
            result.acousticPlus = std::abs(u + a);
            return result;
        }
        if (_solver == Solver::HLLC)
        {
            const real sl = input.outerWaves->left, sr = input.outerWaves->right;
            // Diagnostic wave estimates preserve the legacy Roe-mean contract
            // when a linearization is supplied; bounds remain externally set.
            const real u = input.roe ? Dot(Relative(input.roe->velocity, input.gridVelocity), input.normal) : .5 * (uL + uR);
            const real a = input.roe ? input.roe->soundSpeed : std::max(meanL.soundSpeed, meanR.soundSpeed);
            result.acousticMinus = std::abs(u - a);
            result.contact = std::abs(u);
            result.acousticPlus = std::abs(u + a);
            if (sl >= 0)
            {
                result.flux = fluxLeft;
                return result;
            }
            if (sr <= 0)
            {
                result.flux = fluxRight;
                return result;
            }
            const real denominator = left[0] * (sl - uL) - right[0] * (sr - uR);
            const real faceL = NormalSpeed(input.left, input), faceR = NormalSpeed(input.right, input);
            const real contact = std::abs(denominator) > verySmallReal ? (input.right.flow.pressure - input.left.flow.pressure + left[0] * faceL * (sl - uL) - right[0] * faceR * (sr - uR)) / denominator : 0;
            const bool upstream = contact >= 0;
            const auto &state = upstream ? left : right, &physical = upstream ? fluxLeft : fluxRight;
            const real wave = upstream ? sl : sr, velocity = upstream ? uL : uR, pressure = upstream ? input.left.flow.pressure : input.right.flow.pressure;
            const real div = wave - contact;
            if (std::abs(div) < verySmallReal)
            {
                result.flux = physical;
                return result;
            }
            const real starPressure = pressure + state[0] * (wave - velocity) * (contact - velocity);
            for (int j = 0; j < 5; ++j)
            {
                const real direction = j == 0 ? 0 : j == 4 ? contact
                                                           : input.normal[j - 1];
                result.flux[j] = ((state[j] * wave - physical[j]) * contact + direction * starPressure * wave) / div;
            }
            return result;
        }
        const auto &roe = *input.roe;
        const real un = Dot(roe.velocity, input.normal), u = un - Dot(input.gridVelocity, input.normal), a = roe.soundSpeed;
        result.acousticMinus = std::abs(u - a);
        result.contact = std::abs(u);
        result.acousticPlus = std::abs(u + a);
        const auto waves = Decompose(input);
        if (Roe(_solver))
        {
            EntropyFix(int(_solver) - int(Solver::Roe), meanL.soundSpeed, meanR.soundSpeed, a, uL, uR, u, std::sqrt(Dot(Relative(roe.velocity, input.gridVelocity), Relative(roe.velocity, input.gridVelocity))), settings, result);
            const auto dissipation = Dissipation(input, waves, result);
            for (int j = 0; j < 5; ++j)
                result.flux[j] = .5 * (fluxLeft[j] + fluxRight[j] - dissipation[j]);
        }
        else
        {
            const real sl = std::min(u - a, uL - input.left.soundSpeed), sr = std::max(u + a, uR + input.right.soundSpeed);
            const real sp = std::max(sr, real(0)), sm = std::min(sl, real(0)), dfix = a / (a + std::max(std::abs(u), settings.dLambda * settings.fixScale));
            if (_solver == Solver::HLLEP)
            {
                const real div = sp - sm + signP(sp - sm) * verySmallReal;
                for (int j = 0; j < 5; ++j)
                {
                    const real contactMode = j == 0 ? 1 : j == 4 ? roe.contactEnergy
                                                                 : roe.velocity[j - 1];
                    result.flux[j] = (sp * fluxLeft[j] - sm * fluxRight[j]) / div + (sp * sm / div) * (right[j] - left[j] - dfix * contactMode * waves.contact);
                }
            }
            else
            {
                result.contact = std::abs(((sp + sm) * u - 2 * (1 - dfix) * sp * sm) / (sp - sm));
                result.acousticMinus = std::abs(((sp + sm) * (u - a) - 2 * sp * sm) / (sp - sm));
                result.acousticPlus = std::abs(((sp + sm) * (u + a) - 2 * sp * sm) / (sp - sm));
                const auto dissipation = Dissipation(input, waves, result);
                for (int j = 0; j < 5; ++j)
                    result.flux[j] = .5 * (fluxLeft[j] + fluxRight[j] - dissipation[j]);
            }
        }
        return result;
    }
}
