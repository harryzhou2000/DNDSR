#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Riemann/RiemannFluxMath.hpp"

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
    }
    using Math::Dot;
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
        return Riemann::RequiredInputs(_solver);
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
        return Math::Evaluate(_solver, input, settings);
    }
}
