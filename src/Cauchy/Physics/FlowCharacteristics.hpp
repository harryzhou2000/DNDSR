#pragma once
#include "Cauchy/Physics/FlowState.hpp"
#include <cmath>

namespace DNDS::Cauchy
{
    /** @brief Cell-mean thermal inputs to a frozen-composition flow-five basis.
     * All quantities use the equation's code units. Total enthalpy includes
     * formation/reference energy. beta is dp/d(rho*e) at fixed rho and Y;
     * an ideal mixture supplies R_mix/cv, not gamma-equivalent energy closure.
     * Zero initialization deliberately represents invalid/unprepared inputs.
     */
    struct CharacteristicThermodynamics
    {
        real pressure = 0, soundSpeed = 0, totalEnthalpy = 0;
        real pressureInternalEnergyDerivative = 0;
        DNDS_DEVICE_CALLABLE bool Valid() const noexcept
        {
            return std::isfinite(pressure) && pressure > 0 &&
                   std::isfinite(soundSpeed) && soundSpeed > 0 &&
                   std::isfinite(totalEnthalpy) &&
                   std::isfinite(pressureInternalEnergyDerivative) && pressureInternalEnergyDerivative > 0;
        }
    };

    /// Column-major conservative/characteristic maps for the flow-five block.
    /// Other equation components remain scalars. This record owns only numbers.
    struct FlowCharacteristics
    {
        std::array<real, 25> forward, inverse;
        Vector3 velocity{};
        real soundSpeed = 0;
        bool componentwiseFallback = false;
    };
}
