#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Config/ConfigParam.hpp"

namespace DNDS::Physics::Chemistry
{
    /// Controls for the existing Cantera constant-volume source integrator.
    /// Tolerances use the backend's reactor variables and are not CFD nonlinear
    /// residual tolerances. Nonpositive maxOrder retains Cantera's default order.
    struct ReactorStepSettings
    {
        real rtol = 1e-10, atol = 1e-18;
        int maxOrder = 0, maxSteps = 10000000;
        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(rtol) && rtol > 0 && std::isfinite(atol) && atol > 0 &&
                                      maxOrder <= 5 && maxSteps > 0,
                                  "invalid constant-volume reactor controls");
        }
        DNDS_DECLARE_CONFIG(ReactorStepSettings)
        {
            DNDS_FIELD(rtol, "Cantera reactor relative tolerance", DNDS::Config::range(0.0));
            DNDS_FIELD(atol, "Cantera reactor absolute tolerance", DNDS::Config::range(0.0));
            DNDS_FIELD(maxOrder, "Cantera reactor max order; <=0 leaves default");
            DNDS_FIELD(maxSteps, "Cantera reactor max internal steps", DNDS::Config::range(1));
        }
    };
}
