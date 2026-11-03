#pragma once

#include <limits>

namespace DNDS::Physics::Thermodynamics
{
    /// One thermodynamic point, in the units declared by its producer. Backend
    /// ChemicalSource queries produce SI; solver-facing material queries use
    /// ReferenceScales. Internal energy and enthalpy include thermochemical
    /// reference energy, not the separate bookkeeping base-energy offset.
    struct Properties
    {
        double density = std::numeric_limits<double>::quiet_NaN();
        double temperature = std::numeric_limits<double>::quiet_NaN();
        double pressure = std::numeric_limits<double>::quiet_NaN();
        double soundSpeed = std::numeric_limits<double>::quiet_NaN();
        double internalEnergy = std::numeric_limits<double>::quiet_NaN();
        double enthalpy = std::numeric_limits<double>::quiet_NaN();
        double cp = std::numeric_limits<double>::quiet_NaN();
        double cv = std::numeric_limits<double>::quiet_NaN();
        double gasConstant = std::numeric_limits<double>::quiet_NaN();
    };
}
