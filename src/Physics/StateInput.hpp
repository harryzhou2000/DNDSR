#pragma once

#include "Physics/ReferenceScales.hpp"
#include "nlohmann/json.hpp"
#include <vector>

namespace DNDS::Physics
{
    /// Setup-only physical state input. The vector ordering is supplied by the
    /// equation adapter. Bare arrays preserve Euler's code-unit consSensible
    /// convention. Absence is handled by the caller, never by filling values.
    struct StateInput
    {
        enum class Form
        {
            Conservative,
            Sensible,
            RhoP,
            RhoT,
            TP
        };
        Form form;
        bool physical;
        std::vector<real> values;
        static StateInput Parse(const nlohmann::ordered_json &input);
        static nlohmann::ordered_json Schema();
        /// Convert units only, before EOS conversion. scalarScales contains
        /// primitive RANS scales followed by 1 for each mass fraction. Neither
        /// this operation nor the scales supply a physical reference state.
        void ToCodeUnits(const ReferenceScales &scales, const std::vector<real> &scalarScales);
    };
}
