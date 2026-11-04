#include "Physics/StateInput.hpp"
#include <algorithm>

namespace DNDS::Physics
{
    StateInput StateInput::Parse(const nlohmann::ordered_json &input)
    {
        StateInput result{Form::Sensible, false, {}};
        std::string type = "consSensible";
        if (input.is_array())
            result.values = input.get<std::vector<real>>();
        else
        {
            DNDS_check_throw_info(input.is_object() && input.size() == 2 && input.contains("type") && input.contains("state"),
                                  "state input requires {type,state} or a consSensible array");
            type = input.at("type").get<std::string>();
            result.values = input.at("state").get<std::vector<real>>();
        }
        if (type.size() > 4 && type.substr(type.size() - 4) == "_phy")
        {
            result.physical = true;
            type.resize(type.size() - 4);
        }
        if (type == "cons")
            result.form = Form::Conservative;
        else if (type == "consSensible")
            result.form = Form::Sensible;
        else if (type == "primRhoP")
            result.form = Form::RhoP;
        else if (type == "primRhoT")
            result.form = Form::RhoT;
        else if (type == "primTP")
            result.form = Form::TP;
        else
            DNDS_check_throw_info(false, "unsupported physical state type: " + type);
        DNDS_check_throw_info(result.values.size() >= 5 &&
                                  std::all_of(result.values.begin(), result.values.end(), [](real v)
                                              { return std::isfinite(v); }),
                              "physical state requires finite, complete flow5 and scalar values");
        return result;
    }
    nlohmann::ordered_json StateInput::Schema()
    {
        using Json = nlohmann::ordered_json;
        const Json values{{"type", "array"}, {"items", {{"type", "number"}}}, {"minItems", 5}};
        return {{"oneOf", Json::array({Json{{"type", "null"}}, values,
                                       Json{{"type", "object"}, {"required", {"type", "state"}}, {"additionalProperties", false}, {"properties", {{"type", {{"type", "string"}, {"enum", {"cons", "consSensible", "primRhoP", "primRhoT", "primTP", "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy", "primTP_phy"}}}}, {"state", values}}}}})}};
    }
    void StateInput::ToCodeUnits(const ReferenceScales &scales, const std::vector<real> &scalarScales)
    {
        scales.Validate();
        DNDS_check_throw_info(values.size() == 5 + scalarScales.size(), "state input component count differs from equation");
        if (!physical)
            return;
        const bool conservative = form == Form::Conservative || form == Form::Sensible;
        values[0] /= form == Form::TP ? scales.T0 : scales.rho0;
        for (int d = 1; d <= 3; ++d)
            values[d] /= conservative ? scales.Momentum() : scales.U0;
        values[4] /= form == Form::RhoT ? scales.T0 : scales.Pressure();
        for (index j = 0; j < index(scalarScales.size()); ++j)
        {
            DNDS_check_throw_info(std::isfinite(scalarScales[j]) && scalarScales[j] > 0, "invalid scalar input unit scale");
            values[5 + j] /= scalarScales[j] * (conservative ? scales.rho0 : 1);
        }
        physical = false;
    }
}
