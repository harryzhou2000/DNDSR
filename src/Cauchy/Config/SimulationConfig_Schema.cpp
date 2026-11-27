#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy
{
    Json SimulationConfig::InputSchema()
    {
        const auto inputDomainSchema = [](const char *description)
        {
            auto schema = DomainSettings::schema(description);
            auto &fields = schema.at("properties").at("numerics").at("properties").at("riemann").at("properties");
            for (const auto &[legacy, current] : RiemannSettings::pressureRatioAliases)
            {
                Json alias = fields.at(current);
                alias["description"] = std::string("Deprecated alias for ") + current;
                fields[legacy] = std::move(alias);
            }
            return schema;
        };
        const Json domain{{"type", "object"},
                          {"additionalProperties", false},
                          {"required", {"name"}},
                          {"properties",
                           {{"name", {{"type", "string"}, {"description", "Unique domain name"}}},
                            {"ranks", {{"type", "array"}, {"items", {{"type", "integer"}}}, {"minItems", 1}}},
                            {"profiles", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                            {"overrides", inputDomainSchema("Per-domain settings patch")}}}};
        Json schema{{"$schema", "http://json-schema.org/draft-07/schema#"},
                    {"title", "DNDSR Cauchy simulation input"},
                    {"type", "object"},
                    {"additionalProperties", false},
                    {"properties",
                     {{"schemaVersion", {{"type", "integer"}, {"enum", {1}}, {"default", 1}}},
                      {"profiles", {{"type", "object"}, {"additionalProperties", inputDomainSchema("Named settings patch")}}},
                      {"defaults", inputDomainSchema("Root domain-settings patch")},
                      {"domains", {{"type", "array"}, {"items", domain}, {"minItems", 1}}},
                      {"time", TimeSettings::schema("Physical-time settings patch")},
                      {"output", OutputSettings::schema("Output settings patch")},
                      {"interfaces", {{"type", "array"}, {"items", InterfaceSettings::schema("Directed interface patch")}}}}}};
        return schema;
    }

}
