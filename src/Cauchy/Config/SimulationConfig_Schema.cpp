#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy
{
    Json SimulationConfig::InputSchema()
    {
        const Json domain{{"type", "object"},
                          {"additionalProperties", false},
                          {"required", {"name"}},
                          {"properties",
                           {{"name", {{"type", "string"}, {"description", "Unique domain name"}}},
                            {"ranks", {{"type", "array"}, {"items", {{"type", "integer"}}}, {"minItems", 1}}},
                            {"profiles", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                            {"overrides", DomainSettings::schema("Per-domain settings patch")}}}};
        Json schema{{"$schema", "http://json-schema.org/draft-07/schema#"},
                    {"title", "DNDSR Cauchy simulation input"},
                    {"type", "object"},
                    {"additionalProperties", false},
                    {"properties",
                     {{"schemaVersion", {{"type", "integer"}, {"enum", {1}}, {"default", 1}}},
                      {"profiles", {{"type", "object"}, {"additionalProperties", DomainSettings::schema("Named settings patch")}}},
                      {"defaults", DomainSettings::schema("Root domain-settings patch")},
                      {"domains", {{"type", "array"}, {"items", domain}, {"minItems", 1}}},
                      {"time", TimeSettings::schema("Physical-time settings patch")},
                      {"output", OutputSettings::schema("Output settings patch")},
                      {"interfaces", {{"type", "array"}, {"items", InterfaceSettings::schema("Directed interface patch")}}}}}};
        return schema;
    }

}
