#include "DNDS/Config/JsonSchema.hpp"

#include "DNDS/Core/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::Config
{
    namespace
    {
        std::string Escape(std::string text)
        {
            std::string result;
            for (const char c : text)
                result += c == '~' ? "~0" : c == '/' ? "~1"
                                                     : std::string(1, c);
            return result;
        }

        bool MatchesType(const nlohmann::ordered_json &value,
                         const std::string &type)
        {
            return type.empty() ||
                   (type == "null" && value.is_null()) ||
                   (type == "object" && value.is_object()) ||
                   (type == "array" && value.is_array()) ||
                   (type == "string" && value.is_string()) ||
                   (type == "boolean" && value.is_boolean()) ||
                   (type == "integer" && value.is_number_integer()) ||
                   (type == "number" && value.is_number() &&
                    std::isfinite(value.get<long double>()));
        }

        std::string DisplayPath(const std::string &path)
        {
            return path.empty() ? std::string("/") : path;
        }
    }

    void ValidateJsonSchema(const nlohmann::ordered_json &value,
                            const nlohmann::ordered_json &schema,
                            const std::string &path, JsonSchemaMode mode)
    {
        DNDS_check_throw_info(schema.is_object(),
                              "configuration schema is not an object at " + DisplayPath(path));
        if (schema.contains("oneOf"))
        {
            DNDS_check_throw_info(schema.at("oneOf").is_array(),
                                  "configuration oneOf is not an array at " + DisplayPath(path));
            const nlohmann::ordered_json *selected = nullptr;
            for (const auto &variant : schema.at("oneOf"))
                if (MatchesType(value, variant.value("type", std::string{})))
                {
                    DNDS_check_throw_info(!selected,
                                          "ambiguous configuration variant at " + DisplayPath(path));
                    selected = &variant;
                }
            DNDS_check_throw_info(selected,
                                  "configuration does not match an input variant at " + DisplayPath(path));
            ValidateJsonSchema(value, *selected, path, mode);
            return;
        }

        const auto type = schema.value("type", std::string{});
        DNDS_check_throw_info(MatchesType(value, type),
                              "configuration type mismatch at " + DisplayPath(path));
        if (schema.contains("enum"))
            DNDS_check_throw_info(
                std::find(schema.at("enum").begin(), schema.at("enum").end(), value) !=
                    schema.at("enum").end(),
                "unknown configuration enum value at " + DisplayPath(path));

        if (value.is_number())
        {
            const long double number = value.get<long double>();
            if (schema.contains("minimum"))
                DNDS_check_throw_info(number >= schema.at("minimum").get<long double>(),
                                      "configuration value is below its minimum at " + DisplayPath(path));
            if (schema.contains("maximum"))
                DNDS_check_throw_info(number <= schema.at("maximum").get<long double>(),
                                      "configuration value is above its maximum at " + DisplayPath(path));
        }

        if (value.is_object())
        {
            if (mode == JsonSchemaMode::Complete && schema.contains("required"))
                for (const auto &key : schema.at("required"))
                {
                    const auto name = key.get<std::string>();
                    DNDS_check_throw_info(value.contains(name),
                                          "missing configuration field at " + DisplayPath(path + "/" + Escape(name)));
                }
            const auto hasProperties = schema.contains("properties");
            const auto hasAdditional = schema.contains("additionalProperties");
            for (const auto &[key, child] : value.items())
            {
                const auto childPath = path + "/" + Escape(key);
                if (hasProperties && schema.at("properties").contains(key))
                    ValidateJsonSchema(child, schema.at("properties").at(key), childPath, mode);
                else if (hasAdditional && schema.at("additionalProperties").is_object())
                    ValidateJsonSchema(child, schema.at("additionalProperties"), childPath, mode);
                else if ((hasAdditional && schema.at("additionalProperties") == false) || hasProperties)
                    DNDS_check_throw_info(false, "unknown configuration key " + DisplayPath(childPath));
            }
        }

        if (value.is_array())
        {
            const auto minimum = schema.value("minItems", size_t(0));
            const auto maximum = schema.value("maxItems", std::numeric_limits<size_t>::max());
            DNDS_check_throw_info(value.size() >= minimum && value.size() <= maximum,
                                  "configuration array extent mismatch at " + DisplayPath(path));
            if (schema.contains("items"))
                for (size_t i = 0; i < value.size(); ++i)
                {
                    const auto &items = schema.at("items");
                    if (items.is_object())
                        ValidateJsonSchema(value.at(i), items, path + "/" + std::to_string(i), mode);
                    else if (items.is_array() && i < items.size())
                        ValidateJsonSchema(value.at(i), items.at(i), path + "/" + std::to_string(i), mode);
                }
        }
    }
}
