#pragma once

#include "DNDS/Serializer/JsonUtil.hpp"

#include <string>

namespace DNDS::Config
{
    /// Controls whether JSON-Schema `required` members are enforced. Partial
    /// validation is intended for layered configuration patches; it still
    /// validates every supplied value, nested key, enum and numeric bound.
    enum class JsonSchemaMode
    {
        Complete,
        Partial
    };

    /// @brief Validate one JSON value against the bounded schema vocabulary
    /// used by DNDS configuration registries.
    ///
    /// Supported keywords are `type`, `oneOf`, `enum`, `required`,
    /// `properties`, `additionalProperties`, `items`, `minItems`, `maxItems`,
    /// `minimum` and `maximum`. Unknown instance keys are rejected when the
    /// schema declares properties or `additionalProperties: false`; an empty
    /// schema remains an explicitly opaque value.
    ///
    /// @param value User-supplied JSON value.
    /// @param schema JSON-Schema fragment emitted by a DNDS registry or an
    ///               explicitly reviewed root schema.
    /// @param path JSON-pointer-like diagnostic path.
    /// @param mode Complete input or layered partial-patch validation.
    /// @throws std::runtime_error on the first mismatch.
    void ValidateJsonSchema(const nlohmann::ordered_json &value,
                            const nlohmann::ordered_json &schema,
                            const std::string &path = "",
                            JsonSchemaMode mode = JsonSchemaMode::Complete);

    /// @brief Validate a layered configuration patch while permitting omitted
    /// schema members.
    inline void ValidatePartialJsonSchema(const nlohmann::ordered_json &value,
                                          const nlohmann::ordered_json &schema,
                                          const std::string &path = "")
    {
        ValidateJsonSchema(value, schema, path, JsonSchemaMode::Partial);
    }
}
