#pragma once

#include "Defines.hpp"
#define JSON_ASSERT DNDS_assert
#include "json.hpp"

namespace DNDS
{

    inline Eigen::VectorXd JsonGetEigenVector(const nlohmann::json &arr)
    {
        try
        {
            DNDS_assert(arr.is_array());
            Eigen::VectorXd ret;
            ret.resize(arr.size());
            for (int i = 0; i < ret.size(); i++)
                ret(i) = arr.at(i).get<double>();
            return ret;
        }
        catch (...)
        {
            DNDS_assert_info(false, "array parse bad");
            return Eigen::VectorXd{0};
        }
    }

    inline nlohmann::json EigenVectorGetJson(const Eigen::VectorXd &ve)
    {
        std::vector<real> v;
        v.resize(ve.size());
        for (size_t i = 0; i < ve.size(); i++)
            v[i] = ve[i];
        return nlohmann::json(v);
    }

#define __DNDS__json_to_config(name)                                       \
    {                                                                      \
        if (read)                                                          \
            try                                                            \
            {                                                              \
                (name = jsonObj.at(#name).template get<decltype(name)>()); \
            }                                                              \
            catch (const std::exception &v)                                \
            {                                                              \
                std::cerr << v.what() << std::endl;                        \
                DNDS_assert_info(false, #name);                            \
            }                                                              \
        else                                                               \
            (jsonObj[#name] = name);                                       \
    }
#define DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(Type, ...)                                                                                                   \
    friend void to_json(nlohmann::ordered_json &nlohmann_json_j, const Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__)) } \
    friend void from_json(const nlohmann::ordered_json &nlohmann_json_j, Type &nlohmann_json_t) { NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__)) }
}

namespace Eigen // why doesn't work?
{
    inline void to_json(nlohmann::json &j, const VectorXd &v)
    {
        j = DNDS::EigenVectorGetJson(v);
    }

    inline void from_json(const nlohmann::json &j, VectorXd &v)
    {
        v = DNDS::JsonGetEigenVector(j);
    }

    inline void to_json(nlohmann::ordered_json &j, const VectorXd &v)
    {
        j = DNDS::EigenVectorGetJson(v);
    }

    inline void from_json(const nlohmann::ordered_json &j, VectorXd &v)
    {
        v = DNDS::JsonGetEigenVector(j);
    }

    inline void to_json(nlohmann::ordered_json &j, const Vector3d &v)
    {
        j = DNDS::EigenVectorGetJson(v);
    }

    inline void from_json(const nlohmann::ordered_json &j, Vector3d &v)
    {
        v = DNDS::JsonGetEigenVector(j);
    }
}
