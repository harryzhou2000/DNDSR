#pragma once

#include "Defines.hpp"
#define JSON_ASSERT DNDS_assert
#include "json.hpp"

namespace DNDS
{

    inline Eigen::VectorXd jsonGetEigenVector(const nlohmann::json &arr)
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
}