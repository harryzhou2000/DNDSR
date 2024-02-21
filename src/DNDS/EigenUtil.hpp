#pragma once

#include "Defines.hpp"

#include <fmt/core.h>
#include <Eigen/Core>
#include <fmt/ostream.h>

namespace DNDS
{
    // TODO: lessen copying chance?
    template <class dir>
    std::string to_string(const Eigen::DenseBase<dir> &v,
                          int precision = 5,
                          bool scientific = false)
    {
        std::stringstream ss;
        if (precision > 0)
            ss << std::setprecision(precision);
        if (scientific)
            ss << std::scientific;
        ss << v;
        return ss.str();
    }
}