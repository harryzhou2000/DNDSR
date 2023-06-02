#pragma once
#include "DNDS/Defines.hpp"

namespace DNDS::Geom
{
    using t_index = int32_t;
    const t_index invalid_index = INT32_MAX;
    using t_real = double;
    using tPoint = Eigen::Vector3d;
    using tJacobi = Eigen::Matrix3d;
    using tGPoint = Eigen::Matrix3d;

    static_assert(std::is_signed_v<t_index>);
}

namespace DNDS::Geom
{
    /**
     * @brief get normal vector from facial jacobian
     * note that
     * $$
     * J_{ij} = \partial_{\xi_j} x_i
     * $$
     */
    template <int d>
    inline tPoint FacialJacobianToNormVec(const tJacobi &J)
    {
        static_assert(d == 2 || d == 3);
        if constexpr (d == 2)
        {
            DNDS_assert_info(J(2, 0) == 0, "Must be a line in x-y plane");
            DNDS_assert_info(J(Eigen::all, 1).norm() == 0, "Must be a line in x-y plane");
            DNDS_assert_info(J(Eigen::all, 2).norm() == 0, "Must be a line in x-y plane");
            return tPoint{J(1, 0), -J(0, 0), 0};
        }
        if constexpr (d == 3)
        {
            DNDS_assert_info(J(Eigen::all, 2).norm() == 0, "Must be a face");
            return J(Eigen::all, 0).cross(J(Eigen::all, 1));
        }
    }
}