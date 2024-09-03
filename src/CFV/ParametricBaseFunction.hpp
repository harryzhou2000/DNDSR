#pragma once

#include "BaseFunction.hpp"
#include "Geom/Elements.hpp"
#include "Geom/Quadrature.hpp"

namespace DNDS::CFV
{
    using tParamDiBj = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    struct ParametricDiBjPair
    {
        Geom::tPoint pParam;
        tParamDiBj DiBjParam;
    };

    template <int dim = 3>
    // static const int dim = 3;
    struct ParametricBaseCache
    {
        // [orderOfQuad][elemType][ic2f+1][iGauss]
        std::array<
            std::array<std::array<
                           std::vector<ParametricDiBjPair>,
                           Geom::Elem::ELEM_MAX_FACE_NUM>,
                       int(Geom::Elem::ElemType_NUM)>,
            Geom::Elem::INT_ORDER_MAX + 1>
            cache;

        ParametricBaseCache();

        auto FindMatchingPointDiBj(int intOrder, Geom::Elem::ElemType eType, rowsize ic2f, const Geom::tPoint &pParam)
        {
            const double errSqr = sqr(1e-10);
            int nFind{0};
            tParamDiBj *ptr{nullptr};
            for (auto &p : cache[intOrder][eType][ic2f < 0 ? 0 : ic2f + 1])
                if ((p.pParam - pParam).squaredNorm() <= errSqr)
                    nFind++, ptr = &p.DiBjParam;
            return std::make_tuple(nFind, ptr);
        }
    };
}