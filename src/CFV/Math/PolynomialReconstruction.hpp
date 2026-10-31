#pragma once

#include "Geom/BaseFunction.hpp"
#include "DNDS/Errors.hpp"

namespace DNDS::CFV::Math
{
    /// Legacy monomial order and physical derivatives. No mesh, allocation,
    /// communication, or configuration ownership belongs to this kernel.
    using SmallMatrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, 0, 20, 20>;
    inline int PolynomialSize(int dim, int degree)
    {
        DNDS_check_throw_info((dim == 2 || dim == 3) && degree >= 1 && degree <= 3,
                              "reconstruction supports dimension 2/3 and polynomial degree 1..3");
        return dim == 2 ? Geom::Base::ndiffSizC2D[degree] : Geom::Base::ndiffSizC[degree];
    }
    inline void PolynomialDerivatives(SmallMatrix &out, int dim, int degree,
                                      const Geom::tPoint &point, const Geom::tPoint &center, real scale)
    {
        const int n = PolynomialSize(dim, degree);
        out.resize(n, n);
        const Geom::tPoint x = (point - center) / scale;
        if (dim == 2)
            Geom::Base::FPolynomialFill2D(out, x[0], x[1], 0., scale, scale, 1., n, n);
        else
            Geom::Base::FPolynomialFill3D(out, x[0], x[1], x[2], scale, scale, scale, n, n);
    }
    /// Diagonal weights of the isotropic variational functional, including
    /// symmetric-tensor multiplicities. Inputs are unsquared per-order weights.
    inline void FunctionalWeights(real *out, int dim, int degree, real length, const real *weights)
    {
        using Geom::Base::factorials;
        const int n = PolynomialSize(dim, degree);
        for (int i = 0; i < n; ++i)
        {
            const auto &a = dim == 2 ? Geom::Base::diffOperatorOrderList2D[i] : Geom::Base::diffOperatorOrderList[i];
            const int order = a[0] + a[1] + a[2];
            const real multiplicity = factorials[order] / (factorials[a[0]] * factorials[a[1]] * factorials[a[2]]);
            out[i] = sqr(weights[order]) * std::pow(length, 2 * order) * multiplicity;
        }
    }
    /// Scalar version of legacy WBAP L2 biway. Zero neighbors suppress a mode.
    inline real WBAPBiway(real center, real other)
    {
        const real epsilon = std::pow(verySmallReal, 0.25);
        const real ratio = center / (std::abs(other) + epsilon) * ((other > 0) - (other < 0));
        const real r2 = ratio * ratio;
        return other == 0 ? 0 : center * (1 + r2 * ratio) / (1 + r2 * r2);
    }
}
