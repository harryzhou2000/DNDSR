#pragma once

#include "Geom/Math/BaseFunction.hpp"
#include "DNDS/Core/Errors.hpp"
#include "CFV/Math/WBAPScalar.hpp"

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
    /// Numerical leaf after setup validates dimension, extents (<=20) and
    /// positive scale. Bounded Eigen storage; no allocation or exception-based
    /// validation. Public setup callers use PolynomialDerivatives below.
    inline void PolynomialDerivativesUnchecked(SmallMatrix &out, int dim, int rows, int columns,
                                               const Geom::tPoint &point, const Geom::tPoint &center, real scale) noexcept
    {
        out.resize(rows, columns);
        const Geom::tPoint x = (point - center) / scale;
        if (dim == 2)
            Geom::Base::FPolynomialFill2D(out, x[0], x[1], 0., scale, scale, 1., rows, columns);
        else
            Geom::Base::FPolynomialFill3D(out, x[0], x[1], x[2], scale, scale, scale, rows, columns);
    }
    inline void PolynomialDerivatives(SmallMatrix &out, int dim, int degree,
                                      const Geom::tPoint &point, const Geom::tPoint &center, real scale,
                                      int derivativeOrder = -1)
    {
        const int n = PolynomialSize(dim, degree);
        const int rows = derivativeOrder < 0 ? n : derivativeOrder == 0 ? 1
                                                                        : PolynomialSize(dim, std::min(derivativeOrder, degree));
        PolynomialDerivativesUnchecked(out, dim, rows, n, point, center, scale);
    }
    /// Diagonal weights of the isotropic variational functional, including
    /// symmetric-tensor multiplicities. Inputs are unsquared per-order weights.
    inline void FunctionalWeights(real *out, int dim, int degree, real length, const real *weights, int multiplicityPower = 1)
    {
        using Geom::Base::factorials;
        const int n = PolynomialSize(dim, degree);
        for (int i = 0; i < n; ++i)
        {
            const auto &a = dim == 2 ? Geom::Base::diffOperatorOrderList2D[i] : Geom::Base::diffOperatorOrderList[i];
            const int order = a[0] + a[1] + a[2];
            const real multiplicity = factorials[order] / (factorials[a[0]] * factorials[a[1]] * factorials[a[2]]);
            out[i] = sqr(weights[order]) * std::pow(length, 2 * order) * std::pow(multiplicity, multiplicityPower);
        }
    }
}
