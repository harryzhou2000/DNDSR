#pragma once

#include "PolynomialReconstruction.hpp"

namespace DNDS::CFV::Math
{
    /// Central definition of polynomial/derivative ordering and Cartesian frame
    /// changes for the scalar-scaled Taylor family. Geometry supplies centers,
    /// lengths and moments; a basis owns none of them. Cartesian-tree backends
    /// can specialize reference evaluation while preserving this mode contract.
    class TaylorBasis
    {
        int _dimension, _degree;

    public:
        TaylorBasis(int dimension, int degree) : _dimension(dimension), _degree(degree)
        {
            (void)PolynomialSize(dimension, degree);
        }
        int Size() const { return PolynomialSize(_dimension, _degree); }
        const std::array<int, 3> &MultiIndex(int mode) const
        {
            DNDS_check_throw_info(mode >= 0 && mode < Size(), "Taylor derivative mode outside basis");
            return _dimension == 2 ? Geom::Base::diffOperatorOrderList2D[mode] : Geom::Base::diffOperatorOrderList[mode];
        }
        int Degree(int mode) const
        {
            const auto &alpha = MultiIndex(mode);
            return alpha[0] + alpha[1] + alpha[2];
        }
        void Derivatives(SmallMatrix &values, const Geom::tPoint &point, const Geom::tPoint &center, real scale) const
        {
            PolynomialDerivatives(values, _dimension, _degree, point, center, scale);
        }
        /// Pull derivatives of f(y), y=J*x+b, into the x frame. Each row is
        /// one unique symmetric derivative; expansion supplies multiplicities
        /// in the chain rule. All derivative orders are transformed together.
        /// Translation is already handled by evaluating at the mapped point.
        void Pullback(SmallMatrix &values, const Geom::tGPoint &jacobian) const
        {
            DNDS_check_throw_info(values.rows() == Size() && jacobian.allFinite(), "invalid Taylor derivative pullback");
            if (_dimension == 2)
                DNDS_check_throw_info((jacobian.block<1, 2>(2, 0).norm() < 1e-12 &&
                                       jacobian.block<2, 1>(0, 2).norm() < 1e-12),
                                      "2D basis rotation leaves its geometric plane");
            SmallMatrix transform;
            transform.setZero(Size(), Size());
            for (int row = 0; row < Size(); ++row)
            {
                std::array<real, 20> coefficients{}, next{};
                coefficients[0] = 1;
                const auto &alpha = MultiIndex(row);
                for (int direction = 0; direction < _dimension; ++direction)
                    for (int repeat = 0; repeat < alpha[direction]; ++repeat)
                    {
                        next.fill(0);
                        for (int mode = 0; mode < Size(); ++mode)
                            if (coefficients[mode] != 0)
                                for (int target = 0; target < _dimension; ++target)
                                {
                                    auto beta = MultiIndex(mode);
                                    ++beta[target];
                                    for (int result = 0; result < Size(); ++result)
                                        if (MultiIndex(result) == beta)
                                        {
                                            next[result] += coefficients[mode] * jacobian(target, direction);
                                            break;
                                        }
                                }
                        coefficients = next;
                    }
                for (int col = 0; col < Size(); ++col)
                    transform(row, col) = coefficients[col];
            }
            values = (transform * values).eval();
        }
    };
}
