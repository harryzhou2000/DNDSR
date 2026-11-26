#pragma once
#include "DNDS/Core/Defines.hpp"
#include <limits>

namespace DNDS::Direct
{
    /// Infinity norm of I-A*X, evaluated in the inversion's row-scaled space.
    /// Matrix operands provide (i,j); caller owns all scratch and placement.
    template <class Matrix>
    DNDS_DEVICE_CALLABLE real InverseResidual(const Matrix &a, const Matrix &x, int n)
    {
        real norm = 0;
        for (int i = 0; i < n; ++i)
        {
            real sum = 0;
            for (int j = 0; j < n; ++j)
            {
                real value = i == j ? 1 : 0;
                for (int k = 0; k < n; ++k)
                    value -= a(i, k) * x(k, j);
                if (!std::isfinite(value))
                    return std::numeric_limits<real>::infinity();
                sum += std::abs(value);
            }
            norm = norm > sum ? norm : sum;
        }
        return norm;
    }

    /** @brief Optional guarded Newton correction of a nonsingular inverse.
     * @details X' = X + X(I-AX). In exact arithmetic the right residual squares.
     * Finite arithmetic is not guaranteed to improve: accept only a finite
     * candidate with strictly smaller infinity residual. Stop at the roundoff
     * floor or first rejected correction; never replace a usable inverse with
     * a failed trial. This is numerical arithmetic, with no leases or allocation.
     */
    template <class Matrix>
    DNDS_DEVICE_CALLABLE int RefineInverse(const Matrix &a, Matrix &x, Matrix &residual,
                                           Matrix &trial, int n, int steps)
    {
        int accepted = 0;
        real previous = InverseResidual(a, x, n);
        for (int step = 0; step < steps; ++step)
        {
            if (!std::isfinite(previous) || previous <= 8 * n * std::numeric_limits<real>::epsilon())
                break;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                {
                    real value = i == j ? 1 : 0;
                    for (int k = 0; k < n; ++k)
                        value -= a(i, k) * x(k, j);
                    residual(i, j) = value;
                }
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                {
                    real value = x(i, j);
                    for (int k = 0; k < n; ++k)
                        value += x(i, k) * residual(k, j);
                    trial(i, j) = value;
                }
            const real next = InverseResidual(a, trial, n);
            if (!(std::isfinite(next) && next < previous))
                break;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    x(i, j) = trial(i, j);
            previous = next;
            ++accepted;
        }
        return accepted;
    }

    /** @brief Refine a filtered SVD inverse without restoring discarded modes.
     * @details The cutoff remains epsilon*sigma_max. With every mode retained,
     * correct X directly. Otherwise refine C = U_r^T A V_r within the retained
     * subspace and publish X = V_r C^-1 U_r^T. Discarded slots use decoupled
     * identity padding in scratch only and never contribute to X. Testing
     * I-A*X against zero for a truncated inverse would be mathematically wrong.
     * The caller supplies the original, unmodified normalized matrix, SVD
     * factors, initial filtered inverse and four distinct scratch matrices.
     */
    template <class Matrix, class Singular>
    DNDS_DEVICE_CALLABLE void RefineSVDInverse(const Matrix &a, const Matrix &u, const Matrix &v,
                                               const Singular &singular, Matrix &x, Matrix &c, Matrix &z,
                                               Matrix &residual, Matrix &trial, int n, int steps)
    {
        real maximum = 0;
        for (int k = 0; k < n; ++k)
            maximum = maximum > singular(k) ? maximum : singular(k);
        const real cutoff = std::numeric_limits<real>::epsilon() * maximum;
        int rank = 0;
        for (int k = 0; k < n; ++k)
            rank += singular(k) > cutoff;
        if (rank == n)
        {
            RefineInverse(a, x, residual, trial, n, steps);
            return;
        }
        if (!rank)
            return;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                const bool active = singular(i) > cutoff && singular(j) > cutoff;
                real value = 0;
                if (active)
                    for (int p = 0; p < n; ++p)
                        for (int q = 0; q < n; ++q)
                            value += u(p, i) * a(p, q) * v(q, j);
                c(i, j) = active ? value : (i == j ? 1 : 0);
                z(i, j) = i == j ? (singular(i) > cutoff ? 1 / singular(i) : 1) : 0;
            }
        if (!RefineInverse(c, z, residual, trial, n, steps))
            return;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                real value = 0;
                for (int p = 0; p < n; ++p)
                    if (singular(p) > cutoff)
                        for (int q = 0; q < n; ++q)
                            if (singular(q) > cutoff)
                                value += v(i, p) * z(p, q) * u(j, q);
                trial(i, j) = value;
                if (!std::isfinite(value))
                    return; // Keep the complete original inverse on failure.
            }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                x(i, j) = trial(i, j);
    }
}
