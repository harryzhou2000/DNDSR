#pragma once

#include <cmath>

#include "DNDS/Math/HardEigen.hpp"
#include "DNDS/Core/Defines.hpp"

namespace DNDS::Linear
{

    template <class TDATA>
    class GMRES_LeftPreconditioned
    {
        uint32_t kSubspace;

        std::vector<TDATA> Vs;
        TDATA V_temp;
        TDATA MLb;

    public:
        template <class Finit>
        GMRES_LeftPreconditioned(
            uint32_t NkSubspace,
            Finit &&finit = [](TDATA &) {}) : kSubspace(NkSubspace)
        {
            Vs.resize(kSubspace + 1);
            for (auto &i : Vs)
                finit(i);
            finit(V_temp);
            finit(MLb);
        }

        /**
         * @brief
         *
         * @tparam TFA
         * @tparam TFML
         * @tparam TFstop
         * @param FA  void FA(TDATA &x, TDATA &Ax)
         * @param FML void FML(TDATA &x, TDATA &MLx)
         * @param b   rhs
         * @param x   input and output
         * @param nRestart
         * @param FStop bool FStop(iRestart, res, resB)
         */
        template <class TFA, class TFML, class TFDot, class TFstop>
        bool solve(TFA &&FA, TFML &&FML, TFDot &&fDot, TDATA &b, TDATA &x, uint32_t nRestart, TFstop &&FStop)
        {
            FML(b, MLb); // MLb = ML * b
            real scale_MLb = std::sqrt(fDot(MLb, MLb));
            MatrixXR h;
            h.setZero(kSubspace + 1, kSubspace);
            bool converged = false;
            for (uint32_t iRestart = 0; iRestart <= nRestart; iRestart++)
            {
                h.setZero();
                FA(x, V_temp);                             // V_temp = A * x
                FML(V_temp, Vs[0]);                        // Vs[0] = ML * A * x
                Vs[0].addTo(MLb, -1.0);                    // Vs[0] = ML * A * x - ML * b = -r
                real beta = std::sqrt(fDot(Vs[0], Vs[0])); // beta = norm2(r)
                if (FStop(iRestart, beta, scale_MLb))      // see if converge
                {
                    converged = true;
                    break;
                }
                if (std::abs(beta) < verySmallReal) // beta is floating-point negligible
                {
                    converged = true;
                    break;
                }
                if (iRestart == nRestart)
                    break;
                Vs[0] *= -1.0 / beta; // Vs[0] = r/norm2(r)

                uint32_t j = 0;
                bool luckyBreakdown = false;
                for (; j < kSubspace; j++) // Arnoldi (Modified Gram-Schmidt)
                {
                    FA(Vs[j], V_temp);      // V_temp = A * Vs[j]
                    FML(V_temp, Vs[j + 1]); // Vs[j + 1] = ML * A * Vs[j]
                    real normAv = std::sqrt(fDot(Vs[j + 1], Vs[j + 1]));
                    for (uint32_t i = 0; i <= j; i++)
                    {
                        h(i, j) = fDot(Vs[j + 1], Vs[i]);
                        Vs[j + 1].addTo(Vs[i], -h(i, j));
                    }
                    h(j + 1, j) = std::sqrt(fDot(Vs[j + 1], Vs[j + 1]));
                    if (h(j + 1, j) <= std::max(normAv * 1e-14, verySmallReal))
                    {
                        luckyBreakdown = true;
                        j++;
                        break;
                    }
                    Vs[j + 1] *= 1.0 / h(j + 1, j); // normalize
                }

                if (j < 1)
                    break;
                VectorXR eBeta;
                eBeta.resize(j + 1);
                eBeta.setZero();
                eBeta(0) = beta; // eBeta = e_1 * beta

                MatrixXR y;
                MatrixXR hPart = h(Eigen::seq(0, j), Eigen::seq(0, j - 1));
                DNDS_assert_info(hPart.allFinite(), "GMRES_LeftPreconditioned acquired inf or nan coefficient");
                auto rank = HardEigen::EigenLeastSquareSolve(hPart, eBeta, y);

                for (uint32_t jj = 0; jj < j; jj++) // x = V(:, 0,1,2,...j-1) * y
                    x.addTo(Vs[jj], y(jj));

                if (luckyBreakdown || rank < hPart.cols())
                {
                    // Check true preconditioned residual after lucky-breakdown or rank-deficient update.
                    FA(x, V_temp);
                    FML(V_temp, Vs[0]);
                    Vs[0].addTo(MLb, -1.0);
                    real finalBeta = std::sqrt(fDot(Vs[0], Vs[0]));
                    converged = FStop(iRestart + 1, finalBeta, scale_MLb) || std::abs(finalBeta) < verySmallReal;
                    break;
                }
            }
            return converged;
        }
    };

    template <class TDATA, class TScalar>
    class PCG_PreconditionedRes
    {
        bool initialized = false;
        TDATA r, z, p, Ap;
        TDATA V_temp;

        TScalar zrDot;

        index pHistorySize{0};

    public:
        template <class Finit>
        PCG_PreconditionedRes(
            Finit &&finit = [](TDATA &) {})
        {
            finit(r);
            finit(z);
            finit(p);
            finit(Ap);
            finit(V_temp);
        }

        void reset() { initialized = false, pHistorySize = 0; }

        [[nodiscard]] index getPHistorySize() const { return pHistorySize; }

        template <class TFA, class TFM, class TFResPrec, class TFDot, class TFstop>
        bool solve(TFA &&FA, TFM &&FM, TFResPrec &&FResPrec, TFDot &&fDot, TDATA &x, uint32_t niter, TFstop &&FStop)
        {
            if (!initialized)
            {
                FResPrec(x, z);
                FM(z, r);
                p = z;
                pHistorySize++;
                FA(p, Ap);
            }
            for (int iter = 1; iter <= niter; iter++)
            {
                if (initialized)
                {
                    FResPrec(x, z);
                    FM(z, r);
                }
                TScalar zrDotNew = fDot(z, r);
                TScalar zDotz = fDot(z, z);
                if (FStop(iter, zrDotNew, zDotz))
                    return true;

                if (initialized)
                {
                    TScalar beta = zrDotNew / (zrDot + verySmallReal);
                    p *= beta;
                    p += z;
                    pHistorySize++;
                    FA(p, Ap);
                }

                TScalar alpha = zrDotNew / (fDot(p, Ap) + verySmallReal);
                x.addTo(p, alpha);

                zrDot = zrDotNew;
                initialized = true;
            }
            return false;
        }
    };
}
