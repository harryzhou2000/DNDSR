/**
 * @file ResidualCFLDriver.hpp
 * @brief Residual-feedback CFL law for implicit Euler pseudo-time marching.
 */
#pragma once

#include "DNDS/Config/ConfigEnum.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/EigenUtil.hpp"
#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::Euler
{
    /** @brief Selects how the implicit pseudo-time CFL is advanced. */
    enum class ImplicitCFLMode
    {
        Unknown = 0,
        StaticRamp,
        ResidualBased,
    };

    DNDS_DEFINE_ENUM_JSON(
        ImplicitCFLMode,
        {
            {ImplicitCFLMode::Unknown, nullptr},
            {ImplicitCFLMode::StaticRamp, "StaticRamp"},
            {ImplicitCFLMode::ResidualBased, "ResidualBased"},
        })

    /**
     * @brief Configuration of the residual-based CFL law of Bulgarini et al.
     *
     * CFLMin, CFLOrd, alpha, and pMGSafetyFactor correspond to Eq. (12) in
     * J. Comput. Phys. 525 (2025) 113766. CFLMax and the zero-reference policy
     * in ResidualCFLDriver are DNDSR safeguards because the published law is
     * unbounded as the normalized residual tends to zero and leaves zero
     * reference residuals unspecified.
     */
    struct ResidualCFLControl
    {
        real CFLMin = 10;
        real CFLMax = 1e100;
        real CFLOrd = 0;
        real alpha = 1;
        real pMGSafetyFactor = 1;
        bool useVolumeWeightedL2 = false;

        DNDS_DECLARE_CONFIG(ResidualCFLControl)
        {
            // clang-format off
            DNDS_FIELD(CFLMin, "First-iteration/baseline CFL in the residual law",
                       DNDS::Config::range(std::numeric_limits<real>::min()));
            DNDS_FIELD(CFLMax, "Upper safety cap for residual-driven CFL (DNDSR extension)",
                       DNDS::Config::range(std::numeric_limits<real>::min()));
            DNDS_FIELD(CFLOrd, "Divergence-limit CFL; 0 derives 1/(2*p+1) from reconstruction degree",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(alpha, "Residual-law CFL growth exponent",
                       DNDS::Config::range(std::numeric_limits<real>::min()));
            DNDS_FIELD(pMGSafetyFactor, "Multiplicative safety factor n_sf in the pMG level term",
                       DNDS::Config::range(std::numeric_limits<real>::min()));
            DNDS_FIELD(useVolumeWeightedL2, "Volume-weight the physical-residual L2 norm (false follows the paper's algebraic norm)");

            config.check("residual CFL values and exponents must be finite", [](const T &s)
            {
                return std::isfinite(s.CFLMin) && std::isfinite(s.CFLMax) &&
                       std::isfinite(s.CFLOrd) && std::isfinite(s.alpha) &&
                       std::isfinite(s.pMGSafetyFactor);
            });
            config.check("residual CFLMax must be greater than or equal to CFLMin", [](const T &s)
            {
                return s.CFLMax >= s.CFLMin;
            });
            config.check("explicit residual CFLOrd must be less than CFLMin", [](const T &s)
            {
                return s.CFLOrd == 0 || s.CFLOrd < s.CFLMin;
            });
            // clang-format on
        }

        /** @brief Resolve the paper's CFL_ord, optionally from polynomial degree p. */
        [[nodiscard]] real resolvedCFLOrd(int polynomialDegree) const
        {
            DNDS_check_throw_info(polynomialDegree >= 0,
                                  "residual CFL polynomial degree must be non-negative");
            const real resolved = CFLOrd > 0
                                      ? CFLOrd
                                      : real(1) / (real(2) * polynomialDegree + real(1));
            DNDS_check_throw_info(std::isfinite(resolved) && resolved > 0 && resolved < CFLMin,
                                  "residual CFLOrd must resolve to a finite value in (0, CFLMin)");
            return resolved;
        }
    };

    /** @brief Diagnostics returned for one residual-feedback update. */
    struct ResidualCFLUpdate
    {
        real CFL = 0;
        real CFLOrd = 0;
        real xi = 1;
        real xi2 = 1;
        real xiInf = 1;
        real levelFactor = 1;
        bool referenceInitializedThisUpdate = false;
        bool usedLInfBranch = false;
        bool limitedByCFLMax = false;
    };

    /**
     * @brief Stateful implementation of the residual-based CFL law in Eq. (12).
     *
     * The state is only the first-iteration L2 and L-infinity norm of each
     * physical equation. The caller supplies already MPI-global, component-wise
     * norms of the cell-average nonlinear residual. The driver deliberately has
     * no knowledge of reconstruction DOFs, linear residuals, or line searches.
     */
    class ResidualCFLDriver
    {
        Eigen::Vector<real, Eigen::Dynamic> _referenceL2;
        Eigen::Vector<real, Eigen::Dynamic> _referenceLInf;
        bool _hasReference = false;

        static void ValidateNorms(
            const Eigen::Vector<real, Eigen::Dynamic> &residualL2,
            const Eigen::Vector<real, Eigen::Dynamic> &residualLInf)
        {
            DNDS_check_throw_info(residualL2.size() > 0 && residualL2.size() == residualLInf.size(),
                                  "residual CFL norms must be non-empty vectors of equal size");
            DNDS_check_throw_info(residualL2.allFinite() && residualLInf.allFinite(),
                                  "residual CFL norms must be finite");
            DNDS_check_throw_info((residualL2.array() >= 0).all() && (residualLInf.array() >= 0).all(),
                                  "residual CFL norms must be non-negative");
        }

        static real NormalizedRatio(real current, real reference)
        {
            if (reference > 0)
                return current / reference;
            if (current == 0)
                return 0;
            return std::numeric_limits<real>::infinity();
        }

        static real MaximumRatio(
            const Eigen::Vector<real, Eigen::Dynamic> &current,
            const Eigen::Vector<real, Eigen::Dynamic> &reference)
        {
            real ratioMax = 0;
            for (Eigen::Index i = 0; i < current.size(); ++i)
                ratioMax = std::max(ratioMax, NormalizedRatio(current(i), reference(i)));
            return ratioMax;
        }

    public:
        void Reset()
        {
            _referenceL2.resize(0);
            _referenceLInf.resize(0);
            _hasReference = false;
        }

        [[nodiscard]] bool HasReference() const
        {
            return _hasReference;
        }

        /**
         * @brief Capture/update residual feedback and return the CFL for the next iteration.
         *
         * @param residualL2      MPI-global L2 norm of each physical equation.
         * @param residualLInf    MPI-global L-infinity norm of each equation.
         * @param control         Residual CFL parameters.
         * @param polynomialDegree Reconstruction polynomial degree p used for auto CFLOrd.
         * @param maximumPMGLevel Finest pMG polynomial level.
         * @param currentPMGLevel Current pMG polynomial level; equal to maximum for single grid.
         */
        [[nodiscard]] ResidualCFLUpdate Update(
            const Eigen::Vector<real, Eigen::Dynamic> &residualL2,
            const Eigen::Vector<real, Eigen::Dynamic> &residualLInf,
            const ResidualCFLControl &control,
            int polynomialDegree,
            int maximumPMGLevel = 0,
            int currentPMGLevel = 0)
        {
            ValidateNorms(residualL2, residualLInf);
            DNDS_check_throw_info(std::isfinite(control.CFLMin) && control.CFLMin > 0,
                                  "residual CFLMin must be finite and positive");
            DNDS_check_throw_info(std::isfinite(control.CFLMax) && control.CFLMax >= control.CFLMin,
                                  "residual CFLMax must be finite and no smaller than CFLMin");
            DNDS_check_throw_info(std::isfinite(control.alpha) && control.alpha > 0,
                                  "residual CFL alpha must be finite and positive");
            DNDS_check_throw_info(std::isfinite(control.pMGSafetyFactor) && control.pMGSafetyFactor > 0,
                                  "residual CFL pMG safety factor must be finite and positive");
            DNDS_check_throw_info(maximumPMGLevel >= 0 && currentPMGLevel >= 0 &&
                                      currentPMGLevel <= maximumPMGLevel,
                                  "residual CFL pMG levels must satisfy 0 <= current <= maximum");

            ResidualCFLUpdate result;
            result.CFLOrd = control.resolvedCFLOrd(polynomialDegree);
            result.levelFactor = (real(maximumPMGLevel) - real(currentPMGLevel) + real(1)) *
                                 control.pMGSafetyFactor;
            DNDS_check_throw_info(std::isfinite(result.levelFactor) && result.levelFactor > 0,
                                  "residual CFL pMG level factor must be finite and positive");

            if (!_hasReference)
            {
                _referenceL2 = residualL2;
                _referenceLInf = residualLInf;
                _hasReference = true;
                result.CFL = control.CFLMin;
                result.referenceInitializedThisUpdate = true;
                return result;
            }

            DNDS_check_throw_info(
                residualL2.size() == _referenceL2.size() &&
                    residualLInf.size() == _referenceLInf.size(),
                "residual CFL norm vector size changed after reference initialization");

            result.xi2 = MaximumRatio(residualL2, _referenceL2);
            result.xiInf = MaximumRatio(residualLInf, _referenceLInf);
            result.usedLInfBranch = result.xiInf > 1;
            result.xi = result.usedLInfBranch
                            ? result.xiInf
                            : std::min(real(1), result.xi2);

            if (result.xi <= 1)
            {
                if (result.xi == 0)
                {
                    result.CFL = control.CFLMax;
                    result.limitedByCFLMax = true;
                    return result;
                }

                const real logCFL = std::log(control.CFLMin) - control.alpha * std::log(result.xi);
                const real logCFLMax = std::log(control.CFLMax);
                if (logCFL >= logCFLMax)
                {
                    result.CFL = control.CFLMax;
                    result.limitedByCFLMax = logCFL > logCFLMax;
                }
                else
                    result.CFL = std::exp(logCFL);
                return result;
            }

            const real exponent = std::isinf(result.xi)
                                      ? -std::numeric_limits<real>::infinity()
                                      : control.alpha * (real(1) - result.xi) *
                                            result.levelFactor * control.CFLMin /
                                            (control.CFLMin - result.CFLOrd);
            const real phi = std::exp(exponent);
            result.CFL = result.CFLOrd + phi * (control.CFLMin - result.CFLOrd);
            result.CFL = std::clamp(result.CFL, result.CFLOrd, control.CFLMax);
            return result;
        }
    };
}
