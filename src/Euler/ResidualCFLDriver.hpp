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
#include <map>
#include <string>
#include <vector>

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
    struct ResidualCFLLevelControl
    {
        real CFLMin = 10;
        real CFLMax = 1e100;
        real CFLOrd = 0;
        real alpha = 1;
        real pMGSafetyFactor = 1;
        std::vector<int> equationIndices;

        DNDS_DECLARE_CONFIG(ResidualCFLLevelControl)
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
            DNDS_FIELD(equationIndices, "Equation indices used in residual-ratio maxima; empty selects all equations");

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

    /** @brief Fine-grid residual-CFL settings plus independent coarse-level laws. */
    struct ResidualCFLControl : ResidualCFLLevelControl
    {
        bool useVolumeWeightedL2 = false;
        std::map<std::string, ResidualCFLLevelControl> coarseGridControlList{
            {"1", ResidualCFLLevelControl{}},
            {"2", ResidualCFLLevelControl{}},
        };

        DNDS_DECLARE_CONFIG(ResidualCFLControl)
        {
            // Base-class fields are flattened into the residualCFLDriver object.
            // clang-format off
            config.field(static_cast<real T::*>(&T::CFLMin), "CFLMin",
                         "First-iteration/baseline CFL in the residual law",
                         DNDS::Config::range(std::numeric_limits<real>::min()));
            config.field(static_cast<real T::*>(&T::CFLMax), "CFLMax",
                         "Upper safety cap for residual-driven CFL (DNDSR extension)",
                         DNDS::Config::range(std::numeric_limits<real>::min()));
            config.field(static_cast<real T::*>(&T::CFLOrd), "CFLOrd",
                         "Divergence-limit CFL; 0 derives 1/(2*p+1) from reconstruction degree",
                         DNDS::Config::range(0.0));
            config.field(static_cast<real T::*>(&T::alpha), "alpha",
                         "Residual-law CFL growth exponent",
                         DNDS::Config::range(std::numeric_limits<real>::min()));
            config.field(static_cast<real T::*>(&T::pMGSafetyFactor), "pMGSafetyFactor",
                         "Multiplicative safety factor n_sf in the pMG level term",
                         DNDS::Config::range(std::numeric_limits<real>::min()));
            config.field(static_cast<std::vector<int> T::*>(&T::equationIndices), "equationIndices",
                         "Equation indices used in residual-ratio maxima; empty selects all equations");
            DNDS_FIELD(useVolumeWeightedL2, "Volume-weight the physical-residual L2 norm (false follows the paper's algebraic norm)");
            config.template field_map_of<ResidualCFLLevelControl>(
                &T::coarseGridControlList,
                "coarseGridControlList",
                "Independent residual-CFL law for each coarse pMG level");

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

        [[nodiscard]] static int polynomialDegreeForLevel(int fineDegree, int pMGLevel)
        {
            DNDS_check_throw_info(fineDegree >= 0, "fine-grid polynomial degree must be non-negative");
            DNDS_check_throw_info(pMGLevel >= 0 && pMGLevel <= 2,
                                  "Euler pMG level must be 0, 1, or 2");
            if (pMGLevel == 0)
                return fineDegree;
            return pMGLevel == 1 ? 1 : 0;
        }

        [[nodiscard]] const ResidualCFLLevelControl &controlForLevel(int pMGLevel) const
        {
            if (pMGLevel == 0)
                return *this;
            return coarseGridControlList.at(std::to_string(pMGLevel));
        }
    };

    /** @brief Diagnostics returned for one residual-feedback update. */
    struct ResidualCFLUpdate
    {
        real CFL = 0;
        real CFLMaxEffective = 0;
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
     * The state is the first-iteration and latest fine-iteration L2 and
     * L-infinity norm of each physical equation. Every pMG level evaluates that
     * shared sample with its own control law. The caller supplies already
     * MPI-global, component-wise norms of the cell-average nonlinear residual.
     * The driver deliberately has no knowledge of reconstruction DOFs, linear
     * residuals, or line searches.
     */
    class ResidualCFLDriver
    {
        Eigen::Vector<real, Eigen::Dynamic> _referenceL2;
        Eigen::Vector<real, Eigen::Dynamic> _referenceLInf;
        Eigen::Vector<real, Eigen::Dynamic> _currentL2;
        Eigen::Vector<real, Eigen::Dynamic> _currentLInf;
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

        static void ValidateEquationIndices(
            const std::vector<int> &equationIndices,
            Eigen::Index equationCount)
        {
            for (int equationIndex : equationIndices)
                DNDS_check_throw_info(
                    equationIndex >= 0 && equationIndex < equationCount,
                    "residual CFL equation index must be within the residual norm vector");
        }

        static real MaximumRatio(
            const Eigen::Vector<real, Eigen::Dynamic> &current,
            const Eigen::Vector<real, Eigen::Dynamic> &reference,
            const std::vector<int> &equationIndices)
        {
            real ratioMax = 0;
            if (equationIndices.empty())
            {
                for (Eigen::Index i = 0; i < current.size(); ++i)
                    ratioMax = std::max(ratioMax, NormalizedRatio(current(i), reference(i)));
            }
            else
            {
                for (int equationIndex : equationIndices)
                {
                    const Eigen::Index i = equationIndex;
                    ratioMax = std::max(ratioMax, NormalizedRatio(current(i), reference(i)));
                }
            }
            return ratioMax;
        }

        static void ValidateControl(
            const ResidualCFLLevelControl &control,
            int finestPolynomialDegree,
            int currentPolynomialDegree,
            real CFLFactor)
        {
            DNDS_check_throw_info(std::isfinite(control.CFLMin) && control.CFLMin > 0,
                                  "residual CFLMin must be finite and positive");
            DNDS_check_throw_info(std::isfinite(control.CFLMax) && control.CFLMax >= control.CFLMin,
                                  "residual CFLMax must be finite and no smaller than CFLMin");
            DNDS_check_throw_info(std::isfinite(control.alpha) && control.alpha > 0,
                                  "residual CFL alpha must be finite and positive");
            DNDS_check_throw_info(std::isfinite(control.pMGSafetyFactor) && control.pMGSafetyFactor > 0,
                                  "residual CFL pMG safety factor must be finite and positive");
            DNDS_check_throw_info(finestPolynomialDegree >= 0 && currentPolynomialDegree >= 0 &&
                                      currentPolynomialDegree <= finestPolynomialDegree,
                                  "residual CFL pMG degrees must satisfy 0 <= current <= finest");
            DNDS_check_throw_info(std::isfinite(CFLFactor) && CFLFactor >= 0,
                                  "residual CFL external level factor must be finite and non-negative");
        }

    public:
        void Reset()
        {
            _referenceL2.resize(0);
            _referenceLInf.resize(0);
            _currentL2.resize(0);
            _currentLInf.resize(0);
            _hasReference = false;
        }

        /** @brief Evaluate one pMG level from the latest fine-iteration feedback. */
        [[nodiscard]] ResidualCFLUpdate EvaluateCurrent(
            const ResidualCFLLevelControl &control,
            int polynomialDegree,
            int finestPolynomialDegree = 0,
            int currentPolynomialDegree = 0,
            real CFLFactor = 0) const
        {
            ValidateControl(control, finestPolynomialDegree, currentPolynomialDegree, CFLFactor);

            ResidualCFLUpdate result;
            result.CFLOrd = control.resolvedCFLOrd(polynomialDegree);
            const real resolvedCFLFactor = CFLFactor > 0
                                               ? CFLFactor
                                               : real(finestPolynomialDegree) - real(currentPolynomialDegree) + real(1);
            result.levelFactor = resolvedCFLFactor * control.pMGSafetyFactor;
            result.CFLMaxEffective = resolvedCFLFactor * control.CFLMax;
            DNDS_check_throw_info(std::isfinite(result.levelFactor) && result.levelFactor > 0,
                                  "residual CFL pMG level factor must be finite and positive");
            DNDS_check_throw_info(std::isfinite(result.CFLMaxEffective) &&
                                      result.CFLMaxEffective >= control.CFLMin,
                                  "factor-scaled residual CFLMax must be finite and no smaller than CFLMin");

            if (!_hasReference)
            {
                result.CFL = control.CFLMin;
                return result;
            }

            ValidateEquationIndices(control.equationIndices, _currentL2.size());
            result.xi2 = MaximumRatio(_currentL2, _referenceL2, control.equationIndices);
            result.xiInf = MaximumRatio(_currentLInf, _referenceLInf, control.equationIndices);
            result.usedLInfBranch = result.xiInf > 1;
            result.xi = result.usedLInfBranch
                            ? result.xiInf
                            : std::min(real(1), result.xi2);

            if (result.xi <= 1)
            {
                if (result.xi == 0)
                {
                    result.CFL = result.CFLMaxEffective;
                    result.limitedByCFLMax = true;
                    return result;
                }

                const real logCFL = std::log(control.CFLMin) - control.alpha * std::log(result.xi);
                const real logCFLMax = std::log(result.CFLMaxEffective);
                if (logCFL >= logCFLMax)
                {
                    result.CFL = result.CFLMaxEffective;
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
            result.CFL = std::clamp(result.CFL, result.CFLOrd, result.CFLMaxEffective);
            return result;
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
         * @param finestPolynomialDegree Finest-grid polynomial degree.
         * @param currentPolynomialDegree Current pMG polynomial degree; equal to finest on a single grid.
         * @param CFLFactor External level factor; 0 retains the literature degree-gap factor.
         */
        [[nodiscard]] ResidualCFLUpdate Update(
            const Eigen::Vector<real, Eigen::Dynamic> &residualL2,
            const Eigen::Vector<real, Eigen::Dynamic> &residualLInf,
            const ResidualCFLLevelControl &control,
            int polynomialDegree,
            int finestPolynomialDegree = 0,
            int currentPolynomialDegree = 0,
            real CFLFactor = 0)
        {
            ValidateNorms(residualL2, residualLInf);
            ValidateEquationIndices(control.equationIndices, residualL2.size());
            ValidateControl(control, finestPolynomialDegree, currentPolynomialDegree, CFLFactor);

            bool referenceInitialized = false;
            if (!_hasReference)
            {
                _referenceL2 = residualL2;
                _referenceLInf = residualLInf;
                _hasReference = true;
                referenceInitialized = true;
            }
            else
            {
                DNDS_check_throw_info(
                    residualL2.size() == _referenceL2.size() &&
                        residualLInf.size() == _referenceLInf.size(),
                    "residual CFL norm vector size changed after reference initialization");
            }

            _currentL2 = residualL2;
            _currentLInf = residualLInf;

            auto result = EvaluateCurrent(
                control, polynomialDegree, finestPolynomialDegree, currentPolynomialDegree, CFLFactor);
            result.referenceInitializedThisUpdate = referenceInitialized;
            return result;
        }
    };
}
