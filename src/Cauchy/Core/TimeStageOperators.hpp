#pragma once

#include "Cauchy/Core/TimeStageValues.hpp"
#include "Cauchy/Physics/Positivity.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy
{
    struct PhysicalStepLimit;
    /** @brief Managed explicit-stage arithmetic and physical-step reductions.
     * @details Inputs may have stale ghosts; only owned rows are read/written.
     * The mutable stage is explicitly in-out and distinct from initial/RHS.
     * The same submission path supports inline CPU, CPU workers and CUDA.
     * CUDA supports combined AoS flow5/SA/2EQ. CPU retains segmented fields.
     * Compact numerical status/blend results are observed at an explicit engine
     * drain; MPI agreement and step rejection belong to the controller.
     * No cellwise flow/species increment compression is permitted here.
     */
    class TimeStageOperators
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        TimeStageOperators(CFV::FiniteVolumePlan, Runtime::ExecutionContext);
        /// Rank-local CFL minimum. Caller validates auxiliary freshness and
        /// performs domain MPI min; no scalar is observed before the lane drain.
        real MinimumStableStep(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                               const Runtime::Array<real> &spectralSum, real cfl) const;
        /// Rank-local Euler-compatible thermal dt forecast; does not change
        /// means/residuals or enforce species/RANS bounds on the trial. dt is
        /// returned unchanged along with local alpha/count; the controller
        /// agrees those and applies the factor to the common physical step.
        PhysicalStepLimit LimitPhysicalStep(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                            const EquationDofs &mean, const EquationDofs &residual, AdmissibilityPolicy,
                                            real dt, real relax, real scale, int compressionPower) const;
        /// Pure flow5 high/low residual comparison. No mean is modified.
        ExplicitBlendStatistics MeasureBlend(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                             const EquationDofs &initial, const EquationDofs &stage, const EquationDofs &high,
                                             const EquationDofs &low, ThermalBounds, ExplicitStageCoefficients) const;
        /// Apply the agreed conservative blend. An absent low input requires
        /// alpha=1 and reads only the high residual. No per-cell flow repair.
        void UpdateFlow(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                        const EquationDofs &initial, const EquationDofs &rhs, EquationDofs &stage,
                        OptionalRef<const EquationDofs> low, real alpha, ExplicitStageCoefficients) const;
        /// Raw flow/species SSPRK update, followed only by independent RANS
        /// exponential increment/mean repair using the updated flow density.
        /// Final thermal/species admissibility is a separate acceptance check.
        void UpdateExtended(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                            const EquationDofs &initial, const EquationDofs &rhs, EquationDofs &stage,
                            RANSStateView, ExplicitStageCoefficients) const;
    };
}
