#pragma once

#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy::CUDA
{
    using PPInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using PPOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using PPIndexInput = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPIndexOutput = Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPScalarInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPScalarOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPFlagInput = Runtime::ArrayView<const uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPFlagOutput = Runtime::ArrayView<uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /** Perfect-gas modal PP policy; no mixture metadata or EOS callbacks.
     * PP POLICY -- flow5 is compressed as a vector, RANS separately as scalars.
     * These operators limit reconstruction, never explicit solution increments.
     * The raw-affine construction and final clipped-reference checks are kept
     * distinct, even though their predicates coincide without species. Actual
     * post-storage samples must still pass VerifyReconstructedPoints.
     */
    struct PolynomialPositivityParameters
    {
        ThermalView bounds{};
        RANSReconstructionView rans;
        int components = 5, modes = 0, maxOrder = 0, betaPower = 1;
        real betaCutoff = 0;
        ReconstructionClippedReferencePolicy clippedPolicy = ReconstructionClippedReferencePolicy::FallbackToBase;
        bool segment = false;
        ThermalBounds segmentBounds{};
        void Validate() const;
    };
    struct PolynomialPositivityRequest
    {
        PPInput means;
        PPIndexInput offsets;
        PPStatus status;
        PolynomialPositivityParameters policy;
        void Validate(Runtime::ExecutionContext context) const;
    };
    /// Numerical status is per owned cell. No leaf reduces or synchronizes to
    /// the host. The operator may queue flag/statistic reductions and inspect
    /// them only at an explicit controller decision/drain point.
    void ValidateReconstructionMeans(Runtime::ExecutionContext, PolynomialPositivityRequest);
    void MarkAdmissibleDegrees(Runtime::ExecutionContext, PolynomialPositivityRequest,
                               PPInput samples, PPIndexOutput degrees, PPFlagOutput good);
    void TruncateReferenceModes(Runtime::ExecutionContext, PolynomialPositivityRequest,
                                PPIndexInput degrees, PPOutput coefficients, int keep);
    void SelectReferenceDegree(Runtime::ExecutionContext, PolynomialPositivityRequest,
                               PPInput samples, PPInput reference, PPIndexOutput degrees, PPOutput coefficients, int order);
    /// Optional base samples/reference coefficients are controlled explicitly;
    /// absent operands are empty views and are neither validated nor read.
    void CompressPolynomial(Runtime::ExecutionContext, PolynomialPositivityRequest,
                            PPInput higher, PPInput base, PPInput reference, bool hasBase, bool hasReference,
                            PPIndexInput degrees, PPOutput coefficients, PPScalarOutput beta);
    void MarkPolynomialRecovery(Runtime::ExecutionContext, PolynomialPositivityRequest,
                                PPInput samples, PPScalarOutput beta, PPFlagOutput recover, PPScalarOutput recoveryScale);
    void RecoverPolynomial(Runtime::ExecutionContext, PolynomialPositivityRequest,
                           PPInput reference, bool hasReference, PPOutput coefficients,
                           PPFlagInput recover, PPScalarInput recoveryScale);
    void VerifyReconstructedPoints(Runtime::ExecutionContext, PolynomialPositivityRequest, PPInput samples);

    /** Read-only physical-step forecast; produces alpha per owned cell.
     * PP POLICY -- the complete coupled RHS trial enters the existing thermal
     * estimator. RANS bounds do not restrict alpha. This is a common-dt bound,
     * never a per-cell alteration of an explicit update. Actual explicit-step
     * acceptance/retry remains a separate controller operation.
     */
    void LimitPhysicalStepRows(Runtime::ExecutionContext, PPInput means, PPInput rhs,
                               Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells,
                               ThermalView thermal, real dt, real relax, real scale, int compressionPower,
                               PPScalarOutput alpha, PPStatus status);
}
