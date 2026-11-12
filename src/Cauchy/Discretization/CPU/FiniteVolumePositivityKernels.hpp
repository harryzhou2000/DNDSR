#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::CPU
{
    using PPInput = Algebra::DofComponentView<const real>;
    using PPOutput = Algebra::DofComponentView<real>;
    using PPIndexInput = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPIndexOutput = Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPScalarInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPFlagInput = Runtime::ArrayView<const uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPScalarOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using PPFlagOutput = Runtime::ArrayView<uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    struct PolynomialPositivityRequest
    {
        PPInput means;
        PPIndexInput offsets;
        ThermalView bounds;
        RANSReconstructionView rans;
        index ransFirst = -1;
        int modes = 0, maxOrder = 0;
        real betaPower = 1, betaCutoff = 0;
        ReconstructionClippedReferencePolicy clippedPolicy = ReconstructionClippedReferencePolicy::FallbackToBase;
        std::optional<ThermalBounds> segmentBounds;
    };
    NumericalStatus ValidateReconstructionMeans(const PolynomialPositivityRequest &) noexcept;
    NumericalResult<bool> MarkAdmissibleDegrees(const PolynomialPositivityRequest &, PPInput samples, PPIndexOutput degrees) noexcept;
    void TruncateReferenceModes(const PolynomialPositivityRequest &, PPIndexInput degrees, PPOutput coefficients, int keep);
    void SelectReferenceDegree(const PolynomialPositivityRequest &, PPInput samples, PPInput reference,
                               PPIndexOutput degrees, PPOutput coefficients, int order);
    NumericalStatus CompressPolynomial(const PolynomialPositivityRequest &, PPInput higher, std::optional<PPInput> base,
                                       std::optional<PPInput> reference, PPIndexInput degrees, PPOutput coefficients, PPScalarOutput beta) noexcept;
    NumericalResult<bool> MarkPolynomialRecovery(const PolynomialPositivityRequest &, PPInput samples, PPScalarOutput beta,
                                                 PPFlagOutput recover, PPScalarOutput recoveryScale) noexcept;
    void RecoverPolynomial(const PolynomialPositivityRequest &, std::optional<PPInput> reference,
                           PPOutput coefficients, PPFlagInput recover, PPScalarInput recoveryScale);
    NumericalStatus VerifyReconstructedPoints(const PolynomialPositivityRequest &, PPInput samples) noexcept;
    std::pair<index, real> PolynomialPositivityStatistics(PPScalarInput beta);
    /// Read-only whole-equation physical-step forecast. The caller reduces
    /// these statistics and chooses the common dt; no state repair occurs.
    NumericalResult<std::pair<index, real>> LimitPhysicalStepRows(PPInput mean, PPInput rhs, CFV::FiniteVolumeHostView metric,
                                                                  ThermalView thermal, real dt, real relax, real scale, int compressionPower) noexcept;
}
