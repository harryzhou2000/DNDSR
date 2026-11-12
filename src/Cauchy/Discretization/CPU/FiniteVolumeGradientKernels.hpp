#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::CPU
{
    using GradientReadView = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using GradientWriteView = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    struct GradientLimiterRequest
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        Algebra::DofComponentView<const real> means;
        SpanView<const GradientReadView> source;
        SpanView<const GradientWriteView> destination;
        SpanView<const index> logicalField, logicalComponent, vectorStarts;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> scratch;
        index components = 0;
        int dimension = 0;
        bool barth = false;
    };
    /// Componentwise Barth limiting of physical gradients, with periodic vector
    /// and spatial-frame transforms. Read inputs and write outputs are distinct.
    void LimitPhysicalGradients(const GradientLimiterRequest &request);
    struct GradientPositivityRequest
    {
        FiniteVolumeStencilView stencil;
        Algebra::DofComponentView<const real> means;
        SpanView<const GradientWriteView> gradients;
        SpanView<const index> logicalField, logicalComponent;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> samples;
        EquationThermalView bounds;
        RANSReconstructionView rans;
        index components = 0, ransFirst = -1;
        int dimension = 0;
        real betaPower = 1, betaCutoff = 0;
        ReconstructionClippedReferencePolicy clippedPolicy = ReconstructionClippedReferencePolicy::FallbackToBase;
    };
    struct GradientPositivityResult
    {
        index limited = 0;
        real minimum = 1;
    };
    /// Cauchy PP policy: locked flow/species beta, separate RANS repair and final
    /// clipped-composition EOS/T verification. Raw species gradients are retained.
    NumericalResult<GradientPositivityResult> LimitPositivePhysicalGradients(const GradientPositivityRequest &request) noexcept;
    /// Pure perfect-gas mean conversion; caller has frozen physics selection.
    void FillPrimitiveMeans(GradientReadView input, GradientWriteView output, const PerfectGas &gas);
}
