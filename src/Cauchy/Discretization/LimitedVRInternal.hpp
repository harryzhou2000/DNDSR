#pragma once

// Initial compressible CPU model adapter. CFV owns equation-independent gate
// stencils/penalties; this private module supplies physical sampling and O2 PP.
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "CFV/Plans/LimitedVRPlan.hpp"

namespace DNDS::Cauchy
{
    struct FiniteVolume::LimitedVRGeometry
    {
        struct Face
        {
            index cell, other;
            Geom::tPoint normal;
            Geom::tGPoint rotation;
        };
        CFV::GradientPlan gradient;
        CFV::LimitedVRPlan gates;
        std::vector<Face> faces;
        std::optional<CFV::PolynomialSamplingPlan> sampling;
        explicit LimitedVRGeometry(const Data &data);
    };
    struct FiniteVolume::LimitedVRWorkspace
    {
        CFV::ReconstructionWorkspace target;
        CFV::FVField gradients, scalars, branches, seeds, temporary;
        Runtime::Array<real> previous, base, alpha, sampledFaces, sampledPP;
        Runtime::TransferBinding targetHalo, branchHalo, seedHalo;
        explicit LimitedVRWorkspace(const Data &data, const CFV::ReconstructionWorkspace &prototype);
        /// Means are host-visible/current. Gradients and BC values were prepared
        /// for this input; target,alpha are frozen until the next Build. History
        /// mutation is explicit here and must be backed up by the domain driver.
        void Build(const Data &data, const CFV::FVField &mean,
                   CFV::FaceCharacteristicTransformRef transform,
                   Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream);
    };
}
