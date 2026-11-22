#pragma once

// Compressible model adapter. CFV owns equation-independent gate
// stencils/penalties; this private module assembles an accepted physical P1 target.
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "CFV/Plans/LimitedVRPlan.hpp"
#include "Cauchy/Discretization/LimitedVREvaluation.hpp"

namespace DNDS::Cauchy
{
    struct FiniteVolume::LimitedVRGeometry
    {
        CFV::LimitedVRPlan gates;
        explicit LimitedVRGeometry(const Data &data);
    };
    struct FiniteVolume::LimitedVRWorkspace
    {
        std::vector<CFV::FVField> targetCoefficients;
        CFV::FVField scalars, branches, seeds, temporary;
        Runtime::Array<real> base, alpha;
        /// CUDA face sensors are gathered into owned cells without atomic
        /// scatter. Status reductions retain the first failure across leaves.
        struct DeviceStorage
        {
            Runtime::Array<real> sensors;
            Runtime::Array<NumericalStatus> ownedStatus, ghostStatus, faceStatus, partialStatus, resultStatus;
        };
        std::optional<DeviceStorage> device;
        std::optional<Runtime::Array<real>> cpuSensors;
        Runtime::Array<NumericalStatus> emptyStatus{Runtime::RowLayout::Dense(0, 1)};
        Runtime::TransferBinding branchHalo, seedHalo, temporaryHalo;
        explicit LimitedVRWorkspace(const Data &data, const EquationDofs &state);
        /// Means have a current halo. Gradients and BC values are queued before
        /// this call on compute; target,alpha are frozen until the next Build. History
        /// mutation is explicit here and must be backed up by the domain driver.
        /// One schedule serves CPU inline/workers and CUDA; backend dispatch
        /// occurs at individual retained-operand wrappers. Return is completed.
        void Build(const Data &data, const EquationDofs &state, const std::vector<CFV::FVField> &gradients,
                   const CFV::FVField &rawFlowGradient, bool newStage,
                   Runtime::Array<real> &previous,
                   std::vector<Runtime::TransferBinding> &gradientHalo,
                   Runtime::EagerEngine &engine,
                   const Runtime::ExecutionStream &compute,
                   const Runtime::ExecutionStream &communication);
    };
}
