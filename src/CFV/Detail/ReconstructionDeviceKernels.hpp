#pragma once
#include "CFV/Detail/ReconstructionDeviceView.hpp"
namespace DNDS::CFV::Detail
{
    bool FP64TensorCoreAvailable(Runtime::Device);
    void LaunchJacobiTensorCore(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                                Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    void LaunchJacobi(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                      Runtime::PairReadView<real>, Runtime::PairReadView<real>, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
    void LaunchWBAP(Runtime::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks,
                    Runtime::PairReadView<real>, Runtime::ArrayView<real>, Runtime::ArrayView<real>, DeviceCharacteristicView,
                    Runtime::ArrayView<const real>, Runtime::ArrayView<int>, index, index, real, index);
    void LaunchCommit(Runtime::ExecutionContext, Runtime::ArrayView<const real>, Runtime::ArrayView<real>, Runtime::ArrayView<const int>);
}
