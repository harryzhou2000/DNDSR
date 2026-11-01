#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Detail/ReconstructionDeviceView.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Geom
{
    class UnstructuredMeshSnapshot;
}
namespace DNDS::CFV
{
    namespace Detail
    {
        struct ReconstructionPlacementData;
    }
    struct ReconstructionResidencyOptions
    {
        bool mesh = false, metrics = false, basis = false;
    };
    struct ResidentBufferInfo
    {
        std::string name;
        size_t bytes = 0;
        Runtime::Device device = Runtime::Device::Host();
    };
    struct ReconstructionDeviceWorkspace
    {
        FVField coefficients, temporary;
        Runtime::Array<real> scratch;
        Runtime::Array<index> periodicVectorStarts;
        Runtime::Array<int> status;
        index components = 0, blocks = 0;
        uint64_t periodicIdentity = 0, periodicVersion = 0;
        std::shared_ptr<const Detail::ReconstructionPlacementData> plan;
    };
    struct DeviceFaceCharacteristicTransform
    {
        Runtime::Array<real> forward, inverse;
        index first = 0, count = 0;
        uint64_t forwardVersion = 0, inverseVersion = 0, forwardIdentity = 0, inverseIdentity = 0;
        std::shared_ptr<const Detail::ReconstructionPlacementData> plan;
    };
    /// A retained read scope also pins source mesh generations. Its plain view
    /// is borrowed and valid only through the scope's lifetime. Scheduled
    /// wrappers retain this scope through actual GPU completion.
    class ReconstructionDeviceRead
    {
        std::shared_ptr<const Detail::ReconstructionPlacementData> _data;
        std::shared_ptr<FiniteVolumeRead> _geometry;
        explicit ReconstructionDeviceRead(std::shared_ptr<const Detail::ReconstructionPlacementData>);
        friend class ReconstructionDevicePlan;

    public:
        Detail::ReconstructionDeviceView DeviceView() const;
        FiniteVolumeHostView MetricView() const;
        ReconstructionBasisHostView BasisView() const;
    };
    /// Immutable CPU-prepared coefficients published on one CUDA device. Setup
    /// is synchronous; hot submissions enqueue without a host wait. Native CPU
    /// operations remain on VariationalReconstructionPlan. Backend resolution
    /// is explicit: this baseline rejects SOR and unavailable kernels.
    class ReconstructionDevicePlan
    {
        std::shared_ptr<const Detail::ReconstructionPlacementData> _data;
        explicit ReconstructionDevicePlan(std::shared_ptr<const Detail::ReconstructionPlacementData> data) : _data(std::move(data)) {}

    public:
        static ReconstructionDevicePlan Prepare(const VariationalReconstructionPlan &, Runtime::Device,
                                                Runtime::ExecutionContext, ReconstructionResidencyOptions = {});
        Runtime::Device Device() const;
        void Validate() const;
        ReconstructionDeviceRead AcquireRead() const;
        std::vector<ResidentBufferInfo> Buffers() const;
        const Geom::UnstructuredMeshSnapshot &MeshSnapshot() const;
        FVField BuildField(Runtime::RowLayout prototype) const;
        ReconstructionDeviceWorkspace BuildWorkspace(index components, Runtime::RowLayout prototype,
                                                     std::vector<index> periodicVectorStarts = {}) const;
        Runtime::TransferBinding BindHalo(FVField &, Runtime::TransferTransport transport = Runtime::TransferTransport::Direct) const;
        ReconstructionBoundaryValues PrepareBoundary(const ReconstructionBoundaryValues &) const;
        DeviceFaceCharacteristicTransform PrepareTransform(const FaceCharacteristicTransform &) const;
        /// Caller orders conflicting cross-stream accesses and input readiness.
        /// Core receives no ghost operand. Complete halo reception before
        /// submitting dependent work; output is work.temporary, not the input.
        void DoReconstructionIterTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                    const FVField &, const FVField &, ReconstructionDeviceWorkspace &,
                                    const ReconstructionBoundaryValues &, ReconstructionRegion = ReconstructionRegion::All) const;
        void CommitReconstruction(Runtime::EagerEngine &, const Runtime::ExecutionStream &, ReconstructionDeviceWorkspace &) const;
        /// Reusable per-block scratch has bounded capacity. Different streams
        /// sharing a workspace require explicit joins. Exchange coefficients
        /// between descending orders. Observe device status at the boundary.
        void DoLimiterWBAP_3OrderTo(Runtime::EagerEngine &, const Runtime::ExecutionStream &, ReconstructionDeviceWorkspace &, int order,
                                    const DeviceFaceCharacteristicTransform * = nullptr, ReconstructionRegion = ReconstructionRegion::All,
                                    const Runtime::Array<real> *indicator = nullptr, real threshold = 0) const;
        void DoReconstructionIter(const FVField &, ReconstructionDeviceWorkspace &, const ReconstructionBoundaryValues &) const;
        void DoLimiterWBAP_3Order(ReconstructionDeviceWorkspace &, int order, const DeviceFaceCharacteristicTransform * = nullptr) const;
        /// Finish the producing engine prefix before observing status. Invalid
        /// indicators prevent every output update and throw here. This explicit
        /// observation is not a per-kernel completion token.
        void CheckStatus(const ReconstructionDeviceWorkspace &) const;
    };
}
