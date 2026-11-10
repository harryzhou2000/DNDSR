#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include <set>

namespace DNDS::CFV
{
    namespace R = Runtime;
    namespace Detail
    {
        struct ReconstructionPlacementData
        {
            VariationalReconstructionPlan source;
            R::Device device;
            ReconstructionResidencyOptions options;
            bool fp64TensorCore = false;
            std::optional<Algebra::DofSpace> dofs;
            std::vector<ResidentBufferInfo> buffers;
            std::vector<R::Array<real>> realArrays;
            std::vector<R::Array<index>> indexArrays;
            // Immutable device read leases are retained with the publication;
            // scheduled commands retain the publication through completion.
            std::vector<R::Array<real>::ReadLease> realReads;
            std::vector<R::Array<index>::ReadLease> indexReads;
            R::Array<real> empty;
            ReconstructionDeviceView view;
            FiniteVolumeHostView metricView;
            ReconstructionBasisHostView basisView;
            std::shared_ptr<Geom::UnstructuredMeshSnapshot> mesh;
            ReconstructionPlacementData(VariationalReconstructionPlan plan, R::Device placement, ReconstructionResidencyOptions requested)
                : source(std::move(plan)), device(placement), options(requested), empty(R::RowLayout::Dense(0, 0), placement) {}
            template <class T>
            R::ArrayView<const T> Add(const char *name, const R::Array<T> &input, R::ExecutionContext context)
            {
                auto array = input.To(device, context);
                buffers.push_back({name, array.Storage().Bytes(), device});
                if constexpr (std::is_same_v<T, real>)
                {
                    realArrays.push_back(std::move(array));
                    realReads.push_back(realArrays.back().Read({device, nullptr}));
                    return realReads.back().View();
                }
                else
                {
                    indexArrays.push_back(std::move(array));
                    indexReads.push_back(indexArrays.back().Read({device, nullptr}));
                    return indexReads.back().View();
                }
            }
        };
    }
    namespace
    {
        void ValidateDeviceField(const VariationalReconstructionPlan &plan, const FVField &field, index width)
        {
            field.Validate();
            DNDS_check_throw_info(field.Device() == plan.Device() && index(field.Width()) == width, "device FV field width/placement mismatch");
        }
        void ValidateDeviceWorkspace(const std::shared_ptr<const Detail::ReconstructionPlacementData> &data, const ReconstructionWorkspace &work)
        {
            const auto n = data->view.n;
            DNDS_check_throw_info(work.devicePlan == data && work.components > 0 && work.components <= INT_MAX / n &&
                                      work.deviceBlocks > 0 && work.deviceBlocks <= 256,
                                  "device workspace belongs to another publication or has an invalid shape");
            work.coefficients.Validate();
            work.temporary.Validate();
            DNDS_check_throw_info(work.coefficients.Device() == data->device && work.temporary.Device() == data->device &&
                                      work.coefficients.Width() == n * work.components && work.temporary.Width() == n * work.components &&
                                      work.coefficients.Space().Same(work.temporary.Space()) && work.coefficients.Space().Same(*data->dofs),
                                  "device workspace fields invalid");
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(work.coefficients.Owned().Storage(), work.temporary.Owned().Storage()) &&
                                      work.scratch.GetDevice() == data->device && work.scratch.Layout().Kind() == R::RowLayoutKind::Table && work.scratch.RowCount() == work.deviceBlocks * 6 &&
                                      work.scratch.Layout().Width() == n * work.components && work.deviceStatus.GetDevice() == data->device && work.deviceStatus.Layout().Kind() == R::RowLayoutKind::Table &&
                                      work.deviceStatus.RowCount() == 1 && work.deviceStatus.Layout().Width() == 1 &&
                                      work.devicePeriodicVectorStarts.Identity() == work.devicePeriodicIdentity &&
                                      work.devicePeriodicVectorStarts.Version() == work.devicePeriodicVersion &&
                                      work.devicePeriodicVectorStarts.GetDevice() == data->device &&
                                      work.devicePeriodicVectorStarts.Layout().Kind() == R::RowLayoutKind::Table &&
                                      work.devicePeriodicVectorStarts.Layout().Width() == 1,
                                  "device workspace output/scratch/status mismatch");
        }
        Detail::DeviceWorkSelection Selection(const Detail::ReconstructionPlacementData &data, ReconstructionRegion region)
        {
            if (region == ReconstructionRegion::All)
                return {{}, data.view.owned, false};
            DNDS_check_throw_info(region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent, "invalid device work region");
            const auto rows = region == ReconstructionRegion::Core ? data.view.core : data.view.dependent;
            return {rows, rows.rows, true};
        }
        void ValidateDeviceStream(const VariationalReconstructionPlan &plan, const R::ExecutionStream &stream)
        {
            plan.Validate();
            DNDS_check_throw_info(stream.Context().device == plan.Device(), "device operator stream placement mismatch");
        }
        void ValidateArray(const R::Array<real> &array, R::Device device, index rows, index width)
        {
            DNDS_check_throw_info(array.GetDevice() == device && array.IsValid() && !array.Layout().IsVariable() && array.RowCount() == rows &&
                                      array.Layout().Width() == width,
                                  "device operator array shape/placement mismatch");
        }
    }
    VariationalReconstructionPlan VariationalReconstructionPlan::Place(R::Device device, R::ExecutionContext context,
                                                                       ReconstructionResidencyOptions options) const
    {
        if (device == R::Device::Host())
        {
            DNDS_check_throw_info(context.device == device, "host reconstruction placement/context differs");
            Validate();
            return *this;
        }
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(false, "CUDA reconstruction is unavailable in this build");
#endif
        Require(ReconstructionOperation::Sweep);
        Require(ReconstructionOperation::WBAP);
        Validate();
        DNDS_check_throw_info(device.space == R::MemorySpace::CUDA && context.device == device, "device reconstruction requires explicit CUDA placement/context");
        DNDS_check_throw_info(!Settings().SORInstead, "CUDA SOR requires a logical parallel sweep design and is unsupported");
        _data->workPartition->Validate();
        auto geometry = Metrics().AcquireRead();
        auto result = std::make_shared<Detail::ReconstructionPlacementData>(*this, device, options);
        result->fp64TensorCore = Detail::FP64TensorCoreAvailable(device);
        const auto &host = *_data;
        auto &view = result->view;
        view.owned = Metrics().NumCell();
        view.n = NRecDOF();
        view.relax = Settings().jacobiRelax;
        view.wbapCenterWeight = Settings().WBAP_nStd;
        view.rotated = host.hasRotatedPeriodicity;
#define DNDS_REC_ADD(member) view.member = result->Add(#member, host.member, context)
        DNDS_REC_ADD(offsets);
        DNDS_REC_ADD(neighbors);
        DNDS_REC_ADD(faces);
        DNDS_REC_ADD(boundaryOffsets);
        view.matrixAInvB = result->Add("matrixAInvB", *host.matrixAInvB, context);
        view.vectorAInvB = result->Add("vectorAInvB", *host.vectorAInvB, context);
        view.secondary = result->Add("secondary", *host.secondary, context);
        view.boundaryInjection = result->Add("boundaryInjection", *host.boundaryInjection, context);
#undef DNDS_REC_ADD
        view.rotation = result->Add("neighborRotation", host.neighborRotation, context);
        for (bool core : {true, false})
        {
            const auto &list = core ? host.workPartition->Core() : host.workPartition->HaloDependent();
            R::Array<index> rows(R::RowLayout::Dense(list.size(), 1));
            {
                auto write = rows.Write();
                for (index i = 0; i < index(list.size()); ++i)
                    write.View()(i, 0) = list[size_t(i)];
            }
            const auto selected = result->Add(core ? "core" : "dependent", rows, context);
            if (core)
                view.core = selected;
            else
                view.dependent = selected;
        }
        const auto &metric = *Metrics()._data;
        result->dofs = metric.dofs;
        if (options.metrics || options.basis)
        {
            const auto cell = result->Add("cellMetrics", metric.cell, context);
            const auto offsets = result->Add("faceOffsets", metric.faceOffsets, context);
            const auto quadrature = result->Add("faceQuadrature", metric.faceQuadrature, context);
            const auto radial = result->Add("radialMoment", metric.radialMoment, context);
            const auto faceMean = result->Add("faceMean", metric.faceMean, context);
            result->metricView = {R::Detail::ProjectView<const real, false, R::FixedRows<8>>(cell, metric.cell.Layout()),
                                  R::Detail::ProjectView<const index, false, R::FixedRows<1>>(offsets, metric.faceOffsets.Layout()),
                                  R::Detail::ProjectView<const real, false, R::FixedRows<7>>(quadrature, metric.faceQuadrature.Layout()),
                                  R::Detail::ProjectView<const real, false, R::FixedRows<3>>(radial, metric.radialMoment.Layout()),
                                  R::Detail::ProjectView<const real, false, R::FixedRows<7>>(faceMean, metric.faceMean.Layout())};
        }
        if (options.basis)
        {
            DNDS_check_throw_info(Settings().cacheDiffBase, "device basis publication requires prepared cacheDiffBase");
            const auto basis = result->Add("faceBasis", host.faceBasis, context);
            result->basisView = {R::Detail::ProjectView<const real, false, R::TableRows>(basis, host.faceBasis.Layout()), result->metricView.faceOffsets, host.n, host.cachedDerivatives};
        }
        if (options.mesh)
        {
            result->mesh = std::make_shared<Geom::UnstructuredMeshSnapshot>(Geom::UnstructuredMeshSnapshot::Capture(*metric.mesh, Geom::MeshReadRequirements::CellFlux()).To(device, context));
            std::set<uint64_t> allocations;
            result->mesh->VisitAllocations([&](uint64_t identity, size_t capacity, R::Device placement)
                                           {
                if(allocations.insert(identity).second)
                    result->buffers.push_back({"mesh.allocation."+std::to_string(allocations.size()),capacity,placement}); });
        }
        return VariationalReconstructionPlan(_data, std::move(result));
    }
    R::Device VariationalReconstructionPlan::Device() const
    {
        DNDS_check_throw_info(_data, "reconstruction plan was moved/released");
        return _device ? _device->device : R::Device::Host();
    }
    std::vector<ResidentBufferInfo> VariationalReconstructionPlan::ResidentBuffers() const
    {
        Validate();
        return _device ? _device->buffers : std::vector<ResidentBufferInfo>{};
    }
    ReconstructionResidentRead VariationalReconstructionPlan::AcquireResidentRead() const
    {
        Validate();
        DNDS_check_throw_info(_device, "reconstruction plan has no non-host placement");
        return ReconstructionResidentRead(_device);
    }
    Detail::ReconstructionDeviceView ReconstructionResidentRead::DeviceView() const
    {
        DNDS_check_throw_info(_data, "resident reconstruction read was moved/released");
        return _data->view;
    }
    FiniteVolumeHostView ReconstructionResidentRead::MetricView() const
    {
        DNDS_check_throw_info(_data && (_data->options.metrics || _data->options.basis),
                              "resident reconstruction metrics were not requested");
        return _data->metricView;
    }
    ReconstructionBasisHostView ReconstructionResidentRead::BasisView() const
    {
        DNDS_check_throw_info(_data && _data->options.basis,
                              "resident reconstruction basis was not requested");
        return _data->basisView;
    }
    const Geom::UnstructuredMeshSnapshot &VariationalReconstructionPlan::MeshSnapshot() const
    {
        Validate();
        DNDS_check_throw_info(_device && _device->mesh, "device mesh was not requested");
        return *_device->mesh;
    }
    ReconstructionWorkspace VariationalReconstructionPlan::BuildDeviceWorkspace(index components, R::RowLayout prototype,
                                                                                std::vector<index> periodic) const
    {
        Validate();
        DNDS_check_throw_info(_device && components > 0 && components <= INT_MAX / _device->view.n &&
                                  !prototype.IsVariable() && prototype.Width() == components * _device->view.n,
                              "invalid device reconstruction shape");
        Detail::ValidateVectorBlocks(components, periodic);
        const index blocks = std::max<index>(1, std::min<index>(256, _device->view.owned));
        R::Array<index> starts(R::RowLayout::Dense(periodic.size(), 1));
        {
            auto write = starts.Write();
            for (index i = 0; i < index(periodic.size()); ++i)
                write.View()(i, 0) = periodic[size_t(i)];
        }
        auto deviceStarts = starts.To(Device());
        const auto periodicIdentity = deviceStarts.Identity();
        const auto periodicVersion = deviceStarts.Version();
        return {Metrics().BuildField(prototype, Device()), Metrics().BuildField(prototype, Device()),
                R::Array<real>(R::RowLayout::Dense(blocks * 6, prototype.Width()), Device()),
                components, _data, std::move(periodic), std::move(deviceStarts),
                R::Array<int>(R::RowLayout::Dense(1, 1), Device()), blocks,
                periodicIdentity, periodicVersion, _device};
    }
    R::TransferBinding VariationalReconstructionPlan::BindHalo(FVField &field, R::TransferTransport transport, R::TransferOptions options) const
    {
        Validate();
        const auto &metric = Metrics();
        metric.Communication()->LocalStage([&]
                                           {Validate();ValidateDeviceField(*this,field,field.Width());
            DNDS_check_throw_info(field.Space().Same(metric._data->dofs),"device field belongs to another metric space"); });
        return R::PayloadPlan(*metric._data->halo, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<real>()).Bind(field.Owned(), field.Ghost(), transport, std::move(options));
    }
    ReconstructionBoundaryValues VariationalReconstructionPlan::PlaceBoundaryValues(const ReconstructionBoundaryValues &source) const
    {
        Validate();
        DNDS_check_throw_info(_device && source.plan == _data, "boundary values belong to another plan or placement is not device-resident");
        ValidateArray(source.values, R::Device::Host(), NumBoundaryQuadrature(), source.values.Layout().Width());
        return {source.values.To(Device()), _data};
    }
    FaceCharacteristicTransform VariationalReconstructionPlan::PlaceFaceTransform(const FaceCharacteristicTransform &source) const
    {
        Validate();
        DNDS_check_throw_info(_device && source.plan == _data && source.first >= 0 && source.count >= 0 && source.count <= 64,
                              "device characteristic source belongs to another plan or has an invalid range");
        if (source.count == 0)
        {
            FaceCharacteristicTransform result;
            result.first = source.first;
            result.plan = _data;
            return result;
        }
        DNDS_check_throw_info(
            source.forwardVersion == source.forward.Version() && source.inverseVersion == source.inverse.Version() && source.forwardVersion &&
                source.forwardIdentity == source.forward.Identity() && source.inverseIdentity == source.inverse.Identity(),
            "unfinalized device characteristic source");
        FaceCharacteristicTransform result{source.first, source.count, source.forward.To(Device()), source.inverse.To(Device()), _data};
        result.forwardVersion = result.forward.Version();
        result.inverseVersion = result.inverse.Version();
        result.forwardIdentity = result.forward.Identity();
        result.inverseIdentity = result.inverse.Identity();
        result.plan = _data;
        return result;
    }
    void VariationalReconstructionPlan::SubmitDeviceReconstructionIterTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean,
        const FVField &input, ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary,
        ReconstructionRegion region, ReconstructionDeviceKernel kernel) const
    {
        ValidateDeviceStream(*this, stream);
        DNDS_check_throw_info(SupportsKernel(kernel), "requested CUDA reconstruction kernel is unavailable on this device/build");
        ValidateDeviceWorkspace(_device, work);
        ValidateDeviceField(*this, mean, work.components);
        ValidateDeviceField(*this, input, work.components * _device->view.n);
        for (const auto *array : {&mean.Owned(), &mean.Ghost(), &input.Owned(), &input.Ghost(), &boundary.values})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(array->Storage(), work.temporary.Owned().Storage()), "device Jacobi output aliases an input");
        DNDS_check_throw_info(mean.Space().Same(work.coefficients.Space()) && input.Space().Same(work.coefficients.Space()), "device fields belong to different spaces");
        DNDS_check_throw_info(boundary.plan == _data, "device boundary plan mismatch");
        ValidateArray(boundary.values, Device(), NumBoundaryQuadrature(), work.components);
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input.Owned().Storage(), work.temporary.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(input.Ghost().Storage(), work.temporary.Owned().Storage()),
                              "device Jacobi aliases its input");
        const auto selection = Selection(*_device, region);
        auto pin = _device;
        auto launch = [pin, selection, kernel](R::ExecutionContext context, auto mo, auto mg, auto ro, auto rg, auto bc, auto output, auto blocks, auto status)
        {
            if (kernel == ReconstructionDeviceKernel::FP64TensorCoreShared || kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync)
                Detail::LaunchJacobiTensorCoreShared(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status,
                                                     kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync);
            else if (kernel == ReconstructionDeviceKernel::FP64TensorCore)
                Detail::LaunchJacobiTensorCore(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status);
            else
                Detail::LaunchJacobi(context, pin->view, selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status);
        };
        if (region == ReconstructionRegion::Core)
            R::SubmitArrays(engine, stream, [launch](R::ExecutionContext c, auto meanOwned, auto recOwned, auto bc, auto output, auto blocks, auto status)
                            { launch(c, meanOwned, R::ArrayView<const real>{}, recOwned, R::ArrayView<const real>{}, bc, output, blocks, status); }, R::Read(mean.Owned()), R::Read(input.Owned()), R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.devicePeriodicVectorStarts), R::Read(work.deviceStatus));
        else
            R::SubmitArrays(engine, stream, launch, R::Read(mean.Owned()), R::Read(mean.Ghost()), R::Read(input.Owned()), R::Read(input.Ghost()),
                            R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.devicePeriodicVectorStarts), R::Read(work.deviceStatus));
    }
    void VariationalReconstructionPlan::SubmitDeviceCommitReconstruction(
        R::EagerEngine &engine, const R::ExecutionStream &stream, ReconstructionWorkspace &work) const
    {
        // Stable owned-only publication: coefficient ghosts remain stale until
        // exchange. Keep leases and the device-status guard in any future copy
        // optimization; Runtime::Array::CopyFrom alone does not provide the guard.
        ValidateDeviceStream(*this, stream);
        ValidateDeviceWorkspace(_device, work);
        auto pin = _device;
        R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto input, auto output, auto status)
                        { Detail::LaunchCommit(context, input, output, status); }, R::Read(work.temporary.Owned()), R::Write(work.coefficients.Owned()), R::Read(work.deviceStatus));
    }
    void VariationalReconstructionPlan::SubmitDeviceLimiterWBAP_3OrderTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, ReconstructionWorkspace &work, int order,
        FaceCharacteristicTransformRef transform, ReconstructionRegion region, const R::Array<real> *indicator,
        real threshold) const
    {
        ValidateDeviceStream(*this, stream);
        ValidateDeviceWorkspace(_device, work);
        DNDS_check_throw_info(order >= 1 && order <= Settings().maxOrder && std::isfinite(threshold), "invalid device limiter order/threshold");
        for (const auto *input : {&work.coefficients.Owned(), &work.coefficients.Ghost(), &work.scratch})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), work.temporary.Owned().Storage()), "device limiter output aliases its input/scratch");
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(work.coefficients.Owned().Storage(), work.scratch.Storage()) &&
                                  !Algebra::Detail::Overlaps(work.coefficients.Ghost().Storage(), work.scratch.Storage()),
                              "device limiter scratch aliases coefficients");
        const auto selection = Selection(*_device, region);
        const auto n = _device->view.n;
        const index begin = order == 1 ? 0 : Math::PolynomialSize(Metrics().Dimension(), order - 1) - 1;
        const index modes = Math::PolynomialSize(Metrics().Dimension(), order) - 1 - begin;
        if (indicator)
        {
            ValidateArray(*indicator, Device(), _device->view.owned, 1);
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(indicator->Storage(), work.temporary.Owned().Storage()) &&
                                      !Algebra::Detail::Overlaps(indicator->Storage(), work.scratch.Storage()),
                                  "device indicator aliases mutable output/scratch");
        }
        const auto *selectedTransform = transform && transform->get().count ? &transform->get() : nullptr;
        if (transform && !selectedTransform)
            DNDS_check_throw_info(transform->get().plan == _data && transform->get().first >= 0 && transform->get().first <= work.components,
                                  "disabled device characteristic transform belongs to another plan or has an invalid offset");
        if (selectedTransform)
        {
            DNDS_check_throw_info(selectedTransform->plan == _data && selectedTransform->first >= 0 && selectedTransform->count > 0 && selectedTransform->count <= 64 && selectedTransform->count <= work.components - selectedTransform->first &&
                                      selectedTransform->forwardIdentity == selectedTransform->forward.Identity() && selectedTransform->inverseIdentity == selectedTransform->inverse.Identity() &&
                                      selectedTransform->forwardVersion == selectedTransform->forward.Version() && selectedTransform->inverseVersion == selectedTransform->inverse.Version(),
                                  "device characteristic data changed/unfinalized");
            ValidateArray(selectedTransform->forward, Device(), Metrics().NumFaceProc(), selectedTransform->count * selectedTransform->count);
            ValidateArray(selectedTransform->inverse, Device(), Metrics().NumFaceProc(), selectedTransform->count * selectedTransform->count);
            for (const auto *input : {&selectedTransform->forward, &selectedTransform->inverse})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), work.temporary.Owned().Storage()) &&
                                          !Algebra::Detail::Overlaps(input->Storage(), work.scratch.Storage()),
                                      "device transform aliases mutable output/scratch");
        }
        auto pin = _device;
        const auto first = selectedTransform ? selectedTransform->first : 0, count = selectedTransform ? selectedTransform->count : 0,
                   blocksCount = work.deviceBlocks;
        // Optional read operands are empty owned arrays on the right device,
        // retained in the closure. No fabricated pointers or implicit transfer.
        const auto &forward = selectedTransform ? selectedTransform->forward : _device->empty,
                   &inverse = selectedTransform ? selectedTransform->inverse : _device->empty,
                   &activation = indicator ? *indicator : _device->empty;
        auto launch = [pin, selection, first, count, begin, modes, threshold, blocksCount](R::ExecutionContext context, auto owned, auto ghost, auto output, auto scratch, auto blocks, auto left, auto right, auto active, auto status)
        { Detail::LaunchWBAP(context, pin->view, selection, {blocks}, {owned, ghost}, output, scratch, {left, right, first, count}, active, status, begin, modes, threshold, blocksCount); };
        if (region == ReconstructionRegion::Core)
            R::SubmitArrays(engine, stream, [launch](R::ExecutionContext c, auto owned, auto output, auto scratch, auto blocks, auto left, auto right, auto active, auto status)
                            { launch(c, owned, R::ArrayView<const real>{}, output, scratch, blocks, left, right, active, status); }, R::Read(work.coefficients.Owned()), R::Write(work.temporary.Owned()), R::ReadWrite(work.scratch), R::Read(work.devicePeriodicVectorStarts), R::Read(forward), R::Read(inverse), R::Read(activation), R::ReadWrite(work.deviceStatus));
        else
            R::SubmitArrays(engine, stream, launch, R::Read(work.coefficients.Owned()), R::Read(work.coefficients.Ghost()), R::Write(work.temporary.Owned()),
                            R::ReadWrite(work.scratch), R::Read(work.devicePeriodicVectorStarts), R::Read(forward), R::Read(inverse), R::Read(activation), R::ReadWrite(work.deviceStatus));
    }
    void VariationalReconstructionPlan::CheckStatus(const ReconstructionWorkspace &work) const
    {
        Validate();
        if (!_device)
            return;
        ValidateDeviceWorkspace(_device, work);
        const auto host = work.deviceStatus.To(R::Device::Host());
        DNDS_check_throw_info(host.Read().View()(0, 0) == 0, "device limiter indicator contains nonfinite values; output was not updated");
    }
    bool VariationalReconstructionPlan::SupportsKernel(ReconstructionDeviceKernel kernel) const
    {
        DNDS_check_throw_info(_data, "reconstruction plan is empty");
        if (!_device)
            return kernel == ReconstructionDeviceKernel::FP64;
        return kernel == ReconstructionDeviceKernel::FP64 ||
               ((kernel == ReconstructionDeviceKernel::FP64TensorCore || kernel == ReconstructionDeviceKernel::FP64TensorCoreShared ||
                 kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync) &&
                _device->fp64TensorCore);
    }
#ifndef DNDS_USE_CUDA
    namespace Detail
    {
        bool FP64TensorCoreAvailable(R::Device) { return false; }
        void LaunchJacobiTensorCoreShared(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks, R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>, bool)
        {
            DNDS_check_throw_info(false, "CUDA shared tensor-core Jacobi unavailable");
        }
        void LaunchJacobiTensorCore(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks, R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>)
        {
            DNDS_check_throw_info(false, "CUDA FP64 tensor-core Jacobi unavailable");
        }
        void LaunchJacobi(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks, R::PairReadView<real>, R::PairReadView<real>, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>)
        {
            DNDS_check_throw_info(false, "CUDA Jacobi unavailable");
        }
        void LaunchWBAP(R::ExecutionContext, ReconstructionDeviceView, DeviceWorkSelection, DevicePeriodicBlocks, R::PairReadView<real>, R::ArrayView<real>, R::ArrayView<real>, DeviceCharacteristicView, R::ArrayView<const real>, R::ArrayView<int>, index, index, real, index)
        {
            DNDS_check_throw_info(false, "CUDA WBAP unavailable");
        }
        void LaunchCommit(R::ExecutionContext, R::ArrayView<const real>, R::ArrayView<real>, R::ArrayView<const int>)
        {
            DNDS_check_throw_info(false, "CUDA reconstruction commit unavailable");
        }
    }
#endif
}
