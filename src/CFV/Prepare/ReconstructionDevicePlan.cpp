#include "CFV/Plans/ReconstructionDevicePlan.hpp"
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
        void ValidateField(const ReconstructionDevicePlan &plan, const FVField &field, index width)
        {
            field.Validate();
            DNDS_check_throw_info(field.Device() == plan.Device() && index(field.Width()) == width, "device FV field width/placement mismatch");
        }
        void ValidateWorkspace(const std::shared_ptr<const Detail::ReconstructionPlacementData> &data, const ReconstructionDeviceWorkspace &work)
        {
            const auto n = data->view.n;
            DNDS_check_throw_info(work.plan == data && work.components > 0 && work.components <= INT_MAX / n && work.blocks > 0 && work.blocks <= 256,
                                  "device workspace belongs to another publication or has an invalid shape");
            work.coefficients.Validate();
            work.temporary.Validate();
            DNDS_check_throw_info(work.coefficients.Device() == data->device && work.temporary.Device() == data->device &&
                                      work.coefficients.Width() == n * work.components && work.temporary.Width() == n * work.components &&
                                      work.coefficients.Space().Same(work.temporary.Space()) && work.coefficients.Space().Same(*data->dofs),
                                  "device workspace fields invalid");
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(work.coefficients.Owned().Storage(), work.temporary.Owned().Storage()) &&
                                      work.scratch.GetDevice() == data->device && work.scratch.Layout().Kind() == R::RowLayoutKind::Table && work.scratch.RowCount() == work.blocks * 6 &&
                                      work.scratch.Layout().Width() == n * work.components && work.status.GetDevice() == data->device && work.status.Layout().Kind() == R::RowLayoutKind::Table &&
                                      work.status.RowCount() == 1 && work.status.Layout().Width() == 1 &&
                                      work.periodicVectorStarts.Identity() == work.periodicIdentity && work.periodicVectorStarts.Version() == work.periodicVersion &&
                                      work.periodicVectorStarts.GetDevice() == data->device && work.periodicVectorStarts.Layout().Kind() == R::RowLayoutKind::Table && work.periodicVectorStarts.Layout().Width() == 1,
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
        void ValidateStream(const ReconstructionDevicePlan &plan, const R::ExecutionStream &stream)
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
    ReconstructionDevicePlan ReconstructionDevicePlan::Prepare(const VariationalReconstructionPlan &source, R::Device device,
                                                               R::ExecutionContext context, ReconstructionResidencyOptions options)
    {
#ifndef DNDS_USE_CUDA
        DNDS_check_throw_info(false, "CUDA reconstruction is unavailable in this build");
#endif
        source.Require(ReconstructionOperation::Sweep);
        source.Require(ReconstructionOperation::WBAP);
        source.Validate();
        DNDS_check_throw_info(device.space == R::MemorySpace::CUDA && context.device == device, "device reconstruction requires explicit CUDA placement/context");
        DNDS_check_throw_info(!source.Settings().SORInstead, "CUDA SOR requires a logical parallel sweep design and is unsupported");
        source._data->workPartition->Validate();
        auto geometry = source.Metrics().AcquireRead();
        auto result = std::make_shared<Detail::ReconstructionPlacementData>(source, device, options);
        result->fp64TensorCore = Detail::FP64TensorCoreAvailable(device);
        const auto &host = *source._data;
        auto &view = result->view;
        view.owned = source.Metrics().NumCell();
        view.n = source.NRecDOF();
        view.relax = source.Settings().jacobiRelax;
        view.wbapCenterWeight = source.Settings().WBAP_nStd;
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
        const auto &metric = *source.Metrics()._data;
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
            DNDS_check_throw_info(source.Settings().cacheDiffBase, "device basis publication requires prepared cacheDiffBase");
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
        return ReconstructionDevicePlan(std::move(result));
    }
    R::Device ReconstructionDevicePlan::Device() const
    {
        DNDS_check_throw_info(_data, "device plan was moved/released");
        return _data->device;
    }
    void ReconstructionDevicePlan::Validate() const
    {
        DNDS_check_throw_info(_data, "device plan was moved/released");
        _data->source.Validate();
        _data->source._data->workPartition->Validate();
    }
    ReconstructionDeviceRead::ReconstructionDeviceRead(std::shared_ptr<const Detail::ReconstructionPlacementData> data)
        : _data(std::move(data)), _geometry(std::make_shared<FiniteVolumeRead>(_data->source.Metrics().AcquireRead())) {}
    ReconstructionDeviceRead ReconstructionDevicePlan::AcquireRead() const
    {
        Validate();
        return ReconstructionDeviceRead(_data);
    }
    Detail::ReconstructionDeviceView ReconstructionDeviceRead::DeviceView() const
    {
        DNDS_check_throw_info(_data, "device read scope was moved/released");
        return _data->view;
    }
    FiniteVolumeHostView ReconstructionDeviceRead::MetricView() const
    {
        DNDS_check_throw_info(_data, "device read scope was moved/released");
        DNDS_check_throw_info(_data->options.metrics || _data->options.basis, "device metrics were not requested");
        return _data->metricView;
    }
    ReconstructionBasisHostView ReconstructionDeviceRead::BasisView() const
    {
        DNDS_check_throw_info(_data, "device read scope was moved/released");
        DNDS_check_throw_info(_data->options.basis, "device basis was not requested");
        return _data->basisView;
    }
    std::vector<ResidentBufferInfo> ReconstructionDevicePlan::Buffers() const
    {
        Validate();
        return _data->buffers;
    }
    const Geom::UnstructuredMeshSnapshot &ReconstructionDevicePlan::MeshSnapshot() const
    {
        Validate();
        DNDS_check_throw_info(_data->mesh, "device mesh was not requested");
        return *_data->mesh;
    }
    FVField ReconstructionDevicePlan::BuildField(R::RowLayout prototype) const
    {
        Validate();
        return _data->source.Metrics().BuildField(std::move(prototype), Device());
    }
    ReconstructionDeviceWorkspace ReconstructionDevicePlan::BuildWorkspace(index components, R::RowLayout prototype, std::vector<index> periodic) const
    {
        Validate();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / _data->view.n && !prototype.IsVariable() && prototype.Width() == components * _data->view.n,
                              "invalid device reconstruction shape");
        Detail::ValidateVectorBlocks(components, periodic);
        const index blocks = std::max<index>(1, std::min<index>(256, _data->view.owned));
        R::Array<index> starts(R::RowLayout::Dense(periodic.size(), 1));
        {
            auto write = starts.Write();
            for (index i = 0; i < index(periodic.size()); ++i)
                write.View()(i, 0) = periodic[size_t(i)];
        }
        auto deviceStarts = starts.To(Device());
        const auto identity = deviceStarts.Identity(), version = deviceStarts.Version();
        return {BuildField(prototype), BuildField(prototype), R::Array<real>(R::RowLayout::Dense(blocks * 6, prototype.Width()), Device()),
                std::move(deviceStarts), R::Array<int>(R::RowLayout::Dense(1, 1), Device()), components, blocks, identity, version, _data};
    }
    R::TransferBinding ReconstructionDevicePlan::BindHalo(FVField &field, R::TransferTransport transport, R::TransferOptions options) const
    {
        Validate();
        const auto &metric = _data->source.Metrics();
        metric.Communication()->LocalStage([&]
                                           {Validate();ValidateField(*this,field,field.Width());
            DNDS_check_throw_info(field.Space().Same(metric._data->dofs),"device field belongs to another metric space"); });
        return R::PayloadPlan(*metric._data->halo, field.Owned().Layout(), field.Ghost().Layout(), R::ElementSchema::Arithmetic<real>()).Bind(field.Owned(), field.Ghost(), transport, std::move(options));
    }
    ReconstructionBoundaryValues ReconstructionDevicePlan::PrepareBoundary(const ReconstructionBoundaryValues &source) const
    {
        Validate();
        DNDS_check_throw_info(source.plan == _data->source._data, "boundary values belong to another plan");
        ValidateArray(source.values, R::Device::Host(), _data->source.NumBoundaryQuadrature(), source.values.Layout().Width());
        return {source.values.To(Device()), source.plan};
    }
    DeviceFaceCharacteristicTransform ReconstructionDevicePlan::PrepareTransform(const FaceCharacteristicTransform &source) const
    {
        Validate();
        DNDS_check_throw_info(source.plan == _data->source._data && source.count > 0 && source.count <= 64 &&
                                  source.forwardVersion == source.forward.Version() && source.inverseVersion == source.inverse.Version() && source.forwardVersion &&
                                  source.forwardIdentity == source.forward.Identity() && source.inverseIdentity == source.inverse.Identity(),
                              "unfinalized device characteristic source");
        DeviceFaceCharacteristicTransform result{source.forward.To(Device()), source.inverse.To(Device()), source.first, source.count};
        result.forwardVersion = result.forward.Version();
        result.inverseVersion = result.inverse.Version();
        result.forwardIdentity = result.forward.Identity();
        result.inverseIdentity = result.inverse.Identity();
        result.plan = _data;
        return result;
    }
    void ReconstructionDevicePlan::DoReconstructionIterTo(R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean,
                                                          const FVField &input, ReconstructionDeviceWorkspace &work, const ReconstructionBoundaryValues &boundary,
                                                          ReconstructionRegion region, ReconstructionDeviceKernel kernel) const
    {
        ValidateStream(*this, stream);
        DNDS_check_throw_info(SupportsKernel(kernel), "requested CUDA reconstruction kernel is unavailable on this device/build");
        ValidateWorkspace(_data, work);
        ValidateField(*this, mean, work.components);
        ValidateField(*this, input, work.components * _data->view.n);
        for (const auto *array : {&mean.Owned(), &mean.Ghost(), &input.Owned(), &input.Ghost(), &boundary.values})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(array->Storage(), work.temporary.Owned().Storage()), "device Jacobi output aliases an input");
        DNDS_check_throw_info(mean.Space().Same(work.coefficients.Space()) && input.Space().Same(work.coefficients.Space()), "device fields belong to different spaces");
        DNDS_check_throw_info(boundary.plan == _data->source._data, "device boundary plan mismatch");
        ValidateArray(boundary.values, Device(), _data->source.NumBoundaryQuadrature(), work.components);
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input.Owned().Storage(), work.temporary.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(input.Ghost().Storage(), work.temporary.Owned().Storage()),
                              "device Jacobi aliases its input");
        const auto selection = Selection(*_data, region);
        auto pin = std::make_shared<ReconstructionDeviceRead>(AcquireRead());
        auto launch = [pin, selection, kernel](R::ExecutionContext context, auto mo, auto mg, auto ro, auto rg, auto bc, auto output, auto blocks, auto status)
        {
            if (kernel == ReconstructionDeviceKernel::FP64TensorCoreShared || kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync)
                Detail::LaunchJacobiTensorCoreShared(context, pin->DeviceView(), selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status,
                                                     kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync);
            else if (kernel == ReconstructionDeviceKernel::FP64TensorCore)
                Detail::LaunchJacobiTensorCore(context, pin->DeviceView(), selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status);
            else
                Detail::LaunchJacobi(context, pin->DeviceView(), selection, {blocks}, {mo, mg}, {ro, rg}, bc, output, status);
        };
        if (region == ReconstructionRegion::Core)
            R::SubmitArrays(engine, stream, [launch](R::ExecutionContext c, auto meanOwned, auto recOwned, auto bc, auto output, auto blocks, auto status)
                            { launch(c, meanOwned, R::ArrayView<const real>{}, recOwned, R::ArrayView<const real>{}, bc, output, blocks, status); }, R::Read(mean.Owned()), R::Read(input.Owned()), R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.periodicVectorStarts), R::Read(work.status));
        else
            R::SubmitArrays(engine, stream, launch, R::Read(mean.Owned()), R::Read(mean.Ghost()), R::Read(input.Owned()), R::Read(input.Ghost()),
                            R::Read(boundary.values), R::Write(work.temporary.Owned()), R::Read(work.periodicVectorStarts), R::Read(work.status));
    }
    void ReconstructionDevicePlan::CommitReconstruction(R::EagerEngine &engine, const R::ExecutionStream &stream, ReconstructionDeviceWorkspace &work) const
    {
        ValidateStream(*this, stream);
        ValidateWorkspace(_data, work);
        auto pin = std::make_shared<ReconstructionDeviceRead>(AcquireRead());
        R::SubmitArrays(engine, stream, [pin](R::ExecutionContext context, auto input, auto output, auto status)
                        { Detail::LaunchCommit(context, input, output, status); }, R::Read(work.temporary.Owned()), R::Write(work.coefficients.Owned()), R::Read(work.status));
    }
    void ReconstructionDevicePlan::DoLimiterWBAP_3OrderTo(R::EagerEngine &engine, const R::ExecutionStream &stream, ReconstructionDeviceWorkspace &work, int order,
                                                          const DeviceFaceCharacteristicTransform *transform, ReconstructionRegion region, const R::Array<real> *indicator, real threshold) const
    {
        ValidateStream(*this, stream);
        ValidateWorkspace(_data, work);
        DNDS_check_throw_info(order >= 1 && order <= _data->source.Settings().maxOrder && std::isfinite(threshold), "invalid device limiter order/threshold");
        for (const auto *input : {&work.coefficients.Owned(), &work.coefficients.Ghost(), &work.scratch})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), work.temporary.Owned().Storage()), "device limiter output aliases its input/scratch");
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(work.coefficients.Owned().Storage(), work.scratch.Storage()) &&
                                  !Algebra::Detail::Overlaps(work.coefficients.Ghost().Storage(), work.scratch.Storage()),
                              "device limiter scratch aliases coefficients");
        const auto selection = Selection(*_data, region);
        const auto n = _data->view.n;
        const index begin = order == 1 ? 0 : Math::PolynomialSize(_data->source.Metrics().Dimension(), order - 1) - 1;
        const index modes = Math::PolynomialSize(_data->source.Metrics().Dimension(), order) - 1 - begin;
        if (indicator)
        {
            ValidateArray(*indicator, Device(), _data->view.owned, 1);
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(indicator->Storage(), work.temporary.Owned().Storage()) &&
                                      !Algebra::Detail::Overlaps(indicator->Storage(), work.scratch.Storage()),
                                  "device indicator aliases mutable output/scratch");
        }
        if (transform)
        {
            DNDS_check_throw_info(transform->plan == _data && transform->first >= 0 && transform->count > 0 && transform->count <= 64 && transform->count <= work.components - transform->first &&
                                      transform->forwardIdentity == transform->forward.Identity() && transform->inverseIdentity == transform->inverse.Identity() &&
                                      transform->forwardVersion == transform->forward.Version() && transform->inverseVersion == transform->inverse.Version(),
                                  "device characteristic data changed/unfinalized");
            ValidateArray(transform->forward, Device(), _data->source.Metrics().NumFaceProc(), transform->count * transform->count);
            ValidateArray(transform->inverse, Device(), _data->source.Metrics().NumFaceProc(), transform->count * transform->count);
            for (const auto *input : {&transform->forward, &transform->inverse})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), work.temporary.Owned().Storage()) &&
                                          !Algebra::Detail::Overlaps(input->Storage(), work.scratch.Storage()),
                                      "device transform aliases mutable output/scratch");
        }
        auto pin = std::make_shared<ReconstructionDeviceRead>(AcquireRead());
        const auto first = transform ? transform->first : 0, count = transform ? transform->count : 0, blocksCount = work.blocks;
        // Optional read operands are empty owned arrays on the right device,
        // retained in the closure. No fabricated pointers or implicit transfer.
        const auto &forward = transform ? transform->forward : _data->empty, &inverse = transform ? transform->inverse : _data->empty,
                   &activation = indicator ? *indicator : _data->empty;
        auto launch = [pin, selection, first, count, begin, modes, threshold, blocksCount](R::ExecutionContext context, auto owned, auto ghost, auto output, auto scratch, auto blocks, auto left, auto right, auto active, auto status)
        { Detail::LaunchWBAP(context, pin->DeviceView(), selection, {blocks}, {owned, ghost}, output, scratch, {left, right, first, count}, active, status, begin, modes, threshold, blocksCount); };
        if (region == ReconstructionRegion::Core)
            R::SubmitArrays(engine, stream, [launch](R::ExecutionContext c, auto owned, auto output, auto scratch, auto blocks, auto left, auto right, auto active, auto status)
                            { launch(c, owned, R::ArrayView<const real>{}, output, scratch, blocks, left, right, active, status); }, R::Read(work.coefficients.Owned()), R::Write(work.temporary.Owned()), R::ReadWrite(work.scratch), R::Read(work.periodicVectorStarts), R::Read(forward), R::Read(inverse), R::Read(activation), R::ReadWrite(work.status));
        else
            R::SubmitArrays(engine, stream, launch, R::Read(work.coefficients.Owned()), R::Read(work.coefficients.Ghost()), R::Write(work.temporary.Owned()),
                            R::ReadWrite(work.scratch), R::Read(work.periodicVectorStarts), R::Read(forward), R::Read(inverse), R::Read(activation), R::ReadWrite(work.status));
    }
    void ReconstructionDevicePlan::CheckStatus(const ReconstructionDeviceWorkspace &work) const
    {
        Validate();
        ValidateWorkspace(_data, work);
        const auto host = work.status.To(R::Device::Host());
        DNDS_check_throw_info(host.Read().View()(0, 0) == 0, "device limiter indicator contains nonfinite values; output was not updated");
    }
    bool ReconstructionDevicePlan::SupportsKernel(ReconstructionDeviceKernel kernel) const
    {
        DNDS_check_throw_info(_data, "device reconstruction publication is empty");
        return kernel == ReconstructionDeviceKernel::FP64 ||
               ((kernel == ReconstructionDeviceKernel::FP64TensorCore || kernel == ReconstructionDeviceKernel::FP64TensorCoreShared ||
                 kernel == ReconstructionDeviceKernel::FP64TensorCoreAsync) &&
                _data->fp64TensorCore);
    }
    void ReconstructionDevicePlan::DoReconstructionIter(const FVField &mean, ReconstructionDeviceWorkspace &work,
                                                        const ReconstructionBoundaryValues &boundary, ReconstructionDeviceKernel kernel) const
    {
        R::EagerEngine engine;
        auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
        DoReconstructionIterTo(engine, stream, mean, work.coefficients, work, boundary, ReconstructionRegion::All, kernel);
        CommitReconstruction(engine, stream, work);
        engine.Synchronize();
        CheckStatus(work);
    }
    void ReconstructionDevicePlan::DoLimiterWBAP_3Order(ReconstructionDeviceWorkspace &work, int order, const DeviceFaceCharacteristicTransform *transform) const
    {
        R::EagerEngine engine;
        auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
        DoLimiterWBAP_3OrderTo(engine, stream, work, order, transform);
        CommitReconstruction(engine, stream, work);
        engine.Synchronize();
        CheckStatus(work);
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
