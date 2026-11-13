#include "CFV/Detail/ReconstructionData.hpp"
#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    using Matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    using ConstMap = Eigen::Map<const Matrix>;
    using Map = Eigen::Map<Matrix>;
    namespace
    {
        void CheckArray(const R::Array<real> &array, index rows, index width)
        {
            DNDS_check_throw_info(array.GetDevice() == R::Device::Host() && array.IsValid() &&
                                      !array.Layout().IsVariable() && array.RowCount() == rows && index(array.Layout().Width()) == width,
                                  "FV operand has wrong shape, validity or CPU placement");
        }
        void CheckWorkspace(const Detail::ReconstructionData *data, const ReconstructionWorkspace &work)
        {
            DNDS_check_throw_info(work.plan.get() == data && work.components > 0, "reconstruction workspace belongs to another plan");
            data->metrics.ValidateField(work.temporary, data->n * work.components);
            CheckArray(work.scratch, data->capabilities.wbap ? 6 * data->hostThreads : 0, data->n * work.components);
            DNDS_check_throw_info(OMP::DistributedThreads() <= data->hostThreads,
                                  "CPU reconstruction team exceeds prepared scratch capacity");
        }
        void CheckWorkspace(const Detail::ReconstructionData *data, const ReconstructionWorkspace &work,
                            const FVField &coefficients)
        {
            CheckWorkspace(data, work);
            data->metrics.ValidateField(coefficients, data->n * work.components);
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(coefficients.Owned().Storage(), work.temporary.Owned().Storage()),
                                  "reconstruction input and temporary storage overlap");
        }
        void AddScheduledAccess(std::vector<R::ScheduledBufferAccess> &accesses, const R::Buffer &buffer, R::AccessMode mode)
        {
            const auto identity = buffer.Identity();
            for (auto &entry : accesses)
                if (entry.allocation == identity)
                {
                    if (entry.mode != mode)
                        entry.mode = R::AccessMode::ReadWrite;
                    return;
                }
            accesses.emplace_back(buffer, mode);
        }
        template <class T>
        void AddScheduledAccess(std::vector<R::ScheduledBufferAccess> &accesses, const R::Array<T> &array, R::AccessMode mode)
        {
            AddScheduledAccess(accesses, array.Storage(), mode);
        }
    }
    const FiniteVolumePlan &VariationalReconstructionPlan::Metrics() const
    {
        DNDS_check_throw_info(_data, "device-only reconstruction placement has no host metric publication");
        return _data->metrics;
    }
    const ReconstructionSettings &VariationalReconstructionPlan::Settings() const
    {
        return _data ? _data->settings : PlacementSettings();
    }
    int VariationalReconstructionPlan::NRecDOF() const { return _data ? _data->n : PlacementNRecDOF(); }
    ReconstructionCapabilities VariationalReconstructionPlan::Capabilities() const
    {
        DNDS_check_throw_info(_device || _data, "reconstruction plan is empty");
        return _device ? PlacementCapabilities() : _data->capabilities;
    }
    void VariationalReconstructionPlan::Require(ReconstructionOperation operation) const
    {
        if (Capabilities().Supports(operation))
            return;
        const char *name = "unknown";
        switch (operation)
        {
        case ReconstructionOperation::Sweep:
            name = "reconstruction sweep";
            break;
        case ReconstructionOperation::WBAP:
            name = "WBAP limiter";
            break;
        case ReconstructionOperation::LinearSystem:
            name = "original linear system";
            break;
        case ReconstructionOperation::LocalPreconditioner:
            name = "local preconditioner";
            break;
        case ReconstructionOperation::InverseDiagnostics:
            name = "inverse diagnostics";
            break;
        }
        DNDS_check_throw_info(false, std::string("VR operation was not prepared: ") + name);
    }
    ReconstructionStorageReport VariationalReconstructionPlan::StorageReport() const
    {
        Validate();
        if (!_data)
            return PlacementStorageReport();
        ReconstructionStorageReport result;
        std::vector<uint64_t> identities;
        const auto add = [&](const char *name, const auto &array)
        {
            using Value = typename std::decay_t<decltype(array)>::value_type;
            const uint64_t payload = uint64_t(array.RowCount()) * array.Layout().Width() * sizeof(Value);
            uint64_t allocation = 0;
            std::vector<uint64_t> tableIdentities;
            array.SharedStorage()->VisitAllocations([&](uint64_t identity, size_t bytes, auto)
                                                    {
                if (std::find(tableIdentities.begin(), tableIdentities.end(), identity) == tableIdentities.end())
                {
                    tableIdentities.push_back(identity);
                    allocation += bytes;
                }
                if (std::find(identities.begin(), identities.end(), identity) == identities.end())
                {
                    identities.push_back(identity);
                    result.retainedAllocationBytes += bytes;
                } });
            result.tables.push_back({name, true, payload, allocation});
            result.retainedPayloadBytes += payload;
        };
        const auto optional = [&](const char *name, const auto &array)
        {
            if (array)
                add(name, *array);
            else
                result.tables.push_back({name, false, 0, 0});
        };
#define DNDS_REC_STORAGE(name) add(#name, _data->name)
        DNDS_REC_STORAGE(moments);
        if (_data->settings.cacheDiffBase)
            DNDS_REC_STORAGE(faceBasis);
        else
            result.tables.push_back({"faceBasis", false, 0, 0});
        DNDS_REC_STORAGE(offsets);
        DNDS_REC_STORAGE(neighbors);
        DNDS_REC_STORAGE(faces);
        DNDS_REC_STORAGE(boundaryOffsets);
        DNDS_REC_STORAGE(neighborRotation);
        DNDS_REC_STORAGE(neighborRotationActive);
        DNDS_REC_STORAGE(boundaryPoints);
#undef DNDS_REC_STORAGE
#define DNDS_REC_OPTIONAL_STORAGE(name) optional(#name, _data->name)
        DNDS_REC_OPTIONAL_STORAGE(matrixAInvB);
        DNDS_REC_OPTIONAL_STORAGE(vectorAInvB);
        DNDS_REC_OPTIONAL_STORAGE(boundaryInjection);
        DNDS_REC_OPTIONAL_STORAGE(secondary);
        DNDS_REC_OPTIONAL_STORAGE(localMatrix);
        DNDS_REC_OPTIONAL_STORAGE(localInverse);
        DNDS_REC_OPTIONAL_STORAGE(rawMatrixB);
        DNDS_REC_OPTIONAL_STORAGE(rawVectorB);
        DNDS_REC_OPTIONAL_STORAGE(rawBoundaryInjection);
        DNDS_REC_OPTIONAL_STORAGE(inverseConditions);
#undef DNDS_REC_OPTIONAL_STORAGE
        result.preparationRuntimeScratchBytes = _data->preparationRuntimeScratchBytes;
        result.preparationRuntimePeakBytes = result.retainedAllocationBytes + result.preparationRuntimeScratchBytes;
        return result;
    }
    ReconstructionInverseDiagnostics VariationalReconstructionPlan::GetInverseDiagnostics() const
    {
        Require(ReconstructionOperation::InverseDiagnostics);
        Validate();
        return _data ? _data->inverseDiagnostics : PlacementInverseDiagnostics();
    }
    HardEigen::SVDInverseDiagnostics VariationalReconstructionPlan::GetCellInverseDiagnostics(index iCell) const
    {
        Require(ReconstructionOperation::InverseDiagnostics);
        Validate();
        DNDS_check_throw_info(_data, "per-cell inverse diagnostics require retained host payloads");
        DNDS_check_throw_info(iCell >= 0 && iCell < Metrics().NumCell(), "inverse diagnostic cell is not owned");
        auto read = _data->inverseConditions->Read();
        const auto values = read.View<Runtime::FixedRows<4>>();
        return {_data->n, Eigen::Index(values(iCell, 0)), values(iCell, 3), values(iCell, 1), values(iCell, 2)};
    }
    void VariationalReconstructionPlan::Validate() const
    {
        DNDS_check_throw_info(_identity, "reconstruction plan is empty");
        if (_data)
            Metrics().Validate();
        if (_device)
            ValidatePlacement();
    }
    ReconstructionWorkspace VariationalReconstructionPlan::BuildWorkspace(index components, std::vector<index> periodicVectorStarts) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n, "invalid reconstruction component count");
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        if (_device)
            return BuildDeviceWorkspace(components, R::RowLayout::Dense(PlacementOwnedRows(), n * components),
                                        std::move(periodicVectorStarts));
        return {Metrics().BuildField(n * components),
                R::Array<real>(R::RowLayout::Dense(_data->capabilities.wbap ? 6 * _data->hostThreads : 0, n * components)), components, _data, std::move(periodicVectorStarts)};
    }
    ReconstructionWorkspace VariationalReconstructionPlan::BuildWorkspace(index components, R::RowLayout prototype,
                                                                          std::vector<index> periodicVectorStarts) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n && !prototype.IsVariable() &&
                                  prototype.Width() == n * components,
                              "reconstruction physical prototype width mismatch");
        Detail::ValidateVectorBlocks(components, periodicVectorStarts);
        if (_device)
            return BuildDeviceWorkspace(components, std::move(prototype), std::move(periodicVectorStarts));
        return {Metrics().BuildField(prototype),
                R::Array<real>(R::RowLayout::Dense(_data->capabilities.wbap ? 6 * _data->hostThreads : 0, n * components)), components, _data, std::move(periodicVectorStarts)};
    }
    FVField VariationalReconstructionPlan::BuildCoefficients(index components) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n, "invalid reconstruction component count");
        if (_device)
            return BuildDeviceField(R::RowLayout::Dense(PlacementOwnedRows(), n * components));
        return Metrics().BuildField(R::RowLayout::Dense(Metrics().NumCell(), n * components), Device());
    }
    FVField VariationalReconstructionPlan::BuildCoefficients(index components, R::RowLayout prototype) const
    {
        Validate();
        const auto n = NRecDOF();
        DNDS_check_throw_info(components > 0 && components <= INT_MAX / n && !prototype.IsVariable() &&
                                  prototype.Width() == n * components,
                              "reconstruction physical prototype width mismatch");
        if (_device)
            return BuildDeviceField(std::move(prototype));
        return Metrics().BuildField(std::move(prototype), Device());
    }
    ReconstructionBoundaryValues VariationalReconstructionPlan::BuildBoundaryValues(index components) const
    {
        Validate();
        DNDS_check_throw_info(components > 0, "boundary value component count must be positive");
        return {R::Array<real>(R::RowLayout::Dense(NumBoundaryQuadrature(), components), Device()), _identity};
    }
    index VariationalReconstructionPlan::NumBoundaryQuadrature() const
    {
        return _data ? _data->boundaryPoints.RowCount() : PlacementBoundaryPoints();
    }
    BoundaryQuadraturePoint VariationalReconstructionPlan::GetBoundaryQuadrature(index iPoint) const
    {
        Validate();
        DNDS_check_throw_info(!_device, "boundary quadrature geometry is a host setup query");
        auto indices = _data->boundaryPoints.Read();
        const auto entry = indices.View<R::FixedRows<4>, true>()[iPoint];
        auto geometry = Metrics()._data->faceQuadrature.Read();
        const auto values = geometry.View<R::FixedRows<7>>()[entry[3]];
        return {entry[0], entry[1], int(entry[2]), {values[0], values[1], values[2]}, {values[3], values[4], values[5]}};
    }
    FaceCharacteristicTransform VariationalReconstructionPlan::BuildFaceTransform(index first, index count) const
    {
        Require(ReconstructionOperation::WBAP);
        Validate();
        DNDS_check_throw_info(!_device,
                              "build mutable characteristic transforms on the host plan, then place them on the selected backend");
        DNDS_check_throw_info(first >= 0 && count >= 0 && count <= 64, "invalid characteristic component range");
        if (count == 0)
            return FaceCharacteristicTransform{first, 0, {}, {}, _identity};
        FaceCharacteristicTransform result{first, count,
                                           R::Array<real>(R::RowLayout::Dense(Metrics().NumFaceProc(), count * count)),
                                           R::Array<real>(R::RowLayout::Dense(Metrics().NumFaceProc(), count * count)), _identity};
        for (auto *array : {&result.forward, &result.inverse})
        {
            auto write = array->Write();
            auto view = write.View<R::TableRows>();
            for (index iFace = 0; iFace < view.RowCount(); ++iFace)
                Map(view[iFace].data(), count, count).setIdentity();
        }
        FinalizeFaceTransform(result);
        return result;
    }
    void VariationalReconstructionPlan::FinalizeFaceTransform(FaceCharacteristicTransform &transform, bool checkInverses) const
    {
        Require(ReconstructionOperation::WBAP);
        Validate();
        DNDS_check_throw_info(!_device, "finalize characteristic transforms before backend placement");
        DNDS_check_throw_info(transform.validation == CharacteristicValidation::Frozen, "queued characteristics do not use host version finalization");
        transform.forwardVersion = transform.inverseVersion = 0;
        DNDS_check_throw_info(transform.plan == _identity && transform.first >= 0 && transform.count >= 0 && transform.count <= 64,
                              "characteristic transform belongs to another plan or has an invalid range");
        const index count = transform.count;
        if (count == 0)
        {
            transform.forwardVersion = transform.inverseVersion = 0;
            transform.forwardIdentity = transform.inverseIdentity = 0;
            return;
        }
        CheckArray(transform.forward, Metrics().NumFaceProc(), count * count);
        CheckArray(transform.inverse, Metrics().NumFaceProc(), count * count);
        if (checkInverses)
        {
            auto forward = transform.forward.Read(), inverse = transform.inverse.Read();
            const auto left = forward.View<R::TableRows>(), right = inverse.View<R::TableRows>();
            Matrix product(count, count);
            for (index iFace = 0; iFace < Metrics().NumFaceProc(); ++iFace)
            {
                const ConstMap L(left[iFace].data(), count, count), R(right[iFace].data(), count, count);
                product.noalias() = L * R;
                DNDS_check_throw_info(L.allFinite() && R.allFinite() && product.isApprox(Matrix::Identity(count, count), 1e-10),
                                      "face characteristic matrices must be finite mutual inverses");
            }
        }
        transform.forwardVersion = transform.forward.Version();
        transform.inverseVersion = transform.inverse.Version();
        transform.forwardIdentity = transform.forward.Identity();
        transform.inverseIdentity = transform.inverse.Identity();
    }
    void VariationalReconstructionPlan::DoReconstructionIterTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                                               const ReconstructionBoundaryValues &boundary, ReconstructionRegion region) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceReconstructionIterTo(engine, stream, mean, input, work, boundary, region,
                                             ReconstructionDeviceKernel::FP64);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        DoReconstructionIterImpl(mean, input, work, boundary, region, nullptr, nullptr);
    }
    void VariationalReconstructionPlan::DoReconstructionIterLimitedTo(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                                                      const ReconstructionBoundaryValues &boundary, const FVField &target,
                                                                      const R::Array<real> &alpha, ReconstructionRegion region) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceReconstructionIterLimitedTo(engine, stream, mean, input, work, boundary, target, alpha, region);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        DoReconstructionIterImpl(mean, input, work, boundary, region, &target, &alpha);
    }
    void VariationalReconstructionPlan::SubmitReconstructionIterTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean, const FVField &input,
        ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary, ReconstructionRegion region,
        ReconstructionDeviceKernel kernel) const
    {
        if (_device)
        {
            SubmitDeviceReconstructionIterTo(engine, stream, mean, input, work, boundary, region, kernel);
            return;
        }
        Require(ReconstructionOperation::Sweep);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU reconstruction submission needs a matching host stream");
        std::vector<R::ScheduledBufferAccess> accesses;
        const auto add = [&](const auto &array, R::AccessMode mode)
        { AddScheduledAccess(accesses, array, mode); };
        add(mean.Owned(), R::AccessMode::Read);
        add(input.Owned(), R::AccessMode::Read);
        if (region != ReconstructionRegion::Core)
        {
            add(mean.Ghost(), R::AccessMode::Read);
            add(input.Ghost(), R::AccessMode::Read);
        }
        add(boundary.values, R::AccessMode::Read);
        add(work.temporary.Owned(), R::AccessMode::Write);
        add(_data->offsets, R::AccessMode::Read);
        add(_data->neighbors, R::AccessMode::Read);
        add(_data->boundaryOffsets, R::AccessMode::Read);
        add(_data->neighborRotation, R::AccessMode::Read);
        add(_data->neighborRotationActive, R::AccessMode::Read);
        for (const auto *array : {&*_data->matrixAInvB, &*_data->vectorAInvB, &*_data->boundaryInjection})
            add(*array, R::AccessMode::Read);
        auto plan = *this;
        auto retainedMean = mean;
        auto retainedInput = input;
        auto retainedWork = work;
        auto retainedBoundary = boundary;
        engine.SubmitOperation(
            stream,
            [plan = std::move(plan), retainedMean = std::move(retainedMean), retainedInput = std::move(retainedInput),
             retainedWork = std::move(retainedWork), retainedBoundary = std::move(retainedBoundary), region](R::ExecutionContext) mutable
            { plan.DoReconstructionIterTo(retainedMean, retainedInput, retainedWork, retainedBoundary, region); },
            [](std::exception_ptr) noexcept {}, std::move(accesses));
    }
    void VariationalReconstructionPlan::SubmitReconstructionIterLimitedTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &mean, const FVField &input,
        ReconstructionWorkspace &work, const ReconstructionBoundaryValues &boundary, const FVField &target,
        const R::Array<real> &alpha, ReconstructionRegion region) const
    {
        if (_device)
        {
            SubmitDeviceReconstructionIterLimitedTo(engine, stream, mean, input, work, boundary, target, alpha, region);
            return;
        }
        Require(ReconstructionOperation::Sweep);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU limited reconstruction submission needs a matching host stream");
        std::vector<R::ScheduledBufferAccess> accesses;
        const auto add = [&](const auto &array, R::AccessMode mode)
        { AddScheduledAccess(accesses, array, mode); };
        add(mean.Owned(), R::AccessMode::Read);
        add(input.Owned(), R::AccessMode::Read);
        if (region != ReconstructionRegion::Core)
        {
            add(mean.Ghost(), R::AccessMode::Read);
            add(input.Ghost(), R::AccessMode::Read);
        }
        add(boundary.values, R::AccessMode::Read);
        add(target.Owned(), R::AccessMode::Read);
        add(alpha, R::AccessMode::Read);
        add(work.temporary.Owned(), R::AccessMode::Write);
        add(_data->offsets, R::AccessMode::Read);
        add(_data->neighbors, R::AccessMode::Read);
        add(_data->boundaryOffsets, R::AccessMode::Read);
        add(_data->neighborRotation, R::AccessMode::Read);
        add(_data->neighborRotationActive, R::AccessMode::Read);
        for (const auto *array : {&*_data->matrixAInvB, &*_data->vectorAInvB, &*_data->boundaryInjection})
            add(*array, R::AccessMode::Read);
        auto plan = *this;
        auto retainedMean = mean;
        auto retainedInput = input;
        auto retainedWork = work;
        auto retainedBoundary = boundary;
        auto retainedTarget = target;
        auto retainedAlpha = alpha;
        engine.SubmitOperation(
            stream,
            [plan = std::move(plan), retainedMean = std::move(retainedMean), retainedInput = std::move(retainedInput),
             retainedWork = std::move(retainedWork), retainedBoundary = std::move(retainedBoundary),
             retainedTarget = std::move(retainedTarget), retainedAlpha = std::move(retainedAlpha), region](R::ExecutionContext) mutable
            { plan.DoReconstructionIterLimitedTo(retainedMean, retainedInput, retainedWork, retainedBoundary,
                                                 retainedTarget, retainedAlpha, region); },
            [](std::exception_ptr) noexcept {}, std::move(accesses));
    }
    void VariationalReconstructionPlan::DoReconstructionIterImpl(const FVField &mean, const FVField &input, ReconstructionWorkspace &work,
                                                                 const ReconstructionBoundaryValues &boundary, ReconstructionRegion region,
                                                                 const FVField *target, const R::Array<real> *alpha) const
    {
        Require(ReconstructionOperation::Sweep);
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        CheckWorkspace(_data.get(), work, input);
        const index nv = work.components, n = _data->n;
        Detail::ValidateVectorBlocks(nv, work.periodicVectorStarts);
        Metrics().ValidateField(mean, nv);
        DNDS_check_throw_info(boundary.plan == _identity, "boundary quadrature values belong to another plan");
        CheckArray(boundary.values, NumBoundaryQuadrature(), nv);
        // All leases and layout projection happen before the numerical loops.
        Metrics().ValidateField(input, n * nv);
        _data->workPartition->Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "invalid reconstruction region");
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(input.Owned().Storage(), work.temporary.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(input.Ghost().Storage(), work.temporary.Owned().Storage()),
                              "reconstruction output aliases input");
        for (const auto *source : {&mean.Owned(), &mean.Ghost(), &boundary.values})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), work.temporary.Owned().Storage()),
                                  "reconstruction output aliases mean or boundary input");
        std::optional<R::Array<real>::ReadLease> targetRead, alphaRead;
        R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows> targetValues, penalty;
        if (target)
        {
            Metrics().ValidateField(*target, n * nv);
            CheckArray(*alpha, Metrics().NumCell(), 1);
            for (const auto *source : {&target->Owned(), &target->Ghost(), alpha})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), work.temporary.Owned().Storage()),
                                      "reconstruction output aliases penalty input");
            targetRead.emplace(target->Owned().Read());
            alphaRead.emplace(alpha->Read());
            targetValues = targetRead->View<R::UniformTableRows>();
            penalty = alphaRead->View<R::UniformTableRows>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                DNDS_check_throw_info(std::isfinite(penalty(iCell, 0)) && penalty(iCell, 0) >= 0 && penalty(iCell, 0) <= 1,
                                      "reconstruction penalty alpha must be finite and in [0,1]");
        }
        auto meanOwned = mean.Owned().Read();
        std::optional<R::Array<real>::ReadLease> meanGhost, recGhost;
        if (region != ReconstructionRegion::Core)
        {
            meanGhost.emplace(mean.Ghost().Read());
            recGhost.emplace(input.Ghost().Read());
        }
        auto recOwned = input.Owned().Read();
        const auto current = recOwned.View<R::UniformTableRows>();
        auto nextWrite = work.temporary.Owned().Write();
        auto next = nextWrite.View<R::UniformTableRows>();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{meanOwned.View<R::UniformTableRows>(), meanGhost ? meanGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        const auto ghost = recGhost ? recGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{};
        auto bRead = boundary.values.Read();
        const auto bValues = bRead.View<R::TableRows>();
        auto oRead = _data->offsets.Read(), nbRead = _data->neighbors.Read(), boRead = _data->boundaryOffsets.Read();
        auto mRead = _data->matrixAInvB->Read(), vRead = _data->vectorAInvB->Read(), injectionRead = _data->boundaryInjection->Read();
        const auto offsets = oRead.View<R::FixedRows<1>>(), neighbors = nbRead.View<R::FixedRows<1>>(), bOffsets = boRead.View<R::FixedRows<1>>();
        const auto matrices = mRead.View<R::TableRows>(), vectors = vRead.View<R::TableRows>(), injection = injectionRead.View<R::TableRows>();
        const real relax = _data->settings.jacobiRelax;
        auto rotationRead = _data->neighborRotation.Read();
        auto rotationActiveRead = _data->neighborRotationActive.Read();
        const auto rotations = rotationRead.View<R::FixedRows<9>>();
        const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
        const auto nextMatrices = Algebra::MakeEigenRowMatrices(next, n, nv);
        const auto currentMatrices = Algebra::MakeEigenRowMatrices(current, n, nv);
        std::optional<decltype(Algebra::MakeEigenRowMatrices(current, n, nv))> targetMatrices;
        if (target)
            targetMatrices.emplace(Algebra::MakeEigenRowMatrices(targetValues, n, nv));
        const R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows> nextRead{
            next.data, next.offsets, next.rows, next.width, next.stride, next.sizes, next.kind, next.device};
        const auto visitedMatrices = Algebra::MakeEigenRowMatrices(nextRead, n, nv);
        const auto ghostMatrices = Algebra::MakeEigenRowMatrices(ghost, n, nv);
        const bool coreFirst = region != ReconstructionRegion::All || Settings().coreFirst;
        const auto position = [&](index iCell)
        { return coreFirst ? _data->coreFirstPosition[iCell] : iCell; };
        auto compute = [&](index iCell, index partBegin, index partEnd)
        {
            Math::SmallMatrix rotatedRec;
            auto output = nextMatrices[iCell];
            output = currentMatrices[iCell] * (1 - relax);
            const real alphaCell = target ? penalty(iCell, 0) : 0;
            const real scale = relax * (1 - alphaCell);
            if (alphaCell > 0)
                output.noalias() += (relax * alphaCell) * (*targetMatrices)[iCell];
            if (scale == 0)
                return;
            for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
            {
                const index other = neighbors(iEntry, 0);
                if (other != UnInitIndex)
                {
                    const bool rotatedEntry = _data->hasRotatedPeriodicity && rotationActive(iEntry, 0);
                    // SOR reads completed predecessors from a separate output,
                    // leaving the whole input immutable for direct MPI sends.
                    const auto otherCoefficients = other < Metrics().NumCell()
                                                       ? (Settings().SORInstead && other >= partBegin && other < partEnd && position(other) < position(iCell) ? visitedMatrices[other] : currentMatrices[other])
                                                       : ghostMatrices[other - Metrics().NumCell()];
                    // Component rotation is independent of spatial derivative
                    // pullback, already included in the prepared B block.
                    // Apply scalar columns directly, then correct vector blocks.
                    output.noalias() += scale * ConstMap(matrices[iEntry].data(), n, n) * otherCoefficients;
                    if (rotatedEntry)
                        for (auto first : work.periodicVectorStarts)
                        {
                            const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                            rotatedRec = otherCoefficients.middleCols(first, 3) * (rotation.transpose() - Geom::tGPoint::Identity());
                            output.middleCols(first, 3).noalias() += scale * ConstMap(matrices[iEntry].data(), n, n) * rotatedRec;
                        }
                    for (index variable = 0; variable < nv; ++variable)
                    {
                        real value = u(other, variable);
                        if (rotatedEntry)
                            for (auto first : work.periodicVectorStarts)
                                if (variable >= first && variable < first + 3)
                                {
                                    const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                                    value = 0;
                                    for (int d = 0; d < 3; ++d)
                                        value += rotation(variable - first, d) * u(other, first + d);
                                }
                        output.col(variable).noalias() += Eigen::Map<const Eigen::VectorXd>(vectors[iEntry].data(), n) * (scale * (value - u(iCell, variable)));
                    }
                }
                else
                    for (index iQ = bOffsets(iEntry, 0); iQ < bOffsets(iEntry + 1, 0); ++iQ)
                        for (index variable = 0; variable < nv; ++variable)
                            output.col(variable).noalias() += Eigen::Map<const Eigen::VectorXd>(injection[iQ].data(), n) * (scale * (bValues(iQ, variable) - u(iCell, variable)));
            }
        };
        const index owned = Metrics().NumCell();
        if (!Settings().SORInstead)
        {
            if (region == ReconstructionRegion::All)
                OMP::ForRows(0, owned, [&](index iCell)
                             { compute(iCell, 0, owned); });
            else
            {
                const auto &rows = region == ReconstructionRegion::Core ? _data->workPartition->Core() : _data->workPartition->HaloDependent();
                OMP::ForRows(0, index(rows.size()), [&](index iRow)
                             { compute(rows[iRow], 0, owned); });
            }
        }
        else
        {
            // OMP=1 is the canonical serial sweep. With multiple threads, each
            // mesh subpartition is one ordered block; cross-block reads remain
            // frozen input. A split call preserves core-first logical ordering.
            const auto sweep = [&](index begin, index end)
            {
                const index coreCount = index(_data->workPartition->Core().size());
                const auto run = [&](bool core)
                {
                    for (index iCell = begin; iCell < end; ++iCell)
                        if ((_data->coreFirstPosition[iCell] < coreCount) == core)
                            compute(iCell, begin, end);
                };
                if (!coreFirst)
                    for (index iCell = begin; iCell < end; ++iCell)
                        compute(iCell, begin, end);
                else
                {
                    if (region != ReconstructionRegion::HaloDependent)
                        run(true);
                    if (region != ReconstructionRegion::Core)
                        run(false);
                }
            };
            if (OMP::DistributedThreads() == 1)
                sweep(0, owned);
            else
                OMP::ForRows(0, index(_data->hostPartitionStarts.size()) - 1, [&](index part)
                             { sweep(_data->hostPartitionStarts[part], _data->hostPartitionStarts[part + 1]); });
        }
    }
    void VariationalReconstructionPlan::DoReconstructionIter(const FVField &mean, FVField &coefficients,
                                                             ReconstructionWorkspace &work,
                                                             const ReconstructionBoundaryValues &boundary,
                                                             ReconstructionDeviceKernel kernel) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceReconstructionIterTo(engine, stream, mean, coefficients, work, boundary,
                                             ReconstructionRegion::All, kernel);
            SubmitDeviceCommitReconstruction(engine, stream, coefficients, work);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        DoReconstructionIterTo(mean, coefficients, work, boundary);
        CommitReconstruction(coefficients, work);
    }
    void VariationalReconstructionPlan::CommitReconstruction(FVField &coefficients, ReconstructionWorkspace &work) const
    {
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceCommitReconstruction(engine, stream, coefficients, work);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        CheckWorkspace(_data.get(), work, coefficients);
        auto source = work.temporary.Owned().Read();
        auto destination = coefficients.Owned().Write();
        const auto src = source.View<R::UniformTableRows>();
        const auto dst = destination.View<R::UniformTableRows>();
        OMP::ForRows(0, Metrics().NumCell(), [&](index iCell)
                     { std::copy(src[iCell].begin(), src[iCell].end(), dst[iCell].begin()); });
    }
    void VariationalReconstructionPlan::SubmitCommitReconstruction(
        R::EagerEngine &engine, const R::ExecutionStream &stream, FVField &coefficients,
        ReconstructionWorkspace &work) const
    {
        if (_device)
        {
            SubmitDeviceCommitReconstruction(engine, stream, coefficients, work);
            return;
        }
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU reconstruction commit needs a matching host stream");
        std::vector<R::ScheduledBufferAccess> accesses;
        AddScheduledAccess(accesses, work.temporary.Owned(), R::AccessMode::Read);
        AddScheduledAccess(accesses, coefficients.Owned(), R::AccessMode::Write);
        auto plan = *this;
        auto retainedWork = work;
        auto retainedCoefficients = coefficients;
        engine.SubmitOperation(
            stream,
            [plan = std::move(plan), retainedCoefficients = std::move(retainedCoefficients),
             retainedWork = std::move(retainedWork)](R::ExecutionContext) mutable
            { plan.CommitReconstruction(retainedCoefficients, retainedWork); },
            [](std::exception_ptr) noexcept {}, std::move(accesses));
    }
    void VariationalReconstructionPlan::DoLimiterWBAP_3OrderTo(const FVField &coefficients,
                                                               ReconstructionWorkspace &work, int order,
                                                               FaceCharacteristicTransformRef transform, ReconstructionRegion region,
                                                               LimiterIndicatorRef indicator, real threshold) const
    {
        const auto *indicatorArray = indicator ? &indicator->get() : nullptr;
        if (_device)
        {
            R::EagerEngine engine;
            const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
            SubmitDeviceLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, transform, region, indicatorArray, threshold);
            engine.Synchronize();
            CheckStatus(work);
            return;
        }
        Require(ReconstructionOperation::WBAP);
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        Validate();
        CheckWorkspace(_data.get(), work, coefficients);
        DNDS_check_throw_info(order >= 1 && order <= Settings().maxOrder, "invalid WBAP polynomial order");
        const index n = NRecDOF(), nv = work.components;
        Detail::ValidateVectorBlocks(nv, work.periodicVectorStarts);
        _data->workPartition->Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "invalid limiter region");
        DNDS_check_throw_info(std::isfinite(threshold), "limiter threshold must be finite");
        std::optional<R::Array<real>::ReadLease> indicatorRead;
        R::ArrayView<const real, false, R::FixedRows<1>> activation;
        if (indicatorArray)
        {
            CheckArray(*indicatorArray, Metrics().NumCell(), 1);
            indicatorRead.emplace(indicatorArray->Read());
            activation = indicatorRead->View<R::FixedRows<1>>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                DNDS_check_throw_info(std::isfinite(activation(iCell, 0)), "limiter indicator must be finite");
        }
        const auto *selectedTransform = transform && transform->get().count ? &transform->get() : nullptr;
        if (transform && !selectedTransform)
            DNDS_check_throw_info(transform->get().plan == _identity && transform->get().first >= 0 && transform->get().first <= nv,
                                  "disabled face characteristic transform belongs to another plan or has an invalid offset");
        if (selectedTransform)
        {
            DNDS_check_throw_info(selectedTransform->plan == _identity && selectedTransform->first >= 0 && selectedTransform->count > 0 &&
                                      selectedTransform->first <= nv && selectedTransform->count <= nv - selectedTransform->first,
                                  "invalid face characteristic transform range/plan");
            CheckArray(selectedTransform->forward, Metrics().NumFaceProc(), selectedTransform->count * selectedTransform->count);
            CheckArray(selectedTransform->inverse, Metrics().NumFaceProc(), selectedTransform->count * selectedTransform->count);
            DNDS_check_throw_info(selectedTransform->forwardVersion == selectedTransform->forward.Version() &&
                                      selectedTransform->inverseVersion == selectedTransform->inverse.Version() && selectedTransform->forwardVersion != 0,
                                  "face characteristic matrices changed; finalize them before limiting");
            DNDS_check_throw_info(selectedTransform->forwardIdentity == selectedTransform->forward.Identity() && selectedTransform->inverseIdentity == selectedTransform->inverse.Identity(),
                                  "face characteristic buffers replaced; finalize them before limiting");
        }
        const int begin = order == 1 ? 0 : Math::PolynomialSize(Metrics().Dimension(), order - 1) - 1;
        const int end = Math::PolynomialSize(Metrics().Dimension(), order) - 1, modes = end - begin;
        {
            auto owned = coefficients.Owned().Read();
            std::optional<R::Array<real>::ReadLease> ghostRead;
            if (region != ReconstructionRegion::Core)
                ghostRead.emplace(coefficients.Ghost().Read());
            R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> rec{owned.View<R::UniformTableRows>(), ghostRead ? ghostRead->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
            auto outputWrite = work.temporary.Owned().Write(), scratchWrite = work.scratch.Write();
            const auto output = outputWrite.View<R::UniformTableRows>();
            const auto scratch = scratchWrite.View<R::TableRows>();
            auto offsetRead = _data->offsets.Read(), neighborRead = _data->neighbors.Read(), faceRead = _data->faces.Read();
            auto secondaryRead = _data->secondary->Read();
            const auto offsets = offsetRead.View<R::FixedRows<1>>(), neighbors = neighborRead.View<R::FixedRows<1>>(), faces = faceRead.View<R::FixedRows<1>>();
            const auto secondary = secondaryRead.View<R::TableRows>();
            auto rotationRead = _data->neighborRotation.Read();
            auto rotationActiveRead = _data->neighborRotationActive.Read();
            const auto rotations = rotationRead.View<R::FixedRows<9>>();
            const auto rotationActive = rotationActiveRead.View<R::FixedRows<1>>();
            std::optional<R::Array<real>::ReadLease> forward, inverse;
            if (selectedTransform)
                forward.emplace(selectedTransform->forward.Read()), inverse.emplace(selectedTransform->inverse.Read());
            const auto forwardView = selectedTransform ? forward->View<R::TableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>{};
            const auto inverseView = selectedTransform ? inverse->View<R::TableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows>{};
            const auto scratchMatrices = Algebra::MakeEigenRowMatrices(scratch, modes, nv);
            const auto outputMatrices = Algebra::MakeEigenRowMatrices(output, n, nv);
            const auto recMatrices = Algebra::MakeEigenRowMatrices(rec, n, nv);
            const real epsilon = std::pow(verySmallReal, .25);
            auto compute = [&](index iCell, index worker)
            {
                auto center = scratchMatrices[6 * worker], other = scratchMatrices[6 * worker + 1], limited = scratchMatrices[6 * worker + 2];
                auto up = scratchMatrices[6 * worker + 3], down = scratchMatrices[6 * worker + 4], valid = scratchMatrices[6 * worker + 5];
                auto out = outputMatrices[iCell];
                out = recMatrices[iCell];
                if (indicatorArray && activation(iCell, 0) < threshold)
                    return;
                up.setConstant(Settings().WBAP_nStd), down.setConstant(Settings().WBAP_nStd), valid.setOnes();
                for (index iEntry = offsets(iCell, 0); iEntry < offsets(iCell + 1, 0); ++iEntry)
                {
                    const index neighbor = neighbors(iEntry, 0);
                    if (neighbor == UnInitIndex)
                        continue;
                    center = out.middleRows(begin, modes);
                    other.noalias() = ConstMap(secondary[iEntry].data(), n, n).middleRows(begin, modes) * recMatrices[neighbor];
                    const auto iFace = faces(iEntry, 0);
                    if (_data->hasRotatedPeriodicity && rotationActive(iEntry, 0))
                        for (auto first : work.periodicVectorStarts)
                        {
                            const Eigen::Map<const Geom::tGPoint> rotation(rotations[iEntry].data());
                            other.middleCols(first, 3) = (other.middleCols(first, 3) * rotation.transpose()).eval();
                        }
                    if (selectedTransform)
                    {
                        const index start = selectedTransform->first, count = selectedTransform->count;
                        const ConstMap left(forwardView[iFace].data(), count, count);
                        // A reusable output scratch avoids Eigen alias temporaries.
                        limited.leftCols(count).noalias() = center.middleCols(start, count) * left.transpose();
                        center.middleCols(start, count) = limited.leftCols(count);
                        limited.leftCols(count).noalias() = other.middleCols(start, count) * left.transpose();
                        other.middleCols(start, count) = limited.leftCols(count);
                    }
                    for (index v = 0; v < nv; ++v)
                        for (int mode = 0; mode < modes; ++mode)
                            limited(mode, v) = Math::WBAPBiway(center(mode, v), other(mode, v));
                    if (selectedTransform)
                    {
                        const index start = selectedTransform->first, count = selectedTransform->count;
                        const ConstMap right(inverseView[iFace].data(), count, count);
                        other.leftCols(count).noalias() = limited.middleCols(start, count) * right.transpose();
                        limited.middleCols(start, count) = other.leftCols(count);
                    }
                    for (index v = 0; v < nv; ++v)
                        for (int mode = 0; mode < modes; ++mode)
                        {
                            const real value = limited(mode, v);
                            const real ratio = out(begin + mode, v) / (std::abs(value) + epsilon) * ((value > 0) - (value < 0));
                            up(mode, v) += ratio * ratio * ratio;
                            down(mode, v) += sqr(sqr(ratio));
                            if (value == 0)
                                valid(mode, v) = 0;
                        }
                }
                for (index v = 0; v < nv; ++v)
                    for (int mode = 0; mode < modes; ++mode)
                        out(begin + mode, v) *= valid(mode, v) * up(mode, v) / down(mode, v);
            };
            const auto &rows = region == ReconstructionRegion::Core ? _data->workPartition->Core() : _data->workPartition->HaloDependent();
            const index count = region == ReconstructionRegion::All ? Metrics().NumCell() : index(rows.size());
            const index workers = _data->hostThreads;
            OMP::ForRows(0, workers, [&](index worker)
                         {
                // Explicit worker slots own six scratch matrices. This remains
                // valid when a nested caller executes the slots serially.
                const index quotient = count / workers, remainder = count % workers;
                const index begin = worker * quotient + std::min(worker, remainder);
                const index end = begin + quotient + (worker < remainder);
                for (index iRow = begin; iRow < end; ++iRow)
                    compute(region == ReconstructionRegion::All ? iRow : rows[iRow], worker); });
        }
    }
    void VariationalReconstructionPlan::SubmitLimiterWBAP_3OrderTo(
        R::EagerEngine &engine, const R::ExecutionStream &stream, const FVField &coefficients,
        ReconstructionWorkspace &work, int order,
        FaceCharacteristicTransformRef transform, ReconstructionRegion region,
        LimiterIndicatorRef indicator, real threshold) const
    {
        const auto *indicatorArray = indicator ? &indicator->get() : nullptr;
        if (_device)
        {
            SubmitDeviceLimiterWBAP_3OrderTo(engine, stream, coefficients, work, order, transform, region, indicatorArray, threshold);
            return;
        }
        Require(ReconstructionOperation::WBAP);
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                              "CPU WBAP submission needs a matching host stream");
        std::vector<R::ScheduledBufferAccess> accesses;
        const auto add = [&](const auto &array, R::AccessMode mode)
        { AddScheduledAccess(accesses, array, mode); };
        add(coefficients.Owned(), R::AccessMode::Read);
        if (region != ReconstructionRegion::Core)
            add(coefficients.Ghost(), R::AccessMode::Read);
        add(work.temporary.Owned(), R::AccessMode::Write);
        add(work.scratch, R::AccessMode::Write);
        add(_data->offsets, R::AccessMode::Read);
        add(_data->neighbors, R::AccessMode::Read);
        add(_data->faces, R::AccessMode::Read);
        add(_data->neighborRotation, R::AccessMode::Read);
        add(_data->neighborRotationActive, R::AccessMode::Read);
        add(*_data->secondary, R::AccessMode::Read);
        std::optional<FaceCharacteristicTransform> retainedTransform;
        if (transform)
        {
            retainedTransform.emplace(transform->get());
            if (retainedTransform->count)
            {
                add(retainedTransform->forward, R::AccessMode::Read);
                add(retainedTransform->inverse, R::AccessMode::Read);
            }
        }
        std::optional<R::Array<real>> retainedIndicator;
        if (indicatorArray)
        {
            retainedIndicator.emplace(*indicatorArray);
            add(*retainedIndicator, R::AccessMode::Read);
        }
        auto plan = *this;
        auto retainedWork = work;
        auto retainedCoefficients = coefficients;
        engine.SubmitOperation(
            stream,
            [plan = std::move(plan), retainedCoefficients = std::move(retainedCoefficients),
             retainedWork = std::move(retainedWork), order,
             retainedTransform = std::move(retainedTransform), region,
             retainedIndicator = std::move(retainedIndicator), threshold](R::ExecutionContext) mutable
            {
                const FaceCharacteristicTransformRef selected = retainedTransform
                                                                    ? FaceCharacteristicTransformRef(std::cref(*retainedTransform))
                                                                    : std::nullopt;
                plan.DoLimiterWBAP_3OrderTo(retainedCoefficients, retainedWork, order, selected, region,
                                            retainedIndicator ? LimiterIndicatorRef(std::cref(*retainedIndicator)) : std::nullopt,
                                            threshold);
            },
            [](std::exception_ptr) noexcept {}, std::move(accesses));
    }
    void VariationalReconstructionPlan::DoLimiterWBAP_3Order(FVField &coefficients,
                                                             ReconstructionWorkspace &work, int order,
                                                             FaceCharacteristicTransformRef transform) const
    {
        DoLimiterWBAP_3OrderTo(coefficients, work, order, transform);
        CommitReconstruction(coefficients, work);
    }
    void VariationalReconstructionPlan::Evaluate(const FVField &mean, const FVField &coefficients,
                                                 const R::Array<index> &cells, const R::Array<real> &points, R::Array<real> &values) const
    {
        EvaluateDerivative(mean, coefficients, cells, points, values, 0);
    }
    void VariationalReconstructionPlan::EvaluateDerivative(const FVField &mean, const FVField &coefficients,
                                                           const R::Array<index> &cells, const R::Array<real> &points,
                                                           R::Array<real> &values, int derivative) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const index nv = mean.Width(), n = NRecDOF();
        DNDS_check_throw_info(derivative >= 0 && derivative <= n, "polynomial derivative ordinal outside basis");
        Metrics().ValidateField(mean, nv), Metrics().ValidateField(coefficients, n * nv);
        CheckArray(points, cells.RowCount(), 3), CheckArray(values, cells.RowCount(), nv);
        DNDS_check_throw_info(cells.GetDevice() == R::Device::Host() && !cells.Layout().IsVariable() && cells.Layout().Width() == 1, "sampling cell index shape");
        auto cellRead = cells.Read();
        const auto ids = cellRead.View<R::FixedRows<1>>();
        for (index i = 0; i < cells.RowCount(); ++i)
            DNDS_check_throw_info(ids(i, 0) >= 0 && ids(i, 0) < Metrics().NumCellProc(), "sampling cell outside prepared closure");
        auto mo = mean.Owned().Read(), mg = mean.Ghost().Read(), co = coefficients.Owned().Read(), cg = coefficients.Ghost().Read();
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> u{mo.View<R::UniformTableRows>(), mg.View<R::UniformTableRows>()}, rec{co.View<R::UniformTableRows>(), cg.View<R::UniformTableRows>()};
        auto pRead = points.Read(), geomRead = Metrics()._data->cell.Read(), momentRead = _data->moments.Read();
        auto output = values.Write();
        const auto p = pRead.View<R::FixedRows<3>>();
        const auto geom = geomRead.View<R::FixedRows<8>>();
        const auto moment = momentRead.View<R::TableRows>();
        const auto out = output.View<R::UniformTableRows>();
        Math::SmallMatrix basis;
        for (index i = 0; i < cells.RowCount(); ++i)
        {
            const auto cell = ids(i, 0);
            Math::PolynomialDerivatives(basis, Metrics().Dimension(), Settings().maxOrder, {p(i, 0), p(i, 1), p(i, 2)},
                                        {geom(cell, 1), geom(cell, 2), geom(cell, 3)}, geom(cell, 7));
            for (index v = 0; v < nv; ++v)
            {
                real value = derivative == 0 ? u(cell, v) : 0;
                for (index mode = 0; mode < n; ++mode)
                    value += (basis(derivative, mode + 1) - (derivative == 0 ? moment(cell, mode + 1) : 0)) * rec(cell, v * n + mode);
                out(i, v) = value;
            }
        }
    }
    Eigen::MatrixXd VariationalReconstructionPlan::GetCachedFaceBasis(index iFace, int side, int iQuadrature) const
    {
        auto meshRead = Metrics()._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        DNDS_check_throw_info(Settings().cacheDiffBase && iFace >= 0 && iFace < Metrics().NumFaceProc() && (side == 0 || side == 1),
                              "basis cache disabled or face/side outside prepared data");
        DNDS_check_throw_info(meshRead.HostView().FaceCells(iFace)[side] != UnInitIndex, "exterior face parent has no basis cache");
        const auto parents = meshRead.HostView().FaceCells(iFace);
        DNDS_check_throw_info(parents[0] < Metrics().NumCell() || (parents[1] != UnInitIndex && parents[1] < Metrics().NumCell()),
                              "face basis outside owned cell stencils was not prepared");
        auto offsets = Metrics()._data->faceOffsets.Read();
        const auto offset = offsets.View<R::FixedRows<1>>();
        const index begin = offset(iFace, 0), end = offset(iFace + 1, 0);
        DNDS_check_throw_info(iQuadrature >= 0 && iQuadrature < end - begin, "basis cache quadrature outside face");
        auto read = _data->faceBasis.Read();
        const auto view = read.View<R::TableRows>();
        return ConstMap(view[2 * (begin + iQuadrature) + side].data(), _data->cachedDerivatives, NRecDOF());
    }
    ReconstructionBasisRead::ReconstructionBasisRead(std::shared_ptr<const Detail::ReconstructionData> data)
        : _data(std::move(data)), _geometry(_data->metrics.AcquireRead()), _moments(_data->moments.Read()),
          _meshRead(std::make_shared<Geom::MeshReadAccess>(_data->metrics._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux())))
    {
        if (_data->settings.cacheDiffBase)
        {
            _basis.emplace(_data->faceBasis.Read());
            _basisView = _basis->View<R::TableRows>();
        }
        // Freeze plain pointers once while the enclosing read scopes pin both
        // metadata and allocations. Face queries acquire no new Array views.
        _momentView = _moments.View<R::TableRows>();
        _evaluation = {_basisView, _momentView, _geometry.HostView(),
                       _meshRead->HostView(), _data->metrics.Dimension(),
                       _data->settings.maxOrder, _data->n,
                       _data->cachedDerivatives};
    }
    ReconstructionBasisHostView ReconstructionBasisRead::HostView() const
    {
        DNDS_check_throw_info(bool(_basis), "face basis caching was disabled");
        return {_basisView, _geometry.HostView().faceOffsets, _data->n, _data->cachedDerivatives};
    }
    Math::ReconstructionBasisView ReconstructionBasisRead::FaceBasis(index face, int side, index point,
                                                                     int derivativeOrder, Math::SmallMatrix &scratch) const
    {
        return CPU::EvaluateFaceBasis(EvaluationView(), face, side, point,
                                      derivativeOrder, scratch);
    }
    const ReconstructionBasisEvaluationView &ReconstructionBasisRead::EvaluationView() const
    {
        return _evaluation;
    }
    ReconstructionBasisRead VariationalReconstructionPlan::AcquireBasisRead() const
    {
        Validate();
        return ReconstructionBasisRead(_data);
    }
}
