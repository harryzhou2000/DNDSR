#pragma once

#include "Cauchy/Implicit/NonlinearOperators.hpp"
#include "Cauchy/Implicit/NonlinearNormSum.hpp"
#include "Cauchy/Implicit/CPU/NonlinearKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "Cauchy/State/MeanStateCheck.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
#include "DNDS/Runtime/Execution/EagerCopy.hpp"
#ifdef DNDS_USE_CUDA
#    include "Cauchy/Implicit/CUDA/NonlinearKernels.hpp"
#endif

namespace DNDS::Cauchy::NonlinearOperatorDetail
{
    namespace R = Runtime;
    using namespace EquationArrayOperands;
    struct Result
    {
        std::vector<real> norms;
        std::vector<real> workerSums;
        AlphaStatistics alpha;
        bool frozen = true;
        NumericalStatus status;
    };
    inline R::MetadataBuffer<uint8_t> Selection(SpanView<const uint8_t> values, index width)
    {
        DNDS_check_throw_info(values.empty() || values.size() == width, "nonlinear selection width differs");
        return R::MetadataBuffer<uint8_t>::Generate(width, [&](index j)
                                                    { return uint8_t(values.empty() || values[j]); });
    }
#ifdef DNDS_USE_CUDA
    inline CUDA::NonlinearSelection DeviceSelection(const R::MetadataBuffer<uint8_t> &values)
    {
        DNDS_check_throw_info(values.size() <= 7, "nonlinear device selection exceeds supported width");
        CUDA::NonlinearSelection selection;
        for (index j = 0; j < values.size(); ++j)
            selection.active[j] = values.HostView()[j];
        return selection;
    }
#endif
    /// CPU-only builds retain the public operator but reject CUDA explicitly.
    struct Unavailable
    {
        template <class... Args>
        void operator()(Args &&...) const
        {
            DNDS_check_throw_info(false, "CUDA nonlinear operators are not compiled");
        }
    };
}

namespace DNDS::Cauchy
{
    struct NonlinearOperators::Data
    {
        Runtime::ExecutionContext context;
        CFV::FiniteVolumePlan geometry;
        AdmissibilityPolicy policy;
        MeanStateCheck meanCheck;
        index flowFirst = 0, ransFirst = -1, speciesFirst = -1, independent = 0;
#ifdef DNDS_USE_CUDA
        Runtime::Array<CUDA::NonlinearSummary> partial, summary;
        std::shared_ptr<Runtime::Array<CUDA::NonlinearSummary>> stagedSummary;
#endif
        Data(CFV::FiniteVolumePlan, AdmissibilityPolicy, Runtime::ExecutionContext);
        void Check(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &) const;
        void CheckPair(const EquationDofs &, const EquationDofs &) const;
        void CheckScalar(const Runtime::Array<real> &) const;
        template <class Host, class Device>
        std::function<NonlinearOperatorDetail::Result()> SubmitReduction(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                                                         index components, NonlinearOperatorDetail::Inputs inputs, NonlinearOperatorDetail::Outputs outputs,
                                                                         Host host, Device device, std::shared_ptr<NonlinearNormSum> global = {})
        {
            namespace R = Runtime;
            using namespace NonlinearOperatorDetail;
            if (context.device == R::Device::Host())
            {
                const auto retained = std::make_shared<Result>();
                retained->norms.resize(components + 1);
                if (OMP::DistributedThreads() > 1)
                    retained->workerSums.resize(index(OMP::DistributedThreads()) * (components + 1));
                R::SubmitArrayBatch(engine, stream, [retained, host](R::ExecutionContext, auto input, auto output)
                                    { host(input, output, *retained); }, std::move(inputs), std::move(outputs));
                if (global)
                {
                    R::SubmitArrays(engine, stream, [retained](auto, auto dst)
                                    { std::copy(retained->norms.begin(), retained->norms.end(), dst.Row(0)); }, R::Write(global->Local()));
                    global->Submit(engine, stream);
                    return [retained, global]
                    {
                        retained->norms = global->Observe();
                        RequireNumericalSuccess(retained->status, "nonlinear norm");
                        return std::move(*retained);
                    };
                }
                return [retained, done = stream.Record()]
                {
                    done.WaitHost();
                    RequireNumericalSuccess(retained->status, "nonlinear operator");
                    return std::move(*retained);
                };
            }
#ifdef DNDS_USE_CUDA
            if (global)
            {
                R::SubmitArrayBatch(engine, stream, [device](R::ExecutionContext where, auto input, auto output, auto scratch, auto value)
                                    { device(where, input, output, scratch, value); }, std::move(inputs), std::move(outputs), std::make_tuple(R::WriteAs<R::FixedRows<1>>(partial), R::WriteAs<R::FixedRows<1>>(global->deviceSummary)));
                static_assert(offsetof(CUDA::NonlinearSummary, norms) == 0);
                auto destination = global->Local().Storage();
                DNDS_check_throw_info(destination.Bytes() <= sizeof(CUDA::NonlinearSummary::norms), "nonlinear norm exceeds summary capacity");
                auto source = global->deviceSummary.Storage().Subspan(0, destination.Bytes());
                R::CopyAsync(engine, stream, source, destination);
                global->Submit(engine, stream);
                return [global]
                {
                    Result result;
                    result.norms = global->Observe();
                    auto host = global->deviceSummary.To(R::Device::Host());
                    RequireNumericalSuccess(host.Read().View()(0, 0).status, "nonlinear norm");
                    return result;
                };
            }
            // A pending observation owns its pinned scalar record. When it has
            // been consumed the next iteration can reuse the same allocation.
            if (!stagedSummary || !stagedSummary.unique())
            {
                auto staging = R::Array<CUDA::NonlinearSummary>::FromStorage(summary.Layout(),
                                                                             R::Buffer::AllocatePinnedHost(summary.Storage().Bytes()));
                staging.SetName("nonlinear/reduction/hostSummary", DNDS_ALLOCATION_SITE());
                stagedSummary = std::make_shared<R::Array<CUDA::NonlinearSummary>>(std::move(staging));
            }
            R::SubmitArrayBatch(engine, stream, [device](R::ExecutionContext where, auto input, auto output, auto scratch, auto value)
                                { device(where, input, output, scratch, value); }, std::move(inputs), std::move(outputs), std::make_tuple(R::WriteAs<R::FixedRows<1>>(partial), R::WriteAs<R::FixedRows<1>>(summary)));
            auto destination = stagedSummary->Storage();
            R::CopyAsync(engine, stream, summary.Storage(), destination);
            return [staging = stagedSummary, components, done = stream.Record()]
            {
                done.WaitHost();
                const auto read = staging->Read();
                const auto value = read.View<R::FixedRows<1>>()(0, 0);
                RequireNumericalSuccess(value.status, "nonlinear operator");
                Result result;
                result.norms.assign(value.norms.begin(), value.norms.begin() + components + 1);
                result.alpha = {value.limited, value.minimum};
                result.frozen = value.frozen;
                result.status = value.status;
                return result;
            };
#else
            DNDS_check_throw_info(false, "CUDA nonlinear operators are not compiled");
            return {};
#endif
        }

        template <class Host, class Device>
        NonlinearOperatorDetail::Result Reduce(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                               index components, NonlinearOperatorDetail::Inputs inputs, NonlinearOperatorDetail::Outputs outputs,
                                               Host host, Device device)
        {
            return SubmitReduction(engine, stream, components, std::move(inputs), std::move(outputs), std::move(host), std::move(device))();
        }
    };
}
