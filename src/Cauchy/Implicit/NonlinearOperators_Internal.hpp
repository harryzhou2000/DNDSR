#pragma once

#include "Cauchy/Implicit/NonlinearOperators.hpp"
#include "Cauchy/Implicit/CPU/NonlinearKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationArrayOperands.hpp"
#include "Cauchy/State/MeanStateCheck.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"
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
#endif
        Data(CFV::FiniteVolumePlan, AdmissibilityPolicy, Runtime::ExecutionContext);
        void Check(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &) const;
        void CheckPair(const EquationDofs &, const EquationDofs &) const;
        void CheckScalar(const Runtime::Array<real> &) const;
        template <class Host, class Device>
        NonlinearOperatorDetail::Result Reduce(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                                               index components, NonlinearOperatorDetail::Inputs inputs, NonlinearOperatorDetail::Outputs outputs,
                                               Host host, Device device)
        {
            namespace R = Runtime;
            using namespace NonlinearOperatorDetail;
            Result result;
            if (context.device == R::Device::Host())
            {
                const auto retained = std::make_shared<Result>();
                retained->norms.resize(components + 1);
                if (OMP::DistributedThreads() > 1)
                    retained->workerSums.resize(index(OMP::DistributedThreads()) * (components + 1));
                R::SubmitArrayBatch(engine, stream, [retained, host](R::ExecutionContext, auto input, auto output)
                                    { host(input, output, *retained); }, std::move(inputs), std::move(outputs));
                engine.Synchronize();
                result = std::move(*retained);
            }
            else
            {
#ifdef DNDS_USE_CUDA
                // Summary is serialized by this operator's bound stream and
                // observed before the next summary-producing call. Captured
                // operands, not the facade lifetime, protect in-flight data.
                R::SubmitArrayBatch(engine, stream, [device](R::ExecutionContext where, auto input, auto output, auto scratch, auto value)
                                    { device(where, input, output, scratch, value); }, std::move(inputs), std::move(outputs), std::make_tuple(R::WriteAs<R::FixedRows<1>>(partial), R::WriteAs<R::FixedRows<1>>(summary)));
                engine.Synchronize();
                const auto hostSummary = summary.To(R::Device::Host(), context);
                const auto read = hostSummary.Read();
                const auto value = read.View<R::FixedRows<1>>()(0, 0);
                result.norms.assign(value.norms.begin(), value.norms.begin() + components + 1);
                result.alpha = {value.limited, value.minimum};
                result.frozen = value.frozen;
                result.status = value.status;
#else
                Unavailable{}();
#endif
            }
            RequireNumericalSuccess(result.status, "nonlinear operator");
            return result;
        }
    };
}
