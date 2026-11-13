#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "CFV/CPU/DistributedInterfaceSamplingKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#ifdef DNDS_USE_CUDA
#    include "CFV/Operators/CUDA/DistributedInterfaceSamplingKernels.hpp"
#endif

namespace DNDS::CFV
{
    namespace R = Runtime;
    DistributedInterfaceSamplingPlan DistributedInterfaceSamplingPlan::Placed(R::Device device, R::ExecutionContext context) const
    {
        Validate();
        DNDS_check_throw_info(context.device == (device == R::Device::Host() ? Device() : device), "sampling placement context differs");
        if (device == Device())
            return *this;
        auto data = std::make_shared<Data>(*_data);
        if (data->metrics)
            data->metrics = data->metrics->Placed(device, context);
        if (data->sampling)
            data->sampling = data->sampling->Placed(device, context);
        data->cells = data->cells.To(device, context);
        data->points = data->points.To(device, context);
        return DistributedInterfaceSamplingPlan(std::move(data));
    }
    void DistributedInterfaceSamplingPlan::EvaluateDonors(
        const std::optional<DonorInputs> &input,
        DistributedSamplingWorkspace &workspace) const
    {
        R::EagerEngine engine;
        R::ExecutionStream stream;
        if (Device() == R::Device::Host())
            stream = engine.CreateCPUStream();
#ifdef DNDS_USE_CUDA
        else
            stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(Device().ordinal));
#else
        else
            DNDS_check_throw_info(false, "requested donor sampling backend is unavailable");
#endif
        EvaluateDonors(engine, stream, input, workspace);
        engine.Close();
    }
    void DistributedInterfaceSamplingPlan::EvaluateDonors(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const std::optional<DonorInputs> &input,
        DistributedSamplingWorkspace &workspace) const
    {
        Validate();
        DNDS_check_throw_info(workspace.plan == _data && stream.Context().device == Device() &&
                                  workspace.donorValues.GetDevice() == Device() &&
                                  workspace.donorValues.RowCount() == _data->cells.RowCount() &&
                                  workspace.donorValues.Layout().Kind() == R::RowLayoutKind::Table,
                              "sampling workspace shape/placement or stream differs from plan");
        DNDS_check_throw_info(bool(input) == bool(_data->metrics),
                              "sampling inputs differ from donor-domain participation");
        if (!input)
            return;
        _data->metrics->ValidateField(input->mean,
                                      workspace.donorValues.Layout().Width());
        DNDS_check_throw_info(
            (input->kind == DonorReconstruction::Constant) ==
                !input->reconstructed,
            "constant donor sampling must not carry reconstructed data");
        if (input->kind == DonorReconstruction::Variational)
        {
            DNDS_check_throw_info(_data->sampling && input->reconstructed,
                                  "variational donor sampling lacks its plan or coefficients");
            _data->sampling->Evaluate(engine, stream, input->mean, *input->reconstructed,
                                      workspace.donorValues);
            return;
        }
        const auto validateInput = [&](const FVField &field)
        {
            for (const auto &array : {field.Owned(), field.Ghost()})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(array.Storage(), workspace.donorValues.Storage()),
                                      "donor sampling output overlaps its input");
            DNDS_check_throw_info(Device() == R::Device::Host() || field.Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "CUDA donor sampling requires AoS fields");
        };
        validateInput(input->mean);
        const auto launch = [](R::ExecutionContext context, const Detail::DirectDonorSamplingView &request)
        {
            if (context.device == R::Device::Host())
                CPU::EvaluateDirectDonors(request);
#ifdef DNDS_USE_CUDA
            else
                CUDA::EvaluateDirectDonors(context, request);
#else
            else
                DNDS_check_throw_info(false, "CUDA donor sampling is not built");
#endif
        };
        // Retain geometry/placement until completion even when no CPU worker
        // owns the operation after launch. Data arrays are submitted operands.
        auto geometry = std::make_shared<FiniteVolumeRead>(_data->metrics->AcquireRead(stream.Context()));
        // The submitted lambda retains geometry through operation retirement;
        // query/value allocations are declared Array operands below.
        if (input->kind == DonorReconstruction::Gradient)
        {
            const int dimension = _data->metrics->Dimension();
            _data->metrics->ValidateField(
                *input->reconstructed,
                workspace.donorValues.Layout().Width() * dimension);
            validateInput(*input->reconstructed);
            R::SubmitArrays(engine, stream, [launch, geometry, dimension](R::ExecutionContext context, auto cells, auto mean, auto points, auto gradient, auto output)
                            {
                Detail::DirectDonorSamplingView request{cells, mean, points, gradient, geometry->View().cells, output, dimension, true};
                launch(context, request); }, R::ReadAs<R::FixedRows<1>>(_data->cells), R::ReadAs<R::UniformTableRows>(input->mean.Owned()), R::ReadAs<R::UniformTableRows>(_data->points), R::ReadAs<R::UniformTableRows>(input->reconstructed->Owned()), R::WriteAs<R::UniformTableRows>(workspace.donorValues));
        }
        else
        {
            DNDS_check_throw_info(input->kind == DonorReconstruction::Constant, "unknown donor reconstruction mode");
            R::SubmitArrays(engine, stream, [launch, geometry](R::ExecutionContext context, auto cells, auto mean, auto output)
                            {
                Detail::DirectDonorSamplingView request;
                request.cells = cells; request.means = mean; request.output = output;
                launch(context, request); }, R::ReadAs<R::FixedRows<1>>(_data->cells), R::ReadAs<R::UniformTableRows>(input->mean.Owned()), R::WriteAs<R::UniformTableRows>(workspace.donorValues));
        }
    }
}
