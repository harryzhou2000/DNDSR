#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "CFV/CPU/DistributedInterfaceSamplingKernels.hpp"

namespace DNDS::CFV
{
    void DistributedInterfaceSamplingPlan::EvaluateDonors(
        const std::optional<DonorInputs> &input,
        DistributedSamplingWorkspace &workspace) const
    {
        Validate();
        DNDS_check_throw_info(workspace.plan == _data,
                              "sampling workspace belongs to another plan");
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
            _data->sampling->Evaluate(input->mean, *input->reconstructed,
                                      workspace.donorValues);
            return;
        }
        const auto cells = _data->cells.Read();
        const auto mean = input->mean.Owned().Read();
        auto outputWrite = workspace.donorValues.Write();
        std::optional<Runtime::Array<real>::ReadLease> pointRead;
        std::optional<Runtime::Array<real>::ReadLease> gradientRead;
        std::optional<FiniteVolumeRead> geometry;
        CPU::DirectDonorSamplingRequest request;
        request.cells = cells.View<Runtime::FixedRows<1>>();
        request.means = mean.View<Runtime::UniformTableRows>();
        request.output = outputWrite.View<Runtime::UniformTableRows>();
        request.gradient = input->kind == DonorReconstruction::Gradient;
        if (request.gradient)
        {
            DNDS_check_throw_info(input->reconstructed,
                                  "gradient donor sampling lacks its gradient field");
            request.dimension = _data->metrics->Dimension();
            _data->metrics->ValidateField(
                *input->reconstructed,
                workspace.donorValues.Layout().Width() * request.dimension);
            pointRead.emplace(_data->points.Read());
            gradientRead.emplace(input->reconstructed->Owned().Read());
            geometry.emplace(_data->metrics->AcquireRead());
            request.points = pointRead->View<Runtime::UniformTableRows>();
            request.gradients =
                gradientRead->View<Runtime::UniformTableRows>();
            request.geometry = geometry->HostView();
        }
        CPU::EvaluateDirectDonors(request);
    }
}
