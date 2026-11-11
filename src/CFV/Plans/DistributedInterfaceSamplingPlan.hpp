#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "DNDS/Runtime/Communication/SparseBatchExchange.hpp"

namespace DNDS::CFV
{
    enum class DonorReconstruction
    {
        Constant,
        Gradient,
        Variational
    };
    /// Managed donor bundle. Constant reads means, Gradient reads a physical
    /// component*dimension gradient, and Variational reads modal coefficients.
    /// The named constructors prevent confusing two identically typed fields.
    struct DonorInputs
    {
        FVField mean;
        std::optional<FVField> reconstructed;
        DonorReconstruction kind = DonorReconstruction::Constant;
        explicit DonorInputs(const FVField &u) : mean(u) {}
        DonorInputs(const FVField &u, const FVField &rec)
            : mean(u), reconstructed(rec), kind(DonorReconstruction::Variational) {}
        static DonorInputs Gradient(const FVField &u, const FVField &gradient)
        {
            DonorInputs result(u);
            result.reconstructed = gradient;
            result.kind = DonorReconstruction::Gradient;
            return result;
        }
    };

    /// Explicit reusable donor-side output. Private to one active evaluation;
    /// no hidden mutable solver state belongs to the sampling plan.
    struct DistributedSamplingWorkspace
    {
        Runtime::Array<real> donorValues;
        std::shared_ptr<const void> plan;
    };

    /// Collective preparation routes selected query coordinates to the actual
    /// donor-cell owners. Evaluation uses those owners' means/polynomials and
    /// returns samples in receiver query order via canonical direct transfers.
    /// Values remain in the donor reference/component frame. Equation adapters
    /// handle vector/tensor frame conversion; there is no conservative flux.
    /// Collective setup/evaluation uses the correspondence map's immutable
    /// preflight policy. Off-mode local rejection aborts before more MPI; all
    /// ranks still enter count, routing and payload protocols in matching order.
    class DistributedInterfaceSamplingPlan
    {
        struct Data
        {
            std::optional<FiniteVolumePlan> metrics;
            std::optional<VariationalReconstructionPlan> reconstruction;
            Geom::DistributedInterfacePointMap map;
            Runtime::Array<index> cells;
            Runtime::Array<real> points;
            Runtime::RowTransferPlan returnRoute;
            std::optional<PolynomialSamplingPlan> sampling;
            Data(std::optional<FiniteVolumePlan> finiteVolume,
                 std::optional<VariationalReconstructionPlan> rec,
                 Geom::DistributedInterfacePointMap correspondence,
                 Runtime::Array<index> rows, Runtime::Array<real> locations, Runtime::RowTransferPlan route)
                : metrics(std::move(finiteVolume)), reconstruction(std::move(rec)),
                  map(std::move(correspondence)), cells(std::move(rows)),
                  points(std::move(locations)), returnRoute(std::move(route))
            {
                if (reconstruction)
                    sampling.emplace(*reconstruction, cells, points);
            }
        };
        std::shared_ptr<const Data> _data;

    public:
        DistributedInterfaceSamplingPlan(VariationalReconstructionPlan reconstruction, Geom::DistributedInterfacePointMap map)
            : DistributedInterfaceSamplingPlan(
                  std::nullopt, std::make_optional(std::move(reconstruction)),
                  std::move(map)) {}
        /// Optional donor-domain profile on a larger coupling communicator.
        /// Nonmembers carry routes/queries only, never a dummy mesh or field.
        DistributedInterfaceSamplingPlan(std::optional<VariationalReconstructionPlan> reconstruction, Geom::DistributedInterfacePointMap map)
            : DistributedInterfaceSamplingPlan(std::nullopt,
                                               std::move(reconstruction),
                                               std::move(map)) {}
        /// Metrics-only construction supports P0 and physical-gradient P1.
        /// A local donor rank supplies metrics even when it has no VR plan;
        /// nonmembers supply neither while retaining the collective route.
        DistributedInterfaceSamplingPlan(
            std::optional<FiniteVolumePlan> metrics,
            std::optional<VariationalReconstructionPlan> reconstruction,
            Geom::DistributedInterfacePointMap map)
        {
            const auto context = map.Context();
            const MPIInfo mpi(context->Communicator());
            // Do not derive Metrics() in the delegating-constructor argument
            // list: C++17 may move `reconstruction` before evaluating another
            // argument. Resolve it here while the optional still owns its plan.
            if (!metrics && reconstruction)
                metrics.emplace(reconstruction->Metrics());
            std::vector<std::vector<index>> destinations;
            Runtime::Array<index> cells;
            Runtime::Array<real> points;
            std::optional<FiniteVolumeRead> geometryPin;
            context->LocalStage([&]
                                {
                map.Validate();
                DNDS_check_throw_info(bool(metrics) == map.HasLocalDonor(),
                                      "sampling donor-domain participation differs from the geometric map");
                if (metrics)
                {
                    metrics->Validate();
                    geometryPin.emplace(metrics->AcquireRead());
                    DNDS_check_throw_info(
                        map.Donor().Cells().SameOrder(metrics->Cells()),
                        "distributed interface and reconstruction mesh/order differ");
                    if (reconstruction)
                    {
                        reconstruction->Validate();
                        DNDS_check_throw_info(
                            reconstruction->Metrics().Cells().SameOrder(metrics->Cells()),
                            "interface VR and finite-volume metrics differ");
                    }
                }
                destinations.resize(size_t(mpi.size));
                cells = Runtime::Array<index>(Runtime::RowLayout::Dense(map.Size(), 1));
                points = Runtime::Array<real>(Runtime::RowLayout::Dense(map.Size(), 3));
                auto c = cells.Write();
                auto p = points.Write();
                for (index iQuery = 0; iQuery < map.Size(); ++iQuery)
                {
                    const auto match = map.Get(iQuery);
                    DNDS_check_throw_info(match.donorRank >= 0 && match.donorRank < mpi.size, "invalid donor rank");
                    destinations[match.donorRank].push_back(iQuery);
                    c.View()(iQuery, 0) = match.donorOwnedCell;
                    for (int d = 0; d < 3; ++d)
                        p.View()(iQuery, d) = match.geometry.donorPoint[d];
                } });
            const auto route = Runtime::SparseBatchExchange::Build(context, map.Size(), std::move(destinations));
            cells = route.Exchange(cells);
            points = route.Exchange(points);
            context->LocalStage([&]
                                {
                auto read = cells.Read();
                for (index iQuery = 0; iQuery < cells.RowCount(); ++iQuery)
                    DNDS_check_throw_info(
                        metrics && read.View()(iQuery, 0) >= 0 &&
                            read.View()(iQuery, 0) < metrics->NumCell(),
                        "selected donor row is not owned by its destination");
                _data = std::make_shared<Data>(
                    std::move(metrics), std::move(reconstruction), std::move(map),
                    std::move(cells), std::move(points), route.Route().Reverse()); });
        }
        index Size() const { return _data->map.Size(); }
        /// Shared coupling owner for frontend input agreement, not a new namespace.
        std::shared_ptr<Runtime::CommunicationContext> Communication() const { return _data->returnRoute.Context(); }
        void Validate() const
        {
            _data->map.Validate();
            if (_data->metrics)
                _data->metrics->Validate();
            if (_data->reconstruction)
                _data->reconstruction->Validate();
        }
        DistributedSamplingWorkspace BuildWorkspace(index components) const
        {
            Validate();
            DNDS_check_throw_info(components > 0, "sampling component count must be positive");
            return {Runtime::Array<real>(Runtime::RowLayout::Dense(_data->cells.RowCount(), components)), _data};
        }
        /// Rank-local computation only. Callers may bind the result transfer
        /// persistently with BindResult and schedule it explicitly afterward.
        /// Source halos must be ready under the current polynomial evaluator
        /// lease contract. Every rank validates local geometry before transfer.
        void EvaluateDonors(const FVField &mean, const FVField &coefficients, DistributedSamplingWorkspace &workspace) const
        {
            EvaluateDonors(DonorInputs(mean, coefficients), workspace);
        }
        /// An empty bundle denotes nonmembership; owned field handles retain
        /// both inputs throughout synchronous evaluation.
        void EvaluateDonors(const std::optional<DonorInputs> &input, DistributedSamplingWorkspace &workspace) const
        {
            Validate();
            DNDS_check_throw_info(workspace.plan == _data, "sampling workspace belongs to another plan");
            DNDS_check_throw_info(bool(input) == bool(_data->metrics), "sampling inputs differ from donor-domain participation");
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
            const auto cellRows = cells.View<Runtime::FixedRows<1>>();
            const auto source = mean.View<Runtime::UniformTableRows>();
            auto outputWrite = workspace.donorValues.Write();
            const auto output = outputWrite.View<Runtime::UniformTableRows>();
            std::optional<Runtime::Array<real>::ReadLease> pointRead;
            std::optional<Runtime::Array<real>::ReadLease> gradientRead;
            Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                               Runtime::UniformTableRows>
                locations, gradients;
            std::optional<FiniteVolumeRead> geometry;
            if (input->kind == DonorReconstruction::Gradient)
            {
                DNDS_check_throw_info(input->reconstructed,
                                      "gradient donor sampling lacks its gradient field");
                _data->metrics->ValidateField(
                    *input->reconstructed,
                    workspace.donorValues.Layout().Width() *
                        _data->metrics->Dimension());
                pointRead.emplace(_data->points.Read());
                gradientRead.emplace(input->reconstructed->Owned().Read());
                locations = pointRead->View<Runtime::UniformTableRows>();
                gradients = gradientRead->View<Runtime::UniformTableRows>();
                geometry.emplace(_data->metrics->AcquireRead());
            }
            for (index iQuery = 0; iQuery < output.RowCount(); ++iQuery)
            {
                const index iCell = cellRows(iQuery, 0);
                Geom::tPoint center{};
                if (input->kind == DonorReconstruction::Gradient)
                    center = geometry->HostView().GetCellCent(iCell);
                for (index j = 0; j < output.Width(); ++j)
                {
                    real value = source(iCell, j);
                    if (input->kind == DonorReconstruction::Gradient)
                    {
                        for (int d = 0; d < _data->metrics->Dimension(); ++d)
                            value += gradients(iCell, j * _data->metrics->Dimension() + d) *
                                     (locations(iQuery, d) - center[d]);
                    }
                    output(iQuery, j) = value;
                }
            }
        }
        /// Collective setup. Reuse this binding for repeated fixed-buffer
        /// sampling; wait for donors before Begin and before reading receivers.
        Runtime::TransferBinding BindResult(const DistributedSamplingWorkspace &workspace,
                                            Runtime::Array<real> &values, Runtime::TransferMode mode = Runtime::TransferMode::Persistent,
                                            int tag = 0) const
        {
            _data->returnRoute.Context()->LocalStage([&]
                                                     {
                Validate();
                DNDS_check_throw_info(workspace.plan == _data && values.RowCount() == Size() &&
                                          values.GetDevice() == Runtime::Device::Host() && values.Layout().Kind() == Runtime::RowLayoutKind::Table,
                                      "sampling output/workspace mismatch or non-host output"); });
            Runtime::PayloadPlan payload(_data->returnRoute, workspace.donorValues.Layout(), values.Layout(), Runtime::ElementSchema::Arithmetic<real>());
            return payload.Bind(workspace.donorValues, values, Runtime::TransferTransport::Direct,
                                {mode, _data->returnRoute.Context(), tag});
        }
        /// Collective synchronous convenience. Preparation failures agree before
        /// payload communication. The caller owns workspace/output lifetimes and
        /// arranges new maps/plans after motion, deformation or redistribution.
        void Evaluate(const FVField &mean, const FVField &coefficients, DistributedSamplingWorkspace &workspace,
                      Runtime::Array<real> &values) const
        {
            EvaluateCoupled(DonorInputs(mean, coefficients), workspace, values);
        }
        /// Coupling-participant convenience. Pass an empty bundle when this rank
        /// has no donor-domain reconstruction; every rank still enters MPI.
        void EvaluateCoupled(const std::optional<DonorInputs> &input, DistributedSamplingWorkspace &workspace,
                             Runtime::Array<real> &values) const
        {
            _data->returnRoute.Context()->LocalStage([&]
                                                     { EvaluateDonors(input, workspace); });
            auto binding = BindResult(workspace, values, Runtime::TransferMode::Transient);
            binding.Begin().Wait();
        }
    };
}
