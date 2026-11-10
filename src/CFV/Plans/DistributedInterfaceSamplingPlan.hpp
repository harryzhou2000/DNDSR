#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "DNDS/Runtime/Communication/SparseBatchExchange.hpp"

namespace DNDS::CFV
{
    /// Sharing donor bundle: participation always supplies both fields.
    struct DonorInputs
    {
        FVField mean, coefficients;
        DonorInputs(const FVField &u, const FVField &rec) : mean(u), coefficients(rec) {}
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
            std::optional<VariationalReconstructionPlan> reconstruction;
            Geom::DistributedInterfacePointMap map;
            Runtime::Array<index> cells;
            Runtime::Array<real> points;
            Runtime::RowTransferPlan returnRoute;
            std::optional<PolynomialSamplingPlan> sampling;
            Data(std::optional<VariationalReconstructionPlan> rec, Geom::DistributedInterfacePointMap correspondence,
                 Runtime::Array<index> rows, Runtime::Array<real> locations, Runtime::RowTransferPlan route)
                : reconstruction(std::move(rec)), map(std::move(correspondence)), cells(std::move(rows)),
                  points(std::move(locations)), returnRoute(std::move(route))
            {
                if (reconstruction)
                    sampling.emplace(*reconstruction, cells, points);
            }
        };
        std::shared_ptr<const Data> _data;

    public:
        DistributedInterfaceSamplingPlan(VariationalReconstructionPlan reconstruction, Geom::DistributedInterfacePointMap map)
            : DistributedInterfaceSamplingPlan(std::make_optional(std::move(reconstruction)), std::move(map)) {}
        /// Optional donor-domain profile on a larger coupling communicator.
        /// Nonmembers carry routes/queries only, never a dummy mesh or field.
        DistributedInterfaceSamplingPlan(std::optional<VariationalReconstructionPlan> reconstruction, Geom::DistributedInterfacePointMap map)
        {
            const auto context = map.Context();
            const MPIInfo mpi(context->Communicator());
            std::vector<std::vector<index>> destinations;
            Runtime::Array<index> cells;
            Runtime::Array<real> points;
            std::optional<FiniteVolumeRead> geometryPin;
            context->LocalStage([&]
                                {
                map.Validate();
                DNDS_check_throw_info(bool(reconstruction)==map.HasLocalDonor(),"sampling donor-domain participation differs from the geometric map");
                if(reconstruction)
                {
                    reconstruction->Validate();
                    geometryPin.emplace(reconstruction->Metrics().AcquireRead());
                    DNDS_check_throw_info(map.Donor().Cells().SameOrder(reconstruction->Metrics().Cells()),
                                          "distributed interface and reconstruction mesh/order differ");
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
                    DNDS_check_throw_info(reconstruction && read.View()(iQuery, 0) >= 0 && read.View()(iQuery, 0) < reconstruction->Metrics().NumCell(),
                                          "selected donor row is not owned by its destination");
                _data = std::make_shared<Data>(std::move(reconstruction), std::move(map), std::move(cells), std::move(points), route.Route().Reverse()); });
        }
        index Size() const { return _data->map.Size(); }
        void Validate() const
        {
            _data->map.Validate();
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
            DNDS_check_throw_info(bool(input) == bool(_data->sampling), "sampling inputs differ from donor-domain participation");
            if (_data->sampling)
                _data->sampling->Evaluate(input->mean, input->coefficients, workspace.donorValues);
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
