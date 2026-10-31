#pragma once

#include "FiniteVolumePlan.hpp"
#include "PolynomialSamplingPlan.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "DNDS/Runtime/SparseBatchExchange.hpp"

namespace DNDS::CFV
{
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
    class DistributedInterfaceSamplingPlan
    {
        struct Data
        {
            VariationalReconstructionPlan reconstruction;
            Geom::DistributedInterfacePointMap map;
            Runtime::Array<index> cells;
            Runtime::Array<real> points;
            Runtime::RowTransferPlan returnRoute;
            PolynomialSamplingPlan sampling;
            Data(VariationalReconstructionPlan rec, Geom::DistributedInterfacePointMap correspondence,
                 Runtime::Array<index> rows, Runtime::Array<real> locations, Runtime::RowTransferPlan route)
                : reconstruction(std::move(rec)), map(std::move(correspondence)), cells(std::move(rows)),
                  points(std::move(locations)), returnRoute(std::move(route)), sampling(reconstruction, cells, points) {}
        };
        std::shared_ptr<const Data> _data;

    public:
        DistributedInterfaceSamplingPlan(VariationalReconstructionPlan reconstruction, Geom::DistributedInterfacePointMap map)
        {
            MPIInfo mpi;
            mpi.comm = reconstruction.Metrics().Communicator();
            MPI_Comm_rank(mpi.comm, &mpi.rank);
            MPI_Comm_size(mpi.comm, &mpi.size);
            auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
            std::vector<std::vector<index>> destinations;
            Runtime::Array<index> cells;
            Runtime::Array<real> points;
            std::optional<FiniteVolumeRead> geometryPin;
            MPI::LocalStage(mpi, [&]
                            {
                map.Validate();
                reconstruction.Validate();
                geometryPin.emplace(reconstruction.Metrics().AcquireRead());
                DNDS_check_throw_info(map.Donor().Cells().SameOrder(reconstruction.Metrics().Cells()),
                                      "distributed interface and reconstruction mesh/order differ");
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
            MPI::LocalStage(mpi, [&]
                            {
                auto read = cells.Read();
                for (index iQuery = 0; iQuery < cells.RowCount(); ++iQuery)
                    DNDS_check_throw_info(read.View()(iQuery, 0) >= 0 && read.View()(iQuery, 0) < reconstruction.Metrics().NumCell(),
                                          "selected donor row is not owned by its destination");
                _data = std::make_shared<Data>(std::move(reconstruction), std::move(map), std::move(cells), std::move(points), route.Route().Reverse()); });
        }
        index Size() const { return _data->map.Size(); }
        void Validate() const
        {
            _data->map.Validate();
            _data->reconstruction.Validate();
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
            Validate();
            DNDS_check_throw_info(workspace.plan == _data, "sampling workspace belongs to another plan");
            _data->sampling.Evaluate(mean, coefficients, workspace.donorValues);
        }
        /// Collective setup. Reuse this binding for repeated fixed-buffer
        /// sampling; wait for donors before Begin and before reading receivers.
        Runtime::TransferBinding BindResult(const DistributedSamplingWorkspace &workspace,
                                            Runtime::Array<real> &values, Runtime::TransferMode mode = Runtime::TransferMode::Persistent,
                                            int tag = 0) const
        {
            MPIInfo mpi;
            mpi.comm = _data->returnRoute.Communicator();
            MPI_Comm_rank(mpi.comm, &mpi.rank);
            MPI_Comm_size(mpi.comm, &mpi.size);
            MPI::LocalStage(mpi, [&]
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
            MPIInfo mpi;
            mpi.comm = _data->returnRoute.Communicator();
            MPI_Comm_rank(mpi.comm, &mpi.rank);
            MPI_Comm_size(mpi.comm, &mpi.size);
            MPI::LocalStage(mpi, [&]
                            { EvaluateDonors(mean, coefficients, workspace); });
            auto binding = BindResult(workspace, values, Runtime::TransferMode::Transient);
            binding.Begin().Wait();
        }
    };
}
