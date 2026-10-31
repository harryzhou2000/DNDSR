#pragma once

#include "RowTransfer.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <numeric>

namespace DNDS::Runtime
{
    /// Collective fixed-record batch routing. Payload is sparse; count discovery
    /// uses O(number of ranks) metadata. Lists name existing source rows, which
    /// may be unsent or sent to several peers. Records travel through canonical
    /// direct transient MPI bindings, with unique receiver slots grouped by
    /// source rank and matching the sender's per-peer list order.
    ///
    /// All ranks, including empty peers, call Build and Exchange in the same
    /// order. Finish CPU producers before Exchange; caller retains the MPI
    /// communicator and drains operations before finalization. Equal counts
    /// cannot certify semantic row correspondence: request keys/epochs belong
    /// to the application protocol. This is not a rank-local mailbox.
    class SparseBatchExchange
    {
        RowTransferPlan _route;
        std::vector<index> _receiveOffsets;
        explicit SparseBatchExchange(RowTransferPlan route, std::vector<index> offsets)
            : _route(std::move(route)), _receiveOffsets(std::move(offsets)) {}

    public:
        static SparseBatchExchange Build(std::shared_ptr<CommunicationContext> context, index sourceRows,
                                         std::vector<std::vector<index>> destinations)
        {
            DNDS_check_throw_info(context, "batch exchange requires a communication context");
            MPIInfo mpi;
            mpi.comm = context->Communicator();
            MPI_Comm_rank(mpi.comm, &mpi.rank);
            MPI_Comm_size(mpi.comm, &mpi.size);
            std::vector<index> sendCounts, receiveCounts, offsets;
            MPI::LocalStage(mpi, [&]
                            {
                DNDS_check_throw_info(sourceRows >= 0 && destinations.size() == size_t(mpi.size), "invalid batch extent or peer count");
                sendCounts.resize(size_t(mpi.size));
                receiveCounts.resize(size_t(mpi.size));
                offsets.assign(size_t(mpi.size) + 1, 0);
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    DNDS_check_throw_info(destinations[peer].size() <= size_t(INT64_MAX), "batch send count overflows local index");
                    sendCounts[peer] = index(destinations[peer].size());
                    for (auto row : destinations[peer])
                        DNDS_check_throw_info(row >= 0 && row < sourceRows, "batch send row outside source");
                } });
            DNDS_check_throw_info(MPI_Alltoall(sendCounts.data(), 1, DNDS_MPI_INDEX, receiveCounts.data(), 1,
                                               DNDS_MPI_INDEX, mpi.comm) == MPI_SUCCESS,
                                  "batch count discovery failed");
            std::vector<std::vector<index>> receive;
            MPI::LocalStage(mpi, [&]
                            {
                receive.resize(size_t(mpi.size));
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    DNDS_check_throw_info(receiveCounts[peer] >= 0 && receiveCounts[peer] <= INT64_MAX - offsets[peer],
                                          "batch receive count overflows local index");
                    offsets[peer + 1] = offsets[peer] + receiveCounts[peer];
                    receive[peer].resize(size_t(receiveCounts[peer]));
                    std::iota(receive[peer].begin(), receive[peer].end(), offsets[peer]);
                } });
            auto route = RowTransferPlan::Known(context, sourceRows, offsets.back(), std::move(destinations), std::move(receive));
            return SparseBatchExchange(std::move(route), std::move(offsets));
        }

        const RowTransferPlan &Route() const { return _route; }
        const std::vector<index> &ReceiveOffsets() const { return _receiveOffsets; }

        /// Synchronous host convenience over transient nonblocking payload
        /// transport. Repeated asynchronous payloads can bind Route() directly.
        /// A fixed record may be a POD element or a table row of components.
        template <class T>
        Array<T> Exchange(const Array<T> &source, int tag = 0) const
        {
            MPIInfo mpi;
            mpi.comm = _route.Communicator();
            MPI_Comm_rank(mpi.comm, &mpi.rank);
            MPI_Comm_size(mpi.comm, &mpi.size);
            Array<T> destination;
            std::optional<ElementSchema> schema;
            index width = 0;
            MPI::LocalStage(mpi, [&]
                            {
                const auto shape = source.Layout();
                DNDS_check_throw_info(source.GetDevice() == Device::Host() && shape.Kind() == RowLayoutKind::Table &&
                                          shape.RowCount() == _route.SourceRows(), "batch exchange needs a host fixed-record table");
                destination = Array<T>(RowLayout::Dense(_route.DestinationRows(), shape.Width()));
                width = shape.Width();
                schema.emplace(ElementSchema::For<T>()); });
            index controls[]{width, -width, index(tag), -index(tag)};
            MPI_Allreduce(MPI_IN_PLACE, controls, 4, DNDS_MPI_INDEX, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(controls[0] == -controls[1] && controls[2] == -controls[3],
                                  "batch record widths or tags differ between ranks");
            PayloadPlan payload(_route, source.Layout(), destination.Layout(), *schema);
            auto binding = payload.Bind(source, destination, TransferTransport::Direct,
                                        {TransferMode::Transient, _route.Context(), tag});
            binding.Begin().Wait();
            return destination;
        }
    };
}
