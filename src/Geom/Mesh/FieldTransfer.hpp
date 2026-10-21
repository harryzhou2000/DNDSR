#pragma once

#include "ReorderPlan.hpp"
#include "DNDS/Runtime/Field.hpp"
#include "DNDS/Runtime/Exchange.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"

namespace DNDS::Geom
{
    /// Extract the exact correspondence once, then share it across native Arrays
    /// (including separate SoA components). No geometry or interpolation in DNDS.
    inline Runtime::RowTransferPlan NativeRowTransfer(const PermutationTransfer &transfer, const MPIInfo &mpi)
    {
        ArrayAdjacencyPair<2> owners;
        owners.InitPair("NativeRowTransfer.owners", mpi);
        owners.father->Resize(transfer.size());
        for (DNDS::index i = 0; i < transfer.size(); ++i)
        {
            owners(i, 0) = mpi.rank;
            owners(i, 1) = i;
        }
        transfer.transferRows(owners, mpi);
        std::vector<Runtime::RowRequest> requests;
        for (DNDS::index i = 0; i < owners.father->Size(); ++i)
            requests.push_back({int(owners(i, 0)), owners(i, 1), size_t(i)});
        return {mpi.comm, size_t(transfer.size()), requests.size(), requests};
    }

    template <class T, class L>
    CompanionEntry RegisterArray(Runtime::Array<T, L> &array, EntityKind kind, std::string name,
                                 Runtime::ElementSchema schema = Runtime::ElementSchema::Arithmetic<T>(),
                                 Runtime::ExecutionContext context = Runtime::ExecutionContext::Host())
    {
        return {kind, [&array, schema, context](const PermutationTransfer &transfer, const MPIInfo &mpi)
                {
                    int valid = array.Rows() == size_t(transfer.size()) && array.Storage().IsIdle() && array.IsValid();
                    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
                    DNDS_check_throw_info(valid, "native Array migration requires idle valid owned rows matching the mesh");
                    auto route = NativeRowTransfer(transfer, mpi);
                    auto layout = route.DestinationLayout(array.Layout());
                    if constexpr (!std::is_same_v<L, Runtime::RowLayout>)
                        layout = Runtime::RowLayout::Dense(layout.Rows(), L::Default().Width());
                    Runtime::Array<T, L> next(layout, array.GetDevice());
                    Runtime::PayloadPlan(route, array.Layout(), next.Layout(), schema).Bind(array, next).Begin(context).Wait();
                    array = std::move(next);
                },
                std::move(name)};
    }

    /// Adapt an owned native field to the existing mesh permutation machinery.
    /// Stable IDs, states and time histories are ordinary registered fields.
    /// The initial implementation is host-only; ghosts are rebuilt separately.
    /// The field reference must outlive the synchronous reorder call.
    template <class T>
    CompanionEntry RegisterField(Runtime::Field<T> &field, EntityKind kind, std::string name)
    {
        return {kind, [&field](const PermutationTransfer &transfer, const MPIInfo &mpi)
                {
                    int valid = field.Values().GetDevice() == Runtime::Device::Host() &&
                                field.Layout().rows == size_t(transfer.size()) &&
                                field.GetDistribution().ghosts == 0 &&
                                field.GetDistribution().epoch != UINT64_MAX && field.Values().IsIdle();
                    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
                    DNDS_check_throw_info(valid, "native field migration requires idle host owned rows matching the mesh");

                    // Move source addresses with the mesh's exact correspondence,
                    // including local permutation and distributed receive order.
                    ArrayAdjacencyPair<2> owners;
                    owners.InitPair("FieldTransfer.owners", mpi);
                    owners.father->Resize(transfer.size());
                    for (DNDS::index i = 0; i < transfer.size(); ++i)
                    {
                        owners(i, 0) = mpi.rank;
                        owners(i, 1) = i;
                    }
                    transfer.transferRows(owners, mpi);
                    std::vector<Runtime::RemoteRow> leaves(size_t(owners.father->Size()));
                    for (size_t i = 0; i < leaves.size(); ++i)
                        leaves[i] = {static_cast<int>(owners(i, 0)), owners(i, 1)};
                    Runtime::ExchangePlan exchange(mpi.comm, field.Layout().rows, leaves);
                    auto layout = field.Layout();
                    layout.rows = leaves.size();
                    auto distribution = field.GetDistribution();
                    distribution.owned = leaves.size();
                    ++distribution.epoch;
                    Runtime::Field<T> next(layout, distribution, field.Association());
                    auto destination = next.Values();
                    exchange.BeginCopy(field.Values(), destination,
                                       CheckedSize::Multiply(layout.stride, sizeof(T)))
                        .Wait();
                    field = std::move(next);
                },
                std::move(name)};
    }
}
