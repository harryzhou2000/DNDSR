#pragma once

#include "RowTransfer.hpp"

namespace DNDS::Runtime
{
    struct RemoteRow
    {
        int rank = 0;
        int64_t row = 0;
    };

    /// Compatibility facade for the first runtime prototype. New callers retain
    /// RowTransferPlan/PayloadPlan/TransferBinding separately for setup reuse.
    class ExchangePlan
    {
        RowTransferPlan _route;

    public:
        ExchangePlan(MPI_Comm communicator, size_t localRoots, const std::vector<RemoteRow> &leaves,
                     std::vector<VersionStamp> dependencies = {});
        using Operation = TransferOperation;
        Operation BeginCopy(const Buffer &roots, Buffer &leaves, size_t rowBytes) const;
        Operation BeginAdd(const Buffer &leaves, Buffer &roots, size_t components) const;
    };
}
