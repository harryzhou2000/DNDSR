#include "Exchange.hpp"

namespace DNDS::Runtime
{
    namespace
    {
        std::vector<RowRequest> Requests(const std::vector<RemoteRow> &leaves)
        {
            std::vector<RowRequest> requests;
            for (size_t i = 0; i < leaves.size(); ++i)
                requests.push_back({leaves[i].rank, leaves[i].row, i});
            return requests;
        }
        void Check(bool valid, MPI_Comm comm)
        {
            int all = valid;
            MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, comm);
            DNDS_check_throw_info(all, "prototype Exchange facade requires host arrays and positive row width");
        }
    }
    ExchangePlan::ExchangePlan(MPI_Comm communicator, size_t localRoots, const std::vector<RemoteRow> &leaves,
                               std::vector<VersionStamp> dependencies)
        : _route(communicator, localRoots, leaves.size(), Requests(leaves), std::move(dependencies)) {}

    ExchangePlan::Operation ExchangePlan::BeginCopy(const Buffer &roots, Buffer &leaves, size_t rowBytes) const
    {
        Check(rowBytes && roots.GetDevice() == Device::Host() && leaves.GetDevice() == Device::Host(), _route.Communicator());
        return PayloadPlan(_route, RowLayout::Dense(_route.SourceRows(), rowBytes),
                           RowLayout::Dense(_route.DestinationRows(), rowBytes), ElementSchema::Arithmetic<uint8_t>())
            .Bind(roots, leaves)
            .Begin();
    }

    ExchangePlan::Operation ExchangePlan::BeginAdd(const Buffer &leaves, Buffer &roots, size_t components) const
    {
        Check(components && roots.GetDevice() == Device::Host() && leaves.GetDevice() == Device::Host(), _route.Communicator());
        auto reverse = _route.Reverse();
        return PayloadPlan(reverse, RowLayout::Dense(reverse.SourceRows(), components),
                           RowLayout::Dense(reverse.DestinationRows(), components), ElementSchema::Arithmetic<double>())
            .Bind(leaves, roots, Reducer::Builtin<double>(Reduction::Sum))
            .Begin();
    }
}
