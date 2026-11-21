#include "Cauchy/Implicit/NonlinearNormSum.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    NonlinearNormSum::NonlinearNormSum(index count, const std::shared_ptr<R::CommunicationContext> &comm,
                                       const R::ExecutionStream &lane, const std::shared_ptr<R::NCCLTransferProvider> &nccl,
                                       R::Device computeDevice) : _lane(lane)
    {
        int ranks = 0;
        MPI_Comm_size(comm->Communicator(), &ranks);
        const auto layout = R::RowLayout::Dense(1, count);
        const bool stage = ranks > 1 && !nccl && computeDevice != R::Device::Host();
        _local = stage ? R::Array<real>::FromStorage(layout, R::Buffer::AllocatePinnedHost(size_t(count) * sizeof(real)))
                       : R::Array<real>(layout, computeDevice);
        _local.SetName("nonlinear/norm/local", DNDS_ALLOCATION_SITE());
#ifdef DNDS_USE_CUDA
        if (computeDevice.space == R::MemorySpace::CUDA)
        {
            deviceSummary = R::Array<CUDA::NonlinearSummary>(R::RowLayout::Dense(1, 1), computeDevice);
            deviceSummary.SetName("nonlinear/norm/deviceSummary", DNDS_ALLOCATION_SITE());
        }
#endif
        _sum = _local;
        if (ranks > 1)
        {
            _sum = R::Array<real>(layout, _local.GetDevice());
            _sum.SetName("nonlinear/norm/global", DNDS_ALLOCATION_SITE());
            _binding = nccl ? nccl->BindAllreduce(_local, _sum, R::Reduction::Sum)
                            : R::BindMPIAllreduce(comm, _local, _sum, R::Reduction::Sum);
        }
    }
    void NonlinearNormSum::Submit(R::EagerEngine &engine, const R::ExecutionStream &producer)
    {
        _completion = producer.Record();
        if (_binding)
        {
            if (!_lane.SameLane(producer))
                _lane.WaitFor(_completion);
            _completion = _binding->Submit(engine, _lane);
        }
    }
    std::vector<real> NonlinearNormSum::Observe() const
    {
        _completion.WaitHost();
        const auto host = _sum.GetDevice() == R::Device::Host() ? _sum : _sum.To(R::Device::Host());
        const auto read = host.Read();
        const auto *row = read.View().Row(0);
        return {row, row + _sum.Layout().Width()};
    }
}
