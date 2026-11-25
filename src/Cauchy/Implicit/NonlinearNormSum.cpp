#include "Cauchy/Implicit/NonlinearNormSum.hpp"
#include "DNDS/Runtime/Execution/EagerCopy.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    NonlinearNormSum::NonlinearNormSum(index count, const NumericalReductions &reductions, int ranks, const R::ExecutionStream &compute, const R::ExecutionStream &communication)
        : _lane(communication)
    {
        const auto computeDevice = compute.Context().device;
        const auto layout = R::RowLayout::Dense(1, count);
        _local = R::Array<real>(layout, computeDevice);
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
            _binding = reductions.Bind(_local, _sum, R::Reduction::Sum);
        }
    }
    void NonlinearNormSum::Submit(R::EagerEngine &engine, const R::ExecutionStream &producer)
    {
        _resultLane = _binding ? _lane : producer;
        _completion = producer.Record();
        if (_binding)
        {
            if (!_lane.SameLane(producer)) _lane.WaitFor(_completion);
            _completion = _binding->Submit(engine, _lane);
        }
    }
    const R::Array<real> &NonlinearNormSum::Consume(R::EagerEngine &engine, const R::ExecutionStream &consumer)
    {
        if (!consumer.SameLane(_resultLane))
            consumer.WaitFor(_completion);
        return _sum;
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
