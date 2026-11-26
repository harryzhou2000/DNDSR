#include "Cauchy/Core/NumericalReductions.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerCopy.hpp"
#include <mutex>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        template <typename T>
        class StagedReduction final : public R::GlobalReductionBinding
        {
            R::Array<T> _source, _destination;
            R::Array<T> _hostInput, _hostOutput;
            std::shared_ptr<const R::GlobalReductionBinding> _mpi;
            mutable std::once_flag _hostLaneOnce;
            mutable std::optional<R::ExecutionStream> _hostLane;

        public:
            StagedReduction(const std::shared_ptr<R::CommunicationContext> &comm, R::Array<T> source,
                            R::Array<T> destination, R::Reduction op)
                : _source(std::move(source)), _destination(std::move(destination)),
                  _hostInput(R::Array<T>::FromStorage(_source.Layout(), R::Buffer::AllocatePinnedHost(_source.Storage().Bytes()))),
                  _hostOutput(R::Array<T>::FromStorage(_destination.Layout(), R::Buffer::AllocatePinnedHost(_destination.Storage().Bytes()))),
                  _mpi(R::BindMPIAllreduce(comm, _hostInput, _hostOutput, op)) {}

            R::StreamEvent Submit(R::EagerEngine &engine, const R::ExecutionStream &stream) const override
            {
                std::call_once(_hostLaneOnce, [&]
                               { _hostLane = engine.CreateCPUStream(); });
                auto input = _hostInput.Storage();
                auto destination = _destination.Storage();
                R::CopyAsync(engine, stream, _source.Storage(), input);
                _hostLane->WaitFor(stream.Record());
                // The MPI controller drives request progress before enqueuing
                // the GPU consumer. Older MPI libraries cannot wait in a CUDA lane.
                _mpi->Submit(engine, *_hostLane).WaitHost();
                R::CopyAsync(engine, stream, _hostOutput.Storage(), destination);
                return stream.Record();
            }
        };
        // Controller agreements are small. A fixed prepared extent also handles
        // longer diagnostics in chunks without binding a new collective mid-step.
        template <class T>
        struct ObservedSlot
        {
            static constexpr size_t width = 16;
            R::Array<T> host, device;
            std::array<std::shared_ptr<const R::GlobalReductionBinding>, 3> bindings;
            ObservedSlot(R::Device where, const R::NCCLTransferProvider &provider)
                : host(R::RowLayout::Dense(1, width)), device(host.Layout(), where)
            {
                size_t i = 0;
                for (auto op : {R::Reduction::Sum, R::Reduction::Min, R::Reduction::Max})
                    bindings[i++] = provider.BindAllreduce(device, device, op);
            }
            void Reduce(T *data, size_t count, R::Reduction op, R::EagerEngine &engine, const R::ExecutionStream &stream)
            {
                const auto &binding = bindings[op == R::Reduction::Sum ? 0 : op == R::Reduction::Min ? 1
                                                                                                     : 2];
                while (count)
                {
                    const size_t n = std::min(count, width);
                    {
                        auto out = host.Write();
                        auto *dst = out.View().data;
                        std::copy_n(data, n, dst);
                        std::fill(dst + n, dst + width, T{});
                    }
                    device.CopyFrom(host, stream.Context()).Wait();
                    binding->Submit(engine, stream).WaitHost();
                    host.CopyFrom(device, stream.Context()).Wait();
                    auto read = host.Read();
                    std::copy_n(read.View().data, n, data);
                    data += n;
                    count -= n;
                }
            }
        };
    }
    struct NumericalReductions::DeviceState
    {
        R::EagerEngine engine;
        R::ExecutionStream lane;
        std::shared_ptr<R::NCCLTransferProvider> provider;
        ObservedSlot<int> integers;
        ObservedSlot<int64_t> indices;
        ObservedSlot<double> reals;
        DeviceState(std::shared_ptr<R::NCCLTransferProvider> selected, R::Device where)
            : lane(engine.CreateStream(std::make_shared<R::CUDALauncher>(where.ordinal))),
              provider(std::move(selected)),
              integers(where, *provider), indices(where, *provider), reals(where, *provider) {}
    };
    NumericalReductions::NumericalReductions(const std::shared_ptr<R::CommunicationContext> &comm, R::Device device)
        : _comm(comm->Communicator()), _context(comm), _device(std::make_shared<DeviceState>(R::NCCLTransferProvider::Create(comm, device), device)) {}
    template <class T>
    std::shared_ptr<const R::GlobalReductionBinding> NumericalReductions::Bind(const R::Array<T> &source, R::Array<T> &destination, R::Reduction op) const
    {
        if (_device)
            return _device->provider->BindAllreduce(source, destination, op);
        if (source.GetDevice() == R::Device::Host())
        {
            // A CPU producer may become ready at different times on different
            // ranks. Isolate its deferred MPI_Iallreduce from controller
            // agreements and other independently scheduled reductions.
            return R::BindMPIAllreduce(R::CommunicationContext::Duplicate(_comm), source, destination, op);
        }
        const auto context = _context ? _context : R::CommunicationContext::Duplicate(_comm);
        return std::make_shared<StagedReduction<T>>(context, source, destination, op);
    }
    template std::shared_ptr<const R::GlobalReductionBinding> NumericalReductions::Bind(const R::Array<real> &, R::Array<real> &, R::Reduction) const;
    template std::shared_ptr<const R::GlobalReductionBinding> NumericalReductions::Bind(const R::Array<index> &, R::Array<index> &, R::Reduction) const;
    template std::shared_ptr<const R::GlobalReductionBinding> NumericalReductions::Bind(const R::Array<int> &, R::Array<int> &, R::Reduction) const;
    void NumericalReductions::ReduceHost(int *p, size_t n, R::Reduction op) const { _device->integers.Reduce(p, n, op, _device->engine, _device->lane); }
    void NumericalReductions::ReduceHost(int64_t *p, size_t n, R::Reduction op) const { _device->indices.Reduce(p, n, op, _device->engine, _device->lane); }
    void NumericalReductions::ReduceHost(double *p, size_t n, R::Reduction op) const { _device->reals.Reduce(p, n, op, _device->engine, _device->lane); }
}
