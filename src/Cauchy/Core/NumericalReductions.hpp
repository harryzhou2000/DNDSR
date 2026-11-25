#pragma once

#include "DNDS/Core/Span.hpp"

#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include "DNDS/Runtime/Communication/MPIGlobalReduction.hpp"

namespace DNDS::Cauchy
{
    /// Domain-scoped numerical collectives. Resident CUDA operators bind their
    /// output Arrays once. Host-side decisions use a separate, reusable slot.
    /// All calls on a communicator must have matching order across ranks.
    class NumericalReductions
    {
        struct DeviceState;
        MPI_Comm _comm;
        std::shared_ptr<Runtime::CommunicationContext> _context;
        std::shared_ptr<DeviceState> _device;
        void ReduceHost(int *, size_t, Runtime::Reduction) const;
        void ReduceHost(int64_t *, size_t, Runtime::Reduction) const;
        void ReduceHost(double *, size_t, Runtime::Reduction) const;

    public:
        explicit NumericalReductions(MPI_Comm comm) : _comm(comm) {}
        NumericalReductions(const std::shared_ptr<Runtime::CommunicationContext> &, Runtime::Device);
        explicit NumericalReductions(std::shared_ptr<Runtime::CommunicationContext> comm)
            : _comm(comm->Communicator()), _context(std::move(comm)) {}
        /// Collective setup on matching compact arrays. Submit on the producer's
        /// stream. MPI stages CUDA operands, NCCL consumes their device buffers.
        template <class T>
        std::shared_ptr<const Runtime::GlobalReductionBinding> Bind(
            const Runtime::Array<T> &source, Runtime::Array<T> &destination, Runtime::Reduction op) const;
        /// Bind a compact scalar range in an operator's resident POD summary.
        /// The binding retains the underlying storage. All ranks use the same layout.
        template <class T>
        std::shared_ptr<const Runtime::GlobalReductionBinding> BindInPlace(
            Runtime::Buffer buffer, size_t byteOffset, index count, Runtime::Reduction op) const
        {
            auto view = Runtime::Array<T>::FromStorage(Runtime::RowLayout::Dense(1, count),
                                                       buffer.Subspan(byteOffset, size_t(count) * sizeof(T)));
            return Bind(view, view, op);
        }
        bool UsesNCCL() const { return bool(_device); }

        /// Validate a local action collectively before a dependent operation.
        /// Preserve the originating exception on that rank and fail its peers.
        template <class F>
        void LocalStage(F &&action, const char *message = "local numerical preparation failed on a peer") const
        {
            std::exception_ptr failure;
            try { std::forward<F>(action)(); }
            catch (...)
            {
                MPI::AbortIfSolverFailure(message);
                failure = std::current_exception();
            }
            const int valid = Value<Runtime::Reduction::Min>(int(!failure));
            if (failure) std::rethrow_exception(failure);
            DNDS_check_throw_info(valid, message);
        }
        template <Runtime::Reduction Op, typename T>
        void Reduce(SpanView<T> values) const
        {
            static_assert(Op == Runtime::Reduction::Sum || Op == Runtime::Reduction::Min || Op == Runtime::Reduction::Max);
            if (_device)
                return ReduceHost(values.data(), size_t(values.size()), Op);
            constexpr auto mpiOp = Op == Runtime::Reduction::Sum ? MPI::Reduction::Sum :
                                   Op == Runtime::Reduction::Min ? MPI::Reduction::Min : MPI::Reduction::Max;
            MPI::AllreduceInPlace<mpiOp>(MPI::ContiguousView<T>{values.data(), size_t(values.size())}, _comm);
        }
        template <Runtime::Reduction Op, typename T>
        T Value(T value) const
        {
            Reduce<Op>(SpanView<T>{&value, 1});
            return value;
        }
    };
}
