#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace DNDS::Cauchy
{
    /// Backend-neutral eager schedule used by Cauchy operator orchestration.
    /// It records the same submission order for inline CPU, worker CPU and a
    /// future resolved device launcher. Backend-specific waiting/progress stays
    /// in ExecutionStream and TransferBinding. The schedule owns no numerical
    /// views; each submitted operation record must capture its retained scope.
    /// Numerical collectives inside a queued segment must be managed operations
    /// with producer/completion events, never blocking calls inside Submit.
    /// A dynamic convergence/early-exit decision belongs outside that segment
    /// and may Join explicitly. In-queue mutation and device driver kernels for
    /// eliminating such decisions' host synchronization are future work.
    /// Candidate device controllers include CDP with device-managed streams,
    /// or a persistent/mega driver kernel managing asynchronous work explicitly.
    /// This contract does not claim either controller or sync-free GMRES today.
    /// TODO(sync-free-iteration-control): implement/validate a device-side
    /// dynamic stopping controller without introducing hidden host waits.
    class OperatorSchedule
    {
        Runtime::EagerEngine &_engine;
        Runtime::ExecutionStream _compute, _communication;
        bool _overlap;
        int _exceptions = std::uncaught_exceptions();

    public:
        OperatorSchedule(Runtime::EagerEngine &engine,
                         Runtime::ExecutionStream compute,
                         Runtime::ExecutionStream communication, bool overlap)
            : _engine(engine), _compute(std::move(compute)),
              _communication(std::move(communication)), _overlap(overlap)
        {
            DNDS_check_throw_info(_engine.Owns(_compute) &&
                                      _engine.Owns(_communication),
                                  "operator schedule stream belongs to another runtime");
        }
        OperatorSchedule(const OperatorSchedule &) = delete;
        OperatorSchedule &operator=(const OperatorSchedule &) = delete;
        ~OperatorSchedule()
        {
            if (std::uncaught_exceptions() <= _exceptions)
                return;
            MPI::AbortIfSolverFailure("solver unwinding before a Cauchy operator stream join");
            try
            {
                Join();
            }
            catch (...)
            {
                // Never replace the controller exception. Solver mode has
                // already taken the terminal path; coordinated tests may catch.
                MPI::AbortIfSolverFailure("Cauchy operator stream join failed");
            }
        }

        bool RunsCoreSeparately() const { return _overlap; }
        CFV::ReconstructionRegion DependentRegion() const
        {
            return _overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All;
        }
        Runtime::TransferOperation Begin(Runtime::TransferBinding &binding) const
        {
            return binding.SubmitReady(_engine, _communication);
        }
        std::vector<Runtime::TransferOperation> Begin(std::vector<Runtime::TransferBinding> &bindings) const
        {
            std::vector<Runtime::TransferOperation> operations;
            operations.reserve(bindings.size());
            for (auto &binding : bindings)
                operations.push_back(Begin(binding));
            return operations;
        }
        /// Exchange a payload produced by the already submitted compute prefix.
        /// Native providers import that prefix onto the communication lane and
        /// enqueue without a host wait. Portable MPI must observe its CPU/CUDA
        /// producer before posting; its managed bridge waits only that prefix,
        /// not every lane in the engine. MPI progress remains controller-owned.
        Runtime::TransferOperation BeginAfterCompute(Runtime::TransferBinding &binding) const
        {
            return binding.SubmitAfter(_engine, _communication, _compute.Record());
        }
        std::vector<Runtime::TransferOperation> BeginAfterCompute(std::vector<Runtime::TransferBinding> &bindings) const
        {
            std::vector<Runtime::TransferOperation> events;
            events.reserve(bindings.size());
            for (auto &binding : bindings)
                events.push_back(BeginAfterCompute(binding));
            return events;
        }
        /// Insert full transfer completion as a compute-lane dependency. Core
        /// work already submitted remains ahead of this edge; later dependent
        /// work cannot start until the runtime progresses the managed transfer.
        void Await(Runtime::TransferOperation &operation) const
        {
            operation.EnqueueWait(_compute);
            operation = {};
        }
        void Await(std::vector<Runtime::TransferOperation> &operations) const
        {
            for (auto &operation : operations)
                operation.EnqueueWait(_compute);
            operations.clear();
        }
        void Await(std::vector<Runtime::StreamEvent> &events) const
        {
            for (auto &event : events)
                _compute.WaitFor(std::move(event));
            events.clear();
        }
        void Await(Runtime::StreamEvent event) const
        {
            _compute.WaitFor(std::move(event));
        }
        template <class Operation>
        void AwaitPublished(Operation &&operation) const
        {
            auto publication = std::make_shared<std::decay_t<Operation>>(
                std::forward<Operation>(operation));
            publication->EnqueuePublication(_engine, _compute, publication);
        }
        template <class Action>
        void Submit(Action &&action) const
        {
            DNDS_check_throw_info(_compute.Context().device == Runtime::Device::Host(),
                                  "CPU compound callback requires a host stream; resolve a prepared device operation instead");
            _engine.Submit(_compute, [operation = std::forward<Action>(action)](Runtime::ExecutionContext) mutable
                           { operation(); });
        }
        /// Submit a prepared backend operation through the same schedule.
        /// The record owns every non-Array scope/descriptor it borrows and
        /// supplies Accesses(), Launch(context), and noexcept Retire(error).
        /// Accesses are checked at submission; native Launch only enqueues work.
        /// Retire runs after actual execution and must complete/fail its access
        /// reservations without throwing. Captured ownership survives both the
        /// launch callback and every queued device use. Missing device kernels
        /// must be rejected during preparation, never silently run CPU code.
        /// For Array operands prefer SubmitArrays: it implements this retained
        /// allocation/layout/access contract without a custom operation record.
        template <class Operation>
        void SubmitPrepared(std::shared_ptr<Operation> operation) const
        {
            static_assert(noexcept(operation->Retire(std::declval<std::exception_ptr>())),
                          "prepared operation retirement must be noexcept");
            DNDS_check_throw_info(operation, "prepared operation is empty");
            auto accesses = operation->Accesses();
            _engine.SubmitOperation(_compute, [operation](Runtime::ExecutionContext context)
                                    { operation->Launch(context); }, [operation](std::exception_ptr error) noexcept
                                    { operation->Retire(error); }, std::move(accesses));
        }
        template <class Action, class... Operands>
        void SubmitArrays(Action &&action, Operands &&...operands) const
        {
            Runtime::SubmitArrays(_engine, _compute, std::forward<Action>(action),
                                  std::forward<Operands>(operands)...);
        }
        Runtime::StreamEvent RecordCompute() const { return _compute.Record(); }
        Runtime::StreamEvent RecordCommunication() const
        {
            return _communication.Record();
        }
        void ComputeWaitFor(Runtime::StreamEvent event) const
        {
            _compute.WaitFor(std::move(event));
        }
        /// Quiescent operator boundary. The runtime progresses registered MPI
        /// work while draining compute dependencies; no transfer wait appears
        /// between queued core and dependent submissions.
        void Join() const { _engine.Synchronize(); }
    };
}
