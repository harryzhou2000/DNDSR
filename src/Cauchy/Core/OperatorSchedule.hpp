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
            return _engine.ChecksAccess()
                       ? binding.BeginReadyChecked(_communication)
                       : binding.BeginReadyScheduled(_communication);
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
        Runtime::StreamEvent BeginAfterCompute(Runtime::TransferBinding &binding) const
        {
            if (binding.SupportsStreamEnqueue())
            {
                _communication.WaitFor(_compute.Record());
                binding.Enqueue(_engine, _communication);
                return _communication.Record();
            }
            auto operation = _engine.ChecksAccess()
                                 ? binding.BeginChecked(_compute)
                                 : binding.BeginScheduled(_compute);
            return operation.CompletionEvent();
        }
        std::vector<Runtime::StreamEvent> BeginAfterCompute(std::vector<Runtime::TransferBinding> &bindings) const
        {
            std::vector<Runtime::StreamEvent> events;
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
            _compute.WaitFor(operation.CompletionEvent());
            operation = {};
        }
        void Await(std::vector<Runtime::TransferOperation> &operations) const
        {
            for (auto &operation : operations)
                _compute.WaitFor(operation.CompletionEvent());
            operations.clear();
        }
        void Await(std::vector<Runtime::StreamEvent> &events) const
        {
            for (auto &event : events)
                _compute.WaitFor(std::move(event));
            events.clear();
        }
        template <class Operation>
        void AwaitPublished(Operation &&operation) const
        {
            auto publication = std::make_shared<std::decay_t<Operation>>(
                std::forward<Operation>(operation));
            _compute.WaitFor(publication->CompletionEvent());
            _engine.Submit(
                _compute,
                [publication](Runtime::ExecutionContext)
                { publication->PublishCompleted(); });
        }
        template <class Action>
        void Submit(Action &&action) const
        {
            _engine.Submit(_compute, [operation = std::forward<Action>(action)](Runtime::ExecutionContext) mutable
                           { operation(); });
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
