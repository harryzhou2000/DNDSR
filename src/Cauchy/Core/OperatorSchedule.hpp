#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include <exception>
#include <utility>
#include <vector>

namespace DNDS::Cauchy
{
    /// Backend-neutral eager schedule used by Cauchy operator orchestration.
    /// It records the same submission order for inline CPU, worker CPU and a
    /// future resolved device launcher. Backend-specific waiting/progress stays
    /// in ExecutionStream and TransferBinding. The schedule owns no numerical
    /// views; each submitted operation record must capture its retained scope.
    class OperatorSchedule
    {
        Runtime::EagerEngine &_engine;
        Runtime::ExecutionStream _stream;
        bool _overlap;
        int _exceptions = std::uncaught_exceptions();

    public:
        OperatorSchedule(Runtime::EagerEngine &engine, Runtime::ExecutionStream stream, bool overlap)
            : _engine(engine), _stream(std::move(stream)), _overlap(overlap)
        {
            DNDS_check_throw_info(_engine.Owns(_stream), "operator schedule stream belongs to another runtime");
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
            return _engine.ChecksAccess() ? binding.BeginChecked(_stream) : binding.BeginScheduled(_stream);
        }
        std::vector<Runtime::TransferOperation> Begin(std::vector<Runtime::TransferBinding> &bindings) const
        {
            std::vector<Runtime::TransferOperation> operations;
            operations.reserve(bindings.size());
            for (auto &binding : bindings)
                operations.push_back(Begin(binding));
            return operations;
        }
        template <class Operation>
        static void Wait(Operation &operation)
        {
            operation.Wait();
        }
        template <class Operation>
        static void Wait(std::vector<Operation> &operations)
        {
            for (auto &operation : operations)
                operation.Wait();
            operations.clear();
        }
        template <class Action>
        void Submit(Action &&action) const
        {
            _engine.Submit(_stream, [operation = std::forward<Action>(action)](Runtime::ExecutionContext) mutable
                           { operation(); });
        }
        template <class Action, class... Operands>
        void SubmitArrays(Action &&action, Operands &&...operands) const
        {
            Runtime::SubmitArrays(_engine, _stream, std::forward<Action>(action),
                                  std::forward<Operands>(operands)...);
        }
        void Join() const { _stream.Record().WaitHost(); }
    };
}
