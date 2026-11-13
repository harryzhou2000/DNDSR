#pragma once

#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Core/Span.hpp"

namespace DNDS::Runtime
{
    /**
     * @brief Submit a runtime-sized set of typed array operands.
     * @details This is the dynamic-field counterpart of SubmitArrays. The
     * launcher receives two host spans of bare whole-array views, in operand
     * order. Value and shape allocations are retained through completion;
     * reservations, mutation epochs, dependency checks and ordered handoff use
     * exactly the ordinary Array operand machinery. No payload is copied.
     *
     * The descriptor spans themselves are host metadata, even for CUDA arrays.
     * A CUDA launcher may select/pass individual POD views by value; it must
     * not pass these host span addresses to a device kernel. An operator needing
     * a device table of descriptors must prepare that table explicitly.
     *
     * @pre Declare all mutable input/output arrays and order conflicts with
     * stream edges. Write does not initialize storage. Writable operands may
     * use ReadWriteAs when the kernel also consumes their previous contents.
     */
    template <class T, class ReadPolicy, class WritePolicy, class Kernel>
    void SubmitArrayBatch(EagerEngine &engine, const ExecutionStream &stream,
                          Kernel kernel,
                          std::vector<ArrayOperand<T, false, ReadPolicy>> inputs,
                          std::vector<ArrayOperand<T, true, WritePolicy>> outputs)
    {
        using Input = Detail::PreparedOperand<ArrayOperand<T, false, ReadPolicy>>;
        using Output = Detail::PreparedOperand<ArrayOperand<T, true, WritePolicy>>;
        using InputView = decltype(std::declval<Input>().view);
        using OutputView = decltype(std::declval<Output>().view);
        struct Retained
        {
            std::vector<Input> inputs;
            std::vector<Output> outputs;
            std::vector<InputView> inputViews;
            std::vector<OutputView> outputViews;

            void Validate() const
            {
                for (const auto &value : inputs)
                    value.access.Validate();
                for (const auto &value : outputs)
                    value.access.Validate();
            }
            void Handoff(ExecutionContext context)
            {
                for (auto &value : inputs)
                    value.Handoff(context);
                for (auto &value : outputs)
                    value.Handoff(context);
            }
            void Complete(bool success) noexcept
            {
                for (auto &value : inputs)
                    value.Complete(success);
                for (auto &value : outputs)
                    value.Complete(success);
            }
        };
        const auto context = stream.Context();
        auto retained = std::make_shared<Retained>();
        retained->inputs.reserve(inputs.size());
        retained->outputs.reserve(outputs.size());
        retained->inputViews.reserve(inputs.size());
        retained->outputViews.reserve(outputs.size());
        std::vector<ScheduledBufferAccess> accesses;
        accesses.reserve(inputs.size() + outputs.size());
        for (auto &operand : inputs)
        {
            accesses.push_back({operand.storage->Values(), operand.mode});
            retained->inputs.emplace_back(std::move(operand), context, engine.ChecksAccess());
            retained->inputViews.push_back(retained->inputs.back().view);
        }
        for (auto &operand : outputs)
        {
            accesses.push_back({operand.storage->Values(), operand.mode});
            retained->outputs.emplace_back(std::move(operand), context, engine.ChecksAccess());
            retained->outputViews.push_back(retained->outputs.back().view);
        }
        engine.SubmitOperation(stream, [retained, kernel = std::move(kernel)](ExecutionContext context) mutable
                               {
                try
                {
                    retained->Validate();
                    kernel(context,
                           SpanView<const InputView>{retained->inputViews.data(), index(retained->inputViews.size())},
                           SpanView<const OutputView>{retained->outputViews.data(), index(retained->outputViews.size())});
                }
                catch (...)
                {
                    // Match SubmitArrays: native work may have launched before
                    // failure, so ordered release follows that issued prefix.
                    const auto error = std::current_exception();
                    for (auto &value : retained->inputs) { try { value.Handoff(context); } catch (...) {} }
                    for (auto &value : retained->outputs) { try { value.Handoff(context); } catch (...) {} }
                    std::rethrow_exception(error);
                }
                retained->Handoff(context); }, [retained](std::exception_ptr error) noexcept
                               { retained->Complete(!error); }, accesses);
    }
}
