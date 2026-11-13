#pragma once

#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Core/Span.hpp"
#include <optional>
#include <tuple>

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
    template <class T, class ReadPolicy, class WritePolicy, class Kernel, class... Extra>
    void SubmitArrayBatch(EagerEngine &engine, const ExecutionStream &stream,
                          Kernel kernel,
                          std::vector<ArrayOperand<T, false, ReadPolicy>> inputs,
                          std::vector<ArrayOperand<T, true, WritePolicy>> outputs,
                          std::tuple<Extra...> extra)
    {
        using Input = Detail::PreparedOperand<ArrayOperand<T, false, ReadPolicy>>;
        using Output = Detail::PreparedOperand<ArrayOperand<T, true, WritePolicy>>;
        using InputView = decltype(std::declval<Input>().view);
        using OutputView = decltype(std::declval<Output>().view);
        struct Retained
        {
            Kernel kernel;
            explicit Retained(Kernel body) : kernel(std::move(body)) {}
            std::vector<Input> inputs;
            std::vector<Output> outputs;
            std::vector<InputView> inputViews;
            std::vector<OutputView> outputViews;
            std::optional<std::tuple<Detail::PreparedOperand<Extra>...>> extra;

            void Validate() const
            {
                for (const auto &value : inputs)
                    value.access.Validate();
                for (const auto &value : outputs)
                    value.access.Validate();
                std::apply([](const auto &...value)
                           { (value.access.Validate(), ...); }, *extra);
            }
            void Handoff(ExecutionContext context)
            {
                for (auto &value : inputs)
                    value.Handoff(context);
                for (auto &value : outputs)
                    value.Handoff(context);
                std::apply([&](auto &...value)
                           { (value.Handoff(context), ...); }, *extra);
            }
            void Complete(bool success) noexcept
            {
                for (auto &value : inputs)
                    value.Complete(success);
                for (auto &value : outputs)
                    value.Complete(success);
                std::apply([&](auto &...value)
                           { (value.Complete(success), ...); }, *extra);
            }
        };
        const auto context = stream.Context();
        auto retained = std::make_shared<Retained>(std::move(kernel));
        retained->inputs.reserve(inputs.size());
        retained->outputs.reserve(outputs.size());
        retained->inputViews.reserve(inputs.size());
        retained->outputViews.reserve(outputs.size());
        std::vector<ScheduledBufferAccess> accesses;
        accesses.reserve(inputs.size() + outputs.size() + sizeof...(Extra));
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
        std::apply([&](auto &...operand)
                   {
            (accesses.push_back({operand.storage->Values(), operand.mode}), ...);
            retained->extra.emplace(Detail::PreparedOperand<Extra>(std::move(operand), context, engine.ChecksAccess())...); }, extra);
        engine.SubmitOperation(stream, [retained](ExecutionContext context) mutable
                               {
                try
                {
                    retained->Validate();
                    std::apply([&](auto &...value) { retained->kernel(context,
                           SpanView<const InputView>{retained->inputViews.data(), index(retained->inputViews.size())},
                           SpanView<const OutputView>{retained->outputViews.data(), index(retained->outputViews.size())}, value.view...); }, *retained->extra);
                }
                catch (...)
                {
                    // Match SubmitArrays: native work may have launched before
                    // failure, so ordered release follows that issued prefix.
                    const auto error = std::current_exception();
                    for (auto &value : retained->inputs) { try { value.Handoff(context); } catch (...) {} }
                    for (auto &value : retained->outputs) { try { value.Handoff(context); } catch (...) {} }
                    std::apply([&](auto &...value)
                    {
                        auto handoff = [context](auto &item) { try { item.Handoff(context); } catch (...) {} };
                        (handoff(value), ...);
                    }, *retained->extra);
                    std::rethrow_exception(error);
                }
                retained->Handoff(context); }, [retained](std::exception_ptr error) noexcept
                               { retained->Complete(!error); }, accesses);
    }
    /// Homogeneous batches retain their original calling shape. The overload
    /// with a final tuple appends fixed typed operands (e.g. index/flag/status
    /// arrays) after the two dynamic view spans, under the same reservation,
    /// ownership and retirement contract. No second operation is introduced.
    template <class T, class ReadPolicy, class WritePolicy, class Kernel>
    void SubmitArrayBatch(EagerEngine &engine, const ExecutionStream &stream, Kernel kernel,
                          std::vector<ArrayOperand<T, false, ReadPolicy>> inputs,
                          std::vector<ArrayOperand<T, true, WritePolicy>> outputs)
    {
        SubmitArrayBatch(engine, stream, std::move(kernel), std::move(inputs), std::move(outputs), std::tuple<>{});
    }
}
