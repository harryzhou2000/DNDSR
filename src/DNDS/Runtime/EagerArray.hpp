#pragma once

#include "Array.hpp"
#include "EagerEngine.hpp"
#include <tuple>

namespace DNDS::Runtime
{
    /// Structural pin and declared intent; construction does not reserve access.
    /// SubmitArrays reserves all operands before accepting the command.
    template <class T, bool Writable>
    struct ArrayOperand
    {
        using Value = std::conditional_t<Writable, T, const T>;
        ArrayStorage::Shared storage;
        AccessMode mode;
    };

    template <class T, class L>
    ArrayOperand<T, false> Read(const Array<T, L> &array) { return {array.SharedStorage(), AccessMode::Read}; }
    template <class T, class L>
    ArrayOperand<T, true> Write(Array<T, L> &array) { return {array.SharedStorage(), AccessMode::Write}; }
    template <class T, class L>
    ArrayOperand<T, true> ReadWrite(Array<T, L> &array) { return {array.SharedStorage(), AccessMode::ReadWrite}; }

    /// Values are initialized in stream order. New GPU tables need no metadata
    /// upload. For variable GPU shapes, prepare/upload metadata during setup and
    /// use AllocateLikeAsync to share it without allocation or upload per field.
    template <class T, class L = RowLayout>
    Array<T, L> AllocateArrayAsync(EagerEngine &engine, const ExecutionStream &stream, RowLayout layout)
    {
        if constexpr (!std::is_same_v<L, RowLayout>)
            L::Validate(layout);
        DNDS_check_throw_info(stream.Context().device == Device::Host() || !layout.IsVariable(),
                              "new variable GPU metadata needs setup; use AllocateLikeAsync with a prepared Array");
        auto values = engine.AllocateBuffer(stream, CheckedSize::Multiply(layout.Elements(), sizeof(T)));
        try
        {
            auto result = Array<T, L>::FromStorage(std::move(layout), values);
            values.ReleaseOn(stream.Context());
            return result;
        }
        catch (...)
        {
            // Keep one allocation owner until metadata/handle construction
            // succeeds, so failure can also free in the allocation stream.
            values.ReleaseOn(stream.Context());
            throw;
        }
    }
    template <class T, class L>
    Array<T, L> AllocateLikeAsync(EagerEngine &engine, const ExecutionStream &stream, const Array<T, L> &prototype)
    {
        auto storage = prototype.SharedStorage();
        auto values = storage->Values();
        DNDS_check_throw_info(values.GetDevice() == stream.Context().device, "async prototype placement mismatch");
        auto fresh = engine.AllocateBuffer(stream, values.Bytes());
        try
        {
            auto result = Array<T, L>::FromSharedStorage(storage->WithValues(fresh, sizeof(T)));
            fresh.ReleaseOn(stream.Context());
            return result;
        }
        catch (...)
        {
            fresh.ReleaseOn(stream.Context());
            throw;
        }
    }

    namespace Detail
    {
        template <class Operand>
        struct PreparedOperand
        {
            using Value = typename Operand::Value;
            ArrayStorage::Shared storage;
            std::shared_ptr<const ArrayLayout> layout;
            ScheduledAccess access;
            ArrayView<Value> view;

            PreparedOperand(Operand operand, Device device)
                : storage(std::move(operand.storage)), layout(storage->SharedLayout()),
                  access(storage->Values().Reserve(operand.mode, device))
            {
                DNDS_check_throw_info(storage->ElementBytes() == sizeof(Value), "eager operand element size mismatch");
                auto *pointer = static_cast<Value *>(access.Pointer(alignof(Value), sizeof(Value)));
                const auto &rows = layout->rows;
                view = {pointer, layout->Offsets(device), rows.Rows(), rows.Width(), rows.Stride(), layout->Sizes(device)};
            }
            void Handoff(ExecutionContext context)
            {
                if (storage && context.device.space == MemorySpace::CUDA && storage->CanReleaseOrdered(context) && layout->CanReleaseOrdered(context))
                {
                    ArrayStorage::ReleaseHandle(storage, context);
                    ArrayLayout::ReleaseHandle(layout, context);
                    access.Handoff(context);
                }
            }
        };
    }

    /// Same orchestration for inline/worker CPU and device launchers. Kernel
    /// receives only bare views. Its return means CPU completion or device
    /// submission; the launcher decides when the retirement callback is safe.
    template <class Kernel, class... Operands>
    void SubmitArrays(EagerEngine &engine, const ExecutionStream &stream, Kernel kernel, Operands... operands)
    {
        using Prepared = std::tuple<Detail::PreparedOperand<Operands>...>;
        const auto device = stream.Context().device;
        auto retained = std::make_shared<Prepared>(Detail::PreparedOperand<Operands>(std::move(operands), device)...);
        engine.SubmitOperation(stream, [retained, kernel = std::move(kernel)](ExecutionContext context) mutable
                               { std::apply([&](auto &...operand)
                                            {
                    try
                    {
                        (operand.access.Validate(), ...);
                        kernel(context, operand.view...);
                    }
                    catch (...)
                    {
                        // A failed launcher may already have issued native work.
                        // Ordered release follows that prefix; publication still
                        // retires as failed. Unsupported providers stay retained.
                        auto error = std::current_exception();
                        auto handoff = [context](auto &value) { try { value.Handoff(context); } catch (...) {} };
                        (handoff(operand), ...);
                        std::rethrow_exception(error);
                    }
                    (operand.Handoff(context), ...); }, *retained); }, [retained](std::exception_ptr error) noexcept
                               { std::apply([&](auto &...operand)
                                            { (operand.access.Complete(!error), ...); }, *retained); });
    }
}
