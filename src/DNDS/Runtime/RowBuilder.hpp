#pragma once

#include "Array.hpp"
#include <cstring>
#include <optional>

namespace DNDS::Runtime
{
    /// CPU setup for known row sizes: allocate the final table/CSR once and
    /// initialize it under one writer. Unlike incremental RowBuilder this needs
    /// no allocation/controller per row. Callbacks must not retain borrowed rows;
    /// a failed fill never publishes an output. Every size is queried once.
    template <class T, class Size, class Fill>
    Array<T> BuildArrayRows(size_t rows, Size &&size, Fill &&fill, std::optional<size_t> width = {})
    {
        std::vector<size_t> lengths(rows);
        for (size_t row = 0; row < rows; ++row)
        {
            lengths[row] = size(row);
            DNDS_check_throw_info(!width || lengths[row] == *width, "known table row width mismatch");
        }
        Array<T> result(width ? RowLayout::Dense(rows, *width) : RowLayout::Ragged(lengths));
        {
            auto write = result.Write();
            const auto view = write.View();
            for (size_t row = 0; row < rows; ++row)
                fill(row, view[row]);
        }
        return result;
    }

    /// Exclusive CPU construction form. Distinct rows may be resized/filled by
    /// different workers; Commit follows their join. Borrowed row pointers expire
    /// on that row's resize or on Commit. No published Array aliases this draft.
    /// @pre Workers edit distinct rows; the caller joins them before Commit or
    /// destruction. No row pointer may escape its resize/publication boundary.
    /// A builder does not serialize concurrent writes to the same row.
    class RowBuilder
    {
        friend class ExclusiveRows;
        struct State
        {
            size_t rows, elementBytes, capacity, width;
            RowLayoutKind kind;
            Buffer lengths, dense;
            WriteAccess lengthAccess, denseAccess;
            size_t *sizes;
            struct Slab
            {
                Buffer buffer;
                size_t used = 0;
            };
            std::vector<Slab> slabs;
            std::vector<std::byte *> rowPointers;
            std::vector<size_t> rowCapacity;

            State(size_t n, size_t bytes, RowLayoutKind layout, size_t cap, Buffer values = {}, WriteAccess access = {})
                : rows(n), elementBytes(bytes), capacity(cap), width(cap), kind(layout)
            {
                DNDS_check_throw_info(bytes && (layout == RowLayoutKind::CSR || layout == RowLayoutKind::Padded || layout == RowLayoutKind::Table),
                                      "invalid row builder element size/layout");
                lengths = Buffer::Allocate(CheckedSize::Multiply(n, sizeof(size_t)));
                lengthAccess = lengths.Write();
                sizes = lengthAccess.View<size_t>().data;
                if (kind == RowLayoutKind::CSR)
                {
                    rowPointers.resize(n);
                    rowCapacity.resize(n);
                }
                else
                {
                    dense = values.Identity() ? std::move(values) : Buffer::Allocate(CheckedSize::Multiply(CheckedSize::Multiply(n, cap), bytes));
                    denseAccess = access.Active() ? std::move(access) : dense.Write();
                    if (kind == RowLayoutKind::Table && n)
                        std::fill_n(sizes, n, cap);
                }
            }
            std::byte *Pointer(size_t row) const
            {
                if (kind == RowLayoutKind::CSR)
                    return rowPointers[row];
                auto *base = denseAccess.View<std::byte>().data;
                return base ? base + row * capacity * elementBytes : nullptr;
            }
            std::byte *AllocateRow(size_t row, size_t bytes)
            {
                // CSR construction often creates thousands of tiny rows. One
                // allocation per node/cell dominates serial ingress and generic
                // topology setup, so carve stable row spans from large slabs.
                // Oversized rows receive a dedicated slab. A resize still
                // invalidates the old borrowed pointer, as documented above.
                constexpr size_t slabBytes = 64 * 1024;
                constexpr size_t alignment = 64;
                const size_t request = std::max(bytes, size_t(1));
                const size_t wanted = std::max(slabBytes, request + alignment);
                std::byte *pointer = nullptr;
                if (!slabs.empty())
                {
                    auto &slab = slabs.back();
                    auto *base = static_cast<std::byte *>(slab.buffer.LegacyPointer());
                    const auto address = reinterpret_cast<uintptr_t>(base + slab.used);
                    const auto padding = (alignment - address % alignment) % alignment;
                    if (slab.used + padding + request <= slab.buffer.Bytes())
                    {
                        slab.used += padding;
                        pointer = base + slab.used;
                        slab.used += request;
                    }
                }
                if (!pointer)
                {
                    slabs.push_back({Buffer::Allocate(wanted), 0});
                    auto &slab = slabs.back();
                    auto *base = static_cast<std::byte *>(slab.buffer.LegacyPointer());
                    const auto address = reinterpret_cast<uintptr_t>(base);
                    slab.used = (alignment - address % alignment) % alignment;
                    pointer = base + slab.used;
                    slab.used += request;
                }
                rowPointers[row] = pointer;
                rowCapacity[row] = bytes;
                return pointer;
            }
        };
        std::unique_ptr<State> _state;
        State &Live() const
        {
            DNDS_check_throw_info(_state, "row builder already committed or moved");
            return *_state;
        }
        explicit RowBuilder(std::unique_ptr<State> state) : _state(std::move(state)) {}

    public:
        RowBuilder(size_t rows, size_t elementBytes, RowLayoutKind kind = RowLayoutKind::CSR, size_t capacity = 0)
            : _state(std::make_unique<State>(rows, elementBytes, kind, capacity)) {}
        RowBuilder(const RowBuilder &) = delete;
        RowBuilder &operator=(const RowBuilder &) = delete;
        RowBuilder(RowBuilder &&) noexcept = default;
        RowBuilder &operator=(RowBuilder &&) noexcept = default;
        bool Active() const { return bool(_state); }
        /// Explicit detached edit: snapshot logical rows under one source lease.
        /// The original storage and all retained views remain unchanged. Pending
        /// source writes are rejected by the existing access contract.
        static RowBuilder FromStorage(const ArrayStorage &source)
        {
            const auto &shape = source.Layout();
            RowBuilder draft(shape.Rows(), source.ElementBytes(), shape.Kind(),
                             shape.IsVariable() ? shape.Stride() : shape.Width());
            const auto device = source.Values().GetDevice();
            auto read = source.Read({device, nullptr});
            const auto *values = read.View<std::byte>().data;
            for (size_t i = 0; i < shape.Rows(); ++i)
            {
                draft.ResizeRow(i, shape.RowSize(i), false);
                auto row = draft.RowBytes(i);
                if (row.size)
                    Detail::CopyBytes(values + shape.Offset(i) * source.ElementBytes(), device, row.data,
                                      Device::Host(), row.size, {device, nullptr})
                        .Wait();
            }
            return draft;
        }
        size_t Rows() const { return Live().rows; }
        size_t ElementBytes() const { return Live().elementBytes; }
        size_t RowSize(size_t row) const
        {
            const auto &s = Live();
            DNDS_check_throw_info(row < s.rows, "builder row out of range");
            return s.sizes[row];
        }
        void ResizeRow(size_t row, size_t length, bool preserve = true)
        {
            auto &s = Live();
            const auto previous = RowSize(row);
            const auto bytes = CheckedSize::Multiply(length, s.elementBytes);
            DNDS_check_throw_info(s.kind == RowLayoutKind::CSR || length <= s.capacity, "row size exceeds padded capacity");
            DNDS_check_throw_info(s.kind != RowLayoutKind::Table || length == s.width, "table row width is fixed");
            if (s.kind == RowLayoutKind::CSR && bytes > s.rowCapacity[row])
            {
                auto *previousPointer = s.Pointer(row);
                auto *replacement = s.AllocateRow(row, bytes);
                if (preserve && previous)
                    std::memcpy(replacement, previousPointer, std::min(previous, length) * s.elementBytes);
            }
            else if (length && (!preserve || length > previous))
            {
                const auto start = preserve ? previous : 0;
                std::memset(s.Pointer(row) + start * s.elementBytes, 0, (length - start) * s.elementBytes);
            }
            s.sizes[row] = length;
        }
        Span<std::byte> RowBytes(size_t row) const
        {
            const auto n = RowSize(row);
            return {Live().Pointer(row), CheckedSize::Multiply(n, Live().elementBytes), MemorySpace::Host};
        }
        template <class T>
        Span<T> Row(size_t row) const
        {
            static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
            DNDS_check_throw_info(sizeof(T) == Live().elementBytes, "builder element size mismatch");
            auto bytes = RowBytes(row);
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(bytes.data) % alignof(T) == 0, "builder element alignment mismatch");
            return {reinterpret_cast<T *>(bytes.data), bytes.size / sizeof(T), MemorySpace::Host};
        }
        ArrayStorage::Shared Commit()
        {
            auto &s = Live();
            std::vector<size_t> lengths;
            if (s.rows)
                lengths.assign(s.sizes, s.sizes + s.rows);
            auto layout = s.kind == RowLayoutKind::CSR ? RowLayout::Ragged(lengths) : s.kind == RowLayoutKind::Padded ? RowLayout::Padded(lengths, s.capacity)
                                                                                                                      : RowLayout::Dense(s.rows, s.width, s.capacity);
            Buffer values = s.dense;
            if (s.kind == RowLayoutKind::CSR)
            {
                values = Buffer::Allocate(CheckedSize::Multiply(layout.Elements(), s.elementBytes));
                auto write = values.Write();
                auto *target = write.View<std::byte>().data;
                for (size_t i = 0; i < s.rows; ++i)
                    if (lengths[i])
                        std::memcpy(target + layout.Offset(i) * s.elementBytes, s.Pointer(i), lengths[i] * s.elementBytes);
            }
            auto result = ArrayStorage::FromBuffer(std::move(layout), std::move(values), s.elementBytes);
            s.denseAccess.Wait();
            s.lengthAccess.Wait();
            _state.reset();
            return result;
        }
    };

    /// A unique published Array enters exclusive setup state. Alias creation,
    /// normal access and payload binding fail until Commit. Metadata descriptors
    /// remain immutable; new sizes/offsets are published once. Padded/table values
    /// are edited in place; CSR values pack once. Abandonment marks shape invalid.
    class ExclusiveRows
    {
        ArrayStorage::Shared _source;
        std::unique_ptr<RowBuilder> _draft;
        WriteAccess _oldValues;

        void Abort() noexcept
        {
            if (!_source)
                return;
            try
            {
                auto source = std::move(_source);
                std::lock_guard<std::mutex> lock(source->_shapeMutex);
                source->_shapeValid = false;
                source->_editing = false;
                _draft.reset();
                _oldValues.Fail();
            }
            catch (...)
            {
                std::terminate();
            }
        }

    public:
        explicit ExclusiveRows(const ArrayStorage::Shared &source)
        {
            DNDS_check_throw_info(source, "exclusive edit requires storage");
            std::lock_guard<std::mutex> lock(source->_shapeMutex);
            source->CheckPublished();
            DNDS_check_throw_info(source->_values.GetDevice() == Device::Host(), "exclusive row editing requires host storage");
            DNDS_check_throw_info(source.use_count() == 1 && source->_layout.use_count() == 1 &&
                                      source->_layout->rows.Offsets().IsUnique() && source->_layout->rows.Sizes().IsUnique() &&
                                      source->_values.IsUnique(),
                                  "exclusive edit requires unique unpinned storage and metadata; detach explicitly");
            auto write = source->_values.Write();
            try
            {
                const auto &shape = source->_layout->rows;
                const auto bytes = source->_elementBytes;
                if (shape.IsRagged())
                {
                    _draft = std::make_unique<RowBuilder>(shape.Rows(), bytes);
                    auto *values = write.View<std::byte>().data;
                    for (size_t i = 0; i < shape.Rows(); ++i)
                    {
                        _draft->ResizeRow(i, shape.RowSize(i));
                        auto row = _draft->RowBytes(i);
                        if (row.size)
                            std::memcpy(row.data, values + shape.Offset(i) * bytes, row.size);
                    }
                    _oldValues = std::move(write);
                }
                else
                {
                    auto state = std::make_unique<RowBuilder::State>(shape.Rows(), bytes, shape.Kind(),
                                                                     shape.IsPadded() ? shape.Stride() : shape.Width(),
                                                                     source->_values, std::move(write));
                    // Preserve physical table stride, even when width < capacity.
                    state->capacity = shape.Stride();
                    state->width = shape.Width();
                    for (size_t i = 0; i < shape.Rows(); ++i)
                        state->sizes[i] = shape.RowSize(i);
                    _draft.reset(new RowBuilder(std::move(state)));
                }
                _source = source;
                source->_editing = true;
            }
            catch (...)
            {
                write.CancelUnmodified();
                throw;
            }
        }
        ExclusiveRows(const ExclusiveRows &) = delete;
        ExclusiveRows &operator=(const ExclusiveRows &) = delete;
        ExclusiveRows(ExclusiveRows &&) noexcept = default;
        ExclusiveRows &operator=(ExclusiveRows &&other) noexcept
        {
            if (this != &other)
            {
                Abort();
                _source = std::move(other._source);
                _draft = std::move(other._draft);
                _oldValues = std::move(other._oldValues);
            }
            return *this;
        }
        ~ExclusiveRows() { Abort(); }
        RowBuilder &Draft() const
        {
            DNDS_check_throw_info(_source && _draft, "exclusive editor is inactive");
            return *_draft;
        }
        void Commit()
        {
            auto result = Draft().Commit();
            auto source = std::move(_source);
            std::lock_guard<std::mutex> lock(source->_shapeMutex);
            _oldValues.CancelUnmodified();
            source->_values = result->_values;
            source->_layout = result->_layout;
            ++source->_shapeVersion;
            source->_editing = false;
            _draft.reset();
        }
    };

    template <class T, class LayoutPolicy>
    ExclusiveRows Array<T, LayoutPolicy>::EditRows() { return ExclusiveRows(_storage); }
}
