#pragma once

#include "Buffer.hpp"
#include <algorithm>
#include <vector>
#include <utility>

namespace DNDS::Runtime
{
    /// Published host metadata. Copies share the allocation; only const spans
    /// escape. Construction copies once into an ordinary tracked Buffer. Shape
    /// editors publish a new instance instead of editing an escaped descriptor.
    template <class T>
    class MetadataBuffer
    {
        Buffer _buffer;
        const T *_data = nullptr;
        size_t _size = 0;

    public:
        MetadataBuffer() = default;
        MetadataBuffer(const MetadataBuffer &) = default;
        MetadataBuffer &operator=(const MetadataBuffer &) = default;
        MetadataBuffer(MetadataBuffer &&other) noexcept
            : _buffer(std::move(other._buffer)), _data(std::exchange(other._data, nullptr)), _size(std::exchange(other._size, 0)) {}
        MetadataBuffer &operator=(MetadataBuffer &&other) noexcept
        {
            if (this != &other)
            {
                _buffer = std::move(other._buffer);
                _data = std::exchange(other._data, nullptr);
                _size = std::exchange(other._size, 0);
            }
            return *this;
        }
        explicit MetadataBuffer(const std::vector<T> &values)
            : _buffer(Buffer::Allocate(CheckedSize::Multiply(values.size(), sizeof(T)))), _size(values.size())
        {
            auto write = _buffer.Write();
            auto view = write.template View<T>();
            if (_size)
                std::copy(values.begin(), values.end(), view.data);
            _data = view.data;
        }
        /// Fill the final allocation in ordinal order, then publish read-only
        /// metadata. The generator returns values; no mutable pointer escapes.
        /// An exception abandons the unpublished allocation.
        template <class F>
        static MetadataBuffer Generate(size_t count, F &&valueAt)
        {
            DNDS_check_throw_info(count <= size_t(INT64_MAX), "metadata element count exceeds signed index range");
            MetadataBuffer result;
            result._buffer = Buffer::Allocate(CheckedSize::Multiply(count, sizeof(T)));
            result._size = count;
            auto write = result._buffer.Write();
            auto view = write.template View<T>();
            for (index i = 0; i < index(count); ++i)
                view.data[i] = valueAt(i);
            result._data = view.data;
            return result;
        }
        const T *data() const { return _data; }
        size_t size() const { return _size; }
        bool empty() const { return !_size; }
        const T *begin() const { return _data; }
        const T *end() const { return _size ? _data + _size : _data; }
        const T &operator[](size_t i) const { return _data[i]; }
        const T &back() const { return _data[_size - 1]; }
        uint64_t Identity() const { return _buffer.Identity(); }
        /// Read-only accounting of retained allocation capacity, without an access lease.
        template <class F>
        void VisitAllocations(F &&visit) const
        {
            if (_buffer.Identity())
                visit(_buffer.Identity(), _buffer.AllocationHandle().Bytes(), _buffer.GetDevice());
        }
        bool IsUnique() const { return !_buffer.Identity() || _buffer.IsUnique(); }
        ReadAccess Read() const { return _buffer.Read(); }
        bool operator==(const MetadataBuffer &other) const
        {
            return _size == other._size && (_data == other._data || std::equal(begin(), end(), other.begin()));
        }
    };
}
