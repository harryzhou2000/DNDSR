#pragma once

#include "DNDS/Array.hpp"
#include "Array.hpp"
#include "RowTransfer.hpp"
#include <limits>
#include <type_traits>

namespace DNDS::Runtime
{
    namespace Detail
    {
        template <class Integer>
        size_t LegacyExtent(Integer value)
        {
            static_assert(std::is_integral_v<Integer>);
            DNDS_check_throw_info(value >= 0 && uintmax_t(value) <= uintmax_t(SIZE_MAX),
                                  "legacy shape contains negative or overflowing extent");
            return size_t(value);
        }

        /// Convert once at publication. Never reinterpret index/rowsize as size_t.
        template <class T, rowsize RS, rowsize RM, rowsize A>
        RowLayout LegacyHostLayout(const DNDS::Array<T, RS, RM, A> &source)
        {
            using Legacy = DNDS::Array<T, RS, RM, A>;
            constexpr auto kind = Legacy::GetDataLayoutStatic();
            const auto rows = LegacyExtent(source.Size());
            if constexpr (kind == TABLE_StaticFixed || kind == TABLE_Fixed)
                return RowLayout::Dense(rows, LegacyExtent(source.RowSize()));
            else if constexpr (kind == CSR)
            {
                DNDS_check_throw_info(source.IfCompressed(), "publish compressed CSR before host sharing");
                auto starts = source.getRowStart();
                DNDS_check_throw_info(starts && starts->size() == CheckedSize::Add(rows, size_t(1)),
                                      "legacy CSR offset count does not match rows");
                auto read = starts->HostRead();
                const auto *offsets = read.template View<index>().data;
                DNDS_check_throw_info(offsets[0] == 0, "legacy CSR must start at zero");
                std::vector<size_t> lengths;
                lengths.reserve(rows);
                size_t previous = 0;
                for (size_t i = 0; i < rows; ++i)
                {
                    const auto next = LegacyExtent(offsets[i + 1]);
                    DNDS_check_throw_info(next >= previous && next - previous <= size_t(std::numeric_limits<rowsize>::max()),
                                          "legacy CSR offsets decrease or row exceeds rowsize");
                    lengths.push_back(next - previous);
                    previous = next;
                }
                DNDS_check_throw_info(previous == source.DataSize(), "legacy CSR sentinel does not match values");
                return RowLayout::Ragged(lengths);
            }
            else
            {
                const auto capacity = LegacyExtent(source.DataStride());
                auto sizes = source.getRowSizes();
                if (!rows && !sizes)
                    return RowLayout::Padded({}, capacity);
                DNDS_check_throw_info(sizes && sizes->size() == rows, "legacy padded size count does not match rows");
                auto read = sizes->HostRead();
                const auto *used = read.template View<rowsize>().data;
                std::vector<size_t> lengths;
                lengths.reserve(rows);
                for (size_t i = 0; i < rows; ++i)
                    lengths.push_back(LegacyExtent(used[i]));
                return RowLayout::Padded(lengths, capacity);
            }
        }

        template <class T, rowsize RS, rowsize RM, rowsize A>
        ArrayStorage::Shared LegacyHostStorage(const DNDS::Array<T, RS, RM, A> &source, Device device)
        {
            static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> && alignof(T) <= 64,
                          "legacy bridge requires native-compatible POD elements");
            DNDS_check_throw_info(device == Device::Host(), "legacy bridge supports explicit host sharing only");
            auto values = source.HostBufferSnapshot();
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(values.LegacyPointer()) % alignof(T) == 0,
                                  "legacy bridge element alignment mismatch");
            return ArrayStorage::FromBuffer(LegacyHostLayout(source), std::move(values), sizeof(T));
        }

        inline void ValidateLegacyRows(const RowLayout &layout, Device device)
        {
            DNDS_check_throw_info(device == Device::Host(), "legacy host rows require host storage");
            DNDS_check_throw_info(layout.Rows() <= size_t(std::numeric_limits<index>::max()), "native row count exceeds legacy index");
            if (!layout.IsVariable())
                DNDS_check_throw_info(layout.Width() <= size_t(std::numeric_limits<rowsize>::max()), "native row width exceeds legacy rowsize");
            else
                for (size_t i = 0; i < layout.Rows(); ++i)
                    DNDS_check_throw_info(layout.RowSize(i) <= size_t(std::numeric_limits<rowsize>::max()), "native row width exceeds legacy rowsize");
        }
    }

    /// Named writable alias: same allocation, byte extent, element type and access
    /// controller; immutable converted shape. Source structural replacement never
    /// retargets an existing alias. Call only at a legacy setup phase barrier.
    template <class T, rowsize RS, rowsize RM, rowsize A>
    Array<T> ShareLegacyHost(DNDS::Array<T, RS, RM, A> &source, Device device = Device::Host())
    {
        return Array<T>::FromSharedStorage(Detail::LegacyHostStorage(source, device));
    }

    /// A registered MPI/POD schema must describe this actual C++ element type
    /// and sizeof(T) extent. Untagged custom schemas retain their caller contract.
    template <class T, rowsize RS, rowsize RM, rowsize A>
    Array<T> ShareLegacyHost(DNDS::Array<T, RS, RM, A> &source, const ElementSchema &schema,
                             Device device = Device::Host())
    {
        DNDS_check_throw_info(schema.Extent() == sizeof(T) && schema.Compatible(ElementSchema::TypeIdentity<T>()),
                              "legacy bridge element schema/type mismatch");
        return ShareLegacyHost(source, device);
    }

    /// Borrowed pointer/shape bundle for existing row kernels. No ownership,
    /// locks, shared_ptr copies, placement checks or allocation in these methods.
    /// Valid only while its outer LegacyRowsAccess remains active.
    template <class T>
    struct LegacyRowsView
    {
        ArrayView<T> values;
        DNDS_DEVICE_CALLABLE index Size() const { return index(values.Rows()); }
        DNDS_DEVICE_CALLABLE rowsize RowSize(index row) const { return rowsize(values.RowSize(size_t(row))); }
        DNDS_DEVICE_CALLABLE T *operator[](index row) const { return values.Row(size_t(row)); }
        DNDS_DEVICE_CALLABLE Runtime::RowSpan<T> RowSpan(index row) const { return values[size_t(row)]; }
        DNDS_DEVICE_CALLABLE T &operator()(index row, rowsize column) const { return values(size_t(row), size_t(column)); }
    };

    template <class T, bool Writable>
    class LegacyRowsAccess
    {
        using Value = std::conditional_t<Writable, T, const T>;
        ArrayAccess<T, Writable> _access;

    public:
        /// Factories validate host placement and legacy extents before acquiring
        /// a writer, so a rejected adaptation cannot poison valid values.
        explicit LegacyRowsAccess(ArrayAccess<T, Writable> access) : _access(std::move(access)) {}
        LegacyRowsAccess(const LegacyRowsAccess &) = delete;
        LegacyRowsAccess &operator=(const LegacyRowsAccess &) = delete;
        LegacyRowsAccess(LegacyRowsAccess &&) noexcept = default;
        LegacyRowsAccess &operator=(LegacyRowsAccess &&) noexcept = default;
        LegacyRowsView<Value> View() const { return {_access.View()}; }
        ArrayView<Value> NativeView() const { return _access.View(); }
        void Wait() { _access.Wait(); }
    };

    template <class T, class Policy>
    LegacyRowsAccess<T, false> HostRowsRead(const Array<T, Policy> &source)
    {
        Detail::ValidateLegacyRows(source.Layout(), source.GetDevice());
        return LegacyRowsAccess<T, false>(source.Read());
    }
    template <class T, class Policy>
    LegacyRowsAccess<T, true> HostRowsWrite(Array<T, Policy> &source)
    {
        Detail::ValidateLegacyRows(source.Layout(), source.GetDevice());
        return LegacyRowsAccess<T, true>(source.Write());
    }
    template <class T, rowsize RS, rowsize RM, rowsize A>
    LegacyRowsAccess<T, false> HostRowsRead(const DNDS::Array<T, RS, RM, A> &source)
    {
        auto storage = Detail::LegacyHostStorage(source, Device::Host());
        return LegacyRowsAccess<T, false>({storage->Read(), storage->SharedLayout()});
    }
    template <class T, rowsize RS, rowsize RM, rowsize A>
    LegacyRowsAccess<T, true> HostRowsWrite(DNDS::Array<T, RS, RM, A> &source)
    {
        auto storage = Detail::LegacyHostStorage(source, Device::Host());
        return LegacyRowsAccess<T, true>({storage->Write(), storage->SharedLayout()});
    }
}
