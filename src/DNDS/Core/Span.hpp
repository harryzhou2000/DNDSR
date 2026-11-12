#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/Errors.hpp"
#include <type_traits>

namespace DNDS
{
#if defined(DNDS_RUNTIME_ARRAY_BOUNDS_CHECKS)
    inline constexpr bool SpanBoundsChecks = DNDS_RUNTIME_ARRAY_BOUNDS_CHECKS != 0;
#else
    inline constexpr bool SpanBoundsChecks = true;
#endif
    /**
     * @brief Non-owning contiguous range with a signed element count.
     * @details The enclosing operator retains storage and access permission.
     * This trivially copied descriptor performs no allocation, lease lookup or
     * backend synchronization. Device use requires device-accessible storage.
     */
    template <class T, bool Checked = SpanBoundsChecks>
    class SpanView
    {
        T *_data = nullptr;
        index _size = 0;

    public:
        using value_type = T;
        DNDS_DEVICE_CALLABLE SpanView() = default;
        DNDS_DEVICE_CALLABLE SpanView(T *data, index count)
            : _data(data), _size(count)
        {
            DNDS_HD_assert(count >= 0 && (!count || data));
        }
        template <class U, bool OtherChecked,
                  std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
        DNDS_DEVICE_CALLABLE SpanView(SpanView<U, OtherChecked> other)
            : _data(other.data()), _size(other.size()) {}
        DNDS_DEVICE_CALLABLE index size() const { return _size; }
        DNDS_DEVICE_CALLABLE bool empty() const { return !_size; }
        DNDS_DEVICE_CALLABLE T *data() const { return _data; }
        DNDS_DEVICE_CALLABLE T *begin() const { return _data; }
        DNDS_DEVICE_CALLABLE T *end() const { return _size ? _data + _size : _data; }
        DNDS_DEVICE_CALLABLE T &operator[](index i) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(i >= 0 && i < _size);
            return _data[i];
        }
        DNDS_DEVICE_CALLABLE SpanView Subspan(index first, index count) const
        {
            DNDS_HD_assert(first >= 0 && count >= 0 && first <= _size &&
                           count <= _size - first);
            return {_data ? _data + first : nullptr, count};
        }
    };
}
