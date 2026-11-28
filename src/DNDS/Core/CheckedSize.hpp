#pragma once
#include "DNDS/Core/Errors.hpp"
#include <limits>
#include <type_traits>

namespace DNDS::CheckedSize
{
    // Host-side size arithmetic. Validate operands before evaluating expressions.
    template <class T>
    T Add(T a, T b)
    {
        static_assert(std::is_integral_v<T>);
        if constexpr (std::is_signed_v<T>)
            DNDS_check_throw_info(a >= 0 && b >= 0, "negative size in size addition");
        DNDS_check_throw_info(a <= std::numeric_limits<T>::max() - b,
                              "size addition overflow");
        return a + b;
    }

    template <class OutT, class InT>
    OutT Cast(InT v)
    {
        static_assert(std::is_integral_v<OutT> && std::is_integral_v<InT>);
        if constexpr (std::is_signed_v<InT>)
            DNDS_check_throw_info(v >= 0, "negative size cast");
        using WideU = std::common_type_t<std::make_unsigned_t<OutT>, std::make_unsigned_t<InT>>;
        DNDS_check_throw_info(static_cast<WideU>(v) <= static_cast<WideU>(std::numeric_limits<OutT>::max()),
                              "size cast overflow");
        return static_cast<OutT>(v);
    }

    template <class T>
    T Multiply(T a, T b)
    {
        static_assert(std::is_integral_v<T>);
        if constexpr (std::is_signed_v<T>)
            DNDS_check_throw_info(a >= 0 && b >= 0, "negative size in size multiplication");
        DNDS_check_throw_info(b == 0 || a <= std::numeric_limits<T>::max() / b,
                              "size multiplication overflow");
        return a * b;
    }
}
