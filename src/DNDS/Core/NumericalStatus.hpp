#pragma once

#include "DNDS/Core/Defines.hpp"
#include <type_traits>
#include <utility>

namespace DNDS
{
    /// Numerical failures cross kernel boundaries as values, never exceptions.
    /// Backend-specific detail is a stable provider code, not a borrowed
    /// exception message or an allocation made while handling an error.
    enum class NumericalError
    {
        None,
        InvalidInput,
        InvalidState,
        UnsupportedRequest,
        NoConvergence,
        BackendFailure,
        AllocationFailure,
        UnexpectedFailure
    };

    /// Plain status suitable for a CPU return or a device error buffer.
    /// Failed outputs are unspecified and must not be published. The first
    /// failing row/point can be attached without allocating diagnostic text.
    struct [[nodiscard]] NumericalStatus
    {
        NumericalError code = NumericalError::None;
        int detail = 0;
        index row = -1, point = -1;

        DNDS_DEVICE_CALLABLE explicit operator bool() const noexcept
        {
            return code == NumericalError::None;
        }
        DNDS_DEVICE_CALLABLE NumericalStatus At(index iRow, index iPoint = -1) const noexcept
        {
            return {code, detail, iRow, iPoint};
        }
    };

    /// A value accompanies success only. Callers inspect status before reading
    /// value; no implicit throw, allocation or diagnostic emission is involved.
    template <class T>
    struct [[nodiscard]] NumericalResult
    {
        T value{};
        NumericalStatus status{};
        DNDS_DEVICE_CALLABLE NumericalResult(T result) noexcept(std::is_nothrow_move_constructible_v<T>) : value(std::move(result)) {}
        DNDS_DEVICE_CALLABLE NumericalResult(NumericalStatus failure) noexcept(std::is_nothrow_default_constructible_v<T>) : status(failure) {}
        DNDS_DEVICE_CALLABLE explicit operator bool() const noexcept { return bool(status); }
    };
}
