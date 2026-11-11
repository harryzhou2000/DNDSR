#pragma once
/// @file OptionalRef.hpp
/// @brief A lightweight non-owning nullable reference wrapper.
///
/// `OptionalRef` is a synchronous borrow passed by value. It never extends the
/// referenced object's lifetime and therefore must not be captured as the sole
/// operand owner by a queued CPU, GPU, or communication operation. Operation
/// records retain the owning handle separately and may use `OptionalRef` only
/// while resolving their already-owned operands.

#include "DNDS/Core/Errors.hpp"
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace DNDS
{

    template <typename T>
    class OptionalRef
    {
        T *_ptr = nullptr;

    public:
        OptionalRef() = default;

        OptionalRef(std::nullptr_t) {}

        explicit OptionalRef(T &ref) noexcept : _ptr(&ref) {}

        template <class U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
        explicit OptionalRef(std::reference_wrapper<U> ref) noexcept : _ptr(&ref.get()) {}

        /// Permit the usual mutable-to-const borrow conversion without allowing
        /// const removal or unrelated reference types.
        template <class U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
        OptionalRef(OptionalRef<U> other) noexcept : _ptr(other.Get()) {}

        [[nodiscard]] explicit operator bool() const noexcept { return _ptr != nullptr; }

        [[nodiscard]] bool HasValue() const noexcept { return _ptr != nullptr; }
        [[nodiscard]] bool has_value() const noexcept { return HasValue(); }

        T &operator*() const
        {
            return Value();
        }

        T *operator->() const
        {
            DNDS_check_throw_info(_ptr, "empty optional reference access");
            return _ptr;
        }

        T &Value() const
        {
            DNDS_check_throw_info(_ptr, "empty optional reference access");
            return *_ptr;
        }
        T &value() const { return Value(); }

        [[nodiscard]] T *Get() const noexcept { return _ptr; }
        [[nodiscard]] T *get() const noexcept { return Get(); }

        void Reset() noexcept { _ptr = nullptr; }
        void reset() noexcept { Reset(); }

        template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
        void Emplace(U &ref) noexcept
        {
            _ptr = &ref;
        }
        template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
        void emplace(U &ref) noexcept { Emplace(ref); }

        friend bool operator==(const OptionalRef &a, const OptionalRef &b)
        {
            return a._ptr == b._ptr;
        }
        friend bool operator!=(const OptionalRef &a, const OptionalRef &b)
        {
            return a._ptr != b._ptr;
        }
    };

} // namespace DNDS
