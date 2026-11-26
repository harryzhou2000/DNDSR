#pragma once

#include <memory>
#include <mutex>
#include <utility>

namespace DNDS
{
    /// Thread-safe handle wrapper around std::shared_ptr<T> for C++17.
    /// Prevents concurrent unsynchronized member read/write races on the
    /// shared_ptr object itself while letting callers copy out independent
    /// shared_ptr snapshots without holding any lock during subsequent reads.
    template <class T>
    class LockedSharedPtr
    {
        mutable std::mutex _mutex;
        std::shared_ptr<T> _ptr;

    public:
        LockedSharedPtr() = default;
        explicit LockedSharedPtr(std::shared_ptr<T> ptr) : _ptr(std::move(ptr)) {}
        LockedSharedPtr(const LockedSharedPtr &) = delete;
        LockedSharedPtr &operator=(const LockedSharedPtr &) = delete;

        std::shared_ptr<T> Load() const
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return _ptr;
        }
        bool HasValue() const
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return bool(_ptr);
        }
        void Store(std::shared_ptr<T> next)
        {
            std::shared_ptr<T> previous;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                previous = std::move(_ptr);
                _ptr = std::move(next);
            }
        }
        void Reset()
        {
            Store({});
        }
        void Swap(std::shared_ptr<T> &other) noexcept
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _ptr.swap(other);
        }
        template <class F>
        std::shared_ptr<T> Update(F &&transform)
        {
            std::shared_ptr<T> previous;
            std::shared_ptr<T> current;
            {
                std::lock_guard<std::mutex> lock(_mutex);
                current = transform(_ptr);
                previous = std::move(_ptr);
                _ptr = current;
            }
            return current;
        }
    };
}
