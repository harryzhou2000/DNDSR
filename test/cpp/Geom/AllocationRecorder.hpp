#pragma once
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <new>
namespace AllocationRecorder
{
    // Process-lifetime counters prevent dangling observers for allocations
    // released after a phase. Only the calling thread joins the current epoch.
    struct Counters
    {
        std::atomic<size_t> live{0}, peak{0}, calls{0};
    };
    inline Counters counters[32];
    inline thread_local unsigned epoch = 0;
    inline unsigned nextEpoch = 0;
    struct Header
    {
        void *base;
        size_t bytes;
        unsigned epoch;
    };
    inline void *Allocate(size_t bytes, size_t align)
    {
        align = align < alignof(Header) ? alignof(Header) : align;
        if (bytes > std::numeric_limits<size_t>::max() - align - sizeof(Header))
            throw std::bad_alloc();
        void *base = std::malloc(bytes + align + sizeof(Header));
        if (!base)
            throw std::bad_alloc();
        auto address = (reinterpret_cast<uintptr_t>(base) + sizeof(Header) + align - 1) & ~uintptr_t(align - 1);
        auto *header = reinterpret_cast<Header *>(address) - 1;
        *header = {base, bytes, epoch};
        if (epoch)
        {
            auto &c = counters[epoch];
            const auto live = c.live.fetch_add(bytes) + bytes;
            c.calls.fetch_add(1);
            auto peak = c.peak.load();
            while (peak < live && !c.peak.compare_exchange_weak(peak, live))
            {
            }
        }
        return reinterpret_cast<void *>(address);
    }
    inline void Free(void *pointer) noexcept
    {
        if (!pointer)
            return;
        auto *header = reinterpret_cast<Header *>(pointer) - 1;
        if (header->epoch)
            counters[header->epoch].live.fetch_sub(header->bytes);
        std::free(header->base);
    }
    struct Phase
    {
        unsigned id;
        Phase() : id(++nextEpoch)
        {
            if (id >= 32 || epoch)
                std::abort();
            epoch = id;
        }
        ~Phase() { epoch = 0; }
        size_t Peak() const { return counters[id].peak.load(); }
        size_t Calls() const { return counters[id].calls.load(); }
    };
}
void *operator new(size_t n) { return AllocationRecorder::Allocate(n, alignof(std::max_align_t)); }
void *operator new[](size_t n) { return ::operator new(n); }
void *operator new(size_t n, std::align_val_t a) { return AllocationRecorder::Allocate(n, size_t(a)); }
void *operator new[](size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void *p) noexcept { AllocationRecorder::Free(p); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, size_t) noexcept { ::operator delete(p); }
void operator delete(void *p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete(void *p, size_t, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, size_t, std::align_val_t) noexcept { ::operator delete(p); }
void *operator new(size_t n, const std::nothrow_t &) noexcept
{
    try
    {
        return ::operator new(n);
    }
    catch (...)
    {
        return nullptr;
    }
}
void *operator new[](size_t n, const std::nothrow_t &) noexcept { return ::operator new(n, std::nothrow); }
void *operator new(size_t n, std::align_val_t a, const std::nothrow_t &) noexcept
{
    try
    {
        return ::operator new(n, a);
    }
    catch (...)
    {
        return nullptr;
    }
}
void *operator new[](size_t n, std::align_val_t a, const std::nothrow_t &) noexcept { return ::operator new(n, a, std::nothrow); }
void operator delete(void *p, const std::nothrow_t &) noexcept { ::operator delete(p); }
void operator delete[](void *p, const std::nothrow_t &) noexcept { ::operator delete(p); }
void operator delete(void *p, std::align_val_t, const std::nothrow_t &) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::align_val_t, const std::nothrow_t &) noexcept { ::operator delete(p); }
