#pragma once

#include "DNDS/Runtime/Execution/Device.hpp"
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace DNDS::Runtime
{
    /// Call-site descriptor. Strings supplied by DNDS_ALLOCATION_SITE have
    /// static lifetime; a diagnostic record copies them when it is named.
    struct AllocationSite
    {
        const char *file = "", *function = "";
        int line = 0;
    };
    struct AllocationActivity
    {
        size_t readers = 0, pendingReaders = 0, pendingWriters = 0;
        bool writer = false;
    };
    /// One physical allocation identity. Labels are aliases, not extra bytes.
    /// External providers may reserve more memory than their declared capacity.
    struct AllocationRecord
    {
        uint64_t identity = 0;
        Device device;
        size_t bytes = 0;
        bool externalProvider = false, streamOrderedMemory = false, pinnedHostMemory = false;
        std::vector<std::string> names;
        std::string file, function;
        /// Optional creation provenance, captured only for diagnostics. This
        /// identifies the allocation path, not every current sharing handle.
        std::string creationStack;
        int line = 0;
        AllocationActivity activity;
    };
    struct AllocationSnapshot
    {
        std::vector<AllocationRecord> allocations;
        size_t liveBytes = 0, peakBytes = 0, pinnedHostBytes = 0;
        /// Includes inactive ranges retained by pending operations. Does not
        /// claim allocator-pool reservation or physical GPU page residency.
        void Dump(std::ostream &output) const;
    };

    /// Opt-in, process-local registry with no allocation ownership. Enable before
    /// setup to cover the complete working set. Snapshot takes no MPI/CUDA action.
    /// Disabling clears diagnostic records only; it never frees Array storage.
    class AllocationDiagnostics
    {
    public:
        static void Enable(bool enabled = true);
        /// Capture allocation creation stacks while tracking is enabled.
        /// Off by default; DNDS_RUNTIME_ALLOCATION_STACKS=1 enables it at Enable.
        /// Symbol resolution costs setup time and must not be used for timings.
        static void CaptureStacks(bool enabled = true);
        static bool Enabled() noexcept;
        static AllocationSnapshot Snapshot();
    };

    namespace Detail
    {
        /// Buffer lifecycle hooks. Activity must capture a weak owner only.
        void TrackAllocation(uint64_t identity, size_t bytes, Device device,
                             bool external, bool streamOrdered, bool pinnedHost,
                             std::function<AllocationActivity()> activity);
        void ForgetAllocation(uint64_t identity) noexcept;
        void NameAllocation(uint64_t identity, const std::string &name, AllocationSite site);
    }
}

#if defined(_MSC_VER)
#    define DNDS_ALLOCATION_SITE() \
        ::DNDS::Runtime::AllocationSite { __FILE__, __FUNCSIG__, __LINE__ }
#elif defined(__GNUC__) || defined(__clang__)
#    define DNDS_ALLOCATION_SITE() \
        ::DNDS::Runtime::AllocationSite { __FILE__, __PRETTY_FUNCTION__, __LINE__ }
#else
#    define DNDS_ALLOCATION_SITE() \
        ::DNDS::Runtime::AllocationSite { __FILE__, __func__, __LINE__ }
#endif
