#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::Runtime::Detail
{
    /// One logical row in peer wire order; offsets/strides are in schema extents.
    /// Owned host metadata compiles these once. Kernels never read host layouts.
    struct DevicePackRow
    {
        index base = 0, stride = 1, packed = 0, width = 0;
    };
    /// Described POD bytes only; padding/untransmitted fields stay untouched.
    struct DevicePackField
    {
        index offset = 0, bytes = 0;
    };
    void LaunchDevicePacking(ExecutionContext context, const void *source, void *destination,
                             const DevicePackRow *rows, index rowCount, const DevicePackField *fields,
                             index fieldCount, index extent, bool unpack);
}
