#pragma once
#include <cstddef>
#include <cstdint>

namespace DNDS::IO
{
    /// Non-cryptographic native-format fingerprint of a keyed contiguous record.
    /// The avalanche finalizer permits order-independent XOR of independently keyed rows.
    inline uint64_t ContentHash(uint64_t key, const void *payload, size_t size)
    {
        uint64_t h = 14695981039346656037ull;
        const auto add = [&](uint8_t byte)
        { h = (h ^ byte) * 1099511628211ull; };
        for (int i = 0; i < 8; ++i)
            add(uint8_t(key >> (8 * i)));
        const auto *bytes = static_cast<const uint8_t *>(payload);
        for (size_t i = 0; i < size; ++i)
            add(bytes[i]);
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
        return h ^ (h >> 31);
    }
}
