#pragma once

#include "NativeMeshCGNS.hpp"

namespace DNDS::Geom
{
    /// BC-zone driven one-to-one identification. Transform group i maps main
    /// (-i) physical coordinates to donor (-(i+3)); tolerance is absolute.
    /// Local, detached serial import operation over Runtime arrays. No MPI or
    /// legacy mesh is allocated. Only explicit serial-reader scratch scales
    /// with the serial file. Validates full high-order node bijections and
    /// rejects ambiguous/incomplete face pairing and inconsistent image chains.
    /// Original root node IDs survive compaction; removed IDs remain reserved.
    NativeSerialMeshData CanonicalizeNativePeriodic(NativeSerialMeshData source,
                                                    const NativePeriodicImportOptions &options);
}
