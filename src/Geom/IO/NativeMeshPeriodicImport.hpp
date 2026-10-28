#pragma once

#include "Geom/IO/NativeMeshCGNS.hpp"

namespace DNDS::Geom
{
    /// BC-zone driven one-to-one identification. Transform group i maps main
    /// (-i) physical coordinates to donor (-(i+3)); tolerance is absolute.
    /// Local, detached serial import operation over Runtime arrays. No MPI or
    /// legacy mesh is allocated. Only explicit serial-reader scratch scales
    /// with the serial file. Validates full high-order node bijections and
    /// rejects ambiguous/incomplete face pairing and inconsistent image chains.
    /// Main-image coordinates and incidence bits are retained, but canonical
    /// node IDs are compacted to [0, number of retained nodes), matching legacy
    /// periodic ingress. Pre-identification source labels are not alias IDs.
    /// Existing ID high-water reservations are not lowered.
    NativeSerialMeshData CanonicalizeNativePeriodic(NativeSerialMeshData source,
                                                    const NativePeriodicImportOptions &options);
}
