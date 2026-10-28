#pragma once

#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"

namespace DNDS::Geom
{
    /// Decode authoritative owned arrays and saved coverage policy without
    /// materializing ghosts. Matching/direct-partition I/O remains unchanged;
    /// redistribution or topology construction may precede final assembly.
    NativeOwnedMeshData ReadNativeMeshArchiveOwned(
        const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer, const std::string &name,
        Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown);
    /// Collective owned-row checkpoint, including every published family,
    /// named definition, periodic companion and persistent identity. The source
    /// remains read-protected. Finish asynchronous producers before calling;
    /// every rank uses the same serializer mode, name and call sequence. POD
    /// values use the existing Runtime homogeneous-ABI codec, not a portable
    /// cross-platform encoding. External solver fields are saved separately.
    void WriteNativeMeshArchive(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                Serializer::SerializerBaseSSP serializer, const std::string &name);

    /// Read directly into native owned arrays and rebuild sparse ghost closure.
    /// Unknown restores saved rank partitions; EvenSplit reads each entity
    /// family's contiguous even partition, including with a different MPI size.
    /// Per-rank JSON supports matching partitions only. Arbitrary overlapping or
    /// permuted read windows are rejected: topology globals require exhaustive,
    /// rank-ordered ownership. Use a subsequent native transition to repartition.
    /// All ranks participate, including empty ones. No legacy mesh is created.
    /// Metadata v2 preserves the full declared GhostSpec; v1 selects the default.
    /// Positive ghostLayers explicitly overrides standard Cell2Cell prefixes
    /// while preserving other saved custom paths. It must match on all ranks.
    std::shared_ptr<const UnstructuredMeshState> ReadNativeMeshArchive(
        const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer, const std::string &name,
        Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown, int ghostLayers = 1);
}
