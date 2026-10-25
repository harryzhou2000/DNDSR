#pragma once

#include "NativeMeshAssembler.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"

namespace DNDS::Geom
{
    /// Collective owned-row checkpoint, including every published family,
    /// named definition, periodic companion and persistent identity. The source
    /// remains read-protected. Finish asynchronous producers before calling;
    /// every rank uses the same serializer mode, name and call sequence. POD
    /// values use the existing Runtime homogeneous-ABI codec, not a portable
    /// cross-platform encoding. External solver fields are saved separately.
    void WriteNativeMeshArchive(const NativeMeshState &source, const MPIInfo &mpi,
                                Serializer::SerializerBaseSSP serializer, const std::string &name);

    /// Read directly into native owned arrays and rebuild sparse ghost closure.
    /// Unknown restores saved rank partitions; EvenSplit reads each entity
    /// family's contiguous even partition, including with a different MPI size.
    /// Per-rank JSON supports matching partitions only. Arbitrary overlapping or
    /// permuted read windows are rejected: topology globals require exhaustive,
    /// rank-ordered ownership. Use a subsequent native transition to repartition.
    /// All ranks participate, including empty ones. No legacy mesh is created.
    std::shared_ptr<const NativeMeshState> ReadNativeMeshArchive(
        const MPIInfo &mpi, Serializer::SerializerBaseSSP serializer, const std::string &name,
        Serializer::ArrayGlobalOffset partition = Serializer::ArrayGlobalOffset_Unknown, int ghostLayers = 1);
}
