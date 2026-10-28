#pragma once

#include "Geom/Unstructured/OwnedMeshData.hpp"
#include "Geom/Access/RelationAccess.hpp"

namespace DNDS::Geom
{
    /// Complete parent/node setup star derived from authoritative owned cones.
    /// Source numbering is retained; appended rows are fetched sparsely through
    /// node support even when the source CellCells graph is incomplete.
    struct NativeCellClosure
    {
        NativeRelation cellNodes;
        Runtime::ArrayPair<ElemInfo> cellInfo;
        std::optional<Runtime::ArrayPair<NodePeriodicBits>> cellNodesPbi;
    };

    /// Temporary construction inputs from owned authoritative global cones.
    /// Does not assemble/publish a mesh or depend on existing compute ghosts.
    NativeCellClosure BuildNativeCellClosure(const NativeOwnedMeshData &source, const MPIInfo &mpi);
    /// Adapter for a construction operation starting from an existing mesh.
    /// Captures owned inputs only, then uses the same temporary closure path.
    NativeCellClosure BuildNativeCellClosure(const UnstructuredMeshState &source, const MPIInfo &mpi);
}
