#pragma once

#include "NativeMeshState.hpp"
#include "RelationAccess.hpp"

namespace DNDS::Geom
{
    /// Complete parent/node setup star derived from authoritative owned cones.
    /// Source numbering is retained; appended rows are fetched sparsely through
    /// node support even when the source CellCells graph is incomplete.
    struct NativeCellClosure
    {
        NativeRelation cellNodes;
        Runtime::ArrayPair<ElemInfo> cellInfo;
    };

    NativeCellClosure BuildNativeCellClosure(const NativeMeshState &source, const MPIInfo &mpi);
}
