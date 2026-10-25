#pragma once

#include "Mesh.hpp"

namespace DNDS::Geom
{
    /// Collective staging over the current authority. No writable mesh pointer
    /// escapes: getters return values, setters change an independent candidate.
    /// Explicit Commit publishes native backing; Abort/destruction discards it.
    /// Initial topology support is CellCells neighbor-order permutation only.
    class MeshEdit
    {
        std::shared_ptr<UnstructuredMesh> _source, _candidate;
        std::optional<NativeMeshData> _nativeCandidate;
        std::unique_ptr<MeshReadAccess> _sourceRead;
        Runtime::VersionStamp _stamp;
        MeshReadRequirements _requirements;
        std::vector<Elem::ElemType> _cellTypes;
        std::vector<std::vector<index>> _neighbors;
        unsigned _changes = 0;
        bool _closed = false;

        MeshEdit() = default;
        void Check() const;
        static void CheckRow(index row, index count);

    public:
        static MeshEdit Begin(std::shared_ptr<UnstructuredMesh> source, int ghostLayers = 1);
        MeshEdit(const MeshEdit &) = delete;
        MeshEdit &operator=(const MeshEdit &) = delete;
        MeshEdit(MeshEdit &&) noexcept = default;
        MeshEdit &operator=(MeshEdit &&) = delete;
        bool Closed() const { return _closed || !_source; }
        index NumNode() const;
        index NumCell() const;
        tPoint Coord(index row) const;
        void SetCoord(index row, const tPoint &value);
        ElemInfo CellInfo(index row) const;
        void SetCellInfo(index row, const ElemInfo &value);
        std::vector<index> CellNeighbors(index row) const;
        void SetCellNeighbor(index row, rowsize slot, index value);
        void Commit();
        void Abort();
    };
}
