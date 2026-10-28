#pragma once

#include "Geom/Access/MeshView.hpp"

namespace DNDS::Geom
{
    /// Borrowed readonly segment. Dot and arrow access cover the historical
    /// device-view and shared-array spellings without an Array allocation.
    template <class T, bool Vector3 = false>
    class MeshLegacyArrayRead
    {
        Runtime::ArrayView<const T> _view{};
        bool _present = false;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(MeshLegacyArrayRead, MeshLegacyArrayRead)
        DNDS_DEVICE_CALLABLE MeshLegacyArrayRead(Runtime::ArrayView<const T> view, bool present)
            : _view(view), _present(present) {}
        DNDS_DEVICE_CALLABLE index Size() const { return index(_view.Rows()); }
        DNDS_DEVICE_CALLABLE rowsize RowSize(index row) const { return rowsize(_view.RowSize(row)); }
        DNDS_DEVICE_CALLABLE bool isBuilt() const { return _present; }
        DNDS_DEVICE_CALLABLE explicit operator bool() const { return _present; }
        DNDS_DEVICE_CALLABLE const MeshLegacyArrayRead *operator->() const { return this; }
        DNDS_DEVICE_CALLABLE const T *rowPtr(index row) const { return _view.Row(row); }
        DNDS_DEVICE_CALLABLE const T &operator()(index row, rowsize column) const { return _view(row, column); }
        DNDS_DEVICE_CALLABLE auto operator[](index row) const
        {
            if constexpr (Vector3)
                return Eigen::Map<const tPoint>(_view.Row(row));
            else
                return MeshRowSpan<T>{_view.Row(row), _view.RowSize(row)};
        }
    };

    /// Combined row indexing preserves separate owned/ghost buffers. These
    /// descriptors retain neither allocations nor leases and expose no writer,
    /// transformer, resize or implicit conversion to a legacy Array reference.
    template <class T, bool Vector3 = false>
    class MeshLegacyPairRead
    {
        bool _present = false;

    public:
        MeshLegacyArrayRead<T, Vector3> father, son;
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(MeshLegacyPairRead, MeshLegacyPairRead)
        DNDS_DEVICE_CALLABLE MeshLegacyPairRead(Runtime::PairReadView<T> view, bool present)
            : _present(present), father(view.owned, present), son(view.ghost, present) {}
        DNDS_DEVICE_CALLABLE index Size() const { return father.Size() + son.Size(); }
        DNDS_DEVICE_CALLABLE rowsize RowSize(index row) const
        {
            return row < father.Size() ? father.RowSize(row) : son.RowSize(row - father.Size());
        }
        DNDS_DEVICE_CALLABLE bool isBuilt() const { return _present; }
        DNDS_DEVICE_CALLABLE const T &operator()(index row, rowsize column) const
        {
            return row < father.Size() ? father(row, column) : son(row - father.Size(), column);
        }
        DNDS_DEVICE_CALLABLE auto operator[](index row) const
        {
            return row < father.Size() ? father[row] : son[row - father.Size()];
        }
    };

    /// Readonly first-column spelling used by the former bnd2faceV vector.
    class MeshLegacyColumnRead
    {
        Runtime::PairReadView<index> _view{};

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(MeshLegacyColumnRead, MeshLegacyColumnRead)
        DNDS_DEVICE_CALLABLE explicit MeshLegacyColumnRead(Runtime::PairReadView<index> view) : _view(view) {}
        DNDS_DEVICE_CALLABLE size_t size() const { return _view.Rows(); }
        DNDS_DEVICE_CALLABLE const index &operator[](index row) const { return _view(row, 0); }
        DNDS_HOST const index &at(index row) const
        {
            DNDS_check_throw_info(row >= 0 && size_t(row) < _view.Rows(), "boundary face row is outside the retained view");
            return _view(row, 0);
        }
    };

    /// Legacy read spelling over immutable native buffers. The semantic base
    /// remains available for shared helpers. Copies contain borrowed descriptors
    /// only; the acquiring MeshReadAccess must outlive every CPU/device use.
    /// No descriptor points back into this object, so copies and moves are safe.
    /// Raw legacy member rows retain dynamic compatibility spans. Use semantic
    /// FaceCells()/BoundaryCells() for compile-time two-slot physical cell rows.
    class UnstructuredMeshLegacyView : public UnstructuredMeshView
    {
        struct NoPeriodicOperation
        {
            DNDS_DEVICE_CALLABLE void operator()() const {}
        };

    public:
        int dim = -1;
        bool isPeriodic = false;
        Periodicity periodicInfo{};
        MeshLegacyPairRead<index> cell2node, bnd2node, bnd2cell, cell2cell, node2cell, node2bnd;
        MeshLegacyPairRead<index> cell2face, face2node, face2cell, face2bnd, bnd2face;
        MeshLegacyPairRead<index> cell2edge, edge2node, edge2cell, cell2cellFace;
        MeshLegacyPairRead<real, true> coords, nodeWallDist;
        MeshLegacyPairRead<ElemInfo> cellElemInfo, bndElemInfo, faceElemInfo, edgeElemInfo;
        MeshLegacyPairRead<NodePeriodicBits> cell2nodePbi, bnd2nodePbi, cell2facePbi, face2nodePbi, cell2edgePbi, edge2nodePbi;
        MeshLegacyColumnRead bnd2faceV;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(UnstructuredMeshLegacyView, UnstructuredMeshLegacyView)
        DNDS_DEVICE_CALLABLE explicit UnstructuredMeshLegacyView(const UnstructuredMeshView &source)
            : UnstructuredMeshView(source), dim(source.Dimension()), isPeriodic(source.IsPeriodic()), periodicInfo(source.Periodic()),
              coords(source._storage.coords, true), nodeWallDist(source._storage.nodeWallDist, source.HasNodeWallDist()),
              cellElemInfo(source._storage.cellElemInfo, true), bndElemInfo(source._storage.bndElemInfo, true),
              faceElemInfo(source._storage.faceElemInfo, source.HasRelation(MeshRelation::FaceNodes)),
              edgeElemInfo(source._storage.edgeElemInfo, source.HasRelation(MeshRelation::EdgeNodes)),
              cell2nodePbi(source._storage.cell2nodePbi, source.IsPeriodic()), bnd2nodePbi(source._storage.bnd2nodePbi, source.IsPeriodic()),
              cell2facePbi(source._storage.cell2facePbi, source.IsPeriodic() && source.HasRelation(MeshRelation::CellFaces)),
              face2nodePbi(source._storage.face2nodePbi, source.IsPeriodic() && source.HasRelation(MeshRelation::FaceNodes)),
              cell2edgePbi(source._storage.cell2edgePbi, source.IsPeriodic() && source.HasRelation(MeshRelation::CellEdges)),
              edge2nodePbi(source._storage.edge2nodePbi, source.IsPeriodic() && source.HasRelation(MeshRelation::EdgeNodes)),
              bnd2faceV(source._storage.relations[size_t(MeshRelation::BoundaryFaces)])
        {
#define DNDS_LEGACY_RELATION(member, kind) member = {source._storage.relations[size_t(MeshRelation::kind)], source.HasRelation(MeshRelation::kind)}
            DNDS_LEGACY_RELATION(cell2node, CellNodes);
            DNDS_LEGACY_RELATION(bnd2node, BoundaryNodes);
            DNDS_LEGACY_RELATION(bnd2cell, BoundaryCells);
            DNDS_LEGACY_RELATION(cell2cell, CellCells);
            DNDS_LEGACY_RELATION(node2cell, NodeCells);
            DNDS_LEGACY_RELATION(node2bnd, NodeBoundaries);
            DNDS_LEGACY_RELATION(cell2face, CellFaces);
            DNDS_LEGACY_RELATION(face2node, FaceNodes);
            DNDS_LEGACY_RELATION(face2cell, FaceCells);
            DNDS_LEGACY_RELATION(face2bnd, FaceBoundaries);
            DNDS_LEGACY_RELATION(bnd2face, BoundaryFaces);
            DNDS_LEGACY_RELATION(cell2edge, CellEdges);
            DNDS_LEGACY_RELATION(edge2node, EdgeNodes);
            DNDS_LEGACY_RELATION(edge2cell, EdgeCells);
            DNDS_LEGACY_RELATION(cell2cellFace, CellFaceCells);
#undef DNDS_LEGACY_RELATION
        }

        template <class FA, class FB>
        DNDS_DEVICE_CALLABLE void CellOtherCellPeriodicHandle(index face, rowsize slot, FA &&donor, FB &&main) const
        {
            UnstructuredMeshView::CellOtherCellPeriodicHandle(face, slot, std::forward<FA>(donor), std::forward<FB>(main), NoPeriodicOperation{});
        }
        using UnstructuredMeshView::CellOtherCellPeriodicHandle;
    };

    static_assert(std::is_trivially_copyable_v<UnstructuredMeshLegacyView>);
}
