#pragma once

#include "Mesh_DeviceView.hpp"
#include "DNDS/Runtime/Array.hpp"
#include <type_traits>

namespace DNDS::Geom
{
    class MeshReadAccess;

    /// Variable logical width with inline storage for every supported element.
    /// Unlike tSmallCoords, resizing this output never allocates device memory.
    using MeshQueryCoords = Eigen::Matrix<real, 3, Eigen::Dynamic, Eigen::ColMajor, 3, Elem::CellNumNodeMax>;

    /// Ordered incidences. Entries retain their original slot and sentinel meanings.
    enum class MeshRelation
    {
        CellNodes,
        BoundaryNodes,
        BoundaryCells,
        CellCells,
        NodeCells,
        NodeBoundaries,
        CellFaces,
        FaceNodes,
        FaceCells,
        FaceBoundaries,
        BoundaryFaces,
        CellEdges,
        EdgeNodes,
        EdgeCells,
        CellFaceCells,
        Count
    };

    template <class T>
    class MeshRowSpan
    {
        const T *_data = nullptr;
        size_t _size = 0;

    public:
        DNDS_DEVICE_CALLABLE MeshRowSpan() = default;
        DNDS_DEVICE_CALLABLE MeshRowSpan(const T *data, size_t size) : _data(data), _size(size) {}
        DNDS_DEVICE_CALLABLE const T &operator[](size_t column) const { return _data[column]; }
        DNDS_DEVICE_CALLABLE const T *data() const { return _data; }
        DNDS_DEVICE_CALLABLE size_t size() const { return _size; }
        DNDS_DEVICE_CALLABLE bool empty() const { return _size == 0; }
        DNDS_DEVICE_CALLABLE const T *begin() const { return _data; }
        DNDS_DEVICE_CALLABLE const T *end() const { return _size ? _data + _size : _data; }
        DNDS_DEVICE_CALLABLE const T *cbegin() const { return begin(); }
        DNDS_DEVICE_CALLABLE const T *cend() const { return end(); }
    };

    /// Assembly data for the access owner. All pointed-to values and row metadata
    /// must remain retained by that owner. Coordinates are rows of three reals;
    /// ElemInfo rows contain one value. Absent optional relations remain empty.
    /// Presence/state and local closure are validated before publishing a view.
    struct MeshViewStorage
    {
        Runtime::PairReadView<index> relations[static_cast<size_t>(MeshRelation::Count)]{};
        MeshAdjState relationStates[static_cast<size_t>(MeshRelation::Count)]{};
        bool relationPresent[static_cast<size_t>(MeshRelation::Count)]{};
        Runtime::PairReadView<real> coords{}, nodeWallDist{};
        Runtime::PairReadView<ElemInfo> cellElemInfo{}, bndElemInfo{}, faceElemInfo{}, edgeElemInfo{};
        Runtime::PairReadView<NodePeriodicBits> cell2nodePbi{}, bnd2nodePbi{};
        Runtime::PairReadView<NodePeriodicBits> cell2facePbi{}, face2nodePbi{};
        Runtime::PairReadView<NodePeriodicBits> cell2edgePbi{}, edge2nodePbi{};
        int dim = -1;
        bool isPeriodic = false;
        bool hasNodeWallDist = false;
        Periodicity periodicInfo{};
    };

    /// A storage-independent kernel query view. It owns no leases or allocations.
    /// Host/device placement and dependency ordering belong to MeshReadAccess.
    /// Copying a view or row span does not extend its owner's lifetime.
    class UnstructuredMeshView
    {
        MeshViewStorage _storage{};
        friend class MeshReadAccess;

        DNDS_DEVICE_CALLABLE explicit UnstructuredMeshView(const MeshViewStorage &storage) : _storage(storage) {}

        DNDS_DEVICE_CALLABLE const Runtime::PairReadView<index> &RelationData(MeshRelation relation) const
        {
            return _storage.relations[static_cast<size_t>(relation)];
        }

        template <class T>
        DNDS_DEVICE_CALLABLE static MeshRowSpan<T> Row(const Runtime::PairReadView<T> &data, index row)
        {
            return {data.Row(row), data.RowSize(row)};
        }

        DNDS_DEVICE_CALLABLE tPoint TransformWallVector(const tPoint &value, NodePeriodicBits bits) const
        {
            tPoint result = value;
            // Match Periodicity::GetVectorByBits: third, second, first transform.
            for (int i = 3; i >= 1; --i)
                if (uint8_t(bits) & (uint8_t(1) << (i - 1)))
                    result = (_storage.periodicInfo.rotation[i].map() * result).eval();
            return result;
        }

        DNDS_DEVICE_CALLABLE tPoint TransformCoordinate(const tPoint &value, NodePeriodicBits bits) const
        {
            tPoint result = value;
            // Decode the portable byte directly. The legacy getP1/getP2/getP3
            // accessors are host-only; this query path must work on the device.
            for (int i = 3; i >= 1; --i)
                if (uint8_t(bits) & (uint8_t(1) << (i - 1)))
                    result = _storage.periodicInfo.TransCoord(result, -i);
            return result;
        }

        template <class TResult, class TCoords, std::enable_if_t<TResult::MaxColsAtCompileTime != Eigen::Dynamic, int> = 0>
        DNDS_DEVICE_CALLABLE void LoadCoords(MeshRowSpan<index> nodes, MeshRowSpan<NodePeriodicBits> bits,
                                             TResult &result, const TCoords &coords) const
        {
            static_assert(TResult::RowsAtCompileTime == 3, "coordinate output requires three rows");
            DNDS_HD_assert(nodes.size() <= size_t(TResult::MaxColsAtCompileTime));
            result.resize(Eigen::NoChange, nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i)
                if (_storage.isPeriodic)
                    result.col(i) = TransformCoordinate(coords[nodes[i]], bits[i]);
                else
                    result.col(i) = coords[nodes[i]];
        }

        template <class TCoords>
        DNDS_HOST void LoadCoords(MeshRowSpan<index> nodes, MeshRowSpan<NodePeriodicBits> bits,
                                  tSmallCoords &result, const TCoords &coords) const
        {
            result.resize(Eigen::NoChange, nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i)
                if (_storage.isPeriodic)
                    result.col(i) = TransformCoordinate(coords[nodes[i]], bits[i]);
                else
                    result.col(i) = coords[nodes[i]];
        }

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(UnstructuredMeshView, UnstructuredMeshView)

        using tAdjacencyRow = MeshRowSpan<index>;
        using tPeriodicRow = MeshRowSpan<NodePeriodicBits>;

        DNDS_DEVICE_CALLABLE int Dimension() const { return _storage.dim; }
        DNDS_DEVICE_CALLABLE bool IsPeriodic() const { return _storage.isPeriodic; }
        DNDS_DEVICE_CALLABLE const Periodicity &Periodic() const { return _storage.periodicInfo; }
        DNDS_DEVICE_CALLABLE bool HasNodeWallDist() const { return _storage.hasNodeWallDist; }
        DNDS_DEVICE_CALLABLE bool HasRelation(MeshRelation relation) const
        {
            return _storage.relationPresent[static_cast<size_t>(relation)];
        }
        DNDS_DEVICE_CALLABLE MeshAdjState RelationState(MeshRelation relation) const
        {
            return _storage.relationStates[static_cast<size_t>(relation)];
        }
        DNDS_DEVICE_CALLABLE tAdjacencyRow Relation(MeshRelation relation, index row) const
        {
            return Row(RelationData(relation), row);
        }

        DNDS_DEVICE_CALLABLE tAdjacencyRow CellNodes(index row) const { return Relation(MeshRelation::CellNodes, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow BoundaryNodes(index row) const { return Relation(MeshRelation::BoundaryNodes, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow BoundaryCells(index row) const { return Relation(MeshRelation::BoundaryCells, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow CellCells(index row) const { return Relation(MeshRelation::CellCells, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow NodeCells(index row) const { return Relation(MeshRelation::NodeCells, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow NodeBoundaries(index row) const { return Relation(MeshRelation::NodeBoundaries, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow CellFaces(index row) const { return Relation(MeshRelation::CellFaces, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow FaceNodes(index row) const { return Relation(MeshRelation::FaceNodes, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow FaceCells(index row) const { return Relation(MeshRelation::FaceCells, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow FaceBoundaries(index row) const { return Relation(MeshRelation::FaceBoundaries, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow BoundaryFaces(index row) const { return Relation(MeshRelation::BoundaryFaces, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow CellEdges(index row) const { return Relation(MeshRelation::CellEdges, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow EdgeNodes(index row) const { return Relation(MeshRelation::EdgeNodes, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow EdgeCells(index row) const { return Relation(MeshRelation::EdgeCells, row); }
        DNDS_DEVICE_CALLABLE tAdjacencyRow CellFaceCells(index row) const { return Relation(MeshRelation::CellFaceCells, row); }

        DNDS_DEVICE_CALLABLE tPeriodicRow CellNodePeriodicBits(index row) const { return Row(_storage.cell2nodePbi, row); }
        DNDS_DEVICE_CALLABLE tPeriodicRow BoundaryNodePeriodicBits(index row) const { return Row(_storage.bnd2nodePbi, row); }
        DNDS_DEVICE_CALLABLE tPeriodicRow CellFacePeriodicBits(index row) const { return Row(_storage.cell2facePbi, row); }
        DNDS_DEVICE_CALLABLE tPeriodicRow FaceNodePeriodicBits(index row) const { return Row(_storage.face2nodePbi, row); }
        DNDS_DEVICE_CALLABLE tPeriodicRow CellEdgePeriodicBits(index row) const { return Row(_storage.cell2edgePbi, row); }
        DNDS_DEVICE_CALLABLE tPeriodicRow EdgeNodePeriodicBits(index row) const { return Row(_storage.edge2nodePbi, row); }

        DNDS_DEVICE_CALLABLE index NumNode() const { return _storage.coords.owned.Rows(); }
        DNDS_DEVICE_CALLABLE index NumNodeGhost() const { return _storage.coords.ghost.Rows(); }
        DNDS_DEVICE_CALLABLE index NumNodeProc() const { return NumNode() + NumNodeGhost(); }
        DNDS_DEVICE_CALLABLE index NumCell() const { return RelationData(MeshRelation::CellNodes).owned.Rows(); }
        DNDS_DEVICE_CALLABLE index NumCellGhost() const { return RelationData(MeshRelation::CellNodes).ghost.Rows(); }
        DNDS_DEVICE_CALLABLE index NumCellProc() const { return NumCell() + NumCellGhost(); }
        DNDS_DEVICE_CALLABLE index NumFace() const { return RelationData(MeshRelation::FaceNodes).owned.Rows(); }
        DNDS_DEVICE_CALLABLE index NumFaceGhost() const { return RelationData(MeshRelation::FaceNodes).ghost.Rows(); }
        DNDS_DEVICE_CALLABLE index NumFaceProc() const { return NumFace() + NumFaceGhost(); }
        DNDS_DEVICE_CALLABLE index NumBnd() const { return RelationData(MeshRelation::BoundaryNodes).owned.Rows(); }
        DNDS_DEVICE_CALLABLE index NumBndGhost() const { return RelationData(MeshRelation::BoundaryNodes).ghost.Rows(); }
        DNDS_DEVICE_CALLABLE index NumBndProc() const { return NumBnd() + NumBndGhost(); }
        DNDS_DEVICE_CALLABLE index NumEdge() const { return RelationData(MeshRelation::EdgeNodes).owned.Rows(); }
        DNDS_DEVICE_CALLABLE index NumEdgeGhost() const { return RelationData(MeshRelation::EdgeNodes).ghost.Rows(); }
        DNDS_DEVICE_CALLABLE index NumEdgeProc() const { return NumEdge() + NumEdgeGhost(); }

        DNDS_DEVICE_CALLABLE Eigen::Map<const tPoint> Coord(index node) const { return Eigen::Map<const tPoint>(_storage.coords.Row(node)); }
        DNDS_DEVICE_CALLABLE Eigen::Map<const tPoint> operator[](index node) const { return Coord(node); }
        DNDS_DEVICE_CALLABLE Eigen::Map<const tPoint> NodeWallDist(index node) const
        {
            DNDS_HD_assert(HasNodeWallDist());
            return Eigen::Map<const tPoint>(_storage.nodeWallDist.Row(node));
        }

        DNDS_DEVICE_CALLABLE const ElemInfo &GetCellInfo(index row) const { return _storage.cellElemInfo(row, 0); }
        DNDS_DEVICE_CALLABLE const ElemInfo &GetFaceInfo(index row) const { return _storage.faceElemInfo(row, 0); }
        DNDS_DEVICE_CALLABLE const ElemInfo &GetBndInfo(index row) const { return _storage.bndElemInfo(row, 0); }
        DNDS_DEVICE_CALLABLE const ElemInfo &GetEdgeInfo(index row) const { return _storage.edgeElemInfo(row, 0); }
        DNDS_DEVICE_CALLABLE Elem::Element GetCellElement(index row) const { return Elem::Element{GetCellInfo(row).getElemType()}; }
        DNDS_DEVICE_CALLABLE Elem::Element GetFaceElement(index row) const { return Elem::Element{GetFaceInfo(row).getElemType()}; }
        DNDS_DEVICE_CALLABLE Elem::Element GetBndElement(index row) const { return Elem::Element{GetBndInfo(row).getElemType()}; }
        DNDS_DEVICE_CALLABLE Elem::Element GetEdgeElement(index row) const { return Elem::Element{GetEdgeInfo(row).getElemType()}; }
        DNDS_DEVICE_CALLABLE t_index GetCellZone(index row) const { return GetCellInfo(row).zone; }
        DNDS_DEVICE_CALLABLE t_index GetFaceZone(index row) const { return GetFaceInfo(row).zone; }
        DNDS_DEVICE_CALLABLE t_index GetBndZone(index row) const { return GetBndInfo(row).zone; }
        DNDS_DEVICE_CALLABLE t_index GetEdgeZone(index row) const { return GetEdgeInfo(row).zone; }

        DNDS_HOST void GetCoordsOnCell(index cell, tSmallCoords &result) const
        {
            GetCoordsOnCell(cell, result, *this);
        }
        template <class TCoords>
        DNDS_HOST void GetCoordsOnCell(index cell, tSmallCoords &result, const TCoords &coords) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::CellNodes) == Adj_PointToLocal);
            LoadCoords(CellNodes(cell), IsPeriodic() ? CellNodePeriodicBits(cell) : tPeriodicRow{}, result, coords);
        }
        template <class TResult, std::enable_if_t<TResult::MaxColsAtCompileTime != Eigen::Dynamic, int> = 0>
        DNDS_DEVICE_CALLABLE void GetCoordsOnCell(index cell, TResult &result) const
        {
            GetCoordsOnCell(cell, result, *this);
        }
        template <class TResult, class TCoords, std::enable_if_t<TResult::MaxColsAtCompileTime != Eigen::Dynamic, int> = 0>
        DNDS_DEVICE_CALLABLE void GetCoordsOnCell(index cell, TResult &result, const TCoords &coords) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::CellNodes) == Adj_PointToLocal);
            LoadCoords(CellNodes(cell), IsPeriodic() ? CellNodePeriodicBits(cell) : tPeriodicRow{}, result, coords);
        }
        DNDS_HOST void GetCoordsOnFace(index face, tSmallCoords &result) const
        {
            GetCoordsOnFace(face, result, *this);
        }
        template <class TCoords>
        DNDS_HOST void GetCoordsOnFace(index face, tSmallCoords &result, const TCoords &coords) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::FaceNodes) == Adj_PointToLocal);
            LoadCoords(FaceNodes(face), IsPeriodic() ? FaceNodePeriodicBits(face) : tPeriodicRow{}, result, coords);
        }
        template <class TResult, std::enable_if_t<TResult::MaxColsAtCompileTime != Eigen::Dynamic, int> = 0>
        DNDS_DEVICE_CALLABLE void GetCoordsOnFace(index face, TResult &result) const
        {
            GetCoordsOnFace(face, result, *this);
        }
        template <class TResult, class TCoords, std::enable_if_t<TResult::MaxColsAtCompileTime != Eigen::Dynamic, int> = 0>
        DNDS_DEVICE_CALLABLE void GetCoordsOnFace(index face, TResult &result, const TCoords &coords) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::FaceNodes) == Adj_PointToLocal);
            LoadCoords(FaceNodes(face), IsPeriodic() ? FaceNodePeriodicBits(face) : tPeriodicRow{}, result, coords);
        }
        DNDS_DEVICE_CALLABLE tPoint GetCoordNodeOnCell(index cell, rowsize slot) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::CellNodes) == Adj_PointToLocal);
            tPoint value = Coord(CellNodes(cell)[slot]);
            return IsPeriodic() ? TransformCoordinate(value, CellNodePeriodicBits(cell)[slot]) : value;
        }
        DNDS_DEVICE_CALLABLE tPoint GetCoordNodeOnFace(index face, rowsize slot) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::FaceNodes) == Adj_PointToLocal);
            tPoint value = Coord(FaceNodes(face)[slot]);
            return IsPeriodic() ? TransformCoordinate(value, FaceNodePeriodicBits(face)[slot]) : value;
        }
        DNDS_DEVICE_CALLABLE tPoint GetCoordWallDistOnCell(index cell, rowsize slot) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::CellNodes) == Adj_PointToLocal);
            tPoint value = NodeWallDist(CellNodes(cell)[slot]);
            return IsPeriodic() ? TransformWallVector(value, CellNodePeriodicBits(cell)[slot]) : value;
        }
        DNDS_DEVICE_CALLABLE tPoint GetCoordWallDistOnFace(index face, rowsize slot) const
        {
            DNDS_HD_assert(RelationState(MeshRelation::FaceNodes) == Adj_PointToLocal);
            tPoint value = NodeWallDist(FaceNodes(face)[slot]);
            return IsPeriodic() ? TransformWallVector(value, FaceNodePeriodicBits(face)[slot]) : value;
        }

        DNDS_DEVICE_CALLABLE bool CellIsFaceBack(index cell, index face, rowsize cellFaceSlot) const
        {
            auto cells = FaceCells(face);
            DNDS_HD_assert(cells[0] == cell || cells[1] == cell);
            if (cells[0] == cell && cells[1] == cell)
            {
                DNDS_HD_assert(cellFaceSlot >= 0 && IsPeriodic());
                return !bool(CellFacePeriodicBits(cell)[cellFaceSlot]);
            }
            return cells[0] == cell;
        }
        DNDS_DEVICE_CALLABLE index CellFaceOther(index cell, index face, rowsize cellFaceSlot) const
        {
            return FaceCells(face)[CellIsFaceBack(cell, face, cellFaceSlot) ? 1 : 0];
        }

        template <class FA, class FB, class F0>
        DNDS_DEVICE_CALLABLE auto CellOtherCellPeriodicHandle(index face, rowsize faceCellSlot, FA &&donor, FB &&main, F0 &&otherwise) const
        {
            if (!IsPeriodic())
                return otherwise();
            auto zone = GetFaceZone(face);
            if (!FaceIDIsPeriodic(zone))
                return otherwise();
            if ((faceCellSlot == 1 && FaceIDIsPeriodicMain(zone)) ||
                (faceCellSlot == 0 && FaceIDIsPeriodicDonor(zone)))
                return donor();
            if ((faceCellSlot == 1 && FaceIDIsPeriodicDonor(zone)) ||
                (faceCellSlot == 0 && FaceIDIsPeriodicMain(zone)))
                return main();
            DNDS_HD_assert(false);
            return otherwise();
        }
    };

    static_assert(std::is_trivially_copyable_v<MeshRowSpan<index>>);
    static_assert(std::is_trivially_copyable_v<MeshViewStorage>);
    static_assert(std::is_trivially_copyable_v<UnstructuredMeshView>);
    static_assert(std::is_standard_layout_v<UnstructuredMeshView>);
}
