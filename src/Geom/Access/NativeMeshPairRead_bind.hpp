#pragma once

#include "Geom/Access/MeshAccess_bind.hpp"
#include <limits>

namespace DNDS::Geom
{
    /// Native-only alias kind. This adapter owns a semantic read, never a mesh
    /// payload copy. Rows and POD references are readonly; explicit snapshots
    /// provide detached mutable values for debugging.
    enum class NativePythonPairKind
    {
        Coords,
        Wall,
        CellInfo,
        BoundaryInfo,
        FaceInfo,
        EdgeInfo,
        CellNodes,
        BoundaryNodes,
        BoundaryCells,
        CellCells,
        CellFaces,
        FaceCells,
        FaceNodes,
        FaceBoundaries,
        BoundaryFaces,
        NodeCells,
        NodeBoundaries,
        CellEdges,
        EdgeCells,
        EdgeNodes,
        CellFaceCells,
        CellNodeBits,
        BoundaryNodeBits,
        FaceNodeBits,
        CellFaceBits,
        CellEdgeBits,
        EdgeNodeBits,
        CellOriginalIds,
        NodeOriginalIds
    };
    class NativePythonInfoRead
    {
        std::shared_ptr<MeshReadAccess> _read;
        const ElemInfo *_value;

    public:
        NativePythonInfoRead(std::shared_ptr<MeshReadAccess> read, const ElemInfo *value)
            : _read(std::move(read)), _value(value) {}
        t_index Type() const
        {
            _read->Stamp().Validate();
            return _value->type;
        }
        t_index Zone() const
        {
            _read->Stamp().Validate();
            return _value->zone;
        }
        Elem::ElemType ElementType() const { return Elem::ElemType(Type()); }
        ElemInfo Snapshot() const
        {
            _read->Stamp().Validate();
            return *_value;
        }
    };
    class NativePythonPairRead
    {
        std::shared_ptr<MeshReadAccess> _read;
        NativePythonPairKind _kind;
        std::shared_ptr<const Runtime::Array<index>::ReadLease> _originalIds;
        // -1 combined, 0 owned/father, 1 ghost/son.
        int _part = -1;
        static bool IsInfo(NativePythonPairKind kind)
        {
            return kind >= NativePythonPairKind::CellInfo && kind <= NativePythonPairKind::EdgeInfo;
        }
        static bool IsBits(NativePythonPairKind kind)
        {
            return kind >= NativePythonPairKind::CellNodeBits && kind <= NativePythonPairKind::EdgeNodeBits;
        }
        static MeshRelation Relation(NativePythonPairKind kind)
        {
            switch (kind)
            {
            case NativePythonPairKind::CellNodes:
                return MeshRelation::CellNodes;
            case NativePythonPairKind::BoundaryNodes:
                return MeshRelation::BoundaryNodes;
            case NativePythonPairKind::BoundaryCells:
                return MeshRelation::BoundaryCells;
            case NativePythonPairKind::CellCells:
                return MeshRelation::CellCells;
            case NativePythonPairKind::CellFaces:
                return MeshRelation::CellFaces;
            case NativePythonPairKind::FaceCells:
                return MeshRelation::FaceCells;
            case NativePythonPairKind::FaceNodes:
                return MeshRelation::FaceNodes;
            case NativePythonPairKind::FaceBoundaries:
                return MeshRelation::FaceBoundaries;
            case NativePythonPairKind::BoundaryFaces:
                return MeshRelation::BoundaryFaces;
            case NativePythonPairKind::NodeCells:
                return MeshRelation::NodeCells;
            case NativePythonPairKind::NodeBoundaries:
                return MeshRelation::NodeBoundaries;
            case NativePythonPairKind::CellEdges:
                return MeshRelation::CellEdges;
            case NativePythonPairKind::EdgeCells:
                return MeshRelation::EdgeCells;
            case NativePythonPairKind::EdgeNodes:
                return MeshRelation::EdgeNodes;
            case NativePythonPairKind::CellFaceCells:
                return MeshRelation::CellFaceCells;
            default:
                throw py::value_error("property is not a mesh incidence relation");
            }
        }
        EntityKind Family() const
        {
            switch (_kind)
            {
            case NativePythonPairKind::NodeOriginalIds:
            case NativePythonPairKind::Coords:
            case NativePythonPairKind::Wall:
            case NativePythonPairKind::NodeCells:
            case NativePythonPairKind::NodeBoundaries:
                return EntityKind::Node;
            case NativePythonPairKind::BoundaryInfo:
            case NativePythonPairKind::BoundaryNodes:
            case NativePythonPairKind::BoundaryCells:
            case NativePythonPairKind::BoundaryFaces:
            case NativePythonPairKind::BoundaryNodeBits:
                return EntityKind::Bnd;
            case NativePythonPairKind::FaceInfo:
            case NativePythonPairKind::FaceCells:
            case NativePythonPairKind::FaceNodes:
            case NativePythonPairKind::FaceBoundaries:
            case NativePythonPairKind::FaceNodeBits:
                return EntityKind::Face;
            case NativePythonPairKind::EdgeInfo:
            case NativePythonPairKind::EdgeCells:
            case NativePythonPairKind::EdgeNodes:
            case NativePythonPairKind::EdgeNodeBits:
                return EntityKind::Edge;
            default:
                return EntityKind::Cell;
            }
        }
        std::pair<index, index> Counts() const
        {
            const auto view = _read->HostView();
            switch (Family())
            {
            case EntityKind::Node:
                return {view.NumNode(), view.NumNodeGhost()};
            case EntityKind::Cell:
                return {view.NumCell(), view.NumCellGhost()};
            case EntityKind::Bnd:
                return {view.NumBnd(), view.NumBndGhost()};
            case EntityKind::Face:
                return {view.NumFace(), view.NumFaceGhost()};
            case EntityKind::Edge:
                return {view.NumEdge(), view.NumEdgeGhost()};
            default:
                throw py::value_error("unsupported native pair family");
            }
        }
        index Local(index row) const
        {
            if (row < 0 || row >= Size())
                throw py::index_error("native pair row out of range");
            return row + (_part == 1 ? Counts().first : 0);
        }
        template <class T>
        py::array Buffer(const T *data, py::ssize_t count) const
        {
            auto owner = py_owned_buffer(std::make_shared<NativePythonPairRead>(*this), data, {count}, {py::ssize_t(sizeof(T))}, true);
            return py::module_::import("numpy").attr("asarray")(owner).template cast<py::array>();
        }
        const ElemInfo *Info(index local) const
        {
            const auto view = _read->HostView();
            switch (_kind)
            {
            case NativePythonPairKind::CellInfo:
                return &view.GetCellInfo(local);
            case NativePythonPairKind::BoundaryInfo:
                return &view.GetBndInfo(local);
            case NativePythonPairKind::FaceInfo:
                return &view.GetFaceInfo(local);
            case NativePythonPairKind::EdgeInfo:
                return &view.GetEdgeInfo(local);
            default:
                throw py::value_error("property does not contain element metadata");
            }
        }
        template <class Function>
        auto WithNumeric(index local, Function &&function) const
        {
            const auto view = _read->HostView();
            if (_originalIds)
                return function(&_originalIds->View()(size_t(local), 0), 1);
            if (_kind == NativePythonPairKind::Coords)
                return function(view.Coord(local).data(), 3);
            if (_kind == NativePythonPairKind::Wall)
                return function(view.NodeWallDist(local).data(), 3);
            if (IsBits(_kind))
            {
                const auto bits = [&]
                {
                    switch (_kind)
                    {
                    case NativePythonPairKind::CellNodeBits:
                        return view.CellNodePeriodicBits(local);
                    case NativePythonPairKind::BoundaryNodeBits:
                        return view.BoundaryNodePeriodicBits(local);
                    case NativePythonPairKind::FaceNodeBits:
                        return view.FaceNodePeriodicBits(local);
                    case NativePythonPairKind::CellFaceBits:
                        return view.CellFacePeriodicBits(local);
                    case NativePythonPairKind::CellEdgeBits:
                        return view.CellEdgePeriodicBits(local);
                    case NativePythonPairKind::EdgeNodeBits:
                        return view.EdgeNodePeriodicBits(local);
                    default:
                        throw py::value_error("invalid periodic pair property");
                    }
                }();
                static_assert(sizeof(NodePeriodicBits) == sizeof(uint8_t));
                return function(reinterpret_cast<const uint8_t *>(bits.data()), py::ssize_t(bits.size()));
            }
            const auto row = view.Relation(Relation(_kind), local);
            return function(row.data(), py::ssize_t(row.size()));
        }
        py::array Numeric(index local) const
        {
            return WithNumeric(local, [&](auto *data, size_t count)
                               { return Buffer(data, py::ssize_t(count)); });
        }
        void CheckAvailable() const
        {
            const auto view = _read->HostView();
            if (Family() == EntityKind::Face && !view.HasRelation(MeshRelation::FaceNodes))
                throw py::value_error("facial topology is not prepared");
            if (Family() == EntityKind::Edge && !view.HasRelation(MeshRelation::EdgeNodes))
                throw py::value_error("edge topology is not prepared");
            if (_kind == NativePythonPairKind::Wall && !view.HasNodeWallDist())
                throw py::value_error("wall distance is unavailable");
            if (IsBits(_kind) && !view.IsPeriodic())
                throw py::value_error("periodic image bits are unavailable");
            if (_kind == NativePythonPairKind::CellFaceBits && !view.HasRelation(MeshRelation::CellFaces))
                throw py::value_error("facial topology is not prepared");
            if (_kind == NativePythonPairKind::CellEdgeBits && !view.HasRelation(MeshRelation::CellEdges))
                throw py::value_error("edge topology is not prepared");
            if (!_originalIds && !IsInfo(_kind) && !IsBits(_kind) && _kind != NativePythonPairKind::Coords && _kind != NativePythonPairKind::Wall && !view.HasRelation(Relation(_kind)))
                throw py::value_error("native incidence relation is unavailable");
        }

    public:
        NativePythonPairRead(UnstructuredMesh &mesh, NativePythonPairKind kind)
            : _read(std::make_shared<MeshReadAccess>(mesh.AcquireRead(mesh.CurrentNativeRequirements()))), _kind(kind)
        {
            if (_kind == NativePythonPairKind::CellOriginalIds || _kind == NativePythonPairKind::NodeOriginalIds)
                _originalIds = mesh.CurrentNativeState()->ReadOriginalEntityIds(Family());
            CheckAvailable();
        }
        index Size() const
        {
            const auto [owned, ghost] = Counts();
            return _part == 0 ? owned : _part == 1 ? ghost
                                                   : owned + ghost;
        }
        NativePythonPairRead Part(int part) const
        {
            auto result = *this;
            result._part = part;
            return result;
        }
        rowsize RowSize(index row) const
        {
            const auto local = Local(row);
            return IsInfo(_kind) ? 1 : WithNumeric(local, [](auto *, size_t count)
                                                   {
                if (count > size_t(std::numeric_limits<rowsize>::max())) throw py::value_error("native row width exceeds rowsize");
                return rowsize(count); });
        }
        py::object Get(index row) const
        {
            const auto local = Local(row);
            if (IsInfo(_kind))
                return py::cast(NativePythonInfoRead(_read, Info(local)));
            return Numeric(local);
        }
        py::object Scalar(index row, index column) const
        {
            if (IsInfo(_kind))
            {
                if (column != 0)
                    throw py::index_error("metadata column must be zero");
                return Get(row);
            }
            return WithNumeric(Local(row), [&](auto *values, size_t count) -> py::object
                               {
                if (column < 0 || size_t(column) >= count) throw py::index_error("native pair column out of range");
                return py::cast(values[column]); });
        }
        /// Explicit zero-copy data only for arithmetic uniform contiguous parts.
        /// CSR rows and disjoint owned/ghost allocations remain readable by row.
        py::array Data() const
        {
            if (IsInfo(_kind))
                throw py::value_error("POD metadata uses readonly row handles; call snapshot() for a detached value");
            const auto [owned, ghost] = Counts();
            if (!_originalIds && _part == -1 && owned && ghost)
                throw py::value_error("owned and ghost values are separate allocations; request father.data() or son.data()");
            const auto count = Size();
            if (!count)
                return NumericEmpty();
            return WithNumeric(Local(0), [&](auto *first, size_t width)
                               {
                using Value = std::remove_cv_t<std::remove_pointer_t<decltype(first)>>;
                if (width && size_t(count) > size_t(std::numeric_limits<py::ssize_t>::max()) / sizeof(Value) / width)
                    throw py::value_error("native buffer extent exceeds Python addressable size");
                for (index row = 1; row < count; ++row)
                    WithNumeric(Local(row), [&](auto *current, size_t currentWidth)
                    {
                        // CSR widths may differ; compare byte addresses without
                        // forming pointers beyond the real allocation extent.
                        const auto base = reinterpret_cast<uintptr_t>(first);
                        const auto offset = size_t(row) * width * sizeof(Value);
                        if (offset > std::numeric_limits<uintptr_t>::max() - base || currentWidth != width ||
                            reinterpret_cast<uintptr_t>(current) != base + offset)
                            throw py::value_error("native pair data is not uniform contiguous storage; read rows or request an explicit snapshot");
                        return 0;
                    });
                return Buffer(first, py::ssize_t(size_t(count) * width)); });
        }

        py::array NumericEmpty() const
        {
            // Retained zero-size buffers use a static dummy, never allocate a
            // mesh payload or infer a host pointer for a CUDA allocation.
            if (_kind == NativePythonPairKind::Coords || _kind == NativePythonPairKind::Wall)
            {
                static const real dummy = 0;
                return Buffer(&dummy, 0);
            }
            if (IsBits(_kind))
            {
                static const uint8_t dummy = 0;
                return Buffer(&dummy, 0);
            }
            static const index dummy = 0;
            return Buffer(&dummy, 0);
        }
    };
    inline void pybind11_NativeMeshPairRead_define(py::module_ &m)
    {
        py::class_<NativePythonInfoRead>(m, "NativeElemInfoRead")
            .def_property_readonly("type", &NativePythonInfoRead::Type)
            .def_property_readonly("zone", &NativePythonInfoRead::Zone)
            .def("getElemType", &NativePythonInfoRead::ElementType)
            .def("snapshot", &NativePythonInfoRead::Snapshot, "Explicit detached mutable ElemInfo copy.");
        py::class_<NativePythonPairRead>(m, "NativeMeshPairRead")
            .def("Size", &NativePythonPairRead::Size)
            .def("__len__", &NativePythonPairRead::Size)
            .def("RowSize", &NativePythonPairRead::RowSize)
            .def_property_readonly("father", [](const NativePythonPairRead &self)
                                   { return self.Part(0); })
            .def_property_readonly("son", [](const NativePythonPairRead &self)
                                   { return self.Part(1); })
            .def("__getitem__", [](const NativePythonPairRead &self, py::object key)
                 {
                if (py::isinstance<py::tuple>(key))
                {
                    auto pair = key.cast<py::tuple>();
                    if (pair.size() != 2) throw py::index_error("native pair index requires row or (row,column)");
                    return self.Scalar(pair[0].cast<index>(),pair[1].cast<index>());
                }
                return self.Get(key.cast<index>()); })
            .def("__call__", &NativePythonPairRead::Scalar)
            .def("data", &NativePythonPairRead::Data, "Retained readonly uniform contiguous arithmetic values. Ragged or split buffers require row access.");
    }
}
