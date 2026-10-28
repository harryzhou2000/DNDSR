#pragma once

#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Compat/LegacyFieldBridge.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"
#include "DNDS/Defines_bind.hpp"
#include <pybind11/numpy.h>

namespace DNDS::Geom
{
    /// Python closes its handle independently of escaped readonly buffers. Each
    /// buffer exporter retains the complete native scope, including row metadata.
    class PythonMeshReadAccess
    {
        std::shared_ptr<MeshReadAccess> _read;

        static void CheckRow(index row, index count)
        {
            if (row < 0 || row >= count)
                throw py::index_error("mesh row index outside requested entity space");
        }
        static index SourceRows(const UnstructuredMeshView &view, MeshRelation kind)
        {
            switch (kind)
            {
            case MeshRelation::CellNodes:
            case MeshRelation::CellCells:
            case MeshRelation::CellFaces:
            case MeshRelation::CellEdges:
            case MeshRelation::CellFaceCells:
                return view.NumCellProc();
            case MeshRelation::BoundaryNodes:
            case MeshRelation::BoundaryCells:
            case MeshRelation::BoundaryFaces:
                return view.NumBndProc();
            case MeshRelation::NodeCells:
            case MeshRelation::NodeBoundaries:
                return view.NumNodeProc();
            case MeshRelation::FaceNodes:
            case MeshRelation::FaceCells:
            case MeshRelation::FaceBoundaries:
                return view.NumFaceProc();
            case MeshRelation::EdgeNodes:
            case MeshRelation::EdgeCells:
                return view.NumEdgeProc();
            default:
                throw py::value_error("invalid mesh relation");
            }
        }
        template <class T>
        py::array Readonly(const T *pointer, std::vector<py::ssize_t> shape,
                           std::vector<py::ssize_t> strides) const
        {
            // A readonly PEP3118 exporter prevents NumPy from re-enabling its
            // write flag. A plain ndarray.setflags(False) would not suffice.
            auto memory = py_owned_buffer(std::shared_ptr<const void>(_read), pointer,
                                          std::move(shape), std::move(strides), true);
            return py::module_::import("numpy").attr("asarray")(memory).template cast<py::array>();
        }
        static py::array Snapshot(const std::vector<real> &values, py::ssize_t rows, py::ssize_t columns)
        {
            // Immutable bytes provide an independent snapshot, never a mutable
            // alias to mesh storage. Empty arrays also retain a valid base.
            auto bytes = py::bytes(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(real));
            return py::module_::import("numpy").attr("frombuffer")(bytes, py::arg("dtype") = py::dtype::of<real>()).attr("reshape")(rows, columns).cast<py::array>();
        }

    public:
        explicit PythonMeshReadAccess(MeshReadAccess access) : _read(std::make_shared<MeshReadAccess>(std::move(access))) {}
        bool Closed() const { return !_read; }
        bool Valid() const { return _read && _read->Stamp().Valid(); }
        void Close() { _read.reset(); }
        UnstructuredMeshView View() const
        {
            if (!_read)
                throw py::value_error("mesh read scope is closed");
            return _read->HostView();
        }
        py::dict Counts() const
        {
            auto view = View();
            py::dict result;
            result["node"] = py::make_tuple(view.NumNode(), view.NumNodeGhost(), view.NumNodeProc());
            result["cell"] = py::make_tuple(view.NumCell(), view.NumCellGhost(), view.NumCellProc());
            result["face"] = py::make_tuple(view.NumFace(), view.NumFaceGhost(), view.NumFaceProc());
            result["boundary"] = py::make_tuple(view.NumBnd(), view.NumBndGhost(), view.NumBndProc());
            result["edge"] = py::make_tuple(view.NumEdge(), view.NumEdgeGhost(), view.NumEdgeProc());
            return result;
        }
        bool HasRelation(MeshRelation kind) const
        {
            if (int(kind) < 0 || int(kind) >= int(MeshRelation::Count))
                throw py::value_error("invalid mesh relation");
            return View().HasRelation(kind);
        }
        py::array Relation(MeshRelation kind, index row) const
        {
            auto view = View();
            CheckRow(row, SourceRows(view, kind));
            if (!view.HasRelation(kind))
                throw py::value_error("mesh relation was not requested by this read scope");
            auto span = view.Relation(kind, row);
            return Readonly(span.data(), {py::ssize_t(span.size())}, {py::ssize_t(sizeof(index))});
        }
        py::array Coord(index node) const
        {
            auto view = View();
            CheckRow(node, view.NumNodeProc());
            return Readonly(view.Coord(node).data(), {3}, {py::ssize_t(sizeof(real))});
        }
        py::array Coordinates(bool ownedOnly) const
        {
            auto view = View();
            const auto count = ownedOnly ? view.NumNode() : view.NumNodeProc();
            std::vector<real> values(size_t(count) * 3);
            for (index node = 0; node < count; ++node)
                for (int d = 0; d < 3; ++d)
                    values[size_t(node) * 3 + d] = view.Coord(node)[d];
            return Snapshot(values, count, 3);
        }
        py::array ElementCoords(index row, bool face) const
        {
            auto view = View();
            CheckRow(row, face ? view.NumFaceProc() : view.NumCellProc());
            if (face && !view.HasRelation(MeshRelation::FaceNodes))
                throw py::value_error("faces were not requested by this read scope");
            MeshQueryCoords coords;
            if (face)
                view.GetCoordsOnFace(row, coords);
            else
                view.GetCoordsOnCell(row, coords);
            std::vector<real> values(size_t(coords.cols()) * 3);
            for (Eigen::Index i = 0; i < coords.cols(); ++i)
                for (int d = 0; d < 3; ++d)
                    values[size_t(i) * 3 + d] = coords(d, i);
            return Snapshot(values, coords.cols(), 3);
        }
        ElemInfo Info(index row, bool face) const
        {
            auto view = View();
            CheckRow(row, face ? view.NumFaceProc() : view.NumCellProc());
            return face ? view.GetFaceInfo(row) : view.GetCellInfo(row);
        }
        ElemInfo BoundaryInfo(index row) const
        {
            auto view = View();
            CheckRow(row, view.NumBndProc());
            return view.GetBndInfo(row);
        }
        py::array BoundaryCoords(index row) const
        {
            auto view = View();
            CheckRow(row, view.NumBndProc());
            const auto nodes = view.BoundaryNodes(row);
            std::vector<real> values(nodes.size() * 3);
            for (size_t slot = 0; slot < nodes.size(); ++slot)
            {
                tPoint point = view.Coord(nodes[slot]);
                if (view.IsPeriodic())
                {
                    const auto bits = uint8_t(view.BoundaryNodePeriodicBits(row)[slot]);
                    for (int bit = 3; bit >= 1; --bit)
                        if (bits & (uint8_t(1) << (bit - 1)))
                            point = view.Periodic().TransCoord(point, -bit);
                }
                for (int dimension = 0; dimension < 3; ++dimension)
                    values[slot * 3 + size_t(dimension)] = point[dimension];
            }
            return Snapshot(values, py::ssize_t(nodes.size()), 3);
        }
        py::array PeriodicBits(index row, bool face) const
        {
            auto view = View();
            CheckRow(row, face ? view.NumFaceProc() : view.NumCellProc());
            if (!view.IsPeriodic())
                throw py::value_error("mesh has no periodic incidence bits");
            auto bits = face ? view.FaceNodePeriodicBits(row) : view.CellNodePeriodicBits(row);
            static_assert(sizeof(NodePeriodicBits) == sizeof(uint8_t));
            return Readonly(reinterpret_cast<const uint8_t *>(bits.data()), {py::ssize_t(bits.size())}, {1});
        }
        py::array WallDistance(index node) const
        {
            auto view = View();
            CheckRow(node, view.NumNodeProc());
            if (!view.HasNodeWallDist())
                throw py::value_error("wall distance was not requested by this read scope");
            return Readonly(view.NodeWallDist(node).data(), {3}, {py::ssize_t(sizeof(real))});
        }
        index OtherCell(index cell, index face, rowsize slot) const
        {
            auto view = View();
            CheckRow(cell, view.NumCellProc());
            CheckRow(face, view.NumFaceProc());
            if (!view.HasRelation(MeshRelation::CellFaces))
                throw py::value_error("faces were not requested by this read scope");
            auto faces = view.CellFaces(cell);
            CheckRow(slot, index(faces.size()));
            if (faces[slot] != face)
                throw py::value_error("face does not match the supplied cell-face slot");
            auto adjacent = view.FaceCells(face);
            if (adjacent[0] != cell && adjacent[1] != cell)
                throw py::value_error("cell is not incident to the supplied face");
            return view.CellFaceOther(cell, face, slot);
        }
        bool IsFaceBack(index cell, index face, rowsize slot) const
        {
            OtherCell(cell, face, slot); // Checked Python input before native assertions.
            return View().CellIsFaceBack(cell, face, slot);
        }
    };

    inline void pybind11_MeshAccess_define(py::module_ &m)
    {
        py::enum_<EntityKind>(m, "EntityKind")
            .value("Cell", EntityKind::Cell)
            .value("Node", EntityKind::Node)
            .value("Face", EntityKind::Face)
            .value("Boundary", EntityKind::Bnd)
            .value("Edge", EntityKind::Edge);
        py::class_<EntitySpace>(m, "EntitySpace")
            .def_property_readonly("kind", &EntitySpace::Kind)
            .def_property_readonly("valid", &EntitySpace::Valid)
            .def_property_readonly("mesh_identity", &EntitySpace::MeshIdentity,
                                   "Opaque process-local identity; never persist it or compare between MPI ranks.")
            .def_property_readonly("owned", &EntitySpace::NumOwned)
            .def_property_readonly("ghost", &EntitySpace::NumGhost)
            .def_property_readonly("local", &EntitySpace::NumLocal)
            .def_property_readonly("global_count", &EntitySpace::NumGlobal)
            .def("local_to_global", &EntitySpace::LocalToGlobal)
            .def("global_to_local", &EntitySpace::GlobalToLocal)
            .def("owner", &EntitySpace::Owner)
            .def("owner_local_row", &EntitySpace::OwnerLocalRow)
            .def("same_domain", &EntitySpace::SameDomain)
            .def("same_order", &EntitySpace::SameOrder);
        m.def("borrow_legacy_field_indexing", [](ArrayTransformer<real, 1> &transformer, const EntitySpace &space)
              { LegacyFieldBridge::BorrowIndexing(transformer, space, true); }, py::arg("transformer"), py::arg("entities"), "Attach cloned entity maps to a scalar real field transformer. Caller owns sizing, TransAttach and MPI request setup.");
        m.def("borrow_legacy_field_indexing", [](ArrayTransformer<real, DynamicSize> &transformer, const EntitySpace &space)
              { LegacyFieldBridge::BorrowIndexing(transformer, space, true); }, py::arg("transformer"), py::arg("entities"), "Attach cloned entity maps to a dynamic-width real field transformer. Other transformer types are not supported.");
        py::class_<UnstructuredMeshSnapshot>(m, "UnstructuredMeshSnapshot")
            .def_property_readonly("source_current", [](const UnstructuredMeshSnapshot &self)
                                   { return self.SourceStamp().Valid(); }, "Whether source provenance is current; independent of snapshot memory validity.")
            .def_property_readonly("device", [](const UnstructuredMeshSnapshot &self)
                                   {
                                       auto device = self.GetDevice();
                                       return device == Runtime::Device::Host() ? std::string("cpu") : "cuda:" + std::to_string(device.ordinal); })
            .def("to_device", [](const UnstructuredMeshSnapshot &self, int ordinal)
                 { return self.To(Runtime::Device::CUDA(ordinal)); }, py::arg("ordinal") = 0, "Return independent CUDA storage; completes the copy before returning.")
            .def("to_host", [](const UnstructuredMeshSnapshot &self)
                 { return self.To(Runtime::Device::Host()); }, "Return an independent CPU snapshot, synchronizing any required device copy.")
            .def("acquire_read", [](const UnstructuredMeshSnapshot &self)
                 { return std::make_shared<PythonMeshReadAccess>(self.AcquireRead()); }, "Retain readonly CPU rows. For a CUDA snapshot, call to_host() explicitly first.");
        m.attr("NativeMeshSnapshot") = m.attr("UnstructuredMeshSnapshot");
        py::enum_<MeshRelation> relation(m, "MeshRelation");
#define DNDS_MESH_RELATION_PY(name) relation.value(#name, MeshRelation::name)
        DNDS_MESH_RELATION_PY(CellNodes);
        DNDS_MESH_RELATION_PY(BoundaryNodes);
        DNDS_MESH_RELATION_PY(BoundaryCells);
        DNDS_MESH_RELATION_PY(CellCells);
        DNDS_MESH_RELATION_PY(NodeCells);
        DNDS_MESH_RELATION_PY(NodeBoundaries);
        DNDS_MESH_RELATION_PY(CellFaces);
        DNDS_MESH_RELATION_PY(FaceNodes);
        DNDS_MESH_RELATION_PY(FaceCells);
        DNDS_MESH_RELATION_PY(FaceBoundaries);
        DNDS_MESH_RELATION_PY(BoundaryFaces);
        DNDS_MESH_RELATION_PY(CellEdges);
        DNDS_MESH_RELATION_PY(EdgeNodes);
        DNDS_MESH_RELATION_PY(EdgeCells);
        DNDS_MESH_RELATION_PY(CellFaceCells);
#undef DNDS_MESH_RELATION_PY
        auto read = py_class_ssp<PythonMeshReadAccess>(m, "MeshReadAccess");
        read.def("close", &PythonMeshReadAccess::Close)
            .def_property_readonly("closed", &PythonMeshReadAccess::Closed)
            .def_property_readonly("valid", &PythonMeshReadAccess::Valid)
            .def("__enter__", [](PythonMeshReadAccess &self) -> PythonMeshReadAccess &
                 { self.View(); return self; }, py::return_value_policy::reference_internal)
            .def("__exit__", [](PythonMeshReadAccess &self, py::object, py::object, py::object)
                 { self.Close(); return false; })
            .def_property_readonly("counts", &PythonMeshReadAccess::Counts, "Owned, ghost, combined counts for each requested entity family.")
            .def_property_readonly("dimension", [](const PythonMeshReadAccess &self)
                                   { return self.View().Dimension(); })
            .def_property_readonly("is_periodic", [](const PythonMeshReadAccess &self)
                                   { return self.View().IsPeriodic(); })
            .def("has_relation", &PythonMeshReadAccess::HasRelation)
            .def("relation", &PythonMeshReadAccess::Relation, py::arg("kind"), py::arg("row"))
            .def("relation_state", [](const PythonMeshReadAccess &self, MeshRelation kind)
                 {
                     auto view = self.View();
                     DNDS_check_throw_info(kind >= MeshRelation::CellNodes && kind < MeshRelation::Count && view.HasRelation(kind),
                                           "relation is not present in this read scope");
                     return view.RelationState(kind); }, py::arg("kind"), "Index state of this retained relation. Native compute publications are local; distributed global numbers differ from persistent IDs.")
            .def("coord", &PythonMeshReadAccess::Coord, py::arg("node"), "Readonly retained coordinate row; slices retain its lease.")
            .def("coordinates", &PythonMeshReadAccess::Coordinates, py::arg("owned_only") = false, "Independent immutable coordinate snapshot.")
            .def("cell_coords", [](const PythonMeshReadAccess &self, index row)
                 { return self.ElementCoords(row, false); })
            .def("face_coords", [](const PythonMeshReadAccess &self, index row)
                 { return self.ElementCoords(row, true); })
            .def("cell_info", [](const PythonMeshReadAccess &self, index row)
                 { return self.Info(row, false); }, "Detached ElemInfo value.")
            .def("face_info", [](const PythonMeshReadAccess &self, index row)
                 { return self.Info(row, true); }, "Detached ElemInfo value.")
            .def("boundary_info", &PythonMeshReadAccess::BoundaryInfo, "Detached boundary ElemInfo value.")
            .def("boundary_coords", &PythonMeshReadAccess::BoundaryCoords, "Independent immutable physical coordinate snapshot.")
            .def("cell_node_periodic_bits", [](const PythonMeshReadAccess &self, index row)
                 { return self.PeriodicBits(row, false); })
            .def("face_node_periodic_bits", [](const PythonMeshReadAccess &self, index row)
                 { return self.PeriodicBits(row, true); })
            .def("wall_distance", &PythonMeshReadAccess::WallDistance)
            .def("cell_face_other", &PythonMeshReadAccess::OtherCell, py::arg("cell"), py::arg("face"), py::arg("slot"))
            .def("cell_is_face_back", &PythonMeshReadAccess::IsFaceBack, py::arg("cell"), py::arg("face"), py::arg("slot"));
        const std::pair<const char *, MeshRelation> methods[]{
            {"cell_nodes", MeshRelation::CellNodes}, {"boundary_nodes", MeshRelation::BoundaryNodes}, {"boundary_cells", MeshRelation::BoundaryCells}, {"cell_cells", MeshRelation::CellCells}, {"node_cells", MeshRelation::NodeCells}, {"node_boundaries", MeshRelation::NodeBoundaries}, {"cell_faces", MeshRelation::CellFaces}, {"face_nodes", MeshRelation::FaceNodes}, {"face_cells", MeshRelation::FaceCells}, {"face_boundaries", MeshRelation::FaceBoundaries}, {"boundary_faces", MeshRelation::BoundaryFaces}, {"cell_edges", MeshRelation::CellEdges}, {"edge_nodes", MeshRelation::EdgeNodes}, {"edge_cells", MeshRelation::EdgeCells}, {"cell_face_cells", MeshRelation::CellFaceCells}};
        for (auto [name, kind] : methods)
            read.def(name, [kind](const PythonMeshReadAccess &self, index row)
                     { return self.Relation(kind, row); }, py::arg("row"));
    }
}
