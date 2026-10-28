#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Access/MeshAccess_bind.hpp"
#include "Geom/Access/NativeMeshPairRead_bind.hpp"
#include "Geom/Transition/MeshTransition.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Geometry/MeshEdit_bind.hpp"
#include "Geom/Unstructured/NativeMeshBuilder_bind.hpp"
#include "Geom/Unstructured/NativeMeshPreparation_bind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11_json/pybind11_json.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Geom
{
    using tPy_ElemInfo = py_class_ssp<ElemInfo>;

    inline void pybind11_ElemInfo_define(py::module_ &m)
    {
        auto ElemInfo_ = tPy_ElemInfo(m, "ElemInfo");
        ElemInfo_
            .def(py::init<>())
            .def("getElemType", &ElemInfo::getElemType)
            .def("setElemType", &ElemInfo::setElemType)
            .def_readwrite("zone", &ElemInfo::zone);
        Runtime::RegisterArrayElement(ElemInfo_, "Geom.ElemInfo.v1");
    }

    inline void pybind11_ArrayElemInfo_define(py::module_ &m)
    {
        pybind11_Array_define<ElemInfo>(m);
        pybind11_ParArray_define<ElemInfo>(m);
        pybind11_ArrayTransformer_define<ParArray<ElemInfo>>(m);
        pybind11_ParArrayPair_define<ElemInfo>(m);
    }

    inline void pybind11_MeshLocDefine(py::module_ &m)
    {
#define DNDS_PY_ENUM_CLASS_MeshLoc_ADD(v) value(#v, MeshLoc::v)
        py::enum_<MeshLoc>(m, "MeshLoc")
            .DNDS_PY_ENUM_CLASS_MeshLoc_ADD(Unknown)
            .DNDS_PY_ENUM_CLASS_MeshLoc_ADD(Node)
            .DNDS_PY_ENUM_CLASS_MeshLoc_ADD(Face)
            .DNDS_PY_ENUM_CLASS_MeshLoc_ADD(Cell);
#undef DNDS_PY_ENUM_CLASS_MeshLoc_ADD
    }

    using tPy_UnstructuredMesh = py_class_ssp<UnstructuredMesh>;

    inline void pybind11_UnstructuredMesh_define(py::module_ &m)
    {
        pybind11_MeshAccess_define(m);
        pybind11_NativeMeshPairRead_define(m);
        pybind11_MeshEdit_define(m);
#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(foo) \
    def(#foo, &UnstructuredMesh::foo)

#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(m_name) \
    def_readonly(#m_name, &UnstructuredMesh::m_name, py::return_value_policy::reference_internal)

#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(m_name, kind) \
    def_property_readonly(#m_name, [](std::shared_ptr<UnstructuredMesh> self) { return NativePythonPairRead(*self, NativePythonPairKind::kind); })

        auto UnstructuredMesh_ = tPy_UnstructuredMesh(m, "UnstructuredMesh");
        py::enum_<CurrentMeshCheckpointPolicy>(m, "CurrentMeshCheckpointPolicy")
            .value("RequireNoAuxiliaryValues", CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues)
            .value("DropAuxiliary", CurrentMeshCheckpointPolicy::DropAuxiliary);
        UnstructuredMesh_
            .def(py::init<>([](const MPIInfo &mpi, int n_dim)
                            { return std::make_shared<UnstructuredMesh>(mpi, n_dim); }))
            .def_property_readonly("has_native_state", &UnstructuredMesh::HasNativeState)
            .def("edit", [](std::shared_ptr<UnstructuredMesh> self, int ghostLayers)
                 { return std::make_shared<MeshEdit>(MeshEdit::Begin(std::move(self), ghostLayers)); }, py::arg("ghost_layers") = 1, "Collectively stage native owned coordinate/cell metadata and neighbor-order edits. Native authority is required. Explicit commit required; context exit discards uncommitted changes.")
            .def("entities", &UnstructuredMesh::Entities, py::arg("kind"), "Retain immutable distribution metadata without exposing an ArrayTransformer. Numbering changes invalidate mapping calls; descriptors do not block mutation.")
            .def("original_entity_id", &UnstructuredMesh::OriginalEntityId, py::arg("kind"), py::arg("row"))
            .def("place_native", [](UnstructuredMesh &self, int ordinal)
                 {
                     if (ordinal == -1)
                     {
#ifdef DNDS_USE_CUDA
                         DNDS_check_throw_info(cudaGetDevice(&ordinal) == cudaSuccess, "CUDA current device query failed");
#else
                         DNDS_check_throw_info(false, "CUDA placement unavailable in a CPU-only build");
#endif
                     }
                     DNDS_check_throw_info(ordinal >= 0, "native placement requires a nonnegative CUDA ordinal or -1 for current device");
                     auto device = Runtime::Device::CUDA(ordinal);
                     self.PlaceNative(device, {device, nullptr}); }, py::arg("ordinal") = 0, "Prepare a CUDA placement of published native data; ordinal -1 selects the current CUDA device. Completes transfers before returning. CPU debug reads remain host-resident.")
            .def("apply_geometry_transform", &UnstructuredMesh::ApplyGeometryTransform, py::arg("rotation"), py::arg("scale") = 1.0, "Collective proper rotation and positive scale of coordinates, periodic frames and wall vectors. Native input publishes a tracked replacement; release retained reads first and rebuild dependent metrics afterward.")
            .def("apply_affine_geometry_transform", [](UnstructuredMesh &self, py::object linear, py::object offset)
                 {
                     Eigen::Matrix3d matrix;
                     tPoint translation;
                     // Agree Python shape/conversion errors before any rank
                     // enters the native operation's collective preflight.
                     FieldTransferDetail::LocalStage(self.getMPI(), [&]
                                                     {
                         matrix = linear.cast<Eigen::Matrix3d>();
                         translation = offset.cast<tPoint>(); });
                     self.ApplyAffineGeometryTransform(matrix, translation); }, py::arg("linear"), py::arg("offset") = tPoint{0, 0, 0}, "Collective native-only x'=linear*x+offset with finite orientation-preserving invertible linear. Updates periodic frames atomically; conjugated rotations must remain proper orthonormal within 1e-10. Invalidates wall distances and metrics. All ranks supply identical arguments and release retained read aliases first.")
            .def("rectify_boundary_coordinates", &UnstructuredMesh::RectifyBoundaryCoordinates, py::arg("component_for_zone"), py::arg("volume_boundary") = true, "Collective boundary snapping: callback returns 0 for unchanged or component 1/2/3. Native input publishes a tracked replacement; release retained reads first and rebuild wall distance if needed.")
            .def("snapshot", [](UnstructuredMesh &self, bool faces, bool inverse, bool edges, bool cellFaceNeighbors, bool wallDistance)
                 {
                     MeshReadRequirements requirements;
                     requirements.faces = faces;
                     requirements.inverse = inverse;
                     requirements.edges = edges;
                     requirements.cellFaceNeighbors = cellFaceNeighbors;
                     requirements.wallDistance = wallDistance;
                     return UnstructuredMeshSnapshot::Capture(self, requirements); }, py::arg("faces") = false, py::arg("inverse") = false, py::arg("edges") = false, py::arg("cell_face_neighbors") = false, py::arg("wall_distance") = false, "Copy a validated immutable native snapshot. Source edits do not change its values or metadata.")
            .def("acquire_read", [](UnstructuredMesh &self, bool faces, bool inverse, bool edges, bool cellFaceNeighbors, bool wallDistance)
                 {
                     MeshReadRequirements requirements;
                     requirements.faces = faces;
                     requirements.inverse = inverse;
                     requirements.edges = edges;
                     requirements.cellFaceNeighbors = cellFaceNeighbors;
                     requirements.wallDistance = wallDistance;
                     return std::make_shared<PythonMeshReadAccess>(self.AcquireRead(requirements)); }, py::arg("faces") = false, py::arg("inverse") = false, py::arg("edges") = false, py::arg("cell_face_neighbors") = false, py::arg("wall_distance") = false, "Retained host read scope. Readonly row arrays/slices keep the lease after close; bulk geometry methods return snapshots.")
            // basic
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(coords, Coords)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2node, CellNodes)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(bnd2node, BoundaryNodes)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(bnd2cell, BoundaryCells)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2cell, CellCells)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cellElemInfo, CellInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(bndElemInfo, BoundaryInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2cellOrig, CellOriginalIds)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(node2nodeOrig, NodeOriginalIds)
            // interpolated
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2face, CellFaces)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(face2cell, FaceCells)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(face2node, FaceNodes)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(faceElemInfo, FaceInfo)
            // edge connectivity
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2edge, CellEdges)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(edge2cell, EdgeCells)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(edge2node, EdgeNodes)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(edgeElemInfo, EdgeInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2edgePbi, CellEdgeBits)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(edge2nodePbi, EdgeNodeBits)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(nodeWallDist, Wall)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(node2cell, NodeCells)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(node2bnd, NodeBoundaries)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(face2bnd, FaceBoundaries)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(bnd2face, BoundaryFaces)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2cellFace, CellFaceCells)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2nodePbi, CellNodeBits)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(bnd2nodePbi, BoundaryNodeBits)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(face2nodePbi, FaceNodeBits)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR(cell2facePbi, CellFaceBits);

        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AssertOnN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AssertOnFaces)
            .def("ConstructBndMesh", [](UnstructuredMesh &self, ssp<UnstructuredMesh> &pbMesh)
                 { self.ConstructBndMesh(*pbMesh); }, py::arg("bndMesh"))
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(GetCell2CellFaceVLocal)
            // ObtainLocalFactFillOrdering
            // ObtainSymmetricSymbolicFactorization
            // .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(ReorderLocalCells) //! it has argument now
            .def("getMPI", [](const UnstructuredMesh &self)
                 { return self.getMPI(); })
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(getDim)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNode)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFace)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBndGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBndProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeGlobal)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellGlobal)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceGlobal)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBndGlobal);
        //!!
        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildO2FromO1Elevation)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildBisectO1FormO2)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(IsO1)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(IsO2)
            .def("PrintMeshCGNS", &UnstructuredMesh::PrintMeshCGNS,
                 py::arg("fname"), py::arg("fbcid2name"), py::arg("allNames"));

        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(getArrayBytes);
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_PAIR
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC

        UnstructuredMesh_
            .def("PrepareNativeTopology", &UnstructuredMesh::PrepareNativeTopology, py::arg("faces") = true, py::arg("edges") = false, py::arg("ghost_layers") = 1,
                 "Collectively prepare native topology on this same facade. Release retained reads and rebuild dependent metrics afterward.")
            .def("ReorderLocalCells", &UnstructuredMesh::ReorderLocalCells, py::arg("nParts") = 1, py::arg("nPartsInner") = 1, py::arg("ghost_layers") = 1)
            .def("write_native_archive", &UnstructuredMesh::WriteNativeArchive,
                 py::arg("serializer"), py::arg("name"),
                 "Collective native checkpoint of all published topology, IDs and companions. Finish async producers first; external solver fields are separate.")
            .def_static("read_native_archive", &UnstructuredMesh::ReadNativeArchive,
                        py::arg("mpi"), py::arg("serializer"), py::arg("name"),
                        py::arg("partition") = Serializer::ArrayGlobalOffset_Unknown, py::arg("ghost_layers") = 1,
                        "Collective native restart with saved or even ownership partitions. All ranks participate, including empty ranks; per-rank JSON supports saved partitions only.")
            .def("WriteCurrentPrimaryMeshSnapshot", &WriteCurrentPrimaryMeshSnapshot,
                 py::arg("serializer"), py::arg("name"), py::arg("ghostLayers") = 1,
                 py::arg("policy") = CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues,
                 "Collective primary-only checkpoint. Rebuild optional topology on restart; wall values require explicit DropAuxiliary.")
            .def("WriteCurrentMeshSnapshot", &WriteCurrentMeshSnapshot,
                 py::arg("serializer"), py::arg("name"), py::arg("ghostLayers") = 1,
                 py::arg("policy") = CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues,
                 "Compatibility spelling for collective WriteCurrentPrimaryMeshSnapshot; registered fields and auxiliary values are not saved.");

        UnstructuredMesh_
            .def(
                "CellFaceOther",
                [](const UnstructuredMesh &self, index iCell, index iFace, rowsize ic2f)
                { return self.CellFaceOther(iCell, iFace, ic2f); },
                py::arg("iCell"), py::arg("iFace"), py::arg("ic2f") = rowsize(-1),
                "Return the cell on the opposite side of iFace. ic2f is only required for self-periodic faces.")
            .def(
                "CellIsFaceBack",
                [](const UnstructuredMesh &self, index iCell, index iFace, rowsize ic2f)
                { return self.CellIsFaceBack(iCell, iFace, ic2f); },
                py::arg("iCell"), py::arg("iFace"), py::arg("ic2f") = rowsize(-1),
                "Return whether iCell is face2cell(iFace,0). ic2f is only required for self-periodic faces.");

        auto WallDistOptions_ = py_class_ssp<UnstructuredMesh::WallDistOptions>(UnstructuredMesh_, "WallDistOptions");
        WallDistOptions_.def(py::init());
        WallDistOptions_
            .def_readwrite("method", &UnstructuredMesh::WallDistOptions::method)
            .def_readwrite("verbose", &UnstructuredMesh::WallDistOptions::verbose)
            .def_readwrite("wallDistExecution", &UnstructuredMesh::WallDistOptions::wallDistExecution)
            .def_readwrite("minWallDist", &UnstructuredMesh::WallDistOptions::minWallDist)
            .def_readwrite("subdivide_quad", &UnstructuredMesh::WallDistOptions::subdivide_quad);

        UnstructuredMesh_
            .def("BuildNodeWallDist", &UnstructuredMesh::BuildNodeWallDist, py::arg("fBndIsWall"), py::arg("options") = UnstructuredMesh::WallDistOptions{});

        UnstructuredMesh_
            .def("transform_coordinates", [](UnstructuredMesh &self, py::function transform)
                 {
                     // Python receives an independent point, so a callback
                     // cannot retain a writable alias beyond the native edit.
                     self.TransformCoords([&transform](const tPoint &point) -> tPoint
                                          {
                                              py::array_t<real> input(3);
                                              for (int d = 0; d < 3; ++d)
                                                  input.mutable_data()[d] = point[d];
                                              auto result = transform(input);
                                              auto output = py::array_t<real, py::array::c_style | py::array::forcecast>::ensure(result.is_none() ? py::object(input) : result);
                                              if (!output || output.ndim() != 1 || output.shape(0) != 3)
                                                  throw py::value_error("coordinate transform must return a point with shape (3,), or modify its input and return None");
                                              tPoint value;
                                              for (int d = 0; d < 3; ++d)
                                                  value[d] = output.data()[d];
                                              if (!value.allFinite())
                                                  throw py::value_error("coordinate transform returned a non-finite point");
                                              return value;
                                          }); }, py::arg("transform"), "Collective tracked point transform. Native input validates and publishes an independent replacement; callback receives a detached copy. Release retained reads first; periodic frames and dependent metrics are caller responsibilities.")
            .def("to_device", [](UnstructuredMesh &self, const std::string &backend)
                 {
                     auto source = self.CurrentNativeState();
                     if (backend == "Host")
                     {
                         source->Snapshot(Runtime::Device::Host());
                         return;
                     }
                     DNDS_check_throw_info(backend == "CUDA", "mesh placement backend must be Host or CUDA");
#ifdef DNDS_USE_CUDA
                     int ordinal = -1;
                     DNDS_check_throw_info(cudaGetDevice(&ordinal) == cudaSuccess, "CUDA current device query failed");
                     auto device = Runtime::Device::CUDA(ordinal);
                     self.PlaceNative(device, {device, nullptr});
#else
                     DNDS_check_throw_info(false, "CUDA placement unavailable in a CPU-only build");
#endif
                 },
                 py::arg("backend"), "Place immutable native topology on the currently selected device. Host authority remains available.")
            .def("to_host", [](UnstructuredMesh &self)
                 { self.CurrentNativeState()->Snapshot(Runtime::Device::Host()); }, "Validate the resident immutable host publication; native mesh data are never device-mutable.");
        pybind11_NativeMeshBuilder_define(m);
        pybind11_NativeMeshPreparation_define(m);
    }

    using tPy_UnstructuredMeshSerialRW = py_class_ssp<UnstructuredMeshSerialRW>;

    inline void pybind11_UnstructuredMeshSerialRW_define(py::module_ &m)
    {
        auto UnstructuredMeshSerialRW_ = tPy_UnstructuredMeshSerialRW(m, "UnstructuredMeshSerialRW");
        UnstructuredMeshSerialRW_
            .def(py::init(
                     [](ssp<UnstructuredMesh> mesh, MPI_int mRank)
                     { return std::make_shared<UnstructuredMeshSerialRW>(mesh, mRank); }),
                 py::arg("mesh"), py::arg("mRank") = 0)
            .def_readwrite("mesh", &UnstructuredMeshSerialRW::mesh)
            .def_readonly("dataIsSerialOut", &UnstructuredMeshSerialRW::dataIsSerialOut)
            .def_readonly("dataIsSerialIn", &UnstructuredMeshSerialRW::dataIsSerialIn)
            .def("BuildSerialOut", &UnstructuredMeshSerialRW::BuildSerialOut,
                 "Collectively prepare a native serial output descriptor. Input readers are separate native ingress functions.");
    }
}
