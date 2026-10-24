#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include "Mesh.hpp"
#include "MeshAccess_bind.hpp"
#include "MeshTransition.hpp"
#include "MeshEdit_bind.hpp"
#include <pybind11/pybind11.h>
#include <pybind11_json/pybind11_json.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

namespace DNDS::Geom
{
    using tPy_ElemInfo = py_class_ssp<ElemInfo>;

    inline void pybind11_ElemInfo_define(py::module_ &m)
    {
        auto ElemInfo_ = tPy_ElemInfo(m, "ElemInfo");
        ElemInfo_
            .def("getElemType", &ElemInfo::getElemType)
            .def("setElemType", &ElemInfo::setElemType)
            .def_readwrite("zone", &ElemInfo::zone);
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
        pybind11_MeshEdit_define(m);
#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(foo) \
    def(#foo, &UnstructuredMesh::foo)

#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(m_name) \
    def_readonly(#m_name, &UnstructuredMesh::m_name, py::return_value_policy::reference_internal)

        auto UnstructuredMesh_ = tPy_UnstructuredMesh(m, "UnstructuredMesh");
        py::enum_<CurrentMeshCheckpointPolicy>(m, "CurrentMeshCheckpointPolicy")
            .value("RequireNoAuxiliaryValues", CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues)
            .value("DropAuxiliary", CurrentMeshCheckpointPolicy::DropAuxiliary);
        UnstructuredMesh_
            .def(py::init<>([](const MPIInfo &mpi, int n_dim)
                            { return std::make_shared<UnstructuredMesh>(mpi, n_dim); }))
            .def_property_readonly("has_native_state", &UnstructuredMesh::HasNativeState)
            .def("edit", [](std::shared_ptr<UnstructuredMesh> self, int ghostLayers)
                 { return std::make_shared<MeshEdit>(MeshEdit::Begin(std::move(self), ghostLayers)); }, py::arg("ghost_layers") = 1, "Collectively stage current-authority owned coordinate/cell metadata and neighbor-order edits. Explicit commit required; context exit discards uncommitted changes.")
            .def("entities", &UnstructuredMesh::Entities, py::arg("kind"), "Retain immutable distribution metadata without exposing an ArrayTransformer. Numbering changes invalidate mapping calls; descriptors do not block mutation.")
            .def("publish_native", [](UnstructuredMesh &self, bool faces, bool inverse, bool edges, bool cellFaceNeighbors, bool wallDistance, bool allFaceCellsLocal)
                 {
                     MeshReadRequirements r;
                     r.faces = faces; r.inverse = inverse; r.edges = edges;
                     r.cellFaceNeighbors = cellFaceNeighbors; r.wallDistance = wallDistance;
                     r.allFaceCellsLocal = allFaceCellsLocal;
                     self.PublishNative(r); }, py::arg("faces") = false, py::arg("inverse") = false, py::arg("edges") = false, py::arg("cell_face_neighbors") = false, py::arg("wall_distance") = false, py::arg("all_face_cells_local") = false, "Collectively validate/import the legacy workspace and publish private native query storage. Release managed rows first; raw workspace edits do not alter the publication.")
            .def("place_native", [](UnstructuredMesh &self, int ordinal)
                 {
                     auto device = Runtime::Device::CUDA(ordinal);
                     self.PlaceNative(device, {device, nullptr}); }, py::arg("ordinal") = 0, "Prepare a CUDA placement of published native data; completes transfers before returning. CPU debug reads remain host-resident.")
            .def("use_legacy_workspace", &UnstructuredMesh::UseLegacyWorkspace, "Collectively leave native publication mode and invalidate all current-state bindings; detached snapshots remain readable.")
            .def("apply_geometry_transform", &UnstructuredMesh::ApplyGeometryTransform, py::arg("rotation"), py::arg("scale") = 1.0, "Collective legacy-workspace proper rotation and positive scale of coordinates, periodic frames and wall vectors. Native authority requires mesh.edit; rebuild dependent metrics afterward.")
            .def("rectify_boundary_coordinates", &UnstructuredMesh::RectifyBoundaryCoordinates, py::arg("component_for_zone"), py::arg("volume_boundary") = true, "Collective legacy-workspace boundary snapping: callback returns 0 for unchanged or component 1/2/3. Native authority requires mesh.edit; wall distance must be rebuilt if needed.")
            .def("snapshot", [](UnstructuredMesh &self, bool faces, bool inverse, bool edges, bool cellFaceNeighbors, bool wallDistance)
                 {
                     MeshReadRequirements requirements;
                     requirements.faces = faces;
                     requirements.inverse = inverse;
                     requirements.edges = edges;
                     requirements.cellFaceNeighbors = cellFaceNeighbors;
                     requirements.wallDistance = wallDistance;
                     return NativeMeshSnapshot::Capture(self, requirements); }, py::arg("faces") = false, py::arg("inverse") = false, py::arg("edges") = false, py::arg("cell_face_neighbors") = false, py::arg("wall_distance") = false, "Copy a validated immutable native snapshot. Source edits do not change its values or metadata.")
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
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(coords)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bnd2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bnd2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cellElemInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bndElemInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2cellOrig)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(node2nodeOrig)
            // interpolated
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2face)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(face2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(face2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(faceElemInfo)
            // edge connectivity
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2edge)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(edge2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(edge2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(edgeElemInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2edgePbi)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(edge2nodePbi)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(nodeWallDist);

        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(RecoverNode2CellAndNode2Bnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(RecoverCell2CellAndBnd2Cell)
            .def("BuildGhostPrimary", &UnstructuredMesh::BuildGhostPrimary, py::arg("nGhostLayers") = 1)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalPrimary)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalPrimary)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalPrimaryForBnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalPrimaryForBnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalFacial)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalFacial)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalC2F)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalC2F)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalEdge)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalEdge)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildGhostN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AssertOnN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(InterpolateFace)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(InterpolateEdge)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AssertOnFaces)
            .def("ConstructBndMesh", [](UnstructuredMesh &self, ssp<UnstructuredMesh> &pbMesh)
                 { self.ConstructBndMesh(*pbMesh); }, py::arg("bndMesh"))
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(GetCell2CellFaceVLocal)
            // ObtainLocalFactFillOrdering
            // ObtainSymmetricSymbolicFactorization
            // .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(ReorderLocalCells) //! it has argument now
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(getMPI)
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
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(RecreatePeriodicNodes)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildVTKConnectivity)
            .def("PrintMeshCGNS", &UnstructuredMesh::PrintMeshCGNS,
                 py::arg("fname"), py::arg("fbcid2name"), py::arg("allNames"));

        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(getArrayBytes);
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC

        UnstructuredMesh_
            .def("ReorderLocalCells", &UnstructuredMesh::ReorderLocalCells, py::arg("nParts") = 1, py::arg("nPartsInner") = 1)
            .def("ReadSerialize", &UnstructuredMesh::ReadSerialize, py::arg("serializer"), py::arg("name"))
            .def("WriteSerialize", &UnstructuredMesh::WriteSerialize, py::arg("serializer"), py::arg("name"))
            .def("WriteCurrentPrimaryMeshSnapshot", &WriteCurrentPrimaryMeshSnapshot,
                 py::arg("serializer"), py::arg("name"), py::arg("ghostLayers") = 1,
                 py::arg("policy") = CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues,
                 "Collective primary-only checkpoint. Rebuild optional topology on restart; wall values require explicit DropAuxiliary.")
            .def("WriteCurrentMeshSnapshot", &WriteCurrentMeshSnapshot,
                 py::arg("serializer"), py::arg("name"), py::arg("ghostLayers") = 1,
                 py::arg("policy") = CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues,
                 "Compatibility spelling for collective WriteCurrentPrimaryMeshSnapshot; registered fields and auxiliary values are not saved.")
            .def(
                "ReadSerializeAndDistribute",
                [](UnstructuredMesh &self, Serializer::SerializerBaseSSP serializer,
                   const std::string &name, py::object options_in)
                {
                    auto options_full = PartitionOptions();
                    auto json_options_full = nlohmann::ordered_json(options_full);
                    if (!options_in.is_none())
                    {
                        auto json_options_in = nlohmann::json(options_in);
                        json_options_full.merge_patch(json_options_in);
                    }
                    self.ReadSerializeAndDistribute(
                        serializer, name,
                        json_options_full.template get<PartitionOptions>());
                },
                py::arg("serializer"), py::arg("name"), py::arg("partitionOptions") = py::none());

        UnstructuredMesh_.def("SetPeriodicGeometry", &UnstructuredMesh::SetPeriodicGeometry,
                              py::arg("translation1") = Geom::tPoint{1, 0, 0},
                              py::arg("rotationCenter1") = Geom::tPoint{0, 0, 0},
                              py::arg("eulerAngles1") = Geom::tPoint{0, 0, 0},
                              py::arg("translation2") = Geom::tPoint{0, 1, 0},
                              py::arg("rotationCenter2") = Geom::tPoint{0, 0, 0},
                              py::arg("eulerAngles2") = Geom::tPoint{0, 0, 0},
                              py::arg("translation3") = Geom::tPoint{0, 0, 1},
                              py::arg("rotationCenter3") = Geom::tPoint{0, 0, 0},
                              py::arg("eulerAngles3") = Geom::tPoint{0, 0, 0});

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
                                          }); }, py::arg("transform"), "Tracked legacy-workspace point transform. Native authority requires mesh.edit. Callback receives a detached copy; exceptions leave the mesh failed. Periodic frames and derived caches are caller responsibilities.")
            .def("TransformCoords", [](py::object self, py::function f)
                 {
                    auto &mesh = self.cast<UnstructuredMesh &>();
                    mesh.RequireLegacyWorkspace("legacy Python TransformCoords");
                    auto make_view = [&self](ssp<DNDS::ArrayEigenVector<3>> &arr) -> py::array_t<real>
                    {
                        index N = arr->Size();
                        std::vector<py::ssize_t> shape = {py::ssize_t(N), py::ssize_t(3)};
                        std::vector<py::ssize_t> strides = {py::ssize_t(sizeof(real) * 3),
                                                            py::ssize_t(sizeof(real))};
                        return {shape, strides, arr->data(), self};
                    };
                    if (mesh.coords.father->Size() > 0)
                        f(make_view(mesh.coords.father));
                    if (mesh.coords.son->Size() > 0)
                        f(make_view(mesh.coords.son)); }, py::arg("func"))
            .def("to_device", [](UnstructuredMesh &self, const std::string &backend)
                 { self.to_device(device_backend_name_to_enum(backend)); }, py::arg("backend"))
            .def("to_host", &UnstructuredMesh::to_host);
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
            .def("ReadFromCGNSSerial", py::overload_cast<const std::string &>(&UnstructuredMeshSerialRW::ReadFromCGNSSerial))
            .def("Deduplicate1to1Periodic", &UnstructuredMeshSerialRW::Deduplicate1to1Periodic)
            .def("BuildCell2Cell", &UnstructuredMeshSerialRW::BuildCell2Cell)
            .def("BuildSerialOut", &UnstructuredMeshSerialRW::BuildSerialOut)
            .def(
                "MeshPartitionCell2Cell",
                [](UnstructuredMeshSerialRW &self, py::object options_in) // use default
                {
                    auto options_full = UnstructuredMeshSerialRW::PartitionOptions();
                    auto json_options_full = nlohmann::ordered_json(options_full); // warning: using {} here makes it a list
                    if (!options_in.is_none())
                    {
                        auto json_options_in = nlohmann::json(options_in);
                        json_options_full.merge_patch(json_options_in);
                    }
                    // std::cout << json_options_full << std::endl;
                    self.MeshPartitionCell2Cell(json_options_full.template get<UnstructuredMeshSerialRW::PartitionOptions>());
                },
                py::arg("options") = py::none())
            .def("PartitionReorderToMeshCell2Cell", &UnstructuredMeshSerialRW::PartitionReorderToMeshCell2Cell);
    }
}
