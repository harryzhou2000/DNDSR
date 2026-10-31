#include "BoundaryInterface.hpp"
#include "DNDS/Runtime/Array_bind.hpp"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace DNDS::Geom
{
    void pybind11_BoundaryInterface_define(pybind11::module_ &m)
    {
        namespace py = pybind11;
        py::enum_<BoundingBoxBackend>(m, "BoundingBoxBackend")
            .value("Native", BoundingBoxBackend::Native)
            .value("CGAL", BoundingBoxBackend::CGAL);
        py::class_<RigidPose>(m, "RigidPose")
            .def(py::init<>())
            .def_readwrite("rotation", &RigidPose::rotation)
            .def_readwrite("translation", &RigidPose::translation)
            .def("Apply", &RigidPose::Apply)
            .def("Inverse", &RigidPose::Inverse);
        py::class_<BoundaryElementRecord>(m, "BoundaryElementRecord")
            .def_readonly("boundary", &BoundaryElementRecord::boundary)
            .def_readonly("face", &BoundaryElementRecord::face)
            .def_readonly("cell", &BoundaryElementRecord::cell)
            .def_readonly("cellFaceSlot", &BoundaryElementRecord::cellFaceSlot)
            .def_readonly("originalId", &BoundaryElementRecord::originalId);
        py::class_<BoundaryPatch>(m, "BoundaryPatch")
            .def_static("Create", &BoundaryPatch::Create, py::arg("mesh"), py::arg("boundaries"), py::arg("pose") = RigidPose{},
                        py::arg("backend") = BoundingBoxBackend::CGAL,
                        "Read selected owned Line2/Tri3/Quad4 boundaries and adjacent volume slots from a prepared mesh.")
            .def("WithPose", &BoundaryPatch::WithPose, "Return a new immutable pose sharing reference geometry/index; old maps retain their pose.")
            .def("WithBackend", &BoundaryPatch::WithBackend)
            .def("Backend", &BoundaryPatch::Backend)
            .def("Validate", &BoundaryPatch::Validate)
            .def("Size", &BoundaryPatch::Size)
            .def("Element", &BoundaryPatch::Element)
            .def("Pose", &BoundaryPatch::Pose)
            .def("SurfaceGeometry", &BoundaryPatch::SurfaceGeometry, "Return an immutable geometry snapshot independent of mesh/volume links.");
        py::class_<InterfaceSearchSettings>(m, "InterfaceSearchSettings")
            .def(py::init<>())
            .def_readwrite("maxGap", &InterfaceSearchSettings::maxGap)
            .def_readwrite("referenceTolerance", &InterfaceSearchSettings::referenceTolerance)
            .def_readwrite("projectionTolerance", &InterfaceSearchSettings::projectionTolerance)
            .def_readwrite("maxIterations", &InterfaceSearchSettings::maxIterations)
            .def_readwrite("coarseBoxesPerRank", &InterfaceSearchSettings::coarseBoxesPerRank);
        m.attr("SurfaceSearchSettings") = m.attr("InterfaceSearchSettings");
        py::class_<SurfaceElementRecord>(m, "SurfaceElementRecord")
            .def(py::init([](index id, Elem::ElemType type)
                          { return SurfaceElementRecord{id, type}; }))
            .def_readonly("id", &SurfaceElementRecord::id)
            .def_readonly("type", &SurfaceElementRecord::type);
        py::class_<SurfacePatch>(m, "SurfacePatch")
            .def_static("Create", &SurfacePatch::Create, py::arg("elements"), py::arg("coordinates"), py::arg("pose") = RigidPose{},
                        py::arg("backend") = BoundingBoxBackend::CGAL,
                        "Retain host Runtime coordinate rows [element,3*nodes]. A read lease freezes shared storage; distributed IDs must be globally unique.")
            .def("WithPose", &SurfacePatch::WithPose, "Return a new pose sharing immutable reference geometry/index; mesh coordinates stay unchanged.")
            .def("WithBackend", &SurfacePatch::WithBackend)
            .def("Backend", &SurfacePatch::Backend)
            .def("Size", &SurfacePatch::Size)
            .def("Element", &SurfacePatch::Element)
            .def("Pose", &SurfacePatch::Pose);
        py::class_<SurfacePointMatch>(m, "SurfacePointMatch")
            .def_readonly("queryId", &SurfacePointMatch::queryId)
            .def_readonly("element", &SurfacePointMatch::element)
            .def_readonly("elementId", &SurfacePointMatch::elementId)
            .def_readonly("geometryRank", &SurfacePointMatch::geometryRank)
            .def_readonly("reference", &SurfacePointMatch::reference)
            .def_readonly("projectedPoint", &SurfacePointMatch::projectedPoint)
            .def_readonly("surfacePoint", &SurfacePointMatch::surfacePoint)
            .def_readonly("normal", &SurfacePointMatch::normal)
            .def_readonly("distance", &SurfacePointMatch::distance);
        py::class_<SurfacePointMap>(m, "SurfacePointMap")
            .def_static("Match", &SurfacePointMap::Match, py::arg("surface"), py::arg("points"), py::arg("queryIds"),
                        py::arg("settings") = SurfaceSearchSettings{}, "Local geometry-only normal projection. Query points are in world coordinates.")
            .def("Size", &SurfacePointMap::Size)
            .def("Get", &SurfacePointMap::Get)
            .def("ValidateFor", &SurfacePointMap::ValidateFor);
        py::enum_<SurfaceRoutingMode>(m, "SurfaceRoutingMode")
            .value("CoarseBoxes", SurfaceRoutingMode::CoarseBoxes)
            .value("AllGatherSurface", SurfaceRoutingMode::AllGatherSurface);
        py::class_<DistributedSurfacePointMap>(m, "DistributedSurfacePointMap")
            .def_static("Match", &DistributedSurfacePointMap::Match, py::arg("mpi"), py::arg("surface"), py::arg("points"), py::arg("queryIds"),
                        py::arg("mode") = SurfaceRoutingMode::CoarseBoxes, py::arg("settings") = SurfaceSearchSettings{},
                        "Collective geometry-only search. All ranks participate with matching controls, including empty owners/queries.")
            .def("Size", &DistributedSurfacePointMap::Size)
            .def("Get", &DistributedSurfacePointMap::Get)
            .def("ValidateFor", &DistributedSurfacePointMap::ValidateFor)
            .def("BuildElementDataRoute", &DistributedSurfacePointMap::BuildElementDataRoute,
                 "Collective canonical route from matched geometry-owner element rows into original query order. Retain the communicator through transfer completion.");
        py::class_<InterfacePointMatch>(m, "InterfacePointMatch")
            .def_readonly("queryId", &InterfacePointMatch::queryId)
            .def_readonly("donorElement", &InterfacePointMatch::donorElement)
            .def_readonly("donorBoundary", &InterfacePointMatch::donorBoundary)
            .def_readonly("donorCell", &InterfacePointMatch::donorCell)
            .def_readonly("donorOriginalId", &InterfacePointMatch::donorOriginalId)
            .def_readonly("reference", &InterfacePointMatch::reference)
            .def_readonly("projectedPoint", &InterfacePointMatch::projectedPoint)
            .def_readonly("donorPoint", &InterfacePointMatch::donorPoint)
            .def_readonly("distance", &InterfacePointMatch::distance);
        py::class_<InterfacePointMap>(m, "InterfacePointMap")
            .def_static("Match", &InterfacePointMap::Match, py::arg("donor"), py::arg("points"), py::arg("queryIds"),
                        py::arg("settings") = InterfaceSearchSettings{},
                        "Single-rank search. Runtime arrays have shapes [queries,3] float64 and [queries,1] int64; query IDs must be unique.")
            .def("Validate", &InterfacePointMap::Validate)
            .def("Size", &InterfacePointMap::Size)
            .def("Get", &InterfacePointMap::Get);
        py::enum_<InterfaceRoutingMode>(m, "InterfaceRoutingMode")
            .value("CoarseBoxes", InterfaceRoutingMode::CoarseBoxes)
            .value("AllGatherBoundary", InterfaceRoutingMode::AllGatherBoundary);
        py::class_<DistributedInterfaceMatch>(m, "DistributedInterfaceMatch")
            .def_readonly("geometry", &DistributedInterfaceMatch::geometry)
            .def_readonly("geometryRank", &DistributedInterfaceMatch::geometryRank)
            .def_readonly("donorRank", &DistributedInterfaceMatch::donorRank)
            .def_readonly("donorOwnedCell", &DistributedInterfaceMatch::donorOwnedCell);
        py::class_<DistributedInterfacePointMap>(m, "DistributedInterfacePointMap")
            .def_static("Match", &DistributedInterfacePointMap::Match, py::arg("donor"), py::arg("points"), py::arg("queryIds"),
                        py::arg("mode") = InterfaceRoutingMode::CoarseBoxes, py::arg("settings") = InterfaceSearchSettings{},
                        "Collective fixed-record candidate/proposal matching; all ranks participate, including empty owners.")
            .def("Size", &DistributedInterfacePointMap::Size)
            .def("Get", &DistributedInterfacePointMap::Get)
            .def("Validate", &DistributedInterfacePointMap::Validate)
            .def("ValidateFor", &DistributedInterfacePointMap::ValidateFor);
    }
}
