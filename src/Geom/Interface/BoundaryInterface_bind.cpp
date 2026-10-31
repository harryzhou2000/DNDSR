#include "BoundaryInterface.hpp"
#include "DNDS/Runtime/Array_bind.hpp"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace DNDS::Geom
{
    void pybind11_BoundaryInterface_define(pybind11::module_ &m)
    {
        namespace py = pybind11;
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
                        "Read selected owned Line2/Tri3/Quad4 boundaries and adjacent volume slots from a prepared mesh.")
            .def("WithPose", &BoundaryPatch::WithPose, "Return a new immutable pose/index; existing maps retain their old pose.")
            .def("Validate", &BoundaryPatch::Validate)
            .def("Size", &BoundaryPatch::Size)
            .def("Element", &BoundaryPatch::Element)
            .def("Pose", &BoundaryPatch::Pose);
        py::class_<InterfaceSearchSettings>(m, "InterfaceSearchSettings")
            .def(py::init<>())
            .def_readwrite("maxGap", &InterfaceSearchSettings::maxGap)
            .def_readwrite("referenceTolerance", &InterfaceSearchSettings::referenceTolerance)
            .def_readwrite("projectionTolerance", &InterfaceSearchSettings::projectionTolerance)
            .def_readwrite("maxIterations", &InterfaceSearchSettings::maxIterations);
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
