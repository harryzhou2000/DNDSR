#include "FiniteVolumePlan.hpp"
#include "InterfaceSamplingPlan.hpp"
#include "DistributedInterfaceSamplingPlan.hpp"
#include "DNDS/Runtime/Array_bind.hpp"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace DNDS::CFV
{
    void pybind11_FiniteVolumePlan_define(pybind11::module_ &m)
    {
        namespace py = pybind11;
        py::class_<ReconstructionSettings>(m, "ReconstructionSettings")
            .def(py::init<>())
            .def_readwrite("maxOrder", &ReconstructionSettings::maxOrder)
            .def_readwrite("intOrder", &ReconstructionSettings::intOrder)
            .def_readwrite("jacobiRelax", &ReconstructionSettings::jacobiRelax)
            .def_readwrite("SORInstead", &ReconstructionSettings::SORInstead)
            .def_readwrite("HQMWeights", &ReconstructionSettings::HQMWeights)
            .def_readwrite("bcWeight", &ReconstructionSettings::bcWeight)
            .def_readwrite("boundaryDerivativeWeights", &ReconstructionSettings::boundaryDerivativeWeights)
            .def_readwrite("svdTolerance", &ReconstructionSettings::svdTolerance)
            .def_readwrite("svdAbsoluteTolerance", &ReconstructionSettings::svdAbsoluteTolerance)
            .def_readwrite("WBAP_nStd", &ReconstructionSettings::WBAP_nStd)
            .def("Validate", &ReconstructionSettings::Validate);
        // This is the DNDS numerical field, not a second Python registration
        // or a CFV-specific payload owner.
        m.attr("FVField") = py::type::of<FVField>();
        py::class_<ReconstructionInverseDiagnostics>(m, "ReconstructionInverseDiagnostics")
            .def_readonly("cells", &ReconstructionInverseDiagnostics::cells)
            .def_readonly("filteredCells", &ReconstructionInverseDiagnostics::filteredCells)
            .def_readonly("discardedModes", &ReconstructionInverseDiagnostics::discardedModes)
            .def_readonly("rankZeroCells", &ReconstructionInverseDiagnostics::rankZeroCells)
            .def_readonly("maxCondition", &ReconstructionInverseDiagnostics::maxCondition);
        py::class_<HardEigen::SVDInverseDiagnostics>(m, "SVDInverseDiagnostics")
            .def_readonly("modes", &HardEigen::SVDInverseDiagnostics::modes)
            .def_readonly("rank", &HardEigen::SVDInverseDiagnostics::rank)
            .def_readonly("relativeCutoff", &HardEigen::SVDInverseDiagnostics::relativeCutoff)
            .def_readonly("rawCondition", &HardEigen::SVDInverseDiagnostics::rawCondition)
            .def_readonly("retainedCondition", &HardEigen::SVDInverseDiagnostics::retainedCondition);
        py::class_<FiniteVolumePlan>(m, "FiniteVolumePlan")
            .def_static("ConstructMetrics", &FiniteVolumePlan::ConstructMetrics, py::arg("mesh"), py::arg("intOrder") = 5,
                        "Collective CPU preparation. All ranks must call with matching settings.")
            .def("Validate", &FiniteVolumePlan::Validate)
            .def("NumCell", &FiniteVolumePlan::NumCell)
            .def("NumCellProc", &FiniteVolumePlan::NumCellProc)
            .def("NumFaceProc", &FiniteVolumePlan::NumFaceProc)
            .def("BuildField", &FiniteVolumePlan::BuildField)
            .def("BindHalo", &FiniteVolumePlan::BindHalo, "Collective persistent direct MPI binding; explicitly begin/wait on each field stage.")
            .def("GetCellVol", &FiniteVolumePlan::GetCellVol)
            .def("GetCellCent", &FiniteVolumePlan::GetCellCent);
        py::class_<ReconstructionBoundaryValues>(m, "ReconstructionBoundaryValues")
            .def_property_readonly("values", [](ReconstructionBoundaryValues &value)
                                   { return Runtime::PythonArray::Share(value.values); });
        py::class_<BoundaryQuadraturePoint>(m, "BoundaryQuadraturePoint")
            .def_readonly("cell", &BoundaryQuadraturePoint::cell)
            .def_readonly("face", &BoundaryQuadraturePoint::face)
            .def_readonly("quadrature", &BoundaryQuadraturePoint::quadrature)
            .def_readonly("point", &BoundaryQuadraturePoint::point)
            .def_readonly("normal", &BoundaryQuadraturePoint::normal);
        py::class_<FaceCharacteristicTransform>(m, "FaceCharacteristicTransform")
            .def_readonly("first", &FaceCharacteristicTransform::first)
            .def_readonly("count", &FaceCharacteristicTransform::count)
            .def_property_readonly("forward", [](FaceCharacteristicTransform &value)
                                   { return Runtime::PythonArray::Share(value.forward); })
            .def_property_readonly("inverse", [](FaceCharacteristicTransform &value)
                                   { return Runtime::PythonArray::Share(value.inverse); });
        py::class_<ReconstructionWorkspace>(m, "ReconstructionWorkspace")
            .def_readonly("coefficients", &ReconstructionWorkspace::coefficients)
            .def_readonly("temporary", &ReconstructionWorkspace::temporary);
        py::class_<DistributedSamplingWorkspace>(m, "DistributedSamplingWorkspace")
            .def_property_readonly("donorValues", [](DistributedSamplingWorkspace &work)
                                   { return Runtime::PythonArray::Share(work.donorValues); });
        py::class_<VariationalReconstructionPlan>(m, "VariationalReconstructionPlan")
            .def_static("ConstructRecCoeff", &VariationalReconstructionPlan::ConstructRecCoeff,
                        py::arg("metrics"), py::arg("settings") = ReconstructionSettings{})
            .def("NRecDOF", &VariationalReconstructionPlan::NRecDOF)
            .def("GetInverseDiagnostics", &VariationalReconstructionPlan::GetInverseDiagnostics)
            .def("GetCellInverseDiagnostics", &VariationalReconstructionPlan::GetCellInverseDiagnostics)
            .def("Validate", &VariationalReconstructionPlan::Validate)
            .def("BuildWorkspace", &VariationalReconstructionPlan::BuildWorkspace)
            .def("BuildBoundaryValues", &VariationalReconstructionPlan::BuildBoundaryValues)
            .def("NumBoundaryQuadrature", &VariationalReconstructionPlan::NumBoundaryQuadrature)
            .def("GetBoundaryQuadrature", &VariationalReconstructionPlan::GetBoundaryQuadrature)
            .def("BuildFaceTransform", &VariationalReconstructionPlan::BuildFaceTransform)
            .def("FinalizeFaceTransform", &VariationalReconstructionPlan::FinalizeFaceTransform,
                 "Validate and publish matrices after filling from cell means; any later write requires re-finalization.")
            .def("DoReconstructionIter", &VariationalReconstructionPlan::DoReconstructionIter,
                 "One synchronous CPU sweep. Release Python array views and refresh input halos first.")
            .def("DoLimiterWBAP_3Order", &VariationalReconstructionPlan::DoLimiterWBAP_3Order,
                 py::arg("workspace"), py::arg("order"), py::arg("transform") = nullptr,
                 "Call in descending order with a coefficient halo exchange after each order.")
            .def("Evaluate", &VariationalReconstructionPlan::Evaluate);
        py::class_<InterfaceSamplingPlan>(m, "InterfaceSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, Geom::InterfacePointMap>())
            .def("Size", &InterfaceSamplingPlan::Size)
            .def("Evaluate", &InterfaceSamplingPlan::Evaluate,
                 "Sample donor-frame values in map query order. The equation adapter transforms vector components.");
        py::class_<DistributedInterfaceSamplingPlan>(m, "DistributedInterfaceSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, Geom::DistributedInterfacePointMap>())
            .def("Size", &DistributedInterfaceSamplingPlan::Size)
            .def("Validate", &DistributedInterfaceSamplingPlan::Validate)
            .def("BuildWorkspace", &DistributedInterfaceSamplingPlan::BuildWorkspace)
            .def("Evaluate", &DistributedInterfaceSamplingPlan::Evaluate,
                 "Collective owner evaluation and direct return transfer. Release views and finish all producers first.");
    }
}
