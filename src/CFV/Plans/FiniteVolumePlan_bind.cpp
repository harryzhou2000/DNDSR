#include "FiniteVolumePlan.hpp"
#include "InterfaceSamplingPlan.hpp"
#include "DistributedInterfaceSamplingPlan.hpp"
#include "GradientPlan.hpp"
#include "DNDS/Runtime/Array_bind.hpp"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace DNDS::CFV
{
    void pybind11_FiniteVolumePlan_define(pybind11::module_ &m)
    {
        namespace py = pybind11;
        py::enum_<ReconstructionRegion>(m, "ReconstructionRegion")
            .value("All", ReconstructionRegion::All)
            .value("Core", ReconstructionRegion::Core)
            .value("HaloDependent", ReconstructionRegion::HaloDependent);
        using F = ReconstructionFunctionalSettings;
        py::enum_<F::DirectionalWeight>(m, "ReconstructionDirectionalWeight")
            .value("Factorial", F::DirectionalWeight::Factorial)
            .value("HQM_OPT", F::DirectionalWeight::HQM_OPT)
            .value("TEST_OPT", F::DirectionalWeight::TEST_OPT)
            .value("ManualDirWeight", F::DirectionalWeight::ManualDirWeight);
        py::enum_<F::GeometricWeight>(m, "ReconstructionGeometricWeight")
            .value("GWNone", F::GeometricWeight::GWNone)
            .value("HQM_SD", F::GeometricWeight::HQM_SD)
            .value("SD_Power", F::GeometricWeight::SD_Power);
        py::enum_<F::LengthScale>(m, "ReconstructionLengthScale")
            .value("BaryDiff", F::LengthScale::BaryDiff)
            .value("MeanAACBB", F::LengthScale::MeanAACBB)
            .value("CellMax", F::LengthScale::CellMax);
        py::class_<F>(m, "ReconstructionFunctionalSettings")
            .def(py::init<>())
            .def_readwrite("dirWeightScheme", &F::dirWeightScheme)
            .def_readwrite("geomWeightScheme", &F::geomWeightScheme)
            .def_readwrite("scaleType", &F::scaleType)
            .def_readwrite("manualDirWeights", &F::manualDirWeights)
            .def_readwrite("scaleMultiplier", &F::scaleMultiplier)
            .def_readwrite("geomWeightBias", &F::geomWeightBias)
            .def_readwrite("geomWeightPower", &F::geomWeightPower)
            .def_readwrite("geomWeightPower1", &F::geomWeightPower1)
            .def_readwrite("geomWeightPower2", &F::geomWeightPower2)
            .def_readwrite("dirWeightCombPowV", &F::dirWeightCombPowV)
            .def("Validate", &F::Validate);
        py::class_<ReconstructionSettings>(m, "ReconstructionSettings")
            .def(py::init<>())
            .def_readwrite("maxOrder", &ReconstructionSettings::maxOrder)
            .def_readwrite("intOrder", &ReconstructionSettings::intOrder)
            .def_readwrite("jacobiRelax", &ReconstructionSettings::jacobiRelax)
            .def_readwrite("SORInstead", &ReconstructionSettings::SORInstead)
            .def_readwrite("coreFirst", &ReconstructionSettings::coreFirst)
            .def_readwrite("HQMWeights", &ReconstructionSettings::HQMWeights)
            .def_readwrite("cacheDiffBase", &ReconstructionSettings::cacheDiffBase)
            .def_readwrite("functional", &ReconstructionSettings::functional)
            .def_readwrite("bcWeight", &ReconstructionSettings::bcWeight)
            .def_readwrite("boundaryDerivativeWeights", &ReconstructionSettings::boundaryDerivativeWeights)
            .def_readwrite("boundaryZoneWeights", &ReconstructionSettings::boundaryZoneWeights)
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
            .def_readonly("maxCondition", &ReconstructionInverseDiagnostics::maxCondition)
            .def_readonly("maxConditionCell", &ReconstructionInverseDiagnostics::maxConditionCell)
            .def_readonly("rankCounts", &ReconstructionInverseDiagnostics::rankCounts);
        py::class_<HardEigen::SVDInverseDiagnostics>(m, "SVDInverseDiagnostics")
            .def_readonly("modes", &HardEigen::SVDInverseDiagnostics::modes)
            .def_readonly("rank", &HardEigen::SVDInverseDiagnostics::rank)
            .def_readonly("relativeCutoff", &HardEigen::SVDInverseDiagnostics::relativeCutoff)
            .def_readonly("rawCondition", &HardEigen::SVDInverseDiagnostics::rawCondition)
            .def_readonly("retainedCondition", &HardEigen::SVDInverseDiagnostics::retainedCondition);
        py::class_<FiniteVolumePlan>(m, "FiniteVolumePlan")
            .def_static("ConstructMetrics", &FiniteVolumePlan::ConstructMetrics, py::arg("mesh"), py::arg("intOrder") = 5, py::arg("axisymmetric") = false,
                        "Collective CPU preparation. All ranks must call with matching settings.")
            .def("Validate", &FiniteVolumePlan::Validate)
            .def("NumCell", &FiniteVolumePlan::NumCell)
            .def("NumCellProc", &FiniteVolumePlan::NumCellProc)
            .def("NumFaceProc", &FiniteVolumePlan::NumFaceProc)
            .def("Axisymmetric", &FiniteVolumePlan::Axisymmetric)
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
        py::class_<PolynomialSamplingPlan>(m, "PolynomialSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, const Runtime::Array<index> &, const Runtime::Array<real> &>())
            .def("Size", &PolynomialSamplingPlan::Size)
            .def("Validate", &PolynomialSamplingPlan::Validate)
            .def("Evaluate", &PolynomialSamplingPlan::Evaluate, py::arg("mean"), py::arg("coefficients"),
                 py::arg("output"), py::arg("derivative") = 0);
        py::class_<VariationalReconstructionPlan>(m, "VariationalReconstructionPlan")
            .def_static("ConstructRecCoeff", &VariationalReconstructionPlan::ConstructRecCoeff,
                        py::arg("metrics"), py::arg("settings") = ReconstructionSettings{})
            .def("NRecDOF", &VariationalReconstructionPlan::NRecDOF)
            .def("GetInverseDiagnostics", &VariationalReconstructionPlan::GetInverseDiagnostics)
            .def("GetCellInverseDiagnostics", &VariationalReconstructionPlan::GetCellInverseDiagnostics)
            .def("Validate", &VariationalReconstructionPlan::Validate)
            .def("BuildWorkspace", &VariationalReconstructionPlan::BuildWorkspace,
                 py::arg("components"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("BuildBoundaryValues", &VariationalReconstructionPlan::BuildBoundaryValues)
            .def("NumBoundaryQuadrature", &VariationalReconstructionPlan::NumBoundaryQuadrature)
            .def("GetBoundaryQuadrature", &VariationalReconstructionPlan::GetBoundaryQuadrature)
            .def("BuildFaceTransform", &VariationalReconstructionPlan::BuildFaceTransform)
            .def("FinalizeFaceTransform", &VariationalReconstructionPlan::FinalizeFaceTransform,
                 "Validate and publish matrices after filling from cell means; any later write requires re-finalization.")
            .def("DoReconstructionIter", &VariationalReconstructionPlan::DoReconstructionIter,
                 "One synchronous CPU sweep. Release Python array views and refresh input halos first.")
            .def("DoReconstructionIterTo", &VariationalReconstructionPlan::DoReconstructionIterTo,
                 py::arg("mean"), py::arg("input"), py::arg("workspace"), py::arg("boundary"), py::arg("region") = ReconstructionRegion::All)
            .def("CommitReconstruction", &VariationalReconstructionPlan::CommitReconstruction)
            .def("DoLimiterWBAP_3Order", &VariationalReconstructionPlan::DoLimiterWBAP_3Order,
                 py::arg("workspace"), py::arg("order"), py::arg("transform") = nullptr,
                 "Call in descending order with a coefficient halo exchange after each order.")
            .def("DoLimiterWBAP_3OrderTo", &VariationalReconstructionPlan::DoLimiterWBAP_3OrderTo,
                 py::arg("workspace"), py::arg("order"), py::arg("transform") = nullptr,
                 py::arg("region") = ReconstructionRegion::All,
                 py::arg("indicator") = nullptr, py::arg("threshold") = 0)
            .def("Evaluate", &VariationalReconstructionPlan::Evaluate)
            .def("EvaluateDerivative", &VariationalReconstructionPlan::EvaluateDerivative)
            .def("GetCachedFaceBasis", &VariationalReconstructionPlan::GetCachedFaceBasis)
            .def("ApplyLinearOperator", &VariationalReconstructionPlan::ApplyLinearOperator,
                 py::arg("coefficients"), py::arg("output"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("BuildRightHandSide", &VariationalReconstructionPlan::BuildRightHandSide,
                 py::arg("mean"), py::arg("boundary"), py::arg("output"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("ComputeResidual", &VariationalReconstructionPlan::ComputeResidual,
                 py::arg("rhs"), py::arg("coefficients"), py::arg("output"), py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("ApplyLocalPreconditioner", &VariationalReconstructionPlan::ApplyLocalPreconditioner);
        py::class_<InterfaceSamplingPlan>(m, "InterfaceSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, Geom::InterfacePointMap>())
            .def("Size", &InterfaceSamplingPlan::Size)
            .def("Evaluate", &InterfaceSamplingPlan::Evaluate,
                 "Sample donor-frame values in map query order. The equation adapter transforms vector components.");
        py::enum_<GradientMethod>(m, "GradientMethod")
            .value("GreenGauss", GradientMethod::GreenGauss)
            .value("LeastSquares", GradientMethod::LeastSquares)
            .value("GGMP", GradientMethod::GGMP);
        py::class_<GradientPlan>(m, "GradientPlan")
            .def_static("Prepare", &GradientPlan::Prepare, py::arg("mesh"), py::arg("metrics"),
                        py::arg("method") = GradientMethod::LeastSquares, py::arg("relativeCutoff") = 1e-13)
            .def("NumBoundaryQuadrature", &GradientPlan::NumBoundaryQuadrature)
            .def("GetBoundaryQuadrature", &GradientPlan::GetBoundaryQuadrature)
            .def("BuildBoundaryValues", [](const GradientPlan &plan, index components)
                 { return Runtime::PythonArray::Share(plan.BuildBoundaryValues(components)); })
            .def("Evaluate", &GradientPlan::Evaluate, py::arg("mean"), py::arg("boundaryValues"), py::arg("gradient"),
                 py::arg("periodicVectorStarts") = std::vector<index>{})
            .def("ConvertToCoefficients", &GradientPlan::ConvertToCoefficients);
        py::class_<DistributedInterfaceSamplingPlan>(m, "DistributedInterfaceSamplingPlan")
            .def(py::init<VariationalReconstructionPlan, Geom::DistributedInterfacePointMap>())
            .def("Size", &DistributedInterfaceSamplingPlan::Size)
            .def("Validate", &DistributedInterfaceSamplingPlan::Validate)
            .def("BuildWorkspace", &DistributedInterfaceSamplingPlan::BuildWorkspace)
            .def("Evaluate", &DistributedInterfaceSamplingPlan::Evaluate,
                 "Collective owner evaluation and direct return transfer. Release views and finish all producers first.");
    }
}
