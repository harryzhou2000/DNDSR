#include "Cauchy/Core/Domain.hpp"
#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <pybind11/stl.h>

PYBIND11_MODULE(cauchy_pybind11, m)
{
    namespace py = pybind11;
    namespace C = DNDS::Cauchy;
    namespace R = DNDS::Runtime;
    py::module_::import("DNDSR.CFV");
    py::enum_<C::DerivedQuantity>(m, "DerivedQuantity")
        .value("Temperature", C::DerivedQuantity::Temperature)
        .value("Pressure", C::DerivedQuantity::Pressure)
        .value("SoundSpeed", C::DerivedQuantity::SoundSpeed)
        .value("MolecularViscosity", C::DerivedQuantity::MolecularViscosity)
        .value("ThermalConductivity", C::DerivedQuantity::ThermalConductivity)
        .value("HeatCapacityP", C::DerivedQuantity::HeatCapacityP)
        .value("HeatCapacityV", C::DerivedQuantity::HeatCapacityV)
        .value("GasConstant", C::DerivedQuantity::GasConstant)
        .value("DiffusionBound", C::DerivedQuantity::DiffusionBound);
    py::class_<C::DerivedFields>(m, "DerivedFields", "Optional point properties; no equation components or automatic exchange. Shared arrays retain storage. Raw debug access does not establish freshness.")
        .def_property_readonly("values", [](const C::DerivedFields &fields)
                               { return R::PythonArray::Share(fields.Values()); }, "Shared property table, with columns in quantities order. External writes invalidate publication.")
        .def_property_readonly("quantities", [](const C::DerivedFields &fields)
                               {
            std::vector<C::DerivedQuantity> quantities;
            for(int q=0;q<C::DerivedQuantityNumber;++q)
                if(fields.Request().Has(C::DerivedQuantity(q))) quantities.push_back(C::DerivedQuantity(q));
            return quantities; })
        .def_property_readonly("ghost_part", &C::DerivedFields::GhostPart)
        .def("current", [](const C::DerivedFields &fields, const C::Domain &domain)
             { return fields.Current(domain.Dofs(), domain.Equation()); }, "Provenance snapshot; does not reserve access or refresh.")
        .def("refresh", [](C::DerivedFields &fields, const C::Domain &domain, bool onlyIfStale)
             { return C::RefreshDerivedFields(domain.Dofs(), domain.Equation(), fields, onlyIfStale); }, py::arg("domain"), py::arg("only_if_stale") = false, py::call_guard<py::gil_scoped_release>(), "Synchronous local pointwise refresh of all enabled properties. No MPI; ghost state must already be ready. Release conflicting views first.");
    py::class_<C::PhysicalStepLimit>(m, "PhysicalStepLimit", "Explicit physical-step PP result; caller advances with dt. Counts/alpha precede growth capping and retries.")
        .def_readonly("dt", &C::PhysicalStepLimit::dt)
        .def_readonly("limited_cells", &C::PhysicalStepLimit::limitedCells)
        .def_readonly("minimum_alpha", &C::PhysicalStepLimit::minimumAlpha)
        .def_readonly("growth_limited", &C::PhysicalStepLimit::growthLimited);
    py::class_<C::NonlinearIterationReport>(m, "NonlinearIterationReport", "Read-only snapshot of one implicit stage's convergence and correction counts.")
        .def_readonly("converged", &C::NonlinearIterationReport::converged)
        .def_readonly("iterations", &C::NonlinearIterationReport::iterations)
        .def_readonly("linear_iterations", &C::NonlinearIterationReport::linearIterations)
        .def_readonly("reconstruction_iterations", &C::NonlinearIterationReport::reconstructionIterations)
        .def_readonly("reconstruction_relative_update", &C::NonlinearIterationReport::reconstructionRelativeUpdate)
        .def_readonly("multigrid_cycles", &C::NonlinearIterationReport::multigridCycles)
        .def_readonly("level_corrections", &C::NonlinearIterationReport::levelCorrections)
        .def_readonly("initial_residual", &C::NonlinearIterationReport::initialResidual)
        .def_readonly("reference_residual", &C::NonlinearIterationReport::referenceResidual)
        .def_readonly("residual_target_ratio", &C::NonlinearIterationReport::residualTargetRatio)
        .def_readonly("residual", &C::NonlinearIterationReport::residual)
        .def_readonly("rhs_limited", &C::NonlinearIterationReport::rhsLimited)
        .def_readonly("increment_limited", &C::NonlinearIterationReport::incrementLimited);
    py::class_<C::Domain>(m, "Domain", "CPU compressible flow5 domain. Collective methods use the mesh communicator. Release all debug views before stepping; no CUDA fallback.")
        .def(py::init([](std::shared_ptr<DNDS::Geom::UnstructuredMesh> mesh, const std::string &json, const std::string &name)
                      {
            const auto mpi=mesh->getMPI();
            C::Json input;
            DNDS::MPI::LocalStage(mpi,[&]{input=C::Json::parse(json);});
            auto config=C::SimulationConfig::ResolveCollective(input,mpi.comm);
            DNDS_check_throw_info(config.Domains().size()==1,"Python Domain constructor requires one resolved domain");
            DNDS_check_throw_info(config.Domains()[0].placement.ranks.size()==size_t(mpi.size),"Python Domain requires every mesh-communicator rank");
            DNDS::MPI::LocalStage(mpi,[&]{DNDS_check_throw_info(!name.empty(),"Python Domain name is empty");});
            auto settings=config.Domains()[0].settings;
            return std::make_unique<C::Domain>(name,std::move(mesh),settings,config.Time()); }),
             py::arg("mesh"), py::arg("config_json") = "{}", py::arg("name") = "fluid", py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("flow", [](C::Domain &domain)
                               { return R::PythonArray::Share(domain.Flow().Owned()); }, "Shared owned flow Array; follows normal retained-view/access contracts.")
        .def("make_derived_fields", [](const C::Domain &domain, const std::vector<C::DerivedQuantity> &quantities, bool ghost)
             {
            C::DerivedRequest request;
            for(auto quantity:quantities)
            {
                const int q=int(quantity);
                DNDS_check_throw_info(q>=0&&q<C::DerivedQuantityNumber,"unknown derived quantity");
                request.enabled[q]=true;
            }
            return C::DerivedFields(domain.Dofs().Space(),request,ghost); }, py::arg("quantities"), py::arg("ghost") = false, "Allocate independent property storage for this domain's rows; call refresh explicitly. This does not alter the domain's solved state or workspace selection.")
        .def_property_readonly("time", &C::Domain::Time)
        .def_property_readonly("previous_dt", &C::Domain::PreviousTimeStep, "Last actual committed physical dt; zero for steady/fresh or unknown old restart history.")
        .def_property_readonly("steady", &C::Domain::Steady)
        .def_property_readonly("coupled_iterations", &C::Domain::LastCoupledIterations)
        .def_property_readonly("coupled_target_ratio", &C::Domain::LastCoupledTargetRatio, "Latest original coupled H/I residual ratio; zero for uncoupled time methods.")
        .def_property_readonly("step", [](const C::Domain &domain)
                               { return domain.Epoch().acceptedStep; })
        .def_property_readonly("face_fallbacks", &C::Domain::LastFaceFallbacks)
        .def_property_readonly("implicit_stages", [](const C::Domain &domain)
                               { return domain.LastImplicitStages(); }, "Copied diagnostics from the latest implicit attempt; empty for SSPRK3. Inspect converged before interpreting a trial.")
        .def_property_readonly("positivity", &C::Domain::LastPositivity, "Last accepted trial: reconstruction cell-stage activations, minimum beta, stage activations, minimum alpha, componentwise characteristic fallbacks.")
        .def("residual", [](C::Domain &domain)
             {
                 const DNDS::CFV::FVField *value;
                 {
                     py::gil_scoped_release release;
                     value = &domain.Residual();
                 }
                 return R::PythonArray::Share(value->Owned()); }, "Collective integrated residual; returned Array aliases overwriteable workspace. Invalidates any prepared candidate.")
        .def("advance", &C::Domain::Advance, py::arg("dt"), py::call_guard<py::gil_scoped_release>(), "Collective configured time method; invalid candidate leaves accepted state/time/history unchanged.")
        .def("prepare_step", &C::Domain::PrepareStep, py::arg("dt"), py::call_guard<py::gil_scoped_release>())
        .def("commit_step", &C::Domain::CommitStep, py::call_guard<py::gil_scoped_release>())
        .def("discard_step", &C::Domain::DiscardStep)
        .def("stable_step", &C::Domain::StableStep, py::arg("cfl"), py::call_guard<py::gil_scoped_release>())
        .def("limit_physical_step", &C::Domain::LimitPhysicalStep, py::arg("dt"), py::call_guard<py::gil_scoped_release>(), "Collective configured Euler PP/growth bound, without advancing state or history. All ranks pass the same positive dt and release conflicting views; advance with result.dt.")
        .def("write_checkpoint", [](const C::Domain &domain, const std::string &path, const std::string &resolved, bool reconstruction)
             {
                 C::Json input;
                 DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&] { input = C::Json::parse(resolved); });
                 domain.WriteCheckpoint(path, input, reconstruction); }, py::arg("path"), py::arg("resolved_json"), py::arg("include_reconstruction") = false, py::call_guard<py::gil_scoped_release>())
        .def("read_checkpoint", [](C::Domain &domain, const std::string &path, const std::string &resolved)
             {
                 C::Json input;
                 DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&] { input = C::Json::parse(resolved); });
                 domain.ReadCheckpoint(path, input); }, py::arg("path"), py::arg("resolved_json"), py::call_guard<py::gil_scoped_release>());
}
