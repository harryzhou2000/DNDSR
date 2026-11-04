#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
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
    py::class_<C::InterfaceDiagnostics>(m, "InterfaceDiagnostics", "Global coupling work counters; donor changes include nonmonotone stage revisits and rejected trials.")
        .def_readonly("query_points", &C::InterfaceDiagnostics::queryPoints)
        .def_readonly("changed_donors", &C::InterfaceDiagnostics::changedDonors)
        .def_readonly("map_builds", &C::InterfaceDiagnostics::mapBuilds)
        .def_readonly("maximum_gap", &C::InterfaceDiagnostics::maximumGap);
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
             { return domain.RefreshDerived(fields, onlyIfStale); }, py::arg("domain"), py::arg("only_if_stale") = false, py::call_guard<py::gil_scoped_release>(), "Synchronous local pointwise refresh of all enabled properties using the domain worker. No MPI; ghost state must already be ready. Release conflicting views first.");
    py::class_<C::PhysicalStepLimit>(m, "PhysicalStepLimit", "Explicit physical-step PP result; caller advances with dt. Counts/alpha precede growth capping and retries.")
        .def_readonly("dt", &C::PhysicalStepLimit::dt)
        .def_readonly("limited_cells", &C::PhysicalStepLimit::limitedCells)
        .def_readonly("minimum_alpha", &C::PhysicalStepLimit::minimumAlpha)
        .def_readonly("growth_limited", &C::PhysicalStepLimit::growthLimited);
    py::class_<C::EquationNonlinearIterationReport>(m, "NonlinearIterationReport", "Read-only snapshot of one implicit stage; norms include every logical equation component.")
        .def_readonly("converged", &C::EquationNonlinearIterationReport::converged)
        .def_readonly("iterations", &C::EquationNonlinearIterationReport::iterations)
        .def_readonly("linear_iterations", &C::EquationNonlinearIterationReport::linearIterations)
        .def_readonly("reconstruction_iterations", &C::EquationNonlinearIterationReport::reconstructionIterations)
        .def_readonly("reconstruction_relative_update", &C::EquationNonlinearIterationReport::reconstructionRelativeUpdate)
        .def_readonly("multigrid_cycles", &C::EquationNonlinearIterationReport::multigridCycles)
        .def_readonly("level_corrections", &C::EquationNonlinearIterationReport::levelCorrections)
        .def_readonly("initial_residual", &C::EquationNonlinearIterationReport::initialResidual)
        .def_readonly("reference_residual", &C::EquationNonlinearIterationReport::referenceResidual)
        .def_readonly("residual_target_ratio", &C::EquationNonlinearIterationReport::residualTargetRatio)
        .def_readonly("residual", &C::EquationNonlinearIterationReport::residual)
        .def_readonly("rhs_limited", &C::EquationNonlinearIterationReport::rhsLimited)
        .def_readonly("increment_limited", &C::EquationNonlinearIterationReport::incrementLimited);
    py::class_<C::Domain>(m, "Domain", "CPU compressible domain with perfect gas or the explicit ideal-mixture profile. Collective methods use the mesh communicator. Release all debug views before stepping; no CUDA fallback.")
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
        .def_property_readonly("fields", [](C::Domain &domain)
                               {
            std::map<std::string, R::PythonArray> result;
            const auto &state = domain.Dofs();
            for (DNDS::index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                result.emplace(state.Fields().Name(iField), R::PythonArray::Share(state.PhysicalField(iField).Owned()));
            return result; }, "Shared physical owned fields, including species. Retained views follow the same lease/publication rules as flow.")
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
        .def_property_readonly("coupled_target_ratio", &C::Domain::LastCoupledTargetRatio, "Selected-component ratio of the DITR I+H stopping residual after both coupled-stage updates; the same value appears in its single nonlinear report. Zero for other methods.")
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
        .def("residual_fields", [](C::Domain &domain)
             {
                 const C::EquationDofs *value;
                 { py::gil_scoped_release release; value = &domain.ResidualState(); }
                 std::map<std::string, R::PythonArray> result;
                 for (DNDS::index iField = 0; iField < value->PhysicalFieldNumber(); ++iField)
                     result.emplace(value->Fields().Name(iField), R::PythonArray::Share(value->PhysicalField(iField).Owned()));
                 return result; }, "Collective full-equation integrated residual fields; shared overwriteable scratch, with the same candidate-invalidating and view-release rules as residual.")
        .def("advance", &C::Domain::Advance, py::arg("dt"), py::call_guard<py::gil_scoped_release>(), "Collective configured time method; invalid candidate leaves accepted state/time/history unchanged.")
        .def("prepare_step", &C::Domain::PrepareStep, py::arg("dt"), py::call_guard<py::gil_scoped_release>())
        .def("commit_step", &C::Domain::CommitStep, py::call_guard<py::gil_scoped_release>())
        .def("discard_step", &C::Domain::DiscardStep)
        .def("stable_step", &C::Domain::StableStep, py::arg("cfl"), py::call_guard<py::gil_scoped_release>())
        .def("limit_physical_step", &C::Domain::LimitPhysicalStep, py::arg("dt"), py::call_guard<py::gil_scoped_release>(), "Collective configured Euler PP/growth bound, without advancing state or history. All ranks pass the same positive dt and release conflicting views; advance with result.dt.")
        .def("write_visualization", &C::Domain::WriteVisualization, py::arg("prefix"), py::arg("hdf_deflate_level") = 0, py::call_guard<py::gil_scoped_release>(), "Collective accepted-state VTKHDF and boundary CSV output in code units, including all species fields. Uses separate observation reconstruction. Release debug views before calling; pass a common path/compression setting.")
        .def("write_checkpoint", [](const C::Domain &domain, const std::string &path, const std::string &resolved, bool reconstruction)
             {
                 C::Json input;
                 DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&] { input = C::Json::parse(resolved); });
                 domain.WriteCheckpoint(path, input, reconstruction); }, py::arg("path"), py::arg("resolved_json"), py::arg("include_reconstruction") = false, py::call_guard<py::gil_scoped_release>(), "Collective native checkpoint of all equation fields and BDF history; physical layouts are preserved. Optional reconstruction supports compatible warm continuation. Pass common metadata/path on the domain communicator.")
        .def("read_checkpoint", [](C::Domain &domain, const std::string &path, const std::string &resolved)
             {
                 C::Json input;
                 DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&] { input = C::Json::parse(resolved); });
                 domain.ReadCheckpoint(path, input); }, py::arg("path"), py::arg("resolved_json"), py::call_guard<py::gil_scoped_release>(), "Collective persistent-cell-ID restart, permitting redistribution and layout changes. Release all retained views, including species. Accepted means/history publish together; missing or incompatible reconstruction cold-starts.");
    py::class_<C::Simulation>(m, "Simulation", "Coordinated CPU domains. Every control rank calls advance; local meshes follow configured domain-rank order. Release debug views before publication. No process-failure recovery.")
        .def(py::init([](const std::vector<std::shared_ptr<DNDS::Geom::UnstructuredMesh>> &meshes, const DNDS::MPIInfo &mpi, const std::string &json)
                      {
            C::Json input;
            DNDS::MPI::LocalStage(mpi,[&]{input=C::Json::parse(json);});
            auto config=C::SimulationConfig::ResolveCollective(input,mpi.comm);
            DNDS::MPI::LocalStage(mpi,[&]
            {
                DNDS_check_throw_info(meshes.size()==config.Domains().size(),"one Python mesh slot is required for each configured domain");
                for(DNDS::index iDomain=0;iDomain<DNDS::index(meshes.size());++iDomain)
                {
                    const auto &ranks=config.Domains()[iDomain].placement.ranks;
                    const bool local=std::find(ranks.begin(),ranks.end(),mpi.rank)!=ranks.end();
                    DNDS_check_throw_info(bool(meshes[iDomain])==local,"Python Simulation mesh must be None exactly on domain nonmembers");
                }
            });
            return std::make_unique<C::Simulation>(std::move(config),mpi.comm,[&](DNDS::index iDomain,const auto &,const auto &domainMPI)
            {
                DNDS::MPI::LocalStage(domainMPI,[&]
                {
                    int comparison=MPI_UNEQUAL;
                    MPI_Comm_compare(meshes[iDomain]->getMPI().comm,domainMPI.comm,&comparison);
                    DNDS_check_throw_info(comparison==MPI_IDENT||comparison==MPI_CONGRUENT,"Python mesh communicator differs from configured domain membership/order");
                });
                return meshes[iDomain];
            }); }),
             py::arg("meshes"), py::arg("mpi"), py::arg("config_json") = "{}", py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("time", &C::Simulation::Time)
        .def_property_readonly("step", &C::Simulation::Step)
        .def("local", &C::Simulation::Local, py::arg("domain"))
        .def("domain", static_cast<C::Domain &(C::Simulation::*)(DNDS::index)>(&C::Simulation::GetDomain), py::arg("domain"), py::return_value_policy::reference_internal,
             "Borrow a local domain, retaining Simulation. Advance through Simulation to keep clocks and interface snapshots coordinated.")
        .def("advance", &C::Simulation::Advance, py::arg("dt"), py::call_guard<py::gil_scoped_release>())
        .def("interface_diagnostics", &C::Simulation::GetInterfaceDiagnostics, py::call_guard<py::gil_scoped_release>(), "Collective query/rematch/map counts and maximum physical projection gap. Does not change the solution or traces.")
        .def_property_readonly("resolved_json", [](const C::Simulation &simulation)
                               { return simulation.Config().Root().dump(); });
}
