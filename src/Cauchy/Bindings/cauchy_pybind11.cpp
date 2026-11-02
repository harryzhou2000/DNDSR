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
            return std::make_unique<C::Domain>(name,std::move(mesh),settings); }),
             py::arg("mesh"), py::arg("config_json") = "{}", py::arg("name") = "fluid", py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("flow", [](C::Domain &domain)
                               { return R::PythonArray::Share(domain.Flow().Owned()); }, "Shared owned flow Array; follows normal retained-view/access contracts.")
        .def_property_readonly("time", &C::Domain::Time)
        .def_property_readonly("step", [](const C::Domain &domain)
                               { return domain.Epoch().acceptedStep; })
        .def_property_readonly("face_fallbacks", &C::Domain::LastFaceFallbacks)
        .def_property_readonly("positivity", &C::Domain::LastPositivity, "Last accepted trial: reconstruction cell-stage activations, minimum beta, stage activations, minimum alpha, componentwise characteristic fallbacks.")
        .def("residual", [](C::Domain &domain)
             {
                 const DNDS::CFV::FVField *value;
                 {
                     py::gil_scoped_release release;
                     value = &domain.Residual();
                 }
                 return R::PythonArray::Share(value->Owned()); }, "Collective integrated residual; returned Array aliases overwriteable workspace. Invalidates any prepared candidate.")
        .def("advance", &C::Domain::Advance, py::arg("dt"), py::call_guard<py::gil_scoped_release>(), "Collective SSPRK3; invalid candidate leaves accepted state/time unchanged.")
        .def("prepare_step", &C::Domain::PrepareStep, py::arg("dt"), py::call_guard<py::gil_scoped_release>())
        .def("commit_step", &C::Domain::CommitStep, py::call_guard<py::gil_scoped_release>())
        .def("discard_step", &C::Domain::DiscardStep)
        .def("stable_step", &C::Domain::StableStep, py::arg("cfl"), py::call_guard<py::gil_scoped_release>())
        .def("write_checkpoint", [](const C::Domain &domain, const std::string &path, const std::string &resolved)
             {
                 C::Json input;
                 DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&] { input = C::Json::parse(resolved); });
                 domain.WriteCheckpoint(path, input); }, py::arg("path"), py::arg("resolved_json"), py::call_guard<py::gil_scoped_release>())
        .def("read_checkpoint", [](C::Domain &domain, const std::string &path, const std::string &resolved)
             {
                 C::Json input;
                 DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&] { input = C::Json::parse(resolved); });
                 domain.ReadCheckpoint(path, input); }, py::arg("path"), py::arg("resolved_json"), py::call_guard<py::gil_scoped_release>());
}
