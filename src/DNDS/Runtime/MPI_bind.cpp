#include "Array_bind.hpp"
#include "DynamicRowTransfer.hpp"
#include "MPIProgress.hpp"
#include <pybind11/stl.h>

namespace DNDS::Runtime
{
    namespace py = pybind11;
    namespace
    {
        struct PythonDynamicTransfer
        {
            PythonArray prototype;
            std::unique_ptr<DynamicRowTransfer> operation;
            PythonDynamicTransfer(const PythonArray &source, RowTransferPlan route, ElementSchema schema,
                                  Device destination, TransferTransport transport, TransferOptions options,
                                  std::optional<Reducer> reducer, ExecutionContext context)
                : prototype(source.New(RowLayout{}, destination)),
                  operation(std::make_unique<DynamicRowTransfer>(std::move(route), source.SharedStorage(), std::move(schema),
                                                                 destination, transport, std::move(options), std::move(reducer), context)) {}
            PythonArray Commit() { return prototype.WithStorage(operation->Commit()); }
        };
    }
    void BindMPIOptions(py::module_ &m)
    {
        py::class_<CommunicationContext, std::shared_ptr<CommunicationContext>>(m, "CommunicationContext")
            .def(py::init([](const MPIInfo &mpi)
                          { return std::make_shared<CommunicationContext>(mpi.comm); }));
        py::enum_<TransferMode>(m, "TransferMode").value("Persistent", TransferMode::Persistent).value("Transient", TransferMode::Transient);
        py::class_<TransferOptions>(m, "TransferOptions")
            .def(py::init([](TransferMode mode, std::shared_ptr<CommunicationContext> context, int tag)
                          { return TransferOptions{mode, std::move(context), tag}; }),
                 py::arg("mode") = TransferMode::Persistent,
                 py::arg("context") = nullptr, py::arg("tag") = 0)
            .def_readwrite("mode", &TransferOptions::mode)
            .def_readwrite("context", &TransferOptions::context)
            .def_readwrite("tag", &TransferOptions::tag);
    }
    void BindMPIEagerPython(py::module_ &m)
    {
        py::class_<MPIProgress>(m, "MPIProgress")
            .def(py::init<>())
            .def("import_operation", [](MPIProgress &progress, TransferOperation &operation)
                 { return progress.Import(std::move(operation)); })
            .def("poll", &MPIProgress::Poll, py::call_guard<py::gil_scoped_release>())
            .def("drain", &MPIProgress::Drain, py::call_guard<py::gil_scoped_release>());
        py::class_<PythonDynamicTransfer>(m, "DynamicRowTransfer")
            .def(py::init<const PythonArray &, RowTransferPlan, ElementSchema, Device, TransferTransport, TransferOptions,
                          std::optional<Reducer>, ExecutionContext>(),
                 py::arg("source"), py::arg("route"), py::arg("schema"), py::arg("device") = Device::Host(),
                 py::arg("transport") = TransferTransport::Direct,
                 py::arg("options") = TransferOptions{TransferMode::Transient, {}, 0},
                 py::arg("reducer") = py::none(), py::arg("context") = ExecutionContext::Host())
            .def("test", [](PythonDynamicTransfer &self)
                 { return self.operation->Test(); }, py::call_guard<py::gil_scoped_release>())
            .def("wait", [](PythonDynamicTransfer &self)
                 { self.operation->Wait(); }, py::call_guard<py::gil_scoped_release>())
            .def("commit", &PythonDynamicTransfer::Commit);
    }
}
