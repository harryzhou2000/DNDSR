#include "Array_bind.hpp"
#include "DynamicRowTransfer.hpp"
#include "MPIProgress.hpp"
#include <pybind11/stl.h>

namespace DNDS::Runtime
{
    namespace py = pybind11;
    std::string DtypeSchema(const py::dtype &dtype);
    namespace
    {
        struct PythonDynamicTransfer
        {
            PythonArray prototype;
            std::unique_ptr<DynamicRowTransfer> operation;
            PythonDynamicTransfer(const PythonArray &source, RowTransferPlan route, ElementSchema schema,
                                  Device destination, TransferTransport transport, TransferOptions options,
                                  std::optional<Reducer> reducer, ExecutionContext context)
                : prototype(source.New(RowLayout{}, destination))
            {
                int compatible = schema.Compatible(DtypeSchema(source.Dtype()));
                MPI_Allreduce(MPI_IN_PLACE, &compatible, 1, MPI_INT, MPI_MIN, route.Communicator());
                DNDS_check_throw_info(compatible, "dynamic transfer source dtype differs from schema on a rank");
                operation = std::make_unique<DynamicRowTransfer>(std::move(route), source.SharedStorage(), std::move(schema),
                                                                 destination, transport, std::move(options), std::move(reducer), context);
            }
            PythonArray Commit() { return prototype.WithStorage(operation->Commit()); }
        };
    }
    void BindMPIOptions(py::module_ &m)
    {
        py::class_<CommunicationContext, std::shared_ptr<CommunicationContext>>(m, "CommunicationContext", "Collective shared communicator. Match setup order on all ranks and use distinct matched tags for concurrent bindings.")
            .def(py::init([](const MPIInfo &mpi)
                          { return std::make_shared<CommunicationContext>(mpi.comm); }));
        py::enum_<TransferMode>(m, "TransferMode").value("Persistent", TransferMode::Persistent).value("Transient", TransferMode::Transient);
        py::class_<TransferOptions>(m, "TransferOptions", "Persistent/transient request policy, independent of transport. Caller matches context/tag across ranks; transient mode remains collective.")
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
        py::class_<MPIProgress>(m, "MPIProgress", "Explicit MPI progress on its creating thread, permitted by the MPI thread level. Drain before dependent CUDA work and MPI_Finalize; do not withhold progress while waiting for dependent CPU workers.")
            .def(py::init<>())
            .def("import_operation", [](MPIProgress &progress, TransferOperation &operation)
                 { return progress.Import(std::move(operation)); }, "Consume the operation and return a portable event. Caller must poll/drain on the creating thread; no background progress is promised.")
            .def("poll", &MPIProgress::Poll, py::call_guard<py::gil_scoped_release>())
            .def("drain", &MPIProgress::Drain, py::call_guard<py::gil_scoped_release>());
        py::class_<PythonDynamicTransfer>(m, "DynamicRowTransfer", "Collectively pin source and build private CSR destination. Caller completes producers, matches construction/commit order including empty ranks, drains before mesh mutation, and rebuilds layout-dependent bindings. Field bundles are not atomic transactions.")
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
            .def("commit", &PythonDynamicTransfer::Commit, "Collectively wait/check and publish this field only. Validate all migrated fields before publishing a new mesh epoch.");
    }
}
