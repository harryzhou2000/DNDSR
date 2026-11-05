#include "DNDS/Runtime/Interop/Array_bind.hpp"
#include "DNDS/Runtime/Communication/DynamicRowTransfer.hpp"
#include "DNDS/Runtime/Communication/MPIProgress.hpp"
#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include <pybind11/stl.h>

namespace DNDS::Runtime
{
    namespace py = pybind11;
    std::string ArrayTypeIdentity(const PythonArray &array);
    namespace
    {
        struct PythonDynamicTransfer
        {
            PythonArray prototype;
            std::unique_ptr<DynamicRowTransfer> operation;
            PythonDynamicTransfer(const PythonArray &source, RowTransferPlan route, ElementSchema schema,
                                  Device destination, TransferTransport transport, TransferOptions options,
                                  std::optional<Reducer> reducer, ExecutionContext context, TransferLayout layout)
                : prototype(source.New(RowLayout{}, destination))
            {
                int compatible = schema.Compatible(ArrayTypeIdentity(source));
                MPI_Allreduce(MPI_IN_PLACE, &compatible, 1, MPI_INT, MPI_MIN, route.Communicator());
                DNDS_check_throw_info(compatible, "dynamic transfer source dtype differs from schema on a rank");
                operation = std::make_unique<DynamicRowTransfer>(std::move(route), source.SharedStorage(), std::move(schema),
                                                                 destination, transport, std::move(options), std::move(reducer), context, layout);
            }
            PythonArray Commit() { return prototype.WithStorage(operation->Commit()); }
        };
    }
    void BindMPIOptions(py::module_ &m)
    {
        py::class_<TransferProvider, std::shared_ptr<TransferProvider>>(m, "TransferProvider");
        py::class_<NCCLTransferProvider, TransferProvider, std::shared_ptr<NCCLTransferProvider>>(m, "NCCLTransferProvider",
                                                                                                  "Explicit stream-ordered CUDA payload provider. One rank per GPU; MPI only bootstraps setup. Match per-peer order and retain streams through completion.")
            .def(py::init([](const MPIInfo &mpi, Device device)
                          { return NCCLTransferProvider::Create(mpi.comm, device); }),
                 py::arg("mpi"), py::arg("device"))
            .def_static("available", &NCCLTransferProvider::Available)
            .def("check_async_error", &NCCLTransferProvider::CheckAsyncError)
            .def("abort", &NCCLTransferProvider::Abort);
        py::class_<CommunicationContext, std::shared_ptr<CommunicationContext>>(m, "CommunicationContext",
                                                                                "Collective shared communicator with immutable collective_preflight (default true). "
                                                                                "Match setup order and selected policy on all ranks; use distinct matched tags for concurrent bindings. "
                                                                                "With safety agreement disabled, rejected local MPI launches abort the communicator. Local access/readiness checks remain.")
            .def(py::init([](const MPIInfo &mpi, bool collectivePreflight)
                          { return std::make_shared<CommunicationContext>(mpi.comm, collectivePreflight); }),
                 py::arg("mpi"), py::arg("collective_preflight") = true)
            .def_property_readonly("collective_preflight", &CommunicationContext::ChecksCollectivePreflight);
        py::enum_<TransferMode>(m, "TransferMode").value("Persistent", TransferMode::Persistent).value("Transient", TransferMode::Transient);
        py::class_<TransferOptions>(m, "TransferOptions", "Persistent/transient request policy, independent of transport. Caller matches context/tag across ranks; transient mode remains collective.")
            .def(py::init([](TransferMode mode, std::shared_ptr<CommunicationContext> context, int tag, std::shared_ptr<TransferProvider> provider)
                          { return TransferOptions{mode, std::move(context), tag, std::move(provider)}; }),
                 py::arg("mode") = TransferMode::Persistent,
                 py::arg("context") = nullptr, py::arg("tag") = 0, py::arg("provider") = nullptr)
            .def_readwrite("mode", &TransferOptions::mode)
            .def_readwrite("context", &TransferOptions::context)
            .def_readwrite("tag", &TransferOptions::tag)
            .def_readwrite("provider", &TransferOptions::provider)
            .def_readwrite("allow_stream_ordered_device_memory", &TransferOptions::allowStreamOrderedDeviceMemory);
    }
    void BindMPIEagerPython(py::module_ &m)
    {
        py::class_<MPIProgress>(m, "MPIProgress", "Explicit MPI progress on its creating thread, permitted by the MPI thread level. Drain before dependent CUDA work and MPI_Finalize; do not withhold progress while waiting for dependent CPU workers.")
            .def(py::init<>())
            .def("import_operation", [](MPIProgress &progress, TransferOperation &operation)
                 { return progress.Import(std::move(operation)); }, "Consume the operation and return a portable event. Caller must poll/drain on the creating thread; no background progress is promised.")
            .def("poll", &MPIProgress::Poll, py::call_guard<py::gil_scoped_release>())
            .def("drain", &MPIProgress::Drain, py::call_guard<py::gil_scoped_release>());
        py::class_<PythonDynamicTransfer>(m, "DynamicRowTransfer", "Collectively pin source and build a private destination preserving storage by default. Caller declares the width/capacity even on empty ranks, completes producers, matches construction/commit order, drains before mesh mutation, and rebuilds layout-dependent bindings. Field bundles are not atomic transactions.")
            .def(py::init<const PythonArray &, RowTransferPlan, ElementSchema, Device, TransferTransport, TransferOptions,
                          std::optional<Reducer>, ExecutionContext, TransferLayout>(),
                 py::arg("source"), py::arg("route"), py::arg("schema"), py::arg("device") = Device::Host(),
                 py::arg("transport") = TransferTransport::Direct,
                 py::arg("options") = TransferOptions{TransferMode::Transient, {}, 0},
                 py::arg("reducer") = py::none(), py::arg("context") = ExecutionContext::Host(), py::arg("layout") = TransferLayout{})
            .def("test", [](PythonDynamicTransfer &self)
                 { return self.operation->Test(); }, py::call_guard<py::gil_scoped_release>())
            .def("wait", [](PythonDynamicTransfer &self)
                 { self.operation->Wait(); }, py::call_guard<py::gil_scoped_release>())
            .def("commit", &PythonDynamicTransfer::Commit, "Collectively wait/check and publish this field only. Validate all migrated fields before publishing a new mesh epoch.");
    }
}
