#include "Array_bind.hpp"
#include "EagerCopy.hpp"
#include "CUDAExecution.hpp"
#include <pybind11/stl.h>

namespace DNDS::Runtime
{
    namespace py = pybind11;
    namespace
    {
        // Keep Python dtype/codec objects on the Python side. Queued ownership
        // contains only native storage, so worker retirement needs no GIL.
        struct StorageOwner
        {
            ArrayStorage::Shared storage;
            bool CanReleaseOrdered(ExecutionContext context) const { return !storage || storage->CanReleaseOrdered(context); }
            void ReleaseOn(ExecutionContext context) { ArrayStorage::ReleaseHandle(storage, context); }
        };

        PythonArray Allocate(EagerEngine &engine, const ExecutionStream &stream, RowLayout layout, py::object dtype)
        {
            const auto context = stream.Context();
            DNDS_check_throw_info(context.device == Device::Host() || !layout.IsVariable(),
                                  "new variable GPU metadata needs setup; use allocate_like with a prepared Array");
            PythonArray prototype(RowLayout{}, std::move(dtype));
            auto values = engine.AllocateBuffer(stream, CheckedSize::Multiply(layout.Elements(), size_t(prototype.Dtype().itemsize())));
            try
            {
                auto result = prototype.WithStorage(ArrayStorage::FromBuffer(std::move(layout), values, prototype.Dtype().itemsize()));
                values.ReleaseOn(context);
                return result;
            }
            catch (...)
            {
                values.ReleaseOn(context);
                throw;
            }
        }

        PythonArray AllocateLike(EagerEngine &engine, const ExecutionStream &stream, const PythonArray &prototype)
        {
            auto storage = prototype.SharedStorage();
            const auto context = stream.Context();
            DNDS_check_throw_info(storage->Values().GetDevice() == context.device, "async prototype placement mismatch");
            auto values = engine.AllocateBuffer(stream, storage->Values().Bytes());
            try
            {
                auto result = prototype.WithStorage(storage->WithValues(values, prototype.Dtype().itemsize()));
                values.ReleaseOn(context);
                return result;
            }
            catch (...)
            {
                values.ReleaseOn(context);
                throw;
            }
        }

        void ReleaseArray(EagerEngine &engine, PythonArray &array, const ExecutionStream &stream)
        {
            auto empty = array.New(RowLayout{}, array.Storage().GetDevice());
            StorageOwner owner{array.SharedStorage()};
            array = std::move(empty);
            try
            {
                engine.ReleaseOn(std::move(owner), stream);
            }
            catch (...)
            {
                array = array.WithStorage(std::move(owner.storage));
                throw;
            }
        }
    }

    void BindEagerPython(py::module_ &m)
    {
        py::class_<StreamEvent>(m, "StreamEvent")
            .def("ready", &StreamEvent::Ready, py::call_guard<py::gil_scoped_release>())
            .def("wait", &StreamEvent::WaitHost, py::call_guard<py::gil_scoped_release>());
        py::class_<ExecutionStream>(m, "ExecutionStream")
            .def("record", &ExecutionStream::Record, py::call_guard<py::gil_scoped_release>())
            .def("wait_for", &ExecutionStream::WaitFor, py::call_guard<py::gil_scoped_release>())
            .def_property_readonly("context", &ExecutionStream::Context);
        py::class_<EagerEngine>(m, "EagerEngine")
            .def(py::init<size_t>(), py::arg("host_threads") = 0)
            .def("cpu_stream", &EagerEngine::CreateCPUStream)
            .def("cuda_stream", [](EagerEngine &engine, int ordinal)
                 { return engine.CreateStream(std::make_shared<CUDALauncher>(ordinal)); }, py::arg("ordinal") = 0)
            .def("synchronize", &EagerEngine::Synchronize, py::call_guard<py::gil_scoped_release>())
            .def("allocate", &Allocate, py::arg("stream"), py::arg("layout"), py::arg("dtype") = py::dtype::of<double>())
            .def("allocate_like", &AllocateLike, py::arg("stream"), py::arg("prototype"))
            .def("copy", [](EagerEngine &engine, const ExecutionStream &stream, const PythonArray &source, PythonArray &destination)
                 {
                     DNDS_check_throw_info(source.SameElement(destination) && source.Layout() == destination.Layout(),
                                           "eager Array copy requires identical dtype and physical layout");
                     auto target = destination.Storage();
                     CopyAsync(engine, stream, source.Storage(), target); }, py::arg("stream"), py::arg("source"), py::arg("destination"))
            .def("release_on", &ReleaseArray, py::arg("array"), py::arg("stream"), "Consume this facade's storage and replace it with an empty Array; existing aliases/views remain owners.");
    }
}
