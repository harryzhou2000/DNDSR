#include "ArrayIO.hpp"
#include "Array_bind.hpp"
#include "RowTransfer.hpp"
#include "DNDS/Defines_bind.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <optional>

namespace DNDS::Runtime
{
    namespace
    {
        std::string DtypeSchema(const py::dtype &dtype)
        {
            const auto kind = py::str(dtype.attr("kind")).cast<std::string>();
            if (kind == "i" || kind == "u" || kind == "f")
                return py::str(dtype.attr("name")).cast<std::string>();
            return "numpy.pod.v1:" + py::str(dtype.attr("descr")).cast<std::string>() + ":" + std::to_string(dtype.itemsize());
        }
        std::string ArraySchema(const PythonArray &array)
        {
            return array.Element() ? array.Element()->schema : DtypeSchema(array.Dtype());
        }
        struct PythonAccess
        {
            ArrayStorage::Shared storage;
            const RowLayout &layout;
            py::dtype dtype;
            PythonElementShared element;
            py::object elementType;
            std::optional<ReadAccess> read;
            std::optional<WriteAccess> write;
            ExecutionContext context;
            PythonAccess(PythonArray &a, bool writable, uintptr_t stream) : storage(a.SharedStorage()), layout(storage->Layout()), dtype(a.Dtype()),
                                                                            element(a.Element()), elementType(a.ElementType()), context(a.Context(stream))
            {
                if (writable)
                    write.emplace(a.Write(a.Context(stream)));
                else
                    read.emplace(a.Read(a.Context(stream)));
            }
            bool Active() const { return write ? write->Active() : read->Active(); }
            Device Placement() const { return write ? write->GetDevice() : read->GetDevice(); }
            const uint8_t *Pointer() const { return write ? write->View<uint8_t>().data : read->View<uint8_t>().data; }
            void LiveCheck() const { DNDS_check_throw_info(Active(), "element access requires a live lease"); }
            py::array Row(size_t row) const
            {
                LiveCheck();
                const auto width = layout.RowSize(row);
                py::array result(dtype, std::vector<py::ssize_t>{py::ssize_t(width)});
                if (width)
                    Detail::CopyBytes(Pointer() + layout.Offset(row) * dtype.itemsize(), Placement(), result.mutable_data(),
                                      Device::Host(), width * dtype.itemsize(), context)
                        .Wait();
                return result;
            }
            void AssignRow(size_t row, py::array value)
            {
                LiveCheck();
                DNDS_check_throw_info(write, "read-only Array lease");
                auto input = py::array::ensure(value, py::array::c_style);
                DNDS_check_throw_info(input && input.dtype().equal(dtype) && input.ndim() == 1 && size_t(input.size()) == layout.RowSize(row),
                                      "row assignment dtype/shape mismatch");
                if (input.size())
                    Detail::CopyBytes(input.data(), Device::Host(), write->View<uint8_t>().data + layout.Offset(row) * dtype.itemsize(),
                                      Placement(), input.nbytes(), context)
                        .Wait();
            }
            py::object Get(size_t row, size_t column) const
            {
                LiveCheck();
                DNDS_check_throw_info(column < layout.RowSize(row), "column outside row");
                if (!element)
                    return Row(row).attr("__getitem__")(column);
                std::vector<uint8_t> value(dtype.itemsize());
                Detail::CopyBytes(Pointer() + (layout.Offset(row) + column) * dtype.itemsize(), Placement(), value.data(),
                                  Device::Host(), value.size(), context)
                    .Wait();
                return element->load(value.data());
            }
            void Set(size_t row, size_t column, py::handle value)
            {
                LiveCheck();
                DNDS_check_throw_info(write && column < layout.RowSize(row), "element write requires a valid write lease/index");
                if (!element)
                {
                    auto values = Row(row);
                    values.attr("__setitem__")(column, value);
                    AssignRow(row, values);
                    return;
                }
                std::vector<uint8_t> bytes(dtype.itemsize());
                element->store(bytes.data(), value);
                Detail::CopyBytes(bytes.data(), Device::Host(), write->View<uint8_t>().data + (layout.Offset(row) + column) * dtype.itemsize(),
                                  Placement(), bytes.size(), context)
                    .Wait();
            }
            void Finish(bool failed)
            {
                if (write)
                {
                    if (failed)
                        write->Fail();
                    else
                        write->Wait();
                }
                else
                    read->Wait();
            }
            void Complete(Completion done)
            {
                if (write)
                    write->Complete(std::move(done));
                else
                    read->Complete(std::move(done));
            }
        };

        PythonArray To(const PythonArray &a, Device device, uintptr_t stream)
        {
            auto result = a.New(a.Layout(), device);
            auto context = device == Device::Host() ? a.Context(stream) : ExecutionContext{device, reinterpret_cast<void *>(stream)};
            result.CopyFrom(a, context).Wait();
            return result;
        }
        py::array Numpy(const PythonArray &a)
        {
            DNDS_check_throw_info(!a.Layout().IsRagged(), "ragged Array uses access.row() snapshots");
            auto host = To(a, Device::Host(), 0);
            auto read = host.Read(ExecutionContext::Host());
            py::array result(a.Dtype(), {py::ssize_t(a.Layout().Rows()), py::ssize_t(a.Layout().Width())});
            for (size_t i = 0; i < a.Layout().Rows(); ++i)
                if (a.Layout().Width())
                    std::memcpy(result.mutable_data(i, 0), read.View<uint8_t>().data + host.Layout().Offset(i) * host.Dtype().itemsize(), a.Layout().Width() * a.Dtype().itemsize());
            return result;
        }
        void Assign(PythonArray &a, py::array input)
        {
            input = py::array::ensure(input, py::array::c_style);
            DNDS_check_throw_info(input && !a.Layout().IsRagged() && input.dtype().equal(a.Dtype()) && input.ndim() == 2 &&
                                      size_t(input.shape(0)) == a.Layout().Rows() && size_t(input.shape(1)) == a.Layout().Width(),
                                  "Array assignment dtype/shape mismatch");
            auto host = a.New(a.Layout(), Device::Host());
            {
                auto write = host.Write(ExecutionContext::Host());
                for (size_t i = 0; i < a.Layout().Rows(); ++i)
                    if (a.Layout().Width())
                        std::memcpy(write.View<uint8_t>().data + host.Layout().Offset(i) * host.Dtype().itemsize(), input.data(i, 0), a.Layout().Width() * a.Dtype().itemsize());
            }
            a.CopyFrom(host, a.Context(0)).Wait();
        }
        ElementSchema Schema(const PythonArray &a)
        {
            // Numeric types retain their MPI type signature. Structured POD
            // defaults to bytes; applications may supply a retained MPI typemap.
            MPI_Datatype type = MPI_DATATYPE_NULL;
#define DNDS_NATIVE_TYPE(T)                  \
    if (a.Dtype().equal(py::dtype::of<T>())) \
    type = BasicType_To_MPIIntType<T>().first
            DNDS_NATIVE_TYPE(double);
            else DNDS_NATIVE_TYPE(float);
            else DNDS_NATIVE_TYPE(int64_t);
            else DNDS_NATIVE_TYPE(uint64_t);
            else DNDS_NATIVE_TYPE(int32_t);
            else DNDS_NATIVE_TYPE(uint32_t);
            else DNDS_NATIVE_TYPE(int16_t);
            else DNDS_NATIVE_TYPE(uint16_t);
            else DNDS_NATIVE_TYPE(int8_t);
            else DNDS_NATIVE_TYPE(uint8_t);
#undef DNDS_NATIVE_TYPE
            if (type != MPI_DATATYPE_NULL)
                return ElementSchema(type, a.Dtype().itemsize(), ArraySchema(a));
            DNDS_check_throw_info(a.Dtype().itemsize() <= INT_MAX, "POD dtype exceeds MPI contiguous count");
            MPI_Type_contiguous(int(a.Dtype().itemsize()), MPI_BYTE, &type);
            MPI_Type_commit(&type);
            ElementSchema result(type, a.Dtype().itemsize(), ArraySchema(a));
            MPI_Type_free(&type);
            return result;
        }
        void CheckSelection(const PythonArray &a, const std::vector<size_t> &rows, const std::vector<size_t> &columns)
        {
            for (auto row : rows)
            {
                DNDS_check_throw_info(row < a.Layout().Rows(), "row outside Array");
                for (auto column : columns)
                    DNDS_check_throw_info(column < a.Layout().RowSize(row), "column outside row");
            }
        }
        py::array ReadElements(PythonArray &a, const std::vector<size_t> &rows, const std::vector<size_t> &columns)
        {
            CheckSelection(a, rows, columns);
            py::array result(a.Dtype(), std::vector<py::ssize_t>{py::ssize_t(rows.size()), py::ssize_t(columns.size())});
            auto read = a.Read(a.Context(0));
            const auto *source = read.View<uint8_t>().data;
            const auto stride = a.Dtype().itemsize();
            const bool contiguous = std::adjacent_find(columns.begin(), columns.end(), [](size_t x, size_t y)
                                                       { return y != x + 1; }) == columns.end();
            for (size_t i = 0; i < rows.size() && !columns.empty(); ++i)
                for (size_t j = 0; j < columns.size(); j += contiguous ? columns.size() : 1)
                    Detail::CopyBytes(source + (a.Layout().Offset(rows[i]) + columns[j]) * stride, a.Storage().GetDevice(),
                                      result.mutable_data(i, j), Device::Host(), (contiguous ? columns.size() : 1) * stride, a.Context(0))
                        .Wait();
            return result;
        }
        void WriteElements(PythonArray &a, const std::vector<size_t> &rows, const std::vector<size_t> &columns, py::array values)
        {
            CheckSelection(a, rows, columns);
            values = py::array::ensure(values, py::array::c_style);
            DNDS_check_throw_info(values && values.dtype().equal(a.Dtype()) && values.ndim() == 2 && size_t(values.shape(0)) == rows.size() &&
                                      size_t(values.shape(1)) == columns.size(),
                                  "element assignment dtype/shape mismatch");
            auto write = a.Write(a.Context(0));
            auto *destination = write.View<uint8_t>().data;
            const auto stride = a.Dtype().itemsize();
            const bool contiguous = std::adjacent_find(columns.begin(), columns.end(), [](size_t x, size_t y)
                                                       { return y != x + 1; }) == columns.end();
            for (size_t i = 0; i < rows.size() && !columns.empty(); ++i)
                for (size_t j = 0; j < columns.size(); j += contiguous ? columns.size() : 1)
                    Detail::CopyBytes(values.data(i, j), Device::Host(), destination + (a.Layout().Offset(rows[i]) + columns[j]) * stride,
                                      a.Storage().GetDevice(), (contiguous ? columns.size() : 1) * stride, a.Context(0))
                        .Wait();
        }
        Reducer Builtin(const py::dtype &dtype, Reduction op, DestinationInit init)
        {
            Reducer reducer;
#define DNDS_NATIVE_REDUCE(T)            \
    if (dtype.equal(py::dtype::of<T>())) \
    reducer = Reducer::Builtin<T>(op, init)
            DNDS_NATIVE_REDUCE(double);
            else DNDS_NATIVE_REDUCE(float);
            else DNDS_NATIVE_REDUCE(int64_t);
            else DNDS_NATIVE_REDUCE(uint64_t);
            else DNDS_NATIVE_REDUCE(int32_t);
            else DNDS_NATIVE_REDUCE(uint32_t);
            else DNDS_NATIVE_REDUCE(int16_t);
            else DNDS_NATIVE_REDUCE(uint16_t);
            else DNDS_NATIVE_REDUCE(int8_t);
            else DNDS_NATIVE_REDUCE(uint8_t);
            else throw py::type_error("built-in reduction requires a supported arithmetic dtype");
#undef DNDS_NATIVE_REDUCE
            reducer.schema = DtypeSchema(dtype);
            return reducer;
        }
    }

    void BindPython(py::module_ &parent)
    {
        auto m = parent.def_submodule("Runtime", "Native leased row Arrays and distributed transfers");
#ifdef DNDS_USE_CUDA
        m.attr("cuda_enabled") = true;
#else
        m.attr("cuda_enabled") = false;
#endif
        py::enum_<MemorySpace>(m, "MemorySpace").value("Host", MemorySpace::Host).value("CUDA", MemorySpace::CUDA);
        py::class_<Device>(m, "Device")
            .def_static("host", &Device::Host)
            .def_static("cuda", &Device::CUDA, py::arg("ordinal") = 0)
            .def_readonly("space", &Device::space)
            .def_readonly("ordinal", &Device::ordinal)
            .def("__eq__", &Device::operator==);
        py::class_<ExecutionContext>(m, "ExecutionContext")
            .def(py::init([](Device d, uintptr_t stream)
                          { return ExecutionContext{d, reinterpret_cast<void *>(stream)}; }),
                 py::arg("device") = Device::Host(), py::arg("stream") = 0);
        py::class_<Completion>(m, "Completion")
            .def_static("record", &Completion::Record)
            .def("wait", &Completion::Wait, py::call_guard<py::gil_scoped_release>())
            .def_property_readonly("ready", &Completion::Ready);
        py::class_<RowLayout>(m, "RowLayout")
            .def_static("dense", &RowLayout::Dense, py::arg("rows"), py::arg("width"), py::arg("stride") = 0)
            .def_static("ragged", &RowLayout::Ragged)
            .def_property_readonly("rows", &RowLayout::Rows)
            .def_property_readonly("width", &RowLayout::Width)
            .def_property_readonly("stride", &RowLayout::Stride)
            .def_property_readonly("is_ragged", &RowLayout::IsRagged)
            .def_property_readonly("elements", &RowLayout::Elements)
            .def("row_size", &RowLayout::RowSize)
            .def_property_readonly("offsets", &RowLayout::Offsets);
        py::class_<PythonAccess>(m, "Access")
            .def("__enter__", [](PythonAccess &a) -> PythonAccess &
                 { DNDS_check_throw(a.Active()); return a; }, py::return_value_policy::reference_internal)
            .def("__exit__", [](PythonAccess &a, py::object error, py::object, py::object)
                 { a.Finish(!error.is_none()); })
            .def_property_readonly("active", &PythonAccess::Active)
            .def("row", &PythonAccess::Row, "Return an independent NumPy row snapshot.")
            .def("assign_row", &PythonAccess::AssignRow)
            .def("__getitem__", [](const PythonAccess &a, std::pair<size_t, size_t> index)
                 { return a.Get(index.first, index.second); })
            .def("__setitem__", [](PythonAccess &a, std::pair<size_t, size_t> index, py::object value)
                 { a.Set(index.first, index.second, value); })
            .def("complete", &PythonAccess::Complete)
            .def("wait", [](PythonAccess &a)
                 { a.Finish(false); })
            .def("unsafe_address", [](const PythonAccess &a)
                 { return reinterpret_cast<uintptr_t>(a.Pointer()); }, "Borrowed pointer for native kernels only; retain lease and complete its stream before release.");
        py::class_<CopyOperation>(m, "CopyOperation")
            .def("wait", &CopyOperation::Wait, py::call_guard<py::gil_scoped_release>());
        py::class_<PythonArray>(m, "Array")
            .def(py::init<RowLayout, py::object, Device>(), py::arg("layout"), py::arg("dtype") = py::dtype::of<double>(), py::arg("device") = Device::Host())
            .def_property_readonly("layout", [](const PythonArray &a)
                                   { return a.Layout(); })
            .def_property_readonly("dtype", [](const PythonArray &a)
                                   { return a.Dtype(); })
            .def_property_readonly("element_type", &PythonArray::ElementType)
            .def_property_readonly("device", [](const PythonArray &a)
                                   { return a.Storage().GetDevice(); })
            .def_property_readonly("identity", [](const PythonArray &a)
                                   { return a.Storage().Identity(); })
            .def_property_readonly("version", [](const PythonArray &a)
                                   { return a.Storage().Version(); })
            .def_property_readonly("valid", [](const PythonArray &a)
                                   { return a.Storage().IsValid(); })
            .def("read", [](PythonArray &a, uintptr_t stream)
                 { return PythonAccess(a, false, stream); }, py::arg("stream") = 0)
            .def("write", [](PythonArray &a, uintptr_t stream)
                 { return PythonAccess(a, true, stream); }, py::arg("stream") = 0)
            .def("alias", [](const PythonArray &a)
                 { return a; })
            .def("clone", [](const PythonArray &a, uintptr_t stream)
                 { return To(a, a.Storage().GetDevice(), stream); }, py::arg("stream") = 0)
            .def("to", &To, py::arg("device"), py::arg("stream") = 0)
            .def("reset", [](PythonArray &a, RowLayout layout, Device device)
                 { a = a.New(layout, device); }, py::arg("layout"), py::arg("device") = Device::Host())
            .def("numpy", &Numpy, "Independent host snapshot; no mutable buffer can escape a lease.")
            .def("assign", &Assign)
            .def("_read_elements", &ReadElements)
            .def("_write_elements", &WriteElements)
            .def("_read_object", [](PythonArray &a, size_t row, size_t column)
                 { return PythonAccess(a, false, 0).Get(row, column); })
            .def("_write_object", [](PythonArray &a, size_t row, size_t column, py::object value)
                 {
                     // Validate the class and index before acquiring mutable access.
                     DNDS_check_throw_info(a.Element() && column < a.Layout().RowSize(row), "invalid POD element write");
                     std::vector<uint8_t> bytes(a.Dtype().itemsize());
                     a.Element()->store(bytes.data(), value);
                     PythonAccess access(a, true, 0);
                     Detail::CopyBytes(bytes.data(), Device::Host(), access.write->View<uint8_t>().data + (access.layout.Offset(row) + column) * bytes.size(),
                                       access.Placement(), bytes.size(), access.context).Wait(); })
            .def("_edit_object", [](PythonArray &a, size_t row, size_t column, py::function edit)
                 {
                     DNDS_check_throw_info(a.Element() && column < a.Layout().RowSize(row), "invalid POD element edit");
                     PythonAccess access(a, true, 0);
                     std::vector<uint8_t> bytes(a.Dtype().itemsize());
                     py::object result;
                     try
                     {
                         auto value = access.Get(row, column);
                         result = edit(value);
                         access.element->store(bytes.data(), value);
                     }
                     catch (...)
                     {
                         access.write->CancelUnmodified(); // Only the host snapshot was edited.
                         throw;
                     }
                     Detail::CopyBytes(bytes.data(), Device::Host(), access.write->View<uint8_t>().data + (access.layout.Offset(row) + column) * bytes.size(),
                                       access.Placement(), bytes.size(), access.context).Wait();
                     return result; })
            .def("copy_from", [](PythonArray &a, const PythonArray &source, uintptr_t stream)
                 { DNDS_check_throw_info(a.Dtype().equal(source.Dtype()) && a.Layout() == source.Layout(), "copy dtype/layout mismatch");
                   return a.CopyFrom(source, a.Storage().GetDevice() == Device::Host() ? source.Context(stream) : a.Context(stream)); }, py::arg("source"), py::arg("stream") = 0)
            .def("element_schema", &Schema)
            .def("write_serializer", [](const PythonArray &a, Serializer::SerializerBaseSSP s, const std::string &name, const MPIInfo &mpi, const std::string &schema)
                 { WriteArrayStorage(a.Storage(), a.Layout(), a.Dtype().itemsize(), s, name, mpi, schema.empty() ? ArraySchema(a) : schema); }, py::arg("serializer"), py::arg("name"), py::arg("mpi"), py::arg("schema") = "")
            .def_static("read_serializer", [](Serializer::SerializerBaseSSP s, const std::string &name, const MPIInfo &mpi, py::object dtype, Serializer::ArrayGlobalOffset partition, Device device, const std::string &schema)
                        {
                     PythonArray prototype(RowLayout{}, dtype);
                     auto archive = ReadArrayStorage(s, name, mpi, partition, device, prototype.Dtype().itemsize(), schema.empty() ? ArraySchema(prototype) : schema);
                     return prototype.WithStorage(ArrayStorage::FromBuffer(std::move(archive.layout), std::move(archive.storage), prototype.Dtype().itemsize())); }, py::arg("serializer"), py::arg("name"), py::arg("mpi"), py::arg("dtype") = py::dtype::of<double>(), py::arg("partition") = Serializer::ArrayGlobalOffset_Unknown, py::arg("device") = Device::Host(), py::arg("schema") = "");
        py::class_<RowRequest>(m, "RowRequest")
            .def(py::init([](int rank, int64_t row, size_t destination)
                          { return RowRequest{rank, row, destination}; }));
        py::class_<RowTransferPlan>(m, "RowTransferPlan")
            .def(py::init([](const MPIInfo &mpi, size_t sources, size_t destinations, const std::vector<RowRequest> &requests)
                          { return RowTransferPlan(mpi.comm, sources, destinations, requests); }))
            .def("reverse", &RowTransferPlan::Reverse)
            .def("destination_layout", &RowTransferPlan::DestinationLayout)
            .def_property_readonly("has_collisions", &RowTransferPlan::HasCollisions)
            .def_property_readonly("source_rows", &RowTransferPlan::SourceRows)
            .def_property_readonly("destination_rows", &RowTransferPlan::DestinationRows);
        py::class_<ElementSchema>(m, "ElementSchema")
            .def_static("from_mpi", [](int handle, size_t bytes, const std::string &name)
                        { return ElementSchema(MPI_Type_f2c(handle), bytes, name); }, "Retain mpi4py datatype.py2f(); identity asserts the same type signature across ranks.")
            .def_property_readonly("extent", &ElementSchema::Extent)
            .def_property_readonly("identity", &ElementSchema::Identity);
        py::enum_<Reduction>(m, "Reduction").value("Sum", Reduction::Sum).value("Min", Reduction::Min).value("Max", Reduction::Max).value("BitOr", Reduction::BitOr).value("BitXor", Reduction::BitXor).value("BitAnd", Reduction::BitAnd);
        py::enum_<DestinationInit>(m, "DestinationInit").value("Preserve", DestinationInit::Preserve).value("Identity", DestinationInit::Identity);
        py::enum_<TransferTransport>(m, "TransferTransport").value("Direct", TransferTransport::Direct).value("HostPacked", TransferTransport::HostPacked);
        py::class_<Reducer>(m, "Reducer")
            .def_static("builtin", &Builtin, py::arg("dtype"), py::arg("operation"), py::arg("initialization") = DestinationInit::Preserve)
            .def_static("custom", [](const ElementSchema &schema, py::dtype dtype, std::string name, py::function combine, py::object identity, DestinationInit init)
                        {
                     DNDS_check_throw_info(schema.Extent() == size_t(dtype.itemsize()) && !py::cast<bool>(dtype.attr("hasobject")), "custom reducer POD dtype mismatch");
                     py::array seed(dtype, std::vector<py::ssize_t>{1});
                     seed.attr("__setitem__")(0, identity);
                     Reducer reducer;
                     reducer.identity = std::move(name);
                     reducer.schema = schema.Identity();
                     reducer.elementBytes = dtype.itemsize();
                     reducer.initialization = init;
                     reducer.host = [dtype, combine, seed, init](const void *input, void *output, const std::vector<Contribution> &groups, ExecutionContext)
                     {
                         py::gil_scoped_acquire gil;
                         size_t previous = SIZE_MAX;
                         const size_t stride = dtype.itemsize();
                         for (auto group : groups)
                         {
                             auto *dst = static_cast<uint8_t *>(output) + group.destinationElement * stride;
                             auto *src = static_cast<const uint8_t *>(input) + group.sourceElement * stride;
                             py::array current(dtype, std::vector<py::ssize_t>{py::ssize_t(group.count)});
                             py::array contribution(dtype, std::vector<py::ssize_t>{py::ssize_t(group.count)});
                             if (init == DestinationInit::Identity && previous != group.destinationElement)
                                 for (size_t j = 0; j < group.count; ++j) std::memcpy(static_cast<uint8_t *>(current.mutable_data()) + j * stride, seed.data(), stride);
                             else std::memcpy(current.mutable_data(), dst, group.count * stride);
                             std::memcpy(contribution.mutable_data(), src, group.count * stride);
                             auto result = py::array::ensure(combine(current, contribution), py::array::c_style);
                             DNDS_check_throw_info(result && result.dtype().equal(dtype) && result.ndim() == 1 && size_t(result.size()) == group.count,
                                                   "custom reducer returned wrong dtype/shape");
                             std::memcpy(dst, result.data(), group.count * stride);
                             previous = group.destinationElement;
                         }
                     };
                     return reducer; }, py::arg("schema"), py::arg("dtype"), py::arg("name"), py::arg("combine"), py::arg("identity"), py::arg("initialization") = DestinationInit::Preserve);
        py::class_<PayloadPlan>(m, "PayloadPlan")
            .def(py::init<RowTransferPlan, RowLayout, RowLayout, ElementSchema>())
            .def_property_readonly("send_segments", &PayloadPlan::SendSegments)
            .def_property_readonly("receive_segments", &PayloadPlan::ReceiveSegments)
            .def("bind", [](const PayloadPlan &p, const PythonArray &src, const PythonArray &dst, std::optional<Reducer> reducer, TransferTransport transport)
                 { p.CheckLayouts(src.Layout(), dst.Layout(), src.Dtype().itemsize(), src.SameElement(dst));
                   return reducer ? p.Bind(src.Storage(), dst.Storage(), *reducer, transport) : p.Bind(src.Storage(), dst.Storage(), transport); }, py::arg("source"), py::arg("destination"), py::arg("reducer") = py::none(), py::arg("transport") = TransferTransport::Direct);
        py::class_<TransferBinding>(m, "TransferBinding")
            .def("begin", &TransferBinding::Begin, py::arg("context") = ExecutionContext::Host());
        py::class_<TransferOperation>(m, "TransferOperation")
            .def("wait", &TransferOperation::Wait, py::call_guard<py::gil_scoped_release>())
            .def("wait_destination_ready", &TransferOperation::WaitDestinationReady, py::call_guard<py::gil_scoped_release>())
            .def("test", &TransferOperation::Test, py::call_guard<py::gil_scoped_release>());
    }
}
