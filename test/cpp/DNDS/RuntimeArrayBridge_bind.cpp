#include "DNDS/Runtime/Array_bind.hpp"
#include "DNDS/Runtime/EagerArray.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include <pybind11/stl.h>
#include <atomic>
#include <thread>
#include <chrono>
#include "../../demos/MockStencil.hpp"

namespace R = DNDS::Runtime;
namespace py = pybind11;
using State = R::Array<double, R::FixedRows<5>>;

struct Metadata
{
    int64_t id;
    double weight;
};

struct ClassMetadata
{
    int64_t tag;
    double weight;
};

// Observe actual ownership without extending it or exposing a production API.
struct StorageWatch
{
    std::weak_ptr<const R::ArrayStorage> storage;
    std::weak_ptr<const R::Detail::ArrayLayout> layout;
    explicit StorageWatch(const R::PythonArray &array)
        : storage(array.SharedStorage()), layout(array.SharedStorage()->SharedLayout()) {}
};

struct AllocationWatch
{
    std::weak_ptr<void> owner;
};

#ifdef DNDS_USE_CUDA
void BridgeScaleCUDA(State &array, double factor);
void BridgeScaleCUDAAsync(R::EagerEngine &, const R::ExecutionStream &, State &, double);
void BridgeHoldCUDA(R::EagerEngine &, const R::ExecutionStream &, std::shared_ptr<std::atomic<bool>>);
#endif

struct AsyncGate
{
    std::shared_ptr<std::atomic<bool>> flag = std::make_shared<std::atomic<bool>>(false);
    ~AsyncGate() { *flag = true; }
    struct Event : R::StreamEvent::State
    {
        std::shared_ptr<std::atomic<bool>> flag;
        explicit Event(std::shared_ptr<std::atomic<bool>> f) : flag(std::move(f)) {}
        bool Ready() const override { return flag->load(); }
        void Wait() const override
        {
            while (!Ready())
                std::this_thread::yield();
        }
    };
};

void ScaleAsync(R::EagerEngine &engine, const R::ExecutionStream &stream, State array, double factor)
{
#ifdef DNDS_USE_CUDA
    if (stream.Context().device.space == R::MemorySpace::CUDA)
    {
        BridgeScaleCUDAAsync(engine, stream, array, factor);
        return;
    }
#endif
    R::SubmitArrays(engine, stream, [factor](R::ExecutionContext, R::ArrayView<double> values)
                    {
        for (size_t i = 0; i < values.Rows(); ++i)
            for (size_t j = 0; j < values.RowSize(i); ++j) values(i, j) *= factor; }, R::ReadWrite(array));
}

struct NativeOwner
{
    State state;
    R::Array<int64_t> ids;
    R::Array<Metadata> metadata;
    R::Array<ClassMetadata> classMetadata;

    explicit NativeOwner(R::Device device)
        : state(R::RowLayout::Dense(3, 5, 8), device),
          ids(R::RowLayout::Ragged({0, 2, 3}), device),
          metadata(R::RowLayout::Ragged({1, 0, 2}), device),
          classMetadata(R::RowLayout::Ragged({1, 0, 2}), device) {}
};

template <class T, class L>
T ReadElement(const R::Array<T, L> &array, size_t row, size_t column)
{
    DNDS_check_throw_info(row < array.Rows() && column < array.RowSize(row), "test element out of range");
    auto context = R::ExecutionContext{array.GetDevice(), nullptr};
    auto access = array.Read(context);
    T value;
    R::Detail::CopyBytes(access.View().data + array.Layout().Offset(row) + column, array.GetDevice(),
                         &value, R::Device::Host(), sizeof(T), context)
        .Wait();
    return value;
}

void Scale(State array, double factor)
{
#ifdef DNDS_USE_CUDA
    if (array.GetDevice() != R::Device::Host())
    {
        BridgeScaleCUDA(array, factor);
        return;
    }
#endif
    auto access = array.Write();
    auto view = access.View();
    for (size_t i = 0; i < view.Rows(); ++i)
        for (size_t j = 0; j < view.RowSize(i); ++j)
            view(i, j) *= factor;
}

PYBIND11_MODULE(runtime_array_bridge_test, m)
{
    py::module_::import("DNDSR.DNDS.Runtime");
    m.def("mock_stencil", &R::Demo::Stencil);
    m.def("delay_cpu", [](R::EagerEngine &engine, const R::ExecutionStream &stream)
          { engine.Submit(stream, [](R::ExecutionContext)
                          { std::this_thread::sleep_for(std::chrono::milliseconds(30)); }); });
    m.def("cuda_aware_mpi", []
          { return DNDS::MPI::isCudaAware(); });
    m.def("native_double_schema", &R::ElementSchema::Arithmetic<double>);
    m.def("bind_native_state", [](const R::PayloadPlan &plan, State source, State destination, R::Reducer reducer)
          { return plan.Bind(source, destination, std::move(reducer)); });
    py::class_<AsyncGate, std::shared_ptr<AsyncGate>>(m, "AsyncGate")
        .def("open", [](AsyncGate &gate)
             { *gate.flag = true; });
    m.def("hold", [](R::EagerEngine &engine, const R::ExecutionStream &stream)
          {
        auto gate = std::make_shared<AsyncGate>();
        if (stream.Context().device.space == R::MemorySpace::Host)
            stream.WaitFor(R::StreamEvent(std::make_shared<AsyncGate::Event>(gate->flag)));
#ifdef DNDS_USE_CUDA
        else BridgeHoldCUDA(engine, stream, gate->flag);
#endif
        return gate; });
    m.def("scale_async", &ScaleAsync);
    m.def("shares_layout", [](const R::PythonArray &a, const R::PythonArray &b)
          { return a.SharedStorage()->SharedLayout() == b.SharedStorage()->SharedLayout(); });
    PYBIND11_NUMPY_DTYPE(Metadata, id, weight);
    m.attr("metadata_dtype") = py::dtype::of<Metadata>();
    auto classMetadata = py::class_<ClassMetadata>(m, "ClassMetadata")
                             .def(py::init([](int64_t tag, double weight)
                                           { return ClassMetadata{tag, weight}; }),
                                  py::arg("tag") = 0, py::arg("weight") = 0)
                             .def_readwrite("tag", &ClassMetadata::tag)
                             .def_readwrite("weight", &ClassMetadata::weight)
                             .def("scale", [](ClassMetadata &value, double factor)
                                  { value.weight *= factor; });
    R::RegisterArrayElement(classMetadata, "test.ClassMetadata.v1"); // No NumPy struct registration.
    py::class_<AllocationWatch>(m, "AllocationWatch")
        .def_property_readonly("alive", [](const AllocationWatch &watch)
                               { return !watch.owner.expired(); });
    m.def("external_state", []
          {
              auto values = std::make_shared<std::vector<double>>(53 * 8, 1.0);
              auto owner = std::shared_ptr<void>(values, values->data());
              auto allocation = R::Allocation::Import(owner, values->size() * sizeof(double));
              auto array = R::Array<double, R::FixedRows<7>>::FromStorage(
                  R::RowLayout::Dense(53, 7, 8), R::Buffer::FromAllocation(allocation, 0, allocation.Bytes()));
              return py::make_tuple(array, AllocationWatch{owner}); });
    py::class_<StorageWatch>(m, "StorageWatch")
        .def(py::init<const R::PythonArray &>())
        .def_property_readonly("storage_alive", [](const StorageWatch &watch)
                               { return !watch.storage.expired(); })
        .def_property_readonly("layout_alive", [](const StorageWatch &watch)
                               { return !watch.layout.expired(); });
    py::class_<NativeOwner>(m, "NativeOwner")
        .def(py::init<R::Device>())
        .def_property_readonly("state", [](const NativeOwner &owner)
                               { return owner.state; })
        .def_property_readonly("ids", [](const NativeOwner &owner)
                               { return owner.ids; })
        .def_property_readonly("metadata", [](const NativeOwner &owner)
                               { return owner.metadata; })
        .def_property_readonly("class_metadata", [](const NativeOwner &owner)
                               { return owner.classMetadata; })
        .def("read_class_metadata", [](const NativeOwner &owner, size_t i, size_t j)
             { return ReadElement(owner.classMetadata, i, j); })
        .def("read_state", [](const NativeOwner &owner, size_t i, size_t j)
             { return ReadElement(owner.state, i, j); })
        .def("read_id", [](const NativeOwner &owner, size_t i, size_t j)
             { return ReadElement(owner.ids, i, j); })
        .def("read_metadata", [](const NativeOwner &owner, size_t i, size_t j)
             { auto value = ReadElement(owner.metadata, i, j); return py::make_tuple(value.id, value.weight); })
        .def("scale", [](NativeOwner &owner, double factor)
             { Scale(owner.state, factor); })
        .def("replace_state", [](NativeOwner &owner, State value)
             { owner.state = std::move(value); })
        .def("reset_state", [](NativeOwner &owner)
             { owner.state.Reset(R::RowLayout::Dense(1, 5), owner.state.GetDevice()); })
        .def("shares_state", [](const NativeOwner &owner, State value)
             { return owner.state.SharedStorage() == value.SharedStorage(); })
        .def("shares_ids", [](const NativeOwner &owner, R::Array<int64_t> value)
             { return owner.ids.SharedStorage() == value.SharedStorage() &&
                      owner.ids.Layout().Offsets().data() == value.Layout().Offsets().data() &&
                      owner.ids.SharedStorage()->SharedLayout()->DeviceOffsetIdentity() == value.SharedStorage()->SharedLayout()->DeviceOffsetIdentity(); });
    m.def("scale", &Scale);
    m.def("roundtrip", [](R::Array<double> array)
          { return array; });
    m.def("roundtrip_ids", [](R::Array<int64_t> array)
          { return array; });
    m.def("roundtrip_metadata", [](R::Array<Metadata> array)
          { return array; });
    m.def("roundtrip_class_metadata", [](R::Array<ClassMetadata> array)
          { return array; });
    // Unusual widths demonstrate that no width-instantiation list is consulted.
    m.def("make_fixed37", []
          { return R::Array<double, R::FixedRows<37>>(R::RowLayout::Dense(2, 37)); });
    m.def("make_fixed113", []
          { return R::Array<int64_t, R::FixedRows<113>>(R::RowLayout::Dense(1, 113)); });
    m.def("with_read", [](State array, py::function callback)
          { auto access = array.Read({array.GetDevice(), nullptr}); callback(); });
    m.def("with_write", [](State array, py::function callback)
          { auto access = array.Write({array.GetDevice(), nullptr}); callback(); });
    m.def("poison", [](State array)
          { auto access = array.Write({array.GetDevice(), nullptr}); access.Fail(); });
}
