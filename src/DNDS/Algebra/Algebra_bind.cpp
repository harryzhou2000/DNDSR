#include "DofAlgebra.hpp"
#include "DNDS/Runtime/Array_bind.hpp"
#include <pybind11/stl.h>

namespace DNDS::Runtime
{
    namespace py = pybind11;
    namespace A = DNDS::Algebra;

    constexpr const char *FieldDoc = "Fixed-width float DOFs over shared native Runtime storage. Fields participating in an operation must share the same DofSpace identity, not merely row counts. Create a new space after reordering rows. Field handle copies and imported arrays share storage; use copy into a separately allocated field for independent values.";
    constexpr const char *ImmediateDoc = "Operate on logical owned entries at a serialized CPU caller boundary. All accessed owned storage must be idle: finish pending work and release active leases first. Ghosts and row padding remain untouched. Field handles share storage; matching operands require the same DofSpace identity.";
    constexpr const char *AsyncDoc = "Submit logical owned work eagerly on the supplied engine and stream, retaining operand storage through completion. Ghosts and row padding remain untouched. Dependencies are not inferred: join producer prefixes with record/wait_for before cross-stream access, and explicitly wait for completion/errors before consuming results. Matching operands require the same DofSpace identity.";
    constexpr const char *GlobalDoc = "Collective owned reduction. All ranks must call matching operations with matching scalar types, component widths and collective order, on an MPI-permitted caller thread. Local completion is explicitly waited before MPI. CPU convenience requires idle owned storage; explicit engine/stream overloads retain operands and require caller-supplied cross-stream joins, with no inferred dependencies. Matching operands require the same DofSpace identity.";

    template <class T>
    void BindField(py::module_ &m, const char *name)
    {
        using Field = A::DofField<T>;
        py::class_<Field>(m, name, FieldDoc)
            .def(py::init<A::DofSpace, size_t, size_t, Device>(), py::arg("space"), py::arg("components"), py::arg("stride") = 0, py::arg("device") = Device::Host(), "Allocate owned and ghost tables for this space; stride=0 selects the logical width.")
            .def(py::init([](A::DofSpace space, const PythonArray &owned, const PythonArray &ghost)
                          { return Field(std::move(space), owned.AsArray<T, TableRows>(), ghost.AsArray<T, TableRows>()); }),
                 py::arg("space"), py::arg("owned"), py::arg("ghost"), "Share existing typed Runtime tables without copying. Their rows must match the space, widths and placements must agree, and owned/ghost ranges must be disjoint.")
            .def_property_readonly("owned", [](const Field &field)
                                   { return PythonArray::Share(field.Owned()); }, "Runtime Array sharing the native owned allocation; a NumPy snapshot remains a copy.")
            .def_property_readonly("ghost", [](const Field &field)
                                   { return PythonArray::Share(field.Ghost()); }, "Runtime Array sharing ghost storage, which algebra operations leave untouched.")
            .def_property_readonly("space", &Field::Space, "Shared row-order identity; create a new space after reordering.")
            .def_property_readonly("components", &Field::Width, "Logical component width, excluding row padding.")
            .def_property_readonly("device", &Field::Device, "Placement shared by owned and ghost tables.")
            .def("to", [](const Field &field, Device device)
                 { return Field(field.Space(), field.Owned().To(device), field.Ghost().To(device)); }, py::arg("device"), "Transfer both tables using Runtime Array transfer semantics, retaining the same space identity. Complete producer prefixes before transfer.");
        m.def("fill", [](Field &out, T value)
              { A::Fill(out, value); }, py::arg("out"), py::arg("value"), ImmediateDoc);
        m.def("scale", [](Field &out, T value)
              { A::Scale(out, value); }, py::arg("out"), py::arg("value"), ImmediateDoc);
        m.def("copy", [](Field &out, const Field &input)
              { A::Copy(out, input); }, py::arg("out"), py::arg("input"), ImmediateDoc);
        m.def("axpy", [](Field &out, T alpha, const Field &input)
              { A::Axpy(out, alpha, input); }, py::arg("out"), py::arg("alpha"), py::arg("input"), ImmediateDoc);
        m.def("linear_combination", [](Field &out, T a, const Field &x, T b, const Field &y)
              { A::LinearCombination(out, a, x, b, y); }, py::arg("out"), py::arg("alpha"), py::arg("x"), py::arg("beta"), py::arg("y"), ImmediateDoc);
        m.def("fill_async", [](EagerEngine &e, const ExecutionStream &s, Field &out, T value)
              { A::Fill(e, s, out, value); }, py::arg("engine"), py::arg("stream"), py::arg("out"), py::arg("value"), AsyncDoc);
        m.def("scale_async", [](EagerEngine &e, const ExecutionStream &s, Field &out, T value)
              { A::Scale(e, s, out, value); }, py::arg("engine"), py::arg("stream"), py::arg("out"), py::arg("value"), AsyncDoc);
        m.def("copy_async", [](EagerEngine &e, const ExecutionStream &s, Field &out, const Field &input)
              { A::Copy(e, s, out, input); }, py::arg("engine"), py::arg("stream"), py::arg("out"), py::arg("input"), AsyncDoc);
        m.def("axpy_async", [](EagerEngine &e, const ExecutionStream &s, Field &out, T alpha, const Field &input)
              { A::Axpy(e, s, out, alpha, input); }, py::arg("engine"), py::arg("stream"), py::arg("out"), py::arg("alpha"), py::arg("input"), AsyncDoc);
        m.def("linear_combination_async", [](EagerEngine &e, const ExecutionStream &s, Field &out, T a, const Field &x, T b, const Field &y)
              { A::LinearCombination(e, s, out, a, x, b, y); }, py::arg("engine"), py::arg("stream"), py::arg("out"), py::arg("alpha"), py::arg("x"), py::arg("beta"), py::arg("y"), AsyncDoc);
        m.def("dot_local", [](const Field &x, const Field &y)
              { return A::DotLocal(x, y); }, py::arg("x"), py::arg("y"), ImmediateDoc, py::call_guard<py::gil_scoped_release>());
        m.def("norm_local", [](const Field &x)
              { return A::NormLocal(x); }, py::arg("x"), ImmediateDoc, py::call_guard<py::gil_scoped_release>());
        m.def("dot_local_async", [](EagerEngine &e, const ExecutionStream &s, const Field &x, const Field &y)
              { return A::DotLocal(e, s, x, y); }, py::arg("engine"), py::arg("stream"), py::arg("x"), py::arg("y"), AsyncDoc);
        m.def("norm_local_async", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::NormLocal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), AsyncDoc);
        m.def("component_sum_local", [](const Field &x)
              { return A::ComponentSumLocal(x); }, py::arg("x"), ImmediateDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_min_local", [](const Field &x)
              { return A::ComponentMinLocal(x); }, py::arg("x"), ImmediateDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_max_local", [](const Field &x)
              { return A::ComponentMaxLocal(x); }, py::arg("x"), ImmediateDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_sum_local_async", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::ComponentSumLocal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), AsyncDoc);
        m.def("component_min_local_async", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::ComponentMinLocal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), AsyncDoc);
        m.def("component_max_local_async", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::ComponentMaxLocal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), AsyncDoc);
        m.def("dot_global", [](const Field &x, const Field &y)
              { return A::DotGlobal(x, y); }, py::arg("x"), py::arg("y"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("norm_global", [](const Field &x)
              { return A::NormGlobal(x); }, py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_sum_global", [](const Field &x)
              { return A::ComponentSumGlobal(x); }, py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_min_global", [](const Field &x)
              { return A::ComponentMinGlobal(x); }, py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_max_global", [](const Field &x)
              { return A::ComponentMaxGlobal(x); }, py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("dot_global", [](EagerEngine &e, const ExecutionStream &s, const Field &x, const Field &y)
              { return A::DotGlobal(e, s, x, y); }, py::arg("engine"), py::arg("stream"), py::arg("x"), py::arg("y"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("norm_global", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::NormGlobal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_sum_global", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::ComponentSumGlobal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_min_global", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::ComponentMinGlobal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
        m.def("component_max_global", [](EagerEngine &e, const ExecutionStream &s, const Field &x)
              { return A::ComponentMaxGlobal(e, s, x); }, py::arg("engine"), py::arg("stream"), py::arg("x"), GlobalDoc, py::call_guard<py::gil_scoped_release>());
    }

    void BindAlgebraPython(py::module_ &runtime)
    {
        auto m = runtime.def_submodule("Algebra", "Owned DOF numerical operations over shared native Runtime Arrays.");
        py::class_<A::DofSpace>(m, "DofSpace", "Immutable shared identity for owned/ghost row order. Equal counts do not establish correspondence: reuse one space for matching fields and create a new space after reordering. Copies retain the same identity.")
            .def(py::init<size_t, size_t, std::string>(), py::arg("owned_rows"), py::arg("ghost_rows") = 0, py::arg("name") = "", "Construct a local row-order identity without MPI. Use distributed for global reductions.")
            .def_static("distributed", [](const MPIInfo &mpi, size_t owned, size_t ghost, std::string name)
                        { return A::DofSpace(std::make_shared<CommunicationContext>(mpi.comm), owned, ghost, std::move(name)); }, py::arg("mpi"), py::arg("owned_rows"), py::arg("ghost_rows") = 0, py::arg("name") = "", "Collectively construct a communication context on mpi.comm, from an MPI-permitted caller thread; all ranks must participate in matching order. Row counts may differ by rank.")
            .def_property_readonly("owned_rows", &A::DofSpace::OwnedRows, "Number of local owned rows.")
            .def_property_readonly("ghost_rows", &A::DofSpace::GhostRows, "Number of local ghost rows.")
            .def("same", &A::DofSpace::Same, py::arg("other"), "Test shared row-order identity, not merely equal counts.");
        py::class_<A::ScalarResult>(m, "ScalarResult", "Retained pending scalar reduction result; get() is the explicit completion/error boundary.")
            .def("get", &A::ScalarResult::Get, "Explicitly wait for the submitted prefix, propagate errors and return the host scalar. CUDA copies only the small result; this does not establish unrelated cross-stream dependencies.", py::call_guard<py::gil_scoped_release>());
        py::class_<A::ComponentResult>(m, "ComponentResult", "Retained pending component reduction result; get() is the explicit completion/error boundary.")
            .def("get", &A::ComponentResult::Get, "Explicitly wait for the submitted prefix, propagate errors and return host component values. CUDA copies only the small result; this does not establish unrelated cross-stream dependencies.", py::call_guard<py::gil_scoped_release>());
        BindField<float>(m, "FloatDofField");
        BindField<double>(m, "DoubleDofField");
        m.def("DofField", [](A::DofSpace space, size_t components, py::dtype dtype, size_t stride, Device device) -> py::object
              {
                  if (dtype.equal(py::dtype::of<float>()))
                      return py::cast(A::DofField<float>(std::move(space), components, stride, device));
                  DNDS_check_throw_info(dtype.equal(py::dtype::of<double>()), "DOF algebra requires float32 or float64");
                  return py::cast(A::DofField<double>(std::move(space), components, stride, device)); }, py::arg("space"), py::arg("components"), py::arg("dtype") = py::dtype::of<double>(), py::arg("stride") = 0, py::arg("device") = Device::Host(), FieldDoc);
    }
}
