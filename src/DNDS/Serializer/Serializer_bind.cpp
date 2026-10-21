#include "Serializer_bind.hpp"

namespace DNDS::Serializer
{
    void pybind11_bind_Serializer(py::module_ m)
    {
        auto m_Serializer = m.def_submodule("Serializer");

        py::class_<ArrayGlobalOffset>(m_Serializer, "ArrayGlobalOffset")
            .def(py::init<DNDS::index, DNDS::index>(), py::arg("size"), py::arg("offset"))
            .def_property_readonly("size", &ArrayGlobalOffset::size)
            .def_property_readonly("offset", &ArrayGlobalOffset::offset)
            .def_static("matching_rank", []
                        { return ArrayGlobalOffset_Unknown; })
            .def_static("even_split", []
                        { return ArrayGlobalOffset_EvenSplit; });
        m_Serializer.attr("ArrayGlobalOffset_Unknown") = py::cast(ArrayGlobalOffset_Unknown);
        m_Serializer.attr("ArrayGlobalOffset_EvenSplit") = py::cast(ArrayGlobalOffset_EvenSplit);

        pybind11_SerializerBase_define(m_Serializer);
        pybind11_SerializerJSON_define(m_Serializer);
        pybind11_SerializerH5_define(m_Serializer);

        pybind11_SerializerFactory_define(m_Serializer);
    }
}
