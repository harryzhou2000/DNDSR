#pragma once

#include "MeshEdit.hpp"
#include "DNDS/Defines_bind.hpp"
#include <pybind11/numpy.h>

namespace DNDS::Geom
{
    inline void pybind11_MeshEdit_define(py::module_ &m)
    {
        py::class_<MeshEdit, std::shared_ptr<MeshEdit>>(m, "MeshEdit")
            .def_property_readonly("closed", &MeshEdit::Closed)
            .def_property_readonly("num_nodes", &MeshEdit::NumNode)
            .def_property_readonly("num_cells", &MeshEdit::NumCell)
            .def("coord", [](const MeshEdit &self, index row)
                 {
                     const auto value = self.Coord(row);
                     py::array_t<real> result(3);
                     for (int d = 0; d < 3; ++d) result.mutable_data()[d] = value[d];
                     return result; }, py::arg("row"), "Return an independent coordinate copy. Use set_coord to stage changes.")
            .def("set_coord", [](MeshEdit &self, index row, const py::array_t<real, py::array::c_style | py::array::forcecast> &value)
                 {
                     if (value.ndim() != 1 || value.shape(0) != 3)
                         throw py::value_error("coordinate edit requires shape (3,)");
                     self.SetCoord(row, tPoint{value.data()[0], value.data()[1], value.data()[2]}); }, py::arg("row"), py::arg("value"))
            .def("cell_info", &MeshEdit::CellInfo, py::arg("row"), "Return an independent ElemInfo value.")
            .def("set_cell_info", &MeshEdit::SetCellInfo, py::arg("row"), py::arg("value"))
            .def("cell_neighbors", [](const MeshEdit &self, index row)
                 {
                     const auto values = self.CellNeighbors(row);
                     py::array_t<index> result(values.size());
                     std::copy(values.begin(), values.end(), result.mutable_data());
                     return result; }, py::arg("row"), "Return an independent neighbor-order copy. Only permutations may commit.")
            .def("set_cell_neighbor", &MeshEdit::SetCellNeighbor, py::arg("row"), py::arg("slot"), py::arg("value"))
            .def("commit", &MeshEdit::Commit, "Collectively validate, refresh ghosts and publish native authority. Release other mesh read scopes first.")
            .def("abort", &MeshEdit::Abort)
            .def("__enter__", [](std::shared_ptr<MeshEdit> self)
                 { (void)self->NumNode(); return self; })
            .def("__exit__", [](MeshEdit &self, const py::object &, const py::object &, const py::object &)
                 { self.Abort(); return false; });
    }
}
