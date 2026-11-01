#pragma once

#include "DNDS/Bindings/Defines_bind.hpp"
#include "CFV/Legacy/FiniteVolume.hpp"
#include "CFV/Legacy/VRDefines_bind.hpp"

namespace DNDS::CFV
{
    using tPy_FiniteVolume = py_class_ssp<FiniteVolume>;

    void pybind11_FiniteVolume_define(py::module_ &m);
}