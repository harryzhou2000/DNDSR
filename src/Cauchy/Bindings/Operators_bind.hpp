#pragma once

#include <pybind11/pybind11.h>

namespace DNDS::Cauchy::Bindings
{
    /// Register the Python operator facades over retained native resources.
    /// This binding does not implement another numerical orchestration path.
    void BindFiniteVolumeOperators(pybind11::module_ &module);
    /// Internal registration partitions, called in order by BindFiniteVolumeOperators.
    void BindOperatorState(pybind11::module_ &module);
    void BindOperatorLinear(pybind11::module_ &module);
    void BindOperatorNonlinear(pybind11::module_ &module);
    void BindOperatorFiniteVolume(pybind11::module_ &module);
}
