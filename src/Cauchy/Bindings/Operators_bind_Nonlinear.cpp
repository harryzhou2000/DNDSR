#include "Cauchy/Bindings/Operators_bind.hpp"
#include "Cauchy/Bindings/OperatorsBindInternal.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include <pybind11/stl.h>

namespace DNDS::Cauchy::Bindings
{
    void BindOperatorNonlinear(py::module_ &m)
    {
        py::class_<NonlinearSolveControl>(m, "NonlinearSolveControl")
            .def(py::init<>())
            .def_readwrite("correction_limit", &NonlinearSolveControl::correctionLimit)
            .def_readwrite("initial_pseudo_cfl", &NonlinearSolveControl::initialPseudoCFL)
            .def_readwrite("update_components", &NonlinearSolveControl::updateComponents);
        py::class_<NonlinearStageSolver>(
            m, "NonlinearStageSolver",
            "Bound native nonlinear controller, sharing the production CorrectOnce/solve logic. Retains state, workspace and engine. The caller supplies frozen normalized resOther and physical time; no time-method history lives here.")
            .def_property_readonly("state",
                                   [](const NonlinearStageSolver &solver)
        {
            return *solver.state;
        })
            .def("solve",
                 [](NonlinearStageSolver &solver, const EquationDofs &source, real mass, real rhsScale, real time,
                    int stage, NonlinearSolveControl control)
        {
            return solver.value->Solve(*solver.state, source, mass, rhsScale, time, stage, control);
        },
                 py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::arg("stage") = 0, py::arg_v("control", NonlinearSolveControl{}, "NonlinearSolveControl()"),
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def(
                "correct_once",
                [](NonlinearStageSolver &solver, const EquationDofs &source, real mass, real rhsScale, real time,
                   int stage, NonlinearSolveControl control)
        {
            return solver.value->CorrectOnce(*solver.state, source, mass, rhsScale, time, stage, control);
        },
                py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                py::arg("stage") = 0, py::arg_v("control", NonlinearSolveControl{}, "NonlinearSolveControl()"),
                py::call_guard<py::gil_scoped_release>(),
                "Collective one-correction native block. No nonlinear convergence loop or acceptance decision; the caller controls CFL/stopping and owns rollback.")
            .def("evaluate_equation",
                 [](NonlinearStageSolver &solver, EquationDofs &output, real mass, real rhsScale, real time)
        {
            solver.value->EvaluateEquation(*solver.state, output, mass, rhsScale, time);
        }, py::arg("output"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::call_guard<py::gil_scoped_release>(), Collective)
            .def("apply_correction",
                 [](NonlinearStageSolver &solver, const EquationDofs &correction, real scale,
                    const std::vector<index> &components)
        {
            return solver.value->ApplyCorrection(*solver.state, correction, scale, components);
        },
                 py::arg("correction"), py::arg("scale") = 1, py::arg("update_components") = std::vector<index>{},
                 py::call_guard<py::gil_scoped_release>(), Collective);
        py::class_<PolynomialMultigrid>(
            m, "PolynomialMultigrid",
            "Native same-cell spatial p-FAS. Retains its finest stage solver and mesh. Apply one cycle inside a caller's nonlinear loop, with other-stage source terms frozen; no temporal-pMG.")
            .def("apply_cycle",
                 [](PolynomialMultigrid &multigrid, const EquationDofs &source, real mass, real rhsScale, real time,
                    int stage, real pseudoCFL)
        {
            return multigrid.ApplyCycle(source, mass, rhsScale, time, stage, pseudoCFL);
        },
                 py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::arg("stage") = 0, py::arg("pseudo_cfl") = 0, py::call_guard<py::gil_scoped_release>(), Collective)
            .def(
                "scratch_pool_statistics", &PolynomialMultigrid::ScratchPoolStatistics,
                "Reserved DoF payload bytes and peak simultaneous borrowed slots; excludes persistent/reconstruction workspaces.")
            .def("solve",
                 [](PolynomialMultigrid &multigrid, const EquationDofs &source, real mass, real rhsScale, real time,
                    int stage)
        {
            return multigrid.Solve(source, mass, rhsScale, time, stage);
        },
                 py::arg("source"), py::arg("inverse_dt"), py::arg("rhs_scale"), py::arg("time") = 0,
                 py::arg("stage") = 0, py::call_guard<py::gil_scoped_release>(), Collective);
    }
}
