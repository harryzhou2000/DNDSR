"""Cauchy CPU flow solver, using shared native mesh and Runtime arrays."""

from DNDSR._loader import preload

preload("cauchy")

from ._ext.cauchy_pybind11 import Domain, Simulation, InterfaceDiagnostics, NonlinearIterationReport, PhysicalStepLimit, DerivedFields, DerivedQuantity

__all__ = ["Domain", "Simulation", "InterfaceDiagnostics", "NonlinearIterationReport", "PhysicalStepLimit", "DerivedFields", "DerivedQuantity"]
