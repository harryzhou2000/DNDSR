"""Cauchy CPU flow solver, using shared native mesh and Runtime arrays."""

from DNDSR._loader import preload

preload("cauchy")

from ._ext.cauchy_pybind11 import Domain, NonlinearIterationReport, PhysicalStepLimit

__all__ = ["Domain", "NonlinearIterationReport", "PhysicalStepLimit"]
