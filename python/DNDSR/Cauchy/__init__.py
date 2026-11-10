"""Cauchy CPU solver and equation operators over shared native fields."""

from DNDSR._loader import preload

preload("cauchy")

from ._ext.cauchy_pybind11 import Domain, Simulation, StepResult, InterfaceDiagnostics, NonlinearIterationReport, PhysicalStepLimit, DerivedFields, DerivedQuantity
from ._ext.cauchy_pybind11 import (EquationDofs, EquationOperators, AuxiliaryRequest,
                                  AuxiliaryValues, MeanGradients, Workspace,
                                  MeanWorkspace, DiagonalStorage, OffDiagonalStorage,
                                  LinearRegion, JacobianPreparation, LinearProblem,
                                  LinearSolveSettings, LinearExecution, LinearIterationReport, LinearStop,
                                  EquationLinearSolver, EquationSGSIteration,
                                  NonlinearSolveControl, EquationStageSolver,
                                  PolynomialMultigrid)

__all__ = ["Domain", "Simulation", "StepResult", "InterfaceDiagnostics", "NonlinearIterationReport", "PhysicalStepLimit", "DerivedFields", "DerivedQuantity"]
__all__ += ["EquationDofs", "EquationOperators", "AuxiliaryRequest", "AuxiliaryValues",
            "MeanGradients", "Workspace", "MeanWorkspace",
            "DiagonalStorage", "OffDiagonalStorage", "LinearRegion",
            "JacobianPreparation", "LinearProblem", "LinearSolveSettings",
            "LinearExecution", "LinearIterationReport", "LinearStop", "EquationLinearSolver",
            "EquationSGSIteration", "NonlinearSolveControl", "EquationStageSolver",
            "PolynomialMultigrid"]
