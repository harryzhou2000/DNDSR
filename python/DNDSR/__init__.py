"""DNDSR — Compact Finite Volume CFD framework.

Submodules:
    DNDS   — Core data structures, MPI arrays, serialization.
    Geom   — Unstructured mesh, CGNS I/O, partitioning.
    CFV    — Compact Finite Volume, variational reconstruction.
    Cauchy — Native finite-volume solver, domain orchestration, and implicit operators.
    EulerP — Legacy-experimental compressible Navier-Stokes evaluator (optional CUDA).
"""

from __future__ import annotations

from pathlib import Path

from . import DNDS
from . import Geom
from . import CFV
from . import EulerP

__all__ = ["DNDS", "Geom", "CFV", "EulerP"]

if any((Path(__file__).resolve().parent / "Cauchy" / "_ext").glob("cauchy_pybind11*.so")):
    from . import Cauchy

    __all__.append("Cauchy")
