"""DNDSR.EulerP — Legacy-experimental compressible Navier-Stokes evaluator (optional CUDA).

Note:
    EulerP is formally marked **legacy-experimental** and is retained only for
    historical comparison. Production CPU/CUDA finite-volume work should use
    :mod:`DNDSR.Cauchy`.
"""

from __future__ import annotations

from DNDSR._loader import preload

preload("eulerP")

from ._ext.eulerP_pybind11 import *  # noqa: F401,F403
