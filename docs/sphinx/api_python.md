# Python API Reference

This section documents the `DNDSR` Python package, which wraps the C++ core
via pybind11 and provides high-level utilities for mesh I/O and solver setup.

The reference is generated from a combination of:
- `.pyi` stub files (produced by `pybind11-stubgen` from the compiled modules)
- Pure-Python wrapper docstrings
- `sphinx.ext.autodoc` runtime introspection (when the package is importable)

```{note}
To build this reference with full autodoc support:
1. Build and install the pybind11 modules:
   `cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 cauchy_pybind11 eulerP_pybind11 -j32`
   `cmake --install build --component py`
2. Run with `PYTHONPATH=python` (or `pip install -e .`)
3. Build docs: `cmake --build build -t docs`
```

## Overview by Module

| Module | Purpose | Key Classes / Functions |
|--------|---------|------------------------|
| `DNDSR.DNDS` | MPI, Runtime arrays, serialization | `MPIInfo`, `Runtime.Array`, `Runtime.ArrayPair`, `SerializerFactory` |
| `DNDSR.Geom` | Mesh I/O and topology | `UnstructuredMesh`, `UnstructuredMeshSerialRW`, `ElemType`, `read_mesh`, `prepare_mesh` |
| `DNDSR.CFV` | Finite volume and reconstruction | `FiniteVolume`, `FiniteVolumePlan`, `VariationalReconstruction` |
| `DNDSR.Cauchy` | Native FV solver, domain and implicit operators | `Domain`, `Simulation`, `EquationDofs`, `FiniteVolumeOperators`, `LinearProblem`, `LinearExecutionStreams` |
| `DNDSR.EulerP` | Legacy-experimental GPU evaluator | `EulerPEvaluator`, `EulerP_Solver` |

## DNDSR Package

```{automodule} DNDSR
:members:
:undoc-members:
:show-inheritance:
```

## DNDSR.DNDS

```{automodule} DNDSR.DNDS
:members:
:undoc-members:
:show-inheritance:
```

### DNDSR.DNDS.Runtime

```{automodule} DNDSR.DNDS.Runtime
:members:
:undoc-members:
:show-inheritance:
```

### DNDSR.DNDS.Wrapper

```{automodule} DNDSR.DNDS.Wrapper
:members:
:undoc-members:
:show-inheritance:
```

## DNDSR.Geom

```{automodule} DNDSR.Geom
:members:
:undoc-members:
:show-inheritance:
```

### DNDSR.Geom.utils

```{automodule} DNDSR.Geom.utils
:members:
:undoc-members:
:show-inheritance:
```

## DNDSR.CFV

```{automodule} DNDSR.CFV
:members:
:undoc-members:
:show-inheritance:
```

## DNDSR.Cauchy

```{automodule} DNDSR.Cauchy
:members:
:undoc-members:
:show-inheritance:
```

## DNDSR.EulerP (Legacy-Experimental)

```{automodule} DNDSR.EulerP
:members:
:undoc-members:
:show-inheritance:
```

### DNDSR.EulerP.EulerP_Solver

```{automodule} DNDSR.EulerP.EulerP_Solver
:members:
:undoc-members:
:show-inheritance:
```
