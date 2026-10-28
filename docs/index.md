(index_page)=
# DNDSR Documentation

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code
implementing Compact Finite Volume methods with MPI parallelism and optional
CUDA GPU support.

The [2026-10-31 VR SVD correction](dev/vr_svd_cutoff_2026-10-31.md) documents the
small-mode cutoff change, setup diagnostics and affected nonzero-tolerance cases.

## Architecture entry points

| Topic | Guide |
| --- | --- |
| Mesh ownership, relation semantics, construction closure and ghost assembly | [Mesh connectivity and construction](architecture/MeshConnectivity.md) |
| Owned construction API and mesh access examples | [Native mesh construction](architecture/NativeMesh.md) |
| Shared arrays, placement, access and transfers | [Array infrastructure](architecture/array_infrastructure.md) |
| Runtime ordering and lifetime responsibilities | [Runtime caller contract](architecture/RuntimeCallerContract.md) |
| Native field algebra | [Native DOF algebra](architecture/NativeDofAlgebra.md) |
| Canonical and compatibility source paths | [Source organization](architecture/source_organization.md) |
| Mesh Python access and operations | [Python mesh guide](guides/python_geom_guide.md) |

The mesh construction correction and its fixture/condition audit are recorded
in the validation ledger below. Earlier infrastructure acceptance was withdrawn
after detecting an omitted boundary ring; see the
[evidence correction](dev/infrastructure_finish_2026-10-27.md). Test totals and
owned numerical parity do not prove per-rank ghost completeness.
The [source comparison and validation ledger](dev/mesh_construction_semantics_2026-10-28.md)
records the final checks, explicit legacy periodic-frame differences and limits.
The [legacy connectivity test-purpose ledger](dev/mesh_connectivity_test_port_2026-10-28.md)
records all 47 original cases, independent expected answers and explicit
replacements for retired containers.

## Core Modules

| Module     | Directory    | Description                                              |
|------------|-------------|----------------------------------------------------------|
| **DNDS**   | `src/DNDS`  | MPI arrays, serialization (JSON, HDF5), profiling, CUDA  |
| **Geom**   | `src/Geom`  | Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis) |
| **CFV**    | `src/CFV`   | Compact Finite Volume, variational reconstruction        |
| **Euler**  | `src/Euler` | Compressible Navier-Stokes solvers (2D/3D, SA, k-omega)  |
| **EulerP** | `src/EulerP`| Alternative evaluator with CUDA GPU support              |
| **Solver** | `src/Solver`| ODE integrators and Krylov solvers (GMRES, PCG)          |

## Quick Start

```sh
# 1. Build external dependencies
git submodule update --init --recursive --depth=1
python3.12 -m venv venv && source venv/bin/activate
pip install -r external/cfd_externals/requirements.txt
cd external/cfd_externals && CC=mpicc CXX=mpicxx python cfd_externals_build.py && cd ../..
bash scripts/install_python_deps.sh

# 2. Build C++ solvers
cmake --preset release-test
cmake --build build -t euler -j32

# 3. Install Python package (editable)
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 pip install -e .

# 4. Run tests
ctest --test-dir build -R dnds_ --output-on-failure   # C++
pytest test/DNDS/test_basic.py -v                      # Python

# 5. Build and serve documentation
cmake --build build -t serve-docs
```

> **See also:** @ref building for the complete build guide (CMake presets,
> Python editable installs, external dependencies, and common issues).

## C++ API (Doxygen)

The Sphinx C++ API pages provide namespace outlines and key class documentation
via Breathe. For the full Doxygen HTML documentation with interactive
**call graphs**, **class inheritance diagrams**, and **include dependency graphs**,
build and serve the Doxygen output separately:

```sh
cmake --build build -t serve-doxygen
```

Once built, the full Doxygen reference is at
<a href="doxygen/index.html">Doxygen C++ API Reference</a> within the same site.

```{toctree}
:maxdepth: 2
:caption: Contents

/sphinx/guides
/sphinx/solver-guide
/sphinx/architecture
/sphinx/theory
/sphinx/tests
/sphinx/dev
/sphinx/api_cpp
/sphinx/api_python
/sphinx/presentations
```
