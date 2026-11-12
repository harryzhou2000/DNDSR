# Single-device AmgX dependency

Plan61 uses one single-device AmgX instance per rank/device. DNDS retains the
global operator and halo exchanges; AmgX approximates the owned-owned block.
There is no delegated multi-rank provider communication. This document records
the build recipe; an enabled dependency is not evidence of a completed solver
adapter or numerical validation.

AmgX is not in the current cfd_externals package. Build its pinned v2.5.0 source
(`cc1cebdbb32b14d33762d4ddabcb2e23c1669f47`) with the independent recipe:

```sh
python3 scripts/build_amgx.py --root workspace/dependencies/amgx \
    --architectures '80;86' --jobs 16
# Use --host-compiler /path/to/g++ and --cuda-compiler /path/to/nvcc
# when the system defaults are not compatible with the selected CUDA toolkit.
cmake -S . -B build-cuda -DDNDS_USE_CUDA=ON -DDNDS_USE_AMGX=ON \
    -DDNDS_AMGX_ROOT="$PWD/workspace/dependencies/amgx/install"
```

For auwd's RTX3060, use architecture `86` and the explicitly selected checkout
`~/projects/DNDSR`. The recipe forces `CMAKE_NO_MPI=ON`, records its configure
command/revision after installation, and refuses to alter a dirty or differently
versioned source tree. It installs only into its dedicated prefix and does not
install OS packages or change cfd_externals. Do not bypass CUDA host-compiler
compatibility checks as a default workaround.

`DNDS_USE_AMGX=OFF` remains valid for CPU/native-Jacobi builds. Explicitly
requesting AmgX requires the imported dependency; no fallback is implicit.
Runtime adapter calls must order input readiness and output completion according
to AmgX's actual API, without assuming it uses the application's stream.

`DNDS::Algebra::AmgXPreconditioner` accepts a scalar CSR expansion of the full
owned-owned principal block. `SetPattern` takes host index views and retains
private Runtime arrays. `UpdateValuesBorrowed` takes FP64 device values;
`ApplyBorrowed` takes device residual/correction views. Both wrappers check
placement and synchronize explicitly. The submitting operator must drain
host-progressed communication before this provider boundary, retain scopes,
and abort on an exception in a collective solve. No MPI communicator is passed
to AmgX. Unsupported builds reject the provider instead of falling back.

The default is aggregation AMG with fixed V cycles, zero starting correction,
Jacobi smoothing and fixed coarsest Jacobi sweeps (`coarse_solver=NOSOLVER`).
Residual monitoring is disabled inside the provider, preserving a fixed linear
preconditioner for ordinary GMRES. Pattern/value updates belong between Krylov
solves. This source adapter and its focused test are not yet numerical evidence;
actual nonsymmetric CFD matrices remain part of Plan61 acceptance.
