# E2 isolated reconstruction storage assessment

Capability-selected CPU storage is implemented and validated. This adds actual
isolated-process memory evidence to its earlier numerical tests. It does not
establish a matched setup speedup.

Each row is the median of two maximum-rank observations. Allocations were
identical in both trials. MiB = 2^20 bytes. RSS includes mesh and metrics;
Runtime peak is retained-plus-known-scratch accounting, not total process memory.

| Geometry, cells/rank | Profile | Retained MiB | Runtime scratch MiB | Runtime peak MiB | Post-setup RSS MiB | Process peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2D, 8,192 | Full | 99.74 | 53.41 | 153.15 | 135.73 | 188.13 |
| 2D, 8,192 | Sweeps | 46.87 | 33.01 | 79.87 | 82.79 | 115.00 |
| 2D, 8,192 | Sweeps + WBAP | 67.12 | 53.41 | 120.53 | 102.91 | 155.76 |
| 3D, 2,048 | Full | 189.61 | 93.64 | 283.25 | 220.82 | 313.50 |
| 3D, 2,048 | Sweeps | 107.86 | 56.98 | 164.84 | 138.99 | 195.25 |
| 3D, 2,048 | Sweeps + WBAP | 141.71 | 93.64 | 235.35 | 173.04 | 265.75 |

Sweeps+WBAP saves **32.7% / 25.3%** of retained plan allocation in 2D/3D;
sweep-only saves **53.0% / 43.1%**. Reduced profiles omit the original-operator
and local-preconditioner tables, including raw boundary injection. Sweep-only
also omits the secondary limiter map. No unused tables are built and discarded.
Basis caching remains unchanged and independent: 20.41/69.47 MiB here. Inverse
diagnostics stay enabled.

The six-row pilot and twelve representative rows passed. A final twelve-row
representative run adds actual CPU-affinity provenance; its evidence is retained
here. Each profile uses a fresh MPI process group, eliminating prior-profile
allocator high-water contamination. Current Linux RSS and process high-water
are sampled before/after VR setup, before workspace and validation allocations.
Runtime scratch excludes native Eigen/std temporaries; whole-process high-water
observes these and earlier mesh/metric phases.

All profiles have matching per-rank bitwise sweep digests; full and WBAP profiles
have matching limiter digests. Values must be finite. The reducer independently
checks row/rank/shape identity, raw-log SHA256, recorded digests and actual
presence/absence of each omitted table record. These supplement existing direct
elementwise and independent mathematical UTs. This is one Jacobi sweep and one
degree-three WBAP pass on deterministic five-component fields, not a fluid
trajectory or complete descending limiter cycle.

Observed setup times (maximum rank, two trials, milliseconds):

| Geometry | Full | Sweeps | Sweeps + WBAP |
| --- | ---: | ---: | ---: |
| 2D | 246.99–258.66 | 195.40–219.03 | 240.37–290.29 |
| 3D | 679.24–697.01 | 606.73–614.22 | 660.50–661.15 |

This shared local runner is noisy: 2D WBAP timing changes direction between
trials. **No timing improvement is accepted.** Memory is the discriminating
result. These observations do not replace allocated-node E3–E7 assessments.

Runner: `gpu704`, Xeon Gold 6326, two sockets/NUMA nodes, Release GCC13,
Open MPI5.0.9, actual MPI_THREAD_MULTIPLE, OMP1, CUDA off, Cantera enabled.
Rank CPU affinities are `[0,32]` and `[1,33]`: separate physical cores with SMT
siblings, one OpenMP thread per rank. Meshes are 128x128 periodic quads and
16x16x16 physical-boundary hexes, P3, quadrature order5. Basis values and first
derivatives are cached. Mean/coefficient transfers use direct MPI with optional
preflight off for every profile; mesh fixtures keep their unchanged policy.
Profile order is full/sweeps/WBAP and then reversed.

Source base: `4197fee8`, plus the benchmark/runner additions whose source,
binary and cache hashes are in [the manifest](manifest.json.gz).
[Summary](summary.json) retains ranges, pre-setup RSS and high-water.
[Raw logs](raw-logs.tar.gz) preserve all twelve results with recorded hashes.
[Measured source bytes](measured-sources.tar.gz) preserve the exact benchmark,
runner and CMake input before commit-hook formatting; their hashes match the
manifest. The complete evidence/report is about 40 KiB. Earlier runs remain in
the ignored independent experiment workspace.

Reproduction from this checkout, with a new run ID:

```sh
cmake -S . -B build-audit
cmake --build build-audit -j8 --target cfv_benchmark_reconstruction_storage
python scripts/benchmark/plan53_storage.py \
  --binary build-audit/test/cpp/cfv_benchmark_reconstruction_storage \
  --mpiexec /path/to/matching/mpiexec \
  --workspace workspace/plan53-targeted-assessment --run-id fresh-e2 --np 2
python scripts/benchmark/reduce_plan53_storage.py \
  workspace/plan53-targeted-assessment/fresh-e2/manifest.json /path/to/evidence
```

Initialize the evidence workspace as an independent Git repository and record
the run plan before launching; set matching external/MPI library paths if needed.
Each process group has a 120-second limit. Failures stop the driver, which refuses
overwrite. The opt-in benchmark adds no ordinary CTest registrations or suite cost.
