# E3 coefficient publication: functional pilot

The opt-in `cfv_benchmark_coefficient_publication` separates production
`CommitReconstruction`, the same lease-protected row copy without mesh/workspace
validation, a frozen-input Jacobi sweep and one top-degree WBAP pass. Setup,
reset, halos, barriers and payload checks are outside timing. It introduces no
production change and no buffer swapping. Ordinary builds and CTest exclude it.

Local np2/OMP1, 16x16 periodic quads (128 owned cells/rank), Release GCC13 and
Open MPI5.0.9: P3 flow5 AoS and P2 split 5+12-component SoA both pass two trials,
reversing phase order. Each copy batch has 100 repetitions; numerical batches
have four. Means and compute-only coefficient generations stay unchanged;
coefficient allocation identity stays fixed. Both copy paths produce exact
owned and ghost values against a global-ID formula through the original
persistent bindings. Numerical outputs remain finite. The two allocations model
flow/species coefficient widths; this is not a reacting RHS or a chemistry test.

Logical read+write bytes per complete physical-field batch are 92,160 for
P3 flow5 and 174,080 for P2 split fields, using
`16 * owned_cells * modes * total_components`. This is not hardware traffic.
No comparison between those different shapes/layouts establishes a layout winner.
Existing CFV mathematical and split-SOR/halo UTs remain the numerical oracles.

Source base is `4a174863`, with measured uncommitted benchmark/CMake additions.
The [manifest](manifest.json.gz) pins exact source/binary/cache/log hashes,
commands, timestamps, environment and binding output; [source bytes](measured-sources.tar.gz)
retain the actual benchmark before commit-hook formatting. [Raw logs](raw-logs.tar.gz)
contain all four accepted records. An initial compile used a nonexistent
field convenience accessor; replacing it with owned/ghost read scopes yields
the successful build and both pilots. This failure is not a numerical pass.

**This is functional acceptance only. E3 remains open.** Matched small/large
cells-per-rank timings, complete limiter-cycle and actual production copy
counts, RHS fractions and the retain/change decision are still required.
The bare-copy comparison does not justify removing required mesh ownership.

```sh
cmake --build build-audit --target cfv_benchmark_coefficient_publication -j8
OMP_NUM_THREADS=1 mpiexec --bind-to core -np 2 \
  build-audit/test/cpp/cfv_benchmark_coefficient_publication 16 3 flow5 aos 100 2
OMP_NUM_THREADS=1 mpiexec --bind-to core -np 2 \
  build-audit/test/cpp/cfv_benchmark_coefficient_publication 16 2 split5+12 soa 100 2
```
