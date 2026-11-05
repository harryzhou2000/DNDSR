# E6: identity dispatch stored once per plan

**Retain the revised exact identity optimization and assembled storage for the
measured flow5 workloads.** The auxiliary regression of the first candidate
is absent at that magnitude. General rotated-frame handling remains available.

Job `12442822` completed `0:0` in 48 seconds, scheduler-selected `amd_512` node
`o2409`, one exclusive node. All twelve rows / 36 trials / 324 phase records
and raw-log hashes pass independent verification. Frozen baseline `b6887a40`
and candidate `513d03e4` run serially within the allocation, with reversed
version/storage order between workloads. OMP1/OPENBLAS1, GCC12.2/OpenMPI3.1.4,
actual MPI_THREAD_MULTIPLE, Release with assertions. No node pin or other job
mutation. Benchmark source hashes and recorded compile/link flags are identical.
Cache-file hashes differ after the new-target CMake regeneration; both exact
hashes are retained rather than calling the cache files identical.

Median maximum-rank milliseconds, 5,000 cells/rank:

|np|storage|version|auxiliary|prepare|local SGS pair|prepare+inverse+5 pairs|prepare+inverse+solve5|
|---:|---|---|---:|---:|---:|---:|---:|
|32|assembled|baseline|7.806|23.805|3.160|36.936|52.092|
|32|assembled|candidate|8.013|14.014|3.112|26.964|42.728|
|32|evaluated|baseline|8.197|14.211|20.707|118.273|174.420|
|32|evaluated|candidate|8.110|10.061|19.192|102.518|158.178|
|8|assembled|baseline|7.261|22.932|2.873|35.703|48.779|
|8|assembled|candidate|7.572|11.888|2.688|23.672|34.865|
|8|evaluated|baseline|7.437|12.481|19.732|111.323|161.485|
|8|evaluated|candidate|7.551|8.126|17.984|95.001|147.608|

Actual combined sequences are timed directly, including 1/5/20 pairs and the
five-iteration communicated SGS solve. The latter completes exactly15
iterations per three-repetition trial with zero stopping tolerances. This is
bounded linear work on a frozen approximate operator, not physical/nonlinear
convergence or a sum of separate medians. Independent complete matrix actions
and SGS outputs agree across storage modes, are finite, and retain input
generations. Small-partition observations and all trial ranges stay in JSON.

The optimization changes no quadrature integration, EOS derivative,
dissipation, boundary treatment, moving-frame term, component projection or
source diagonal. It records exact identity presence once in the immutable FV
plan/frozen geometry, not in each quadrature sample/point. Any rotated
incidence keeps the rank's complete plan on the general path. Exact comparison
uses no near-identity tolerance. Baseline Sample/Point payload fields are
restored; the [first candidate](../e6-sample-flag/README.md) and its auxiliary
regression are preserved separately. The new timings support the smaller
metadata placement, without proving a universal cache-effect explanation.

Large auxiliary medians differ by -1.1% to+4.3% here, versus the earlier
roughly+12–26% regressions. Assembled preparation improves roughly41–48%, and
its actual preparation+inverse+five-pair sequence improves27–34%. Evaluated
storage still loses in every measured preparation+reuse sequence, including
one pair. Keep both storage modes without claiming an unmeasured break-even.

Maximum-rank post-work RSS at np32 is111.0/111.1MiB for assembled baseline/
candidate and121.2/121.1MiB for evaluated. At np8 it is108.5/106.7MiB and
116.6/116.7MiB. These include mesh/VR/workspace allocations and are not exact
snapshot byte counts. Opposite-mode numerical oracles are allocated after all
memory/timing observations. VmRSS and getrusage high-water remain separate.

This report qualifies the CPU flow5 candidate. Whole-equation source/layout
oracles remain required, and a complete nonlinear-stage/reaction timing claim
is outside this frozen linear benchmark. No CUDA or general-geometry speedup
is inferred from Cartesian translation-periodic timing fixtures; existing
rotated/physical/moving tests supply the corresponding correctness gates.
