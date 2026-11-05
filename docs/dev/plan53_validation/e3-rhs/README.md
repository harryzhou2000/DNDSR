# E3 flow5 publication fraction inside the composed RHS

Job `12442797` completed `0:0` in 31 seconds on Slurm-selected `amd_512` node
`o2409`, source `01c06db9`. All six cases / 36 timer-on/off records pass their
identity, completed call/byte-count, finite-output and unchanged-mean checks;
all six byte-exact raw-log hashes match the manifest. One exclusive node,
serial np8/32 invocations, OMP1/OPENBLAS1, GCC12.2/Open MPI3.1.4, actual
MPI_THREAD_MULTIPLE, Release with assertions. No node was pinned.

Generated periodic quads, P3/intOrder5, perfect gas Newtonian/Fourier,
viscosity0.001, polynomial viscous gradients, five fixed Jacobi VR sweeps,
zero reconstruction stopping tolerance, positivity off and core/halo overlap
enabled. Warm workspaces precede three alternating timer-on/off trial pairs,
each with five complete Evaluate calls. The GNU link wrapper adds no production
profiling branch. It times the real synchronous publication call; it does not
estimate RHS fractions by multiplying a separate microbenchmark median.

Median milliseconds per RHS (fractions are observed rank-local extrema):

|np|cells/rank|limiter|RHS timer on|RHS timer off|publication max-rank|local publication fraction min/max|
|---:|---:|---|---:|---:|---:|---:|
|32|32|none|1.029|1.021|0.061|4.83% / 5.96%|
|32|32|WBAP|1.677|1.639|0.099|4.70% / 5.92%|
|32|5000|none|131.684|132.400|4.070|1.27% / 3.10%|
|32|5000|WBAP|194.247|195.635|6.107|1.66% / 3.14%|
|8|5000|none|110.239|110.713|2.440|1.73% / 2.21%|
|8|5000|WBAP|166.579|165.795|4.921|2.13% / 2.98%|

Each instrumented RHS performs exactly five publications without limiting,
or eight with descending P3 WBAP. Logical read+write bytes per publication are
`16 * owned_cells * 45`; actual bytes/call counts are checked in every batch.
They are logical traffic, not measured DRAM bytes. The timer covers acquisition,
workspace validation and row copying within CommitReconstruction. Flux-seed
copying and optional PP/reference copies are excluded. These fractions must
not be called all reconstruction-copy traffic. RHS times include its normal
numerical reductions and communication; no physical advancement is measured.

The warm guess evolves between repeated evaluations; opposite timer order and
fixed sweep count control work but do not make the instrumented/off outputs
identical. Existing independent reconstruction/FV oracles remain the numerical
gate; this probe establishes actual publication accounting and finite results.
Timer overhead is visible at small partitions and inconsistent in sign in large
ones; the report does not subtract it or claim sub-percent improvements.

Together with the [same-invocation copy candidate](../e3-candidate/README.md),
retain production copy ownership and row-copying for flow5. Publication is a
small fraction of these large flow5 RHS calls, and unconditional native copy
regressed the large flow5 case. Split-field RHS attribution remains open:
the earlier separate5+12 CFV microbenchmark is not a reacting RHS measurement.

`manifest.json.gz`, `raw-logs.tar.gz`, `summary.json`, and compressed execution
records retain source/build/binary/cache hashes, flags/linkage, exact commands
and final scheduler state. Login build used16 workers after regenerating the
existing cache: the first attempt found the new target absent, rather than a
compiler or numerical failure. No compute work ran on the login node.
