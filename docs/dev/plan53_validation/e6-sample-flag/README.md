# E6: exact identity-transform candidate with per-sample metadata

**Do not accept this version as the final optimization.** It improves Jacobian
work but also regresses the unchanged auxiliary phase. A revised plan-level
classification must be assessed before making the retain/adopt decision.

BSSCA job `12442736` completed `0:0` in 54 seconds, on scheduler-selected
`amd_512` node `o2409`, one exclusive node. All twelve rows / 36 trials / 324
phase records pass. Frozen baseline `b6887a40` and candidate `8a230e82` have
identical benchmark source hashes, cache hashes and build flags. Both use
GCC 12.2 / Open MPI 3.1.4, actual MPI_THREAD_MULTIPLE, OMP1/OPENBLAS1,
Release with assertions. Each invocation is serial within this allocation;
candidate and storage orders reverse between workloads. No node is pinned.

The workload uses generated periodic quads, P3/quadrature order5, perfect-gas
Newtonian/Fourier flow5, viscosity0.001, frozen LLF convective+diffusive radii,
scalar diagonal, physical mass1000, inverse pseudo-time1 and residualScale1.
There is no chemical/RANS source in this benchmark. Existing np1/2/4 tests
retain independent frozen-flux, moving/physical/rotated/self-periodic and
logical-layout/source checks; these timings do not replace those tests.

Median maximum-rank milliseconds, 5,000 cells/rank (full trial ranges in JSON):

|np|storage|version|prepare|local SGS pair|prepare+inverse+1 pair|prepare+inverse+5 pairs|prepare+inverse+solve5|
|---:|---|---|---:|---:|---:|---:|---:|
|32|assembled|baseline|23.691|2.997|26.193|36.606|51.071|
|32|assembled|candidate|18.289|3.033|20.809|28.992|46.311|
|32|evaluated|baseline|14.580|20.586|35.785|118.368|173.883|
|32|evaluated|candidate|14.486|19.406|31.918|106.694|162.985|
|8|assembled|baseline|23.525|2.952|25.994|35.810|49.495|
|8|assembled|candidate|16.878|2.654|19.002|26.769|39.975|
|8|evaluated|baseline|13.169|19.781|32.749|111.019|160.911|
|8|evaluated|candidate|11.821|18.132|29.558|101.157|150.185|

The combined columns are directly measured sequences, not sums of medians.
The communicated solve completes exactly five SGS iterations each time with
zero stopping tolerances and no mid-lower-upper exchange. Its norms/entry
protocols remain enabled as required. Local pairs are a different workload.
The evaluated producer saves preparation time but loses in every measured
preparation+reuse sequence, including one pair. Keep assembled storage as the
practical choice for these workloads; no universal break-even claim follows.

Auxiliary preparation regresses on the large cases: at np32, assembled
7.967→10.038 ms and evaluated 8.190→10.333 ms; at np8, 7.553→8.869 ms and
7.643→8.577 ms respectively. The candidate changed the `Sample` layout by
adding a boolean that the auxiliary traversal does not use. This is a plausible
cache/layout effect, not proven causal attribution from these timings alone.
The follow-up removes all per-sample/per-point classification fields and uses
one exact plan/geometry flag. Any rank with a rotated sample keeps the general
path for its whole plan, avoiding a new per-quadrature data footprint.

At np32, maximum-rank post-work RSS is 111.0/110.0 MiB for assembled
baseline/candidate and 119.5/120.4 MiB for evaluated. These observations include
the mesh, metrics, reconstruction plan and workspaces, and are not exact
snapshot allocation counts. Linux VmRSS and getrusage high-water are reported
separately; their accounting can differ. Opposite-mode oracles are allocated
only after memory/timing observations, so they do not contaminate these rows.

After timing, complete owned matrix actions and SGS results are compared
elementwise with the other storage mode. All values are finite; means retain
their generation. The twelve archived raw-log hashes match the manifest.
Required build/linkage/source pins and final scheduler state are retained.
Both first-attempt login builds completed with16 workers. A model-capacity
error happened after build/freeze but before submission; the controller
verified terminal builds and absence of any prior E6 job before submitting
once. It did not rebuild or restart a running job.

No full nonlinear-stage, reacting-source, physical trajectory, GPU, or general
geometry speedup is claimed. Small-partition results remain in `summary.json`.
E6 remains open for revised-candidate acceptance and final recommendation.
