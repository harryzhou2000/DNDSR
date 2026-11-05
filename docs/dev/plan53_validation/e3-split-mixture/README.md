# E3 split-mixture publication inside the composed RHS

**Retain production coefficient copying and stable payload bindings.** Actual
publication is about0.5–1.7% of local large-case reacting RHS time in this
bounded assessment. A changed coefficient-ownership contract is not justified
by this fraction. This complements the flow5 result and closes the requested
representative separate flow/species copy attribution.

BSSCA job `12442922` completed0:0 in2m47s, one exclusive scheduler-selected
`amd_512` node `n1810`, source `f773d320`. Six cases/36 timer-on/off records,
all finite/generation/count/byte checks and exact raw-log hashes verified.
GCC12.2/OpenMPI3.1.4/UCX, actual MPI_THREAD_MULTIPLE, Release with assertions,
OMP1/OPENBLAS1,32/8 ranks with core binding. Compared process groups run
serially inside that allocation; no node constraint or independent-job mutation.

The existing real EquationSetup/FiniteVolume path uses two AoS physical fields:
flow5 plus nine independent species,14 total transported components. The
bundled ten-species `h2o2.yaml` hash is
`0efc6c52862741a29e0c29b65d979c7d8cb409db5282bca83b9c5437b3d8c8d4`.
Thermodynamics, mixture transport and chemical sources run inside Evaluate,
with explicit nonunit reference scales and1400K converted to code temperature.
Mean density has a small spatial perturbation; the fixture does not advance
chemistry or claim a physically converged flame.

P3/intOrder5 periodic quads, five fixed Jacobi sweeps, none or descending WBAP,
positivity off, polynomial viscous gradients, characteristic limiting off,
Rusanov flux and core/halo overlap on. Mechanism setup, workers, workspaces,
warmup and validation stay outside timing. Three timer-on/off pairs alternate
order, each with five complete RHS calls. The link-only publication wrapper
adds no production profiling branch.

Median maximum-rank milliseconds; fractions span observed rank-local values
over all three instrumented batches:

|np|cells/rank|limiter|RHS timer on|RHS timer off|publication|local fraction range|
|---:|---:|---|---:|---:|---:|---:|
|32|32|none|7.724|7.749|0.132|1.29–1.71%|
|32|32|WBAP|8.499|8.495|0.212|1.88–2.51%|
|32|5000|none|1115.269|1117.572|13.899|0.53–1.30%|
|32|5000|WBAP|1211.361|1202.662|17.258|0.63–1.74%|
|8|5000|none|1064.655|1064.091|10.087|0.57–1.12%|
|8|5000|WBAP|1146.885|1146.766|17.240|0.78–1.66%|

There are exactly10 publications per unlimited RHS and16 with WBAP, counting
both fields. An instrumented five-call batch therefore reports50/80 calls;
timer-off reports zero. Logical read+write traffic per RHS is
`16 * owned_cells * 9 * 14 * (5 or 8)` bytes. It is not measured DRAM traffic.
The wrapper includes CommitReconstruction validation/acquisition and copying;
flux-seed and PP/reference copies remain excluded. Fractions are measured per
rank inside the same RHS batch, not the quotient of unrelated phase medians.

Warm coefficients evolve between calls. Timer order and fixed work control
cost without making outputs identical between phases. Existing independent
equation/reconstruction/source/layout tests are the mathematical oracle; this
instrument establishes actual finite work and publication accounting. No
time-integration, whole-solver speedup, all-mechanism qualification or relation
between this reacting workload's absolute RHS cost and the flow5 cost is
inferred. Timer overhead/noise is retained rather than subtracted.

`summary.json`, `records.tar.gz` and `SHA256SUMS` retain exact commands,
source/binary/cache/mechanism hashes, flags/linkage, scheduler outcome, raw row
logs and trial ranges. The login-node build used16 workers. The independent
80-node job remained pending(Priority) at the final read-only snapshot and was
not cancelled, held or reprioritized. The same snapshot showed idle nodes in
amd_256/amd_512/amd_2T; availability is a dated observation, not a standing
scheduler guarantee.
