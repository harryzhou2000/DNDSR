# E3 matched coefficient-publication attribution

**Baseline attribution completed; ownership/copy recommendation remains open.**
Source43e099fe6e74ee26b6ac11342df17b62d1783b0a, BSSCA job12442563,
Slurm-selected amd_512 nodeo2415, one exclusive node, np8/32 process groups
run serially. GCC12.2/Open MPI3.1.4, OMP1/OPENBLAS1, Release with assertions,
actual MPI_THREAD_MULTIPLE. All18 rows/72 trials/288 phase records pass their
input/allocation/binding/numerical/shape/hash checks; scheduler COMPLETED0:0,
2m16s. Four alternating forward/reverse phase orders,500 warmed copy calls and
five numeric calls per trial; timing reports max-rank wall time per call.

Representative median microseconds (all rows/ranges are in `summary.json`):

|np|cells/rank|degree|fields/layout|production commit|bare row copy|sweep without commit|top-order WBAP without commit|
|---:|---:|---:|---|---:|---:|---:|---:|
|32|32|3|flow5/AoS|9.50|0.88|64.76|62.72|
|32|5000|2|flow5/AoS|75.86|72.86|3833.25|5924.08|
|32|5000|3|flow5/AoS|172.70|163.11|9416.73|9279.83|
|32|5000|2|separate5+12/AoS|2622.46|2575.90|14292.66|25505.16|
|32|5000|3|separate5+12/AoS|5137.68|5088.47|24407.91|32303.85|
|8|5000|3|separate5+12/AoS|3872.70|3833.79|22509.20|29866.88|
|32|5000|3|separate5+12/SoA|5075.64|5030.52|30575.14|36467.35|

Small partitions mostly expose publication's acquisition/validation cost.
At5000cells/rank, flow5 commit is1.7–2.0% of one reconstruction sweep, while
the separate5+12 cases cost14–21%. The bare-copy comparison shows that the
latter is predominantly copying, not only mesh acquisition. This warrants a
bounded local-copy candidate before changing ownership. It does not establish
a whole-RHS fraction or a full reacting-source cost: these are ordinary CFV
fields, with no Cantera equation evaluation. SoA here is not a tuned layout
competition and does not supersede the earlier layout study.

All copy phases retain the same physical allocations and persistent bindings.
After timing, a real halo uses the retained binding, and owned/ghost values are
checked against a global-ID formula. Compute-only phases preserve coefficient
input generations; all phases preserve means and allocation identity. Finite
numeric outputs complement the established independent reconstruction/WBAP UTs.
The logical read+write volume is16*N_owned*N_modes*sum(component widths) per
commit. It is not measured DRAM traffic. Reset, setup, warmup and checks are
excluded from timing. A single top-order WBAP pass is not a whole limiter cycle.

## Production copy inventory and limits

In flow5 `PrepareReconstruction`, each ofK completed VR sweeps calls
`CommitReconstruction` once. WBAP calls it once for each orderp..1, givingp
additional commits. Thus fixedK/p execution hasK+p commits for one physical
field; separate physical fields multiply those calls byF. Warm unlimited
coefficients are copied once to the separate flux coefficient field before
limiting/PP (`Workspace::PrepareFlux`). Final quadrature PP may use additional
reference/temporary copies according to its configured degree/mean policy.
These are distinct from ordinary mean snapshots and cannot all be erased by a
single handle swap. An externally retained Array alias or persistent binding
must keep referring to its original allocation.

Do not estimate complete-RHS savings by multiplying these isolated medians:
cache residency, work order, early reconstruction stopping and limiter passes
matter. Complete-RHS fractions and an actual retain/change recommendation
remain required for E3 closure. No production change or buffer swap is made by
this baseline. A follow-up retains allocations and compares Runtime whole-array
copy with the row loop in the same invocation/allocation; it is not compared
against these other-node medians as a measured speedup.

## Provenance

`manifest.json.gz` retains source/executable/cache hashes, exact row commands,
environment, actual thread support and all trial records. `raw-logs.tar.gz`
contains byte-exact row logs; all18 member hashes match the manifest.
`build-batch-logs.json.gz`, flags/linkage/build hashes and final accounting retain
the executed environment and two compiler-killed attempts before a successful
third incremental login build with16 workers. No numerical run occurred on the
login node. Core binding is logged in every row. The four permitted partitions
were eligible; no node was pinned and no other job was changed.

The local recorder pilot first selected a nonexistent checkout-local Python;
it failed before any row. The canonical venv then completed both pilots. That
launch mistake and subsequent good gate remain separately recorded in the
assessment workspace. This matched campaign has no failed/capped rows.
