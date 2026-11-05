# E3: matched native-array copy candidate

**Recommendation: retain the production row-copy implementation.** The candidate
improves the measured split fields but regresses the large flow5 case. This is
not evidence for an unconditional replacement or a changed ownership contract.

Job `12442622` completed `0:0` in 1m58s on Slurm-selected `amd_512` node `n1810`,
source `e60b85ba9de090d06f92a2d4c5705fd806d32e2c`. One exclusive node,
OMP1/OPENBLAS1, serial np8/32 invocations, GCC 12.2/Open MPI 3.1.4, actual
MPI_THREAD_MULTIPLE, Release with assertions. The eligible partition list was
amd_256/amd_512/amd_1T/amd_2T; no node was pinned or other job modified.

The candidate calls Runtime `Array::CopyFrom().Wait()` for matching unpadded
physical layouts. It retains allocations and persistent bindings. Production
and bare-copy controls run in the same executable/invocation as the candidate,
with four alternating forward/reverse trials, 500 warmed copy calls and five
numeric calls per trial. Values are median maximum-rank microseconds per call;
split rows include both physical fields. Min/max trial values are retained in
`summary.json` rather than hidden behind the medians.

|np|cells/rank|degree|fields/layout|production commit|bare row copy|native array copy|native/production|
|---:|---:|---:|---|---:|---:|---:|---:|
|32|32|3|flow5/AoS|9.26|0.67|0.52|0.06|
|32|5000|3|flow5/AoS|178.19|150.05|685.08|3.84|
|32|5000|2|separate5+12/AoS|2560.70|2505.30|1207.98|0.47|
|32|5000|3|separate5+12/AoS|5150.40|5105.38|3054.81|0.59|
|8|5000|3|separate5+12/AoS|3650.07|3619.55|2372.79|0.65|
|32|5000|3|separate5+12/SoA|5080.10|5038.48|3055.84|0.60|

The native path validates physical layout and copies the whole allocation
through the Runtime buffer-copy facility; production validates its mesh/CFV
workspace and copies logical rows. These matching unpadded fixtures transfer
the same logical coefficient payload, but do not have identical prologues or
copy-call granularity. The result combines those differences and cache/library
behavior; it does not isolate a universal cause of the size-dependent crossover.
SoA is a supported-layout check, not a tuned layout competition.

All six rows / 24 trials / 120 phase records pass finite/work-count, unchanged-mean
generation, retained-allocation and post-copy real-halo payload checks. All six
byte-exact archived log hashes match the manifest. Repeated operations and the
retained original binding are exercised; there is no handle swap. The two
numeric phases retain the established reconstruction/WBAP behavior. This is
ordinary CFV field traffic, not a reacting RHS or a solver-trajectory result.

The [baseline report](../e3-matched/README.md) records actual copy multiplicity
and logical byte accounting. Do not compare its other-node timing medians as
a speedup of this candidate. Complete-RHS copy fractions remain unmeasured;
E3 stays open for that attribution even though this candidate has a retain
decision. No production change is warranted by these measurements.

`manifest.json.gz`, `raw-logs.tar.gz`, independent verification, final scheduler
accounting, exact submission, flags/linkage, build log and hashes preserve the
bounded execution. No numerical work ran on the login node; its incremental
build used 16 workers. Raw outputs remain in the isolated assessment workspace.
