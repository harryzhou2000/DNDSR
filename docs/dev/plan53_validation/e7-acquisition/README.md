# E7: mesh acquisition cost attribution

**Measured recommendation: retain the current ownership API for this phase.**
Acquisition is measurable at very small partitions; a future implementation
should select required buffers before capturing their leases. The measurements
do not authorize borrowing unprotected storage or bypassing mesh/adjacency
objects. No production ownership change is included in this assessment.

Source **df5a3a7556ac184a7742cd684084474bb71077d6**, BSSCA job **12442096**,
one exclusive Slurm-selected `amd_512` node, `o2414`. GCC12.2, Open MPI3.1.4,
OMP1/OPENBLAS1, CPU Release. Job completed `0:0`,22:28:42–22:29:23 on November5,
2026 (UTC+8). Eight- and32-rank process groups ran serially on the same node,
with MPI core binding. No named-node constraint was used. Each row has four
normal/reversed phase-order trials,2000 entry repetitions and20 numerical
repetitions. Warmup, allocation counting, validation and timing are separate.

## Results

Median max-rank microseconds per call, four trials:

| np | cells/rank | Primary read | CellFlux read | RHS acquisition | auxiliary complete | reconstruction All |
|---:|---:|---:|---:|---:|---:|---:|
|32|32|8.110|7.986|28.758|25.851|65.120|
|32|5000|8.080|8.091|28.559|5175.604|9102.888|
|8|32|8.164|8.096|28.546|23.346|63.476|
|8|5000|8.205|8.096|28.085|4835.367|8558.496|

Primary and CellFlux reads both make **47 C++ allocations /5816 bytes** per
call. Metric read makes48/9624, basis read96/19248, RHS acquisition145/29968,
and auxiliary begin49/13552. These are instrumented `new` calls/bytes on the
calling thread, not live retained memory, hardware traffic, Eigen malloc, MPI
internal allocation or the number of individual leases. The values are
constant across the measured row counts; no mesh payload copies are involved.

Primary and CellFlux cost the same because `UnstructuredMesh::AcquireRead`
captures the snapshot before `Restrict` hides unused groups. Source inspection
confirms this boundary; it is not an inferred per-cell loop regression.
At32cells/rank, a single broad acquisition is about12% of the unsplit
reconstruction call; the empty Core call still costs about11us. At5000cells/
rank, the same8us is under0.1% of a reconstruction call. RHS acquisition is
an entry-only probe, not the whole RHS: do not interpret its28us as an RHS
time or add phase medians into a measured composed sequence.

The small periodic slab partitions have zero conservative core cells. Large
partitions have actual core/dependent work. Split numerical outputs match All
exactly and frozen mean/coefficient generations are unchanged in all16 trials.
`summary.json` retains phase ranges, core sizes and allocation counts.

There is a credible small-partition optimization target, but allocation count
alone does not justify a new ownership contract. Keep coherent publication and
requested-array leases. If revisited, a narrowly scoped requirement-aware
capture should preserve version rejection, hidden-group behavior, retained
scope lifetime and asynchronous completion and be tested against this source.
Do not treat the present result as an implemented speedup or a whole-solver
strong-scaling assessment. This measured retain/defer recommendation closes
E7's assessment requirement; it does not close E3–E6.

## Scheduler diagnosis and provenance

The original job requested only `amd_256` and stayed pending with `Priority`.
The visible80-node job12441217 already had `Nice=100000`; E7 had higher priority.
Slurm has `PrivateData=jobs`, so this was not the full global queue. Idle
resources and rolling start estimates did not prove why the global scheduler
deferred E7. Backfill runs every300s; no reservations were reported.

With the user's previously allowed alternative partition, at22:27:59 E7's
eligible list was changed to `amd_256,amd_512`, preserving ranks, memory,
exclusivity and time limit. It started on `amd_512` at22:28:42 via Backfill.
The80-node job was neither cancelled, held nor reprioritized by this work; its
existing Nice remained100000. Exact scheduler and mutation records are retained.
This establishes the successful alternative, not a particular hidden blocker.

The manifest, build/linkage/source pins,16 trial records, raw row/stdout/stderr
logs and hashes are retained compactly here. Source bytes can be recovered from
the commit. The run driver and submitted script remain in the isolated E7
assessment workspace. Every row exited0; all four row-log hashes match and all
declared numerical oracles passed. Preparation and numerical timing remain
distinct from scheduler runtime and queue delay.
