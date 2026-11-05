# E4/E5: grouped starts and CPU worker placement

**Keep separate direct transfers as the production default.** The opt-in
`TransferGroup` is useful for small messages with preflight enabled; these
measurements do not justify automatically grouping large transfers or adding
a production MPI executor. Both worker placements hide the bounded mock
communication cost. That is not yet a complete Cauchy RHS speedup.

Source `fc9ed384bdce682546ad1513316f09054715fd74`, BSSCA job `12442855`, one
exclusive, scheduler-selected `amd_512` node `o2409`. It completed `0:0` in
44 seconds. GCC12.2, Open MPI3.1.4/UCX, Release, OMP1/OPENBLAS1. All24 rows,
72 trials and raw-log hashes verify. Each row has one warmup batch and three
measured batches of ten repeated stages. Rows run serially in one allocation.
No node pinning or mutation of the independent80-node job was performed.

## Grouping

Two fields have5+12 components, multiplied by either one mean mode or nine
P3 reconstruction modes. They share an actual Runtime row route over a
periodic Cartesian rank graph, four peers per rank at np32. Same requests,
datatypes, buffers, field tags and payload bytes; grouping changes reservation
and preflight placement, not message count. All fields are ready before Begin;
this does not measure delay of an early-ready field waiting for another.

Median maximum-rank microseconds per complete communication-only stage:

|rows/rank; modes|transport|preflight|separate|grouped|
|---|---|---|---:|---:|
|32;1|Direct|on|31.76|25.28|
|32;1|Direct|off|19.96|20.40|
|32;1|HostPacked|on|31.54|23.70|
|32;1|HostPacked|off|19.53|20.32|
|5000;9|Direct|on|890.66|921.24|
|5000;9|Direct|off|916.53|911.67|
|5000;9|HostPacked|on|4358.25|4501.30|
|5000;9|HostPacked|off|4716.84|4529.82|

PMPI counts20 versus10 preflight Allreduces per ten starts with policy on,
and zero with policy off. Both paths start160 requests per batch; the packed
paths each make80 Pack and80 Unpack calls. The small checked improvement is
about20–25%; off-mode differences are small, and the larger cases do not show
a consistent grouping winner. Small-case order is separate/grouped; large-case
order is reversed. Only three trials were measured, so do not introduce a
size threshold from these observations. Optional message coalescing is untested.

## Worker placement

Every comparison reserves the same **two distinct physical cores per rank**,
32 ranks/64 cores in total, including the caller-only controls. Controller and
worker affinity and socket/core/NUMA IDs are recorded and checked for unique
physical cores. Caller-only controls leave the second core idle. This is not
evidence that dedicating cores is free in a fully occupied compute solver.

The owned-only mock stencil uses sqrt/exp arithmetic and independent output;
it reads no ghost rows. Communication uses the real persistent Runtime path.
Source and destination accesses coexist with the computation's input reads,
and exact independent rank/row/component payload and arithmetic checks run
after every timed stage. Frozen source generations are unchanged.

Median maximum-rank milliseconds,5000 rows/rank, nine modes, preflight off:

|transport|schedule; requested MPI level|complete mock stage|owned compute|MPI Pack|
|---|---|---:|---:|---:|
|Direct|caller; SERIALIZED|31.186|29.813|0|
|Direct|compute worker; SERIALIZED|30.241|29.968|0|
|Direct|MPI worker; SERIALIZED|29.855|29.846|0|
|Direct|caller; MULTIPLE|31.139|29.796|0|
|HostPacked|caller; SERIALIZED|35.088|29.800|0.714|
|HostPacked|compute worker; SERIALIZED|30.212|30.136|0.952|
|HostPacked|MPI worker; SERIALIZED|29.959|29.949|0.933|
|HostPacked|caller; MULTIPLE|34.988|29.845|0.774|

The compute-worker schedule launches core work before caller-side Begin and
Wait. The MPI-worker schedule moves the complete unchanged staging/Pack/post/
Wait/Unpack path to one worker while the controller computes. The controller
performs no MPI until it observes that task's completion; initialization,
finalization and MPI-owner destruction stay on the controller. Actual provided
levels are recorded. SERIALIZED suffices for this bounded nonconcurrent-MPI
program; it does not authorize concurrent MPI elsewhere in a production solver.

Relative to the SERIALIZED caller, compute/MPI workers reduce this mock stage
about3.0%/4.3% for Direct and13.9%/14.6% for HostPacked. The small difference
between worker placements gives no compelling reason to add a new production
executor. Pack itself becomes slower under concurrent computation here,
consistent with additional resource contention; the net stage still improves.
This is not an isolated proof of a particular memory-bandwidth mechanism.

## Attribution limits

Large receive payload is1,253,376 bytes/rank/exchange. HostPacked additionally
copies8,626,752 bytes of full source and destination staging per exchange;
Direct has zero explicit staging bytes. These are code-derived logical copy
bytes, separate from peer payload and MPI-internal datatype packing. Setup time
is recorded separately; request construction is excluded from stage timing.

PMPI clocks measure Pack, posting, Waitall, Unpack and preflight separately.
Full-buffer staging has byte accounting but **no separate time clock**; the
remaining Begin/completion time includes staging, leases and controller work.
The compute-worker `exposed_wait` field in the measured source is a zero placeholder,
not an independently measured wait. Do not use it as overlap evidence. The
follow-up recorder source computes the nonnegative communication tail from
actual core/communication completion timestamps; it does not rewrite this run.
Complete
stage and core/communication timestamps supply the scheduling observation.
The reported overlap interval is host interval intersection; it does not prove
that the MPI implementation progressed concurrently throughout that interval.
Phase maxima can come from different ranks and are not additive.

These measurements close the bounded grouped-start and worker-placement
comparison, with a retain recommendation. A composed-RHS net-benefit test,
separate staging clocks, different producer-ready times, cross-node behavior,
coalesced messages and a general MPI executor remain unqualified. Do not label
the full E5 end-to-end acceptance complete from this mock workload.

Compact manifest, raw logs, execution records, independent verification and
summary retain source/binary/cache hashes, exact commands, affinities, scheduler
state and phase ranges. No timing was inferred from job runtime alone.
