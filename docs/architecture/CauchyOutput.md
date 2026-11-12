# Cauchy output contract

Cauchy constructs one `OutputCoordinator` before stepping. It owns the
immutable `OutputPlan`, progress controllers, logger, append cursors and
observer lifetime; the executable retains only timestep selection, retry and
accepted-step control. The plan owns cadence inheritance, deterministic names,
exact physical-time targets and the schemas advertised by manifests. Numerical
operators publish typed progress; they do not open files or select output
cadence.

## Artifact bundle

For prefix `run`, domain `D`, accepted step `S` and exact physical time `T`, the
frontend writes:

```text
run-resolved.json
run-run-manifest.json
run-D<domain>-history.csv
run-D<domain>-history_1.csv ...
run-D<domain>-S<step>.vtkhdf
run-D<domain>-T<round-trip-time>.vtkhdf
run-D<domain>-S<step>_surface_rank<domain-rank>.csv
run-D<domain>-S<step>_<stage>_<iteration>.vtkhdf
run-D0-S<step>_<stage>_<iteration>.h5  # exact internal, single domain
run-S<step>.h5                 # multi-domain simulation archive
run-D0-S<step>.h5              # single-domain archive
run-S<step>-manifest.json
run-summary.json
```

`-final` replaces the step/time token for the unconditional final checkpoint
and visualization. A snapshot manifest records the accepted step and time plus
checkpoint, volume and surface-shard paths. The run manifest records the
detailed-history schema, status enumeration and logical component names once,
so CSV rows remain compact numeric data.

The history is written with `DNDS::CsvLog` and a typed `CsvSchema`. Segments
repeat the header and contain nonlinear-iteration, stage-end and accepted-step
rows. `stage=-1,iteration=-1` denotes an accepted step; `iteration=-1` with a
nonnegative stage denotes a stage end. Empty cells mean that a quantity does
not apply; they are not serialized as false zero. Console messages and CSV rows
are constructed by the same `ProgressOutputController` reports.

Each domain's communicator root writes its history and console progress.
In dislocated placement this can be a nonzero world rank, so the output
coordinator explicitly enables its local console sink. World rank zero also
retains its operational logger. No residual gathering or new collective is
needed for printing. Either the step or implicit console interval can enable
the sink; their individual cadences still control which records are printed.
Monitoring must read all numbered CSV segments, not only the initial file.
Iteration `wall` records observation time, not attempt start or flush time.
Captured rows are emitted at each existing quiescent nonlinear boundary even
if internal checkpoints/visualizations are disabled. This local publication
adds no collective; production sink exceptions propagate to the abort handler.

## Controls

The main controls are:

```text
output.checkpointInterval
output.visualizationInterval            # -1 inherits checkpoint cadence
output.checkpointTimeInterval
output.visualizationTimeInterval        # -1 inherits checkpoint time cadence
output.endpointSnapRelativeTolerance
output.consoleInterval
output.historyInterval
output.historySegmentRows
output.historyPrecision
output.implicit.consoleInterval
output.implicit.historyInterval
output.diagnostics.{perRank,level,maxBytes,rotatedFiles}
```

The deprecated `output.interval` spelling is accepted only when the new
checkpoint spelling is absent. `spdlog` provides synchronous rank-zero console
and optional rotating per-rank operational diagnostics. It is not used for
scientific CSV, MPI aggregation, checkpoints or recovery.

## Exact-time stepping

Positive time cadences and `time.tEnd` are mandatory endpoints. Event times are
computed from an integer ordinal and interval. With proposed step `dt`, current
time `t`, target `target` and configured relative tolerance `r`, Cauchy snaps
when

```text
target - t <= dt + r*dt.
```

The submitted step uses exactly `target-t`; after acceptance the clock is set
to `target`, rather than incremented. If an unsnapped `t+dt` cannot advance the
floating-point clock, the frontend fails explicitly. Step- and time-based
triggers at the same accepted state are coalesced into one snapshot bundle.

## Internal-iteration boundary

Iteration and stage progress are fully represented in the detailed history and
console. A positive `output.implicit.visualizationInterval` writes the current
trial at a quiescent nonlinear observation boundary. The observer copies the
trial mean and the current warm reconstruction into detached storage, then
prepares output fields with a separate workspace. Its halo publication,
reconstruction iterations and diagnostics therefore cannot rebind or mutate
the active time-stepper state. Each per-domain product has an
`internal-snapshot.v1` manifest whose `restartScope` is `visualization_only`
when no checkpoint is written at that boundary.

A positive `output.implicit.checkpointInterval` enables exact internal
checkpoints for a single implicit domain. The archive stores the accepted state
and history, active ODE method payloads and cursor, current nonlinear controller
JSON, trial state, endpoint reconstruction publications and limited-VR history.
DITR additionally stores its midpoint state/rates and independent midpoint
reconstruction publications. BDF2, serialized ESDIRK and DITR restore the
active solve at that boundary rather than restarting its stage; continuation
tests compare the following nonlinear result and accepted step with an
uninterrupted run.

The active time payload uses schema 2 and includes DITR's variant, stage
fraction and transformed-equation weight. Exact internal restore rejects a
different mathematical signature before replacing live stage state; it never
reuses saved stages/rates with new coefficients. Older active payloads without
this signature are rejected. Accepted-step restart remains separately flexible
because it starts a new step rather than resuming an existing stage equation.

Exact internal output is rejected for SSPRK3 and, currently, for multi-domain
simulations. A domain callback cannot publish a truthful simulation-wide exact
snapshot while another colocated or dislocated domain is at a different
nonlinear boundary. Multi-domain exact restart therefore requires a future
simulation-level coordinated quiescent boundary; Cauchy fails configuration
instead of emitting a partial archive. Internal visualization remains
available independently for every implicit domain.

An internal observer is a domain-quiescent callback, not a simulation-wide
barrier. It flushes and broadcasts only that domain's CSV cursor on the domain
communicator. It must not enter simulation-control collectives: dislocated
domains may reach different nonlinear iteration counts. Accepted-step and final
snapshots are coordinated by `OutputCoordinator` after `Simulation::Advance`
and may therefore capture all domain cursors with simulation-wide collectives.

Checkpoint controller metadata includes exact-time event ordinals, resolved
cadence and typed CSV segment cursors. Continuing with the same output prefix
validates and appends at those cursors without duplicating the initial/accepted
row. Choosing a new prefix starts a new output bundle from the restored
numerical state.

