# Cauchy output contract

Cauchy resolves one immutable `OutputPlan` before stepping. The plan owns
cadence inheritance, deterministic names, exact physical-time targets and the
schemas advertised by manifests. Numerical operators publish typed progress;
they do not open files or select output cadence.

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

## Current internal-iteration boundary

Iteration and stage progress are fully represented in the detailed history and
console. Restartable checkpoints or visualization *during* a nonlinear
iteration are rejected during configuration rather than silently ignored. A
correct implementation must archive the active ODE stage/coupled-stage,
nonlinear state, pseudo-time controls, history fields and reconstruction needed
to resume that exact solve. Until that controller state is serializable, use
zero internal checkpoint/visualization cadence.

