# Extracted CPU operator diagnostic

CS performance readings are provisional. Canonical comparisons execute matched
baseline/candidate members serially within one exclusive single-node BSSCA
allocation, with frozen source, controls and workload; no named-node pinning.

The fused mean-property trial retains four assembly batches, but changes their
meaning: `jacobian-freeze` covers mass/frame initialization; `jacobian-source`
includes mean source derivatives, owned EOS publication and remaining ghost or
source-free EOS preparation. `jacobian_source_scope` explicitly identifies this
combined scope. It cannot be subtracted as a source-only accounting correction.
Compare the complete matrix/linear/update parents for fusion acceptance.

Optional `DNDS_CAUCHY_DIAGNOSTIC_AUX_TRACE=1` records mean-transfer launch,
owned boundary completion, wait and dependent auxiliary chronology in the
private extracted pipeline. It is CPU-inline, unsplit and single-node only.
Events use the OS steady clock in nanoseconds, never an assumption that MPI
clocks agree. Capacity is reserved before timing; event capture does not
allocate or communicate. Overflow fails explicitly after timing. Trace gather
and output also happen afterward. Every entry has a per-level ordinal and a
full-evaluation/correction-only path. `compute_end` marks completion of Apply,
before publication. Missing auxiliary inputs and the two-equation gradient
path produce no entry; this trace initially targets the maintained SA case.
`BROAD_AUXILIARY_TRACE` is causal evidence, not an accepted throughput result;
compare traced/untraced parent costs to assess perturbation.

Direct-FV component probes perform untimed gradient-publication checks before
their numerical oracles: ready repeated refresh does not rewrite the target,
a nonzero coefficient change refreshes the analytic physical gradient, and an
active input writer is rejected. The original coefficient and gradient values
are restored exactly before timing. `EXTRACTED_GRADIENT_REFRESH_AUDIT` records
these checks; no new hot-loop counter, MPI agreement or altered source-gradient
policy is introduced. Halo-only and variational probes keep their usual inputs.

`BROAD_WORKLOAD` includes untimed `rank_work_columns` and flattened
`rank_work_rows`, one ten-column row per MPI rank: owned/ghost/core/
halo-dependent cell counts, face incidences, canonical faces, canonical
boundary/wall faces, canonical face quadrature visits and source quadrature
samples. These describe the actual finest FV prepared sets, rather than
assuming equal cell counts imply equal work. The single gather precedes all
timed calls and resets; no counter or extra collective is inserted into a
numerical loop. Coarse operators may use different quadrature, so the finest
source count is a topology/work descriptor, not a coarse visit count.
Use at least three ranks for correlation analysis; an np2 pilot validates
metadata shape and unchanged output only. Correlation of cumulative rank
durations can indicate arrival imbalance but does not identify its cause.

The private pipeline copy also counts correction-only source refreshes after
the existing stale-publication guards. `source_refresh_rank_rows` reports six
integers per rank: P3/P1/P0 refresh calls, then P3/P1/P0 source-point visits,
summed over `source_refresh_repetitions`. The extra gather happens after elapsed
timing. There are no point-loop counters or extra timed collectives. An untimed
positive control forces exactly one stale source publication and validates the
counter before resetting the timed fixture. Zero counts then mean this specific
path was unused by the observed workload; they do not show that all source work
or source quadrature was absent. Euler has no corresponding instrumentation.

The private `ExtractedEquationJacobian.cpp` copy observes four disjoint
assembly batches: input validation/access/allocation setup, thermodynamic
freezing and diagonal initialization, source Jacobian, and face-block filling.
The extractor inserts only external scope declarations and checks exact
round-trip recovery of the original source. It never times each face or
component separately, changes production timers, or inserts communication.
These batches are children of the existing matrix range; the reducer checks
their summed duration against that same rank/level's parent before averaging.
Do not add independent maxima or count them again in composed totals.

## Broad production-process batch (preferred starting point)

```sh
cmake --build build-refactor -t cauchy_diagnostic_driver_phases -j16
OMP_NUM_THREADS=1 mpirun -np 2 \
  build-refactor/test/cpp/cauchy_diagnostic_driver_phases \
  /absolute/run/euler-naca.json /absolute/run/cauchy-naca.json 5 3
```

This reports four large blocks from the same production correction: auxiliary preparation;
boundary/reconstruction/limiting/RHS; residual assembly through matrix assembly,
ILU correction and final update repair; and a directly measured composed
correction. It uses the maintained real
P3 SA case, one VR sweep and ILU(2), with all native production wrappers and
Euler's actual driver callbacks. `extract_driver_diagnostic.py` preserves the
original callback definitions and replaces only the time/output loop with a
synchronous diagnostic hook. The hook uses actual BDF1 `Step`, including
residual/history arithmetic. `extract_pipeline_diagnostic.py` inserts external
scope callbacks into a private FV source copy. No numerical call is removed
or reordered and no production timer counter is consumed. The ordinary native
`CorrectOnce` remains in its production library.

Native FV timing encloses aux preparation; subtracting the external aux scopes
gives the remaining reconstruction/RHS cost. Implicit cost is the whole-call
remainder, retaining entry checks, residual arithmetic, factorization/linear/PP
reductions and update repairs. All local blocks belong to the same composed
call. Rank maxima can have different critical ranks and must not be added.
This partition requires CPU-inline, unsplit execution; subtracting concurrent
worker/device scopes would not produce comparable elapsed costs.

`CorrectOnce` means one nonlinear correction, not a complete nonlinear solve.
It omits stopping-norm reduction by design in production, including when used
as a pMG smoother. The harness must not replace a real convergence-loop timing
with this number. Hierarchy transfer, FAS arithmetic, norm reductions and
convergence-control overhead belong in the maintained end-to-end comparison.

The diagnostic resets means and reconstruction outside every timed repetition.
Each broad timed repetition advances the requested nonlinear updates from that
same initial state; it is not a converged maintained-case run. Output comparisons
report all six component norms/differences and check finite, deterministic
repeats; they do not assert bit parity across different reconstruction bases
or incomplete factors. The standalone face/source oracle below remains a
separate numerical check. Both paths share the same mesh/order and diagnostic
wall-distance samples; neither production distance provider is changed.

Euler retains its RHS-output halo; native output halos remain unspecified by
contract. Native spatial output stays integrated during timing; conversion
to per-volume values is only for the untimed comparison. Native entry checks,
eager-engine orchestration and PP remain part of measured production work.
Independent phase timings must not be summed to claim a composed total.

The hierarchy mode appends `2 4` (local gate) or `2 32` (canonical batch) to
the arguments. It runs the actual pMG controller, including finest stopping
norms, restriction/tau assembly and independent four-sweep coarse levels.
It retains production wall-distance providers; their differences are reported
rather than erased by the single-level input alignment. Per-level RHS counts
and per-rank times are gathered after timing. Native performs a final fresh
finest residual observation; Euler's stopping callback reuses its last rate.

After the broad hierarchy localized excess time to the implicit remainder,
external scopes now additionally report `matrix-linear` and `implicit-other`.
They subdivide the existing implicit block; they are not additional work.
Matrix/linear covers the actual Jacobian/matrix preparation and full linear
driver (including factorization, exchanges and requested defect corrections).
The remainder retains validation, defect/pseudo-time arithmetic, PP, nonlinear
norms, candidate publication and FAS transfer/control. The private nonlinear
source copy adds scopes only. No production arithmetic, reduction, iteration
or stream/halo boundary is replaced. Both source extractors reject changed
anchors; exact source/generated bytes are preserved with the run receipt.
The nonlinear copy is compiled as a separate translation unit, matching its
production boundary; `DiagnosticTiming.hpp` shares only clocks and counters.
The matrix/linear subdivision additionally reports `matrix` (Jacobian/matrix
assembly) and `linear` (full linear driver including factorization). Their
individual rank maxima must not be added to recover `matrix-linear`.

Candidate-acceptance attribution additionally separates `candidate-validation`
(the existing local candidate-check lambda) and `candidate-agreement` (its
existing two-value global minimum). Both are nested inside
`candidate-acceptance`; `mean-validation` is itself inside local validation.
The clocks add no barriers, MPI calls or changes to publication ordering.
Compare subranges on the same rank: a collective's elapsed time includes
arrival imbalance and is not a measurement of communication latency alone.

## Fine extracted operators (use after broad localization)

This opt-in executable isolates operators on frozen inputs. It is not a unit
test, a production solver or a replacement for convergence experiments.

```sh
cmake --build build-refactor -t cauchy_diagnostic_operator_phases -j8
OMP_NUM_THREADS=1 mpirun -np 2 \
  build-refactor/test/cpp/cauchy_diagnostic_operator_phases \
  /absolute/run/euler-naca.json /absolute/run/cauchy-naca.json faces 3 20 3
```

Arguments: Euler case, Cauchy case, phase (`faces`/`source`/`composed`/`halo`),
degree (`0`/`1`/`3`), repetitions per batch and trials. Use the maintained SA
setup with at least 50,000 global cells and 1,000 owned cells per rank; smaller
workloads fail before timing. There is no periodic/synthetic mesh mode. Both
operators share the actual distributed Euler mesh/order and its initial means.
P0/P1 use midpoint rules. Reconstruction coefficients and supplied gradients
are zero; reconstruction/gradient cost is a different phase. The production wall
distance providers remain unchanged. The diagnostic reports their raw difference,
then freezes the native mesh-derived cell/face samples into its Euler evaluator
so source and transport consume identical distance inputs. This explicit input
alignment does not establish equivalence of the providers.
At P3, corresponding quadrature wall samples are copied. At P1/P0, Cauchy's
single midpoint wall sample is replicated through Euler's retained high-order
diagnostic wall row, because its direct source reads that row's mean (`ig=-1`).
The raw wall-input discrepancy therefore also reflects different sampling
positions at low order; it is not an isolated provider-accuracy measurement.
Physics/scaling and BC inputs are resolved from the case files, including the
production generated-default/Euler-case merge.
Put inputs in a fresh run directory: Euler generates an adjacent default file
and uses a diagnostic output prefix beside the Cauchy input. Generated defaults
are not maintained or tracked source. All arrays/plans are constructed before
timing. Do not enable production
profiling environment variables. The driver records external whole-call
maximum-rank times, alternates solver order and checks output outside timing.

`extract_operator_diagnostic.py` copies the current Euler RHS body into a
generated subclass; only phase selection, frozen-gradient preparation omission,
the public physics accessor and internal-clock removal differ. A source hash
and unique extraction anchors make drift visible. The original P3 evaluator
entry independently checks the extracted composition. No production header is
modified. Cauchy's implementation TU is compiled directly into this diagnostic
so its existing workspace face/source methods can be invoked separately;
the static archive must not supply a second definition of that TU.

Face phase excludes volume sources, exchange, gradient construction, VR/PP and
output integrations. Source phase includes sampling and destruction-diagonal
preparation. Cauchy calls the production workspace's `AddMeanSourceJacobian`
operation after its quadrature RHS source. Its diagonal is one-point and
mean-state, while Euler's RHS prepares its diagonal on source quadrature.
Both visit counts are reported; this is not identical Jacobian work at P3.
An independent untimed call to Euler's point-source entry (`ig=-1`, mode1)
checks the Cauchy mean-state diagonal with the same mean wall input and zero
gradients. RHS and diagonal have separate error/magnitude bounds, preventing
a large wall-source diagonal from masking an RHS mismatch. The removed
`ransDestruction` RHS cache is neither recreated nor treated as the oracle.
Halo phase exchanges both means and nonzero sentinel coefficients.
The native residual's volume normalization is inside the timed call, so both
implementations return per-volume residuals. Native unique-face flux evaluation
is followed by signed cell gathering; Euler also traverses faces. Both visit
counts are reported explicitly. Earlier diagnostic receipts used a native
cell-incidence flux loop and therefore measured more face evaluations.
The zero-polynomial initial-state diagnostic does not establish the cost of
nonzero reconstruction, limiter activation or nonlinear convergence. Report
shape/work counts and numerical mismatches before drawing a speed conclusion.
