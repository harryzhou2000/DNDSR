# Native solver setup checkpoint

This checkpoint integrates native OpenFOAM Hex8 import, weighted KWAY/RB
partitioning, reader-owned serial distribution, elevated-node smoothing and
native C++ Euler initialization. CFV benchmark wrappers use retained semantic
CPU/CUDA scopes. Euler serial/distributed volume and boundary output uses native
geometry and separately owned solver fields. GN7 remains open: Python defaults,
retained native pair properties and retirement of legacy construction/edit
authority still require completion.

## Final acceptance

The CPU profile is Release with Open MPI 5.0.9 and OMP_NUM_THREADS=2. The focused
matrix passes 32 CTests across MPI 1/2/4/8: native smoothing (4), weighted graph
and preparation (8), OpenFOAM and compatible archive ingress (8), existing mesh
pipeline (4), CFV reconstruction (4), and Euler evaluator (4). These are focused
tests for this checkpoint, not a fresh execution of all earlier broad tests.

Use these final logs:

- `geom-native-batch5-smoothing-tests.txt`
- `geom-native-batch5-partition-tests.txt`
- `geom-native-batch5-ingress-tests.txt`
- `geom-native-batch5-pipeline-tests.txt`
- CFV passes in `geom-native-batch5-consumer-tests.txt`; that initial combined run
  also contains the subsequently fixed Euler fixture failure.
- `geom-native-batch5-euler-output-precision-tests.txt` is the final seven-case
  Euler suite, including all six ingress/output combinations: CGNS, matching
  archive and repartitioned archive, each with serial and distributed output.
  It checks cell/point geometry, exact binary density/flux fields, serial VTU
  density, persistent cell identity and unit volumes. Existing NS, SA, 3-D,
  wall-cache and LVR serial/MPI numerical oracles are retained.

All four CPU Python bindings were rebuilt and installed. The final profile
verifies eight installed/build ELF IDs and CUDA disabled. The final Python
matrix passes eight tests plus one expected skip at np1, and nine per rank at
np2/4/8, including weighted/RB ingress and the nonconstant EulerP RHS comparison.
Commands, return codes and paths are in `geom-native-batch5-python-verified-matrix.json`;
logs use `python-verified-np*.txt`.

CUDA CFV transfer/read tests pass on GPU 1
(`GPU-0848655f-4530-ea83-ca9b-c091c5c6f36a`). Both legacy and independently native
fixtures execute metric reads plus direct cell-node/face/neighbor/coordinate
reads through the embedded device mesh view. Native fixture legacy arrays are
empty. The final executable is identified in the artifact manifest.

**CUDA sanitizer is not clean.** The MPI-launched memcheck completes all
functional checks but reports two duplicate CUB kernel-registration API errors
(`EmptyKernel` and a Thrust fill kernel); no device-address violation is reported
in that log. Their cause is not established by this checkpoint. Do not turn
this into a zero-error sanitizer claim. The earlier singleton sanitizer launch
timed out before attaching and was terminated; both logs are retained.

## Failures retained and corrections

1. Native Euler setup exposed a direct `cell2face` access in the LVR test's PP
   verification loop. It now uses `CellFaces`; positivity and serial/MPI
   comparisons remain unchanged. The preceding evaluator tests passed.
2. `PrintData` still rejected native serial and boundary output. The new native
   output descriptors and field routes now support those paths; the guard was
   removed and the six-combination test added.
3. The first serial VTU test expected binary precision while inheriting the
   solver's five-digit ASCII setting. The fixture now explicitly selects 17
   digits and checks ASCII at 1e-13; binary fields remain exact. Logs preserve
   both failed assumptions before the configured-precision run.
4. Two Python negative tests treated RB/weighted partitioning as unsupported.
   Their invalid cases now use UNKNOWN/method 2, and positive weighted RB/KWAY
   metric/ID tests were added. An initial pytest invocation without --with-mpi
   skipped every test and is not acceptance evidence.

Native partitioning bounds integer weights to prevent METIS degree/cut overflow;
faces generated solely for partitioning follow cell ownership after distribution.
Existing independent face/edge ownership remains preserved. Missing/malformed
rank-local archives fail collectively and allow a valid retry. Historical
archives with duplicated original IDs remain explicitly rejected.

Smoothing keeps the established algorithms and their documented numerical
conventions. Replicated boundary spatial datasets remain an explicit scalability
limit; sparse ghost and remap construction does not allocate global-entity-sized
local tables. No solver trajectory, accuracy or performance claim follows from
these contract and numerical-kernel tests.

The source/binary manifest is captured before the formatting hook. Log copies
remove terminal escapes/NULs and trailing whitespace; normalization hashes retain
the original identities. Artifact presence alone does not establish execution.
