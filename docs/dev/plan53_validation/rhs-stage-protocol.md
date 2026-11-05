# E5 composed RHS and E6 nonlinear-update protocol

These opt-in executables complete the measurements left open by the mock
scheduling and frozen-operator studies. They are `EXCLUDE_FROM_ALL` targets,
not new routine tests, and do not change a production library or solver policy.

## E5 actual work and ownership

`cauchy_benchmark_rhs_execution` executes the public equation FV evaluator with
P3, integration order five, five fixed Jacobi sweeps, and either no limiter or
WBAP. The two profiles are Newtonian/Fourier perfect gas and actual h2o2
reacting equations with separate flow5 and nine independent-species fields.
The mixture has ten species including the dependent species. Both use the
supported logical-equation Rusanov profile, no PP and no face fallback.

Each rank reserves two different physical cores in every comparison. The
`production` control uses the original Direct/inline binding; its second core
is idle. The three adapter modes use caller computation/MPI, a compute worker
with caller MPI, or caller computation with one MPI worker. Direct and
HostPacked select the existing Runtime transports. The adapters preserve the
row route, payload layout and persistent request implementation. They do not
combine fields, reorder cells or implement a new packing algorithm.

The single MPI worker owns all active request starts, polling, packing and
unpacking. Hot controller MPI calls are synchronous RPCs to that same owner;
numerical kernel callbacks contain no MPI. Setup/resource calls on the
controller occur without active payload requests. The benchmark requests and
checks SERIALIZED support, rather than assuming concurrent MPI calls are safe.
All fields, workspaces and operations drain before the worker, then the main
thread finalizes MPI. This is a bounded assessment adapter, not a general
production communication executor.

An original Direct/inline workspace supplies the independent complete-field
comparison for every adapter. Resetting coefficients is outside timing; fixed
work, finite output and unchanged owned mean versions are checked. Instrumented
and uninstrumented trials alternate. Only uninstrumented elapsed times support
net-benefit claims. Instrumented records retain actual staging bytes/calls,
MPI pack/unpack/post/progress costs, submission/queue/wait times, producer/post/
receive/ghost-readiness timestamps and selected VR-core intervals. The receive
timestamp identifies the receive request group, not the first completed send.
Per-rank maxima of phase times may come from different ranks and are not added.
Intersection of pending communication intervals with VR-core intervals proves
host scheduling concurrency, not autonomous internal network progress.
The adapter conservatively waits for full transfer completion before declaring
destination readiness; it is not an early-receive implementation.

## E6 complete nonlinear update

`cauchy_benchmark_nonlinear_stage` runs the public equation nonlinear solver,
not a synthetic matrix-vector loop. It uses the same generated P3 profiles,
one nonlinear correction, five SGS iterations, two residual evaluations and
physical mass 1e6. Source terms and source Jacobians retain their existing
reconstructed-quadrature producers. Means and reconstruction are reset outside
the timer, and the manufactured source remains unchanged. The energy defect
must decrease, repeated complete fields must agree, and an untimed solve with
the other assembled/evaluated storage mode supplies a full-field complement.
Zero tolerances deliberately retain a capped update; this is not physical-step
convergence or a physical trajectory. Existing nonlinear phase clocks and
complete call time are recorded separately.

## Bounds and records

Build only these two targets. First use local np2 generated fixtures, including
the original-production control. The checked recorder requires a clean exact
source pin, binary/cache/source/mechanism hashes and OMP1/OPENBLAS1. Its local
`--pilot` has 18 serial invocations. The matched BSSCA run has 25 serial
invocations on one exclusive unpinned node: np32, two cores per rank, 180G,
15 minutes. All four user-approved AMD partitions are eligible. Three RHS
profiles (small flow5, large flow5, large split-mixture/WBAP) each have one
production control and six adapter/transport combinations; four large actual
nonlinear invocations compare the two storage modes for each physics profile.
Each invocation has two repetitions and three trials, with a 180-second row
bound and 850-second campaign bound. Stop at the first failed row, retain it
and all unrun rows. Preserve failed local startup records as well.

No other implementation checkout, MG/IV campaign, 80-node job, dependency
installation or production policy is part of this protocol. Initial local
startup used an unsupported face-fallback default and failed explicitly; the
fixture now selects the supported profile instead of weakening validation.
