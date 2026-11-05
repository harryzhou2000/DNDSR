# Coarse pMG operator and distributed failure audit — November 5, 2026

The preceding NACA0012/cylinder campaign exposed two distinct problems: the
bottom Cauchy p-level lost viscous/source gradients, and an exception escaped a
rank-local solve into an unrelated simulation error reduction. The latter left
one rank in simulation agreement while another remained in the linear solver.
These fixes address operator semantics and exception propagation. They do not
establish physical pMG acceleration or identify the original cylinder's exact
nonphysical face state.

## Findings and changes

1. **Bottom operator.** `PolynomialMultigrid::CoarseSettings` now centralizes the
   hierarchy controls. Both lower levels prepare direct conservative gradients;
   the bottom selects `firstOrderConvection`, suppressing reconstructed face
   values while retaining viscous derivatives and volume-source reconstruction.
   This follows Euler's `RHS_Direct_2nd_Rec_1st_Conv` distinction. Finest defaults,
   gas/RANS/source dispatch, layouts and the FAS recursion are unchanged.
   Restart comparison treats a missing switch as its original false default;
   explicitly changing it to true remains a numerical-method mismatch.

2. **Unsafe outer catches.** Simulation construction, interface preparation,
   domain solves, multi-domain I/O and the production domain visitor no longer
   catch arbitrary failures and attempt another control reduction. An unexpected
   exception escapes to `MPI::Abort`, which reports it and terminates without
   agreement or finalization. The scalar placement mock follows the same rule.

3. **Numerical retry is explicit.** `StepRejected` means the domain's acceptance
   decision is already global and all relevant operations completed. Simulation
   promotes that status to its control communicator before retry. The executable
   catches only this type for `maxRetries`; arbitrary runtime/backend/access
   errors never become smaller-dt retries. Explicit conservative blending and
   separate RANS repair are unchanged. Python exports the same exception class.

4. **Publication preflight.** Accepted-state and checkpoint reservation perform
   only memory/lease operations under a single control-communicator LocalStage.
   Reconstruction reservation supports an explicitly local mode for composition
   into that preflight. Every destination is reserved before any domain commits;
   unused reservations cancel without advancing generations. Collective file
   reading and equation solves remain outside that catch.

5. **Nested MPI in local checks.** Linear halo waits and the optional inter-scan
   exchange are outside local-algebra stages. Two-equation RANS mean-gradient
   binding/preparation is outside local nonlinear setup/update stages. Interface
   hooks own their communication and bounded local preflights; the scheduler
   marks failure and rethrows without wrapping a hook in another error reduction.

6. **Native Geom I/O and companion transfer.** HDF5 backend construction,
   open/close and native Array reads/writes are not local preparation. Per-rank
   JSON decoding remains eligible for coordinated local error handling because
   its codec performs no MPI. Shape checks follow file reads as separate local
   stages. Reorder companion callbacks may communicate; their exceptions escape
   directly rather than entering an outer readiness reduction.

7. **Runtime completion and cleanup.** DynamicRowTransfer Commit no longer
   catches an MPI-containing Wait and attempts error agreement. It publishes
   locally after successful completion; success on one rank is not distributed
   acceptance. Active MPI-transfer or pending MPI-progress destruction during
   exception unwinding aborts instead of risking unmatched waits before the
   application reaches its fatal handler. Packing errors after receive posting
   also abort immediately: matching sends might not have been posted. Normal
   RAII retirement still drains.

8. **Retained catches.** Array/metadata/lease preflights, local topology scans,
   local schema/codec validation, cancellation of unused writers and immediate
   rethrows remain valid. CPU/CUDA launchers retain asynchronous errors for the
   controller's completion observation. These are not distributed recovery.
   CFV mathematical setup/kernels keep their existing local-work/transfer split;
   the audit found the unsafe compositions in their Cauchy/Geom callers.

## Validation design

- Distributed coarse-operator regression exercises both flow5 and logical
  equation RHS profiles: bottom convection equals mean-only convection in the
  inviscid case, while nonzero direct-gradient diffusion remains distinguishable
  from the previous constant-reconstruction operator. Existing pMG tests retain
  nonconstant RANS/mixture source, selected-component, layout and DITR coverage.
- Fatal subprocess fixtures inject a factory exception with a peer deliberately
  blocked in MPI, and unwind with an active native transfer. A passing wrapper
  requires the expected diagnostic, a nonzero exit and completion before its
  timeout. An arbitrary crash or a timeout does not pass.
- Existing cross-domain restart tests retain the blocked-writer/no-partial-commit
  oracle, including disjoint and colocated communicator groups. Interface tests
  coordinate their injected local failures inside the hook before retrying.
- Native mesh archive tests retain HDF5 and per-rank JSON round trips, repartition,
  malformed schema and retry checks. Transient transfer tests distinguish a local
  reducer failure from a collectively agreed setup error.

The caller obligations are in [RuntimeCallerContract](../architecture/RuntimeCallerContract.md#distributed-exceptions).
No MPI process-failure tolerance, recovery from a missing rank, or CUDA/NCCL
hardware-error recovery is promised. No additional residual-loop Allreduce is
introduced for fatal-error reporting. Numerical campaign follow-up is separate
from these correctness regressions; the released IV campaign remains untouched.

## Verified local gates

Final refreshed C++ builds passed 32 focused CTest registrations at np2/4,
plus the production frontend np2 gate. The set covers both coarse RHS profiles,
linear/implicit stepping, multi-domain publication, checkpoint compatibility,
native mesh reordering/transitions/archives, transient transfers and both fatal
subprocess modes. Final batch log: `/tmp/cauchy-pmg-failure-final2-tests-20261105.log`.
Earlier failed batches are retained; their fault-injection callbacks were
migrated to own their bounded local error preflights rather than relying on
an outer catch around potentially collective work.

After rebuilding and installing DNDS/Geom/CFV/Cauchy Python bindings, the np2
selection passed three tests on each rank: the distinct numerical rejection,
configured implicit step and conservative SSPRK path. No MPI skip is credited.
Python log: `/tmp/cauchy-pmg-failure-python-tests-20261105.log`.
Local runner uses OpenMPI 5.0.9 and single OMP/OpenBLAS threads. This is focused
CPU validation; no new CUDA/NCCL failure-recovery claim is made.
