# Plan 61: current CUDA execution boundaries

Source inspection: 2026-11-14. This complements the chronological
[implementation inventory](plan61_cuda_inventory.md). It is not a claim that
the final physical acceptance campaign is complete.

## Configured path and public operator scope

`ResolvedExecutionPlan::Resolve` in
`Cauchy/Config/SimulationConfig_Validation.cpp` resolves the requested backend
before a simulation advances. CUDA requires combined AoS perfect gas, flow5
or SA/Wilcox/SST extensions, Jacobi VR and Jacobi or GMRES with Jacobi/AmgX.
Coarse pMG levels are checked individually. Unsupported CPU algorithms are
not run by copying device fields back to host.

The normal workspace builds its auxiliary request in
`FiniteVolume_WorkspaceFactory.cpp`, using `CompleteRHSAuxiliaryRequest({})`.
This selects resident mean/face/cell auxiliary arrays as required by the flux.
It does **not** select the optional public `DerivedFields` cache.

`DerivedFields` currently remains a CPU-only public facility (including its
queued refresh overload). `BuildAuxiliaryValues` and `BeginAuxValues` explicitly
reject a CUDA request containing it. It is not a hidden download or a reached
CPU leaf in the configured CUDA solver. This limitation also applies to Python
operator clients requesting that cache; the CUDA auxiliary arrays are the
supported route. A future general derived-cache port must include residency,
queued freshness publication and explicit consumers, not just a point kernel.

## Source-linked implicit/time path

| Orchestration / dispatch | Numerical work and device implementation | Host boundary retained |
| --- | --- | --- |
| `FiniteVolume_Evaluation.cpp` | Auxiliary preparation, CFV Jacobi/limiting, BC/PP and RHS operations are submitted on shared execution lanes; CUDA RHS dispatch is `FiniteVolume_RHSQueued.cpp` → `CUDA/FiniteVolumeRHSKernels*` | MPI provider readiness/progress, explicitly requested residual/diagnostic observations |
| `NonlinearOperators.cpp` and `_Positivity.cpp` | Defect/time-equation assembly, norm and mean-state statistics, volume scaling, pseudo-time/PP and accepted correction → `Implicit/CUDA/NonlinearKernels*` | Small summary reductions and nonlinear convergence decision |
| `FiniteVolume_Jacobian.cpp` | Device preparation branches before CPU view acquisition; device means/boundary/blocks and cuSOLVER batched-SVD inversion remain resident | Metadata/pattern setup and small status observation |
| `LinearSolver.cpp`, `LinearSolverAlgebra.hpp` | Jacobi and assembled/evaluated action; native managed DOF algebra; device-backed GMRES small-matrix operations | Host-controlled Arnoldi/stopping and MPI reductions; sync-free GMRES is not claimed |
| `PolynomialMultigridOperations.cpp` | Selected-component updates and frozen-component checks dispatch to `Multigrid/CUDA/PolynomialMultigridKernels.cu`; level vectors use native device algebra | Small frozen-check result; FAS recursion/control stays on host |
| `TimeStageOperators.cpp` | Physical dt bound, explicit blend/updates and extended RANS repair dispatch to `Core/CUDA/TimeStageKernels.cu` and device state kernels | Scalar acceptance/retry decision; method state/history ownership is shared |
| `InterfaceCoupling.cpp` | Prepared CFV donor sampling, device packet transfer, frame rotation and receiver publication via `InterfacePacketOperator` | CPU geometric stage matching, CUDA-aware MPI producer/completion bridge and publication boundary |

The first physical AmgX pilot exposed overlapping-resource ownership that the
original tests did not exercise. AmgX resource creation/destruction manages
global device pools, so independently owned resource handles invalidated each
other. DNDS now shares one resource per process/device across matrix owners;
an overlapping-instance/destruction test passes. Different-device concurrent
owners in one process reject explicitly, consistent with one rank per GPU.

The pinned provider supports native row-major 5x5 blocks in LOW_DEG aggregation
and block Jacobi. Flow5 now uses this path, as specified by Plan 61's native-block
preference; widths 6/7 retain scalar CSR expansion because LOW_DEG has no native
6/7 implementation. Scalarizing flow5 permitted cross-component aggregation
and gave poor finite-iteration cylinder behavior, including nonzero transverse
momentum. The native-block test preserves an uncoupled zero transverse equation;
its physical 20-update pilot is positive and far better behaved, but full AmgX
convergence is still being measured. Both earlier failures remain recorded.

The interface path is included. Final required physical cases are single-domain
distributed solves; that does not restrict the implementation to one domain.
Colocated/dislocated CUDA interface UTs exercise the repeated numerical path.
Current Cauchy configuration selects MPI-direct. Runtime NCCL availability
alone does not make Cauchy's interface refresh stream-native or capturable.

### Host-transfer audit

The configured numerical update keeps the mean, reconstruction, RHS, Jacobian
and Krylov fields on device. Source-visible downloads fall into these classes:

| Site | Data crossing to host | Lifetime / purpose |
| --- | --- | --- |
| `FiniteVolumeDeviceJacobian.cpp` constructor | Immutable Jacobian samples, offsets and face areas | Once per workspace, to construct symbolic geometry; no solution values or repeated CPU Jacobian assembly |
| `FiniteVolume_{AuxiliaryQueued,RHSQueued,BoundaryQueued}.cpp` | One numerical-status record | Explicit completed-segment reporting; not a field fallback |
| `NonlinearOperators_Internal.hpp`, `TimeStageOperators.cpp` | Fixed-size norm/PP/acceptance summaries | Host stopping/retry decisions after completion |
| `PolynomialMultigridOperations.cpp` | One frozen-component check result | Verification after the device comparison kernel |
| `StateDiagnostics.cpp`, `BoundaryForceReduction.cpp` | Reduced statistics / integrated forces | Host recording; point physics and integration run on device |
| `DomainStateSnapshot.cpp` | Owned state/history and optional reconstruction | Explicit output snapshot, outside numerical evaluation |
| Runtime direct MPI | Provider-dependent host or device transport | Readiness/completion bridge; host staging inside MPI is not a CPU numerical fallback |

CUDA-only Nsight records for the 20-update cylinder pilots support this
distinction: large transfers occur once during first-use setup; repeated
host/device copies are halo-sized or small control summaries. Those traces
include setup/output and do not attribute MPI internals or prove all possible
configurations. Raw traces and compact summaries are retained in the Phase D
experiment. Cauchy's queued phase timer currently measures CPU lanes only;
its zero device entries must not be interpreted as zero GPU work.

Device reconstruction placement also omits periodic rotation matrices and
activity flags when the publication has no rotational periodic entries. The
same publication-wide flag already guards every consuming sweep/limiter
kernel. Rotational meshes retain their tables; this does not change arithmetic.
The Phase D correction passes three CUDA CTest entries (including two GPUs)
and eight installed-Python cases. A matched 20-update FP64 cylinder replay
has bitwise-identical means and coefficients, with 644,298 bytes removed from
the warm device working set. Evidence: `workspace/plan61-validation/D/rotation-storage-*`.

## Focused evidence and limits

The compact Phase C report lives in the plan repository at
`evidence/cauchy-plan61/phase-c/README.md`; raw logs and source manifests live
under this checkout's `workspace/plan61-validation/C/`. The 32 planned CTest
entries have passing reruns, with additional CPU OMP and Python checks.
`test_SpatialScheduleCUDA.cu` explicitly requests `RHSKernelPolicy::Generic`
for the CPU numerical anchor; CPU-inline, CPU-worker and CUDA use the same
managed schedule. `test_DomainCUDA.cpp` exercises complete implicit BDF2 and
ESDIRK4/pMG advancement and BDF restart continuation. These small tests do not
replace physical cylinder/NACA convergence qualification.

Sanitizer executions reach all assertions but exit 99 on an OpenMPI CUDA
finalization error reproduced by a separate MPI+cudart-only program. Preserve
that qualification: neither a clean sanitizer result nor a DNDS device-memory
violation is established by that external API shutdown error.

Final cylinder pilots and comparisons are recorded separately in canonical
`workspace/plan61-device-qualification/`, with raw outputs in
`data/plan61-device-qualification/`. All four CPU/CUDA SG/pMG pilots reached
their deliberate 20-update cap and returned frontend status 2. Full conservative
fields agree within 2e-13 after persistent-ID alignment. Longer CPU SG and pMG
runs converged at 5325 and 1361 updates/cycles; matched full CUDA convergence
remains open. The direct RTX3060 `auwd` port times out, but WSL execution works
through `auw`; a frozen source tree and the locally built SM80/86 executable
with verified runtime dependencies are staged in the requested desktop checkout.
Twenty-update flow5/SA pilots agree near roundoff. SST fails the original
pointwise bound in a small number of cells. A same-mesh generic CPU/CUDA probe
shows near-roundoff RHS/Jacobian agreement, but the existing CPU Eigen-SVD
fallback is less accurate than cuSOLVER for highly nonnormal wall blocks
(normalized condition numbers near 1e9). An independent 80-digit solve confirms
this; an offline inverse refinement improves both actions. The decision to
adopt guarded Cauchy refinement is pending. Neither the comparison bound nor
the production algorithm has been changed to make this pass.
The full two-device AmgX SG run reached its 540-second bound
with decreasing residual; pMG has only capped pilot evidence.

The optional reconstruction precision benchmark passed its FP64/FP32/TF32
operator error bounds. FP32 did not improve runtime for that tested shape;
TF32 did, with a larger error. This is not physical-solution accuracy evidence.
Final physical gates and full completion remain open. Compact application
records are in the plan repository's `evidence/cauchy-plan61/phase-d/`.

### Explicit reconstruction kernel selection

`execution.reconstructionKernel` selects `fp64` (unchanged default),
`fp64Tensor`, `fp64TensorShared`, `fp64TensorAsync`, `fp32`, or `tf32`.
The operator resolves this once when placing its CFV plan, quantizes only
already-inverted sweep maps for the last two choices, and verifies the device
capability before advancing. The common queued sweep passes the frozen policy
to CFV; it contains no CUDA launch or backend-specific schedule. Coefficients,
means, relaxation/publication and WBAP tables remain FP64.

CPU rejects nondefault choices. LVR currently rejects nondefault choices
because its penalized sweep has a distinct kernel contract; it must not silently
ignore a requested policy. Gradient-only pMG levels have no VR sweep, so the
inherited selection has no numerical work there. FP64 tensor modes require
device FP64 tensor support; TF32 requires its corresponding device capability.
These explicit optional modes do not change the required FP64 acceptance gate.
The Phase D selector build and focused config/domain/spatial checks passed,
including one-rank device1 and two-rank/two-device execution. Rebuilt/installed
Python operator tests passed three cases; the mixture case skipped because its
installed mechanism was unavailable. Logs are under
`workspace/plan61-validation/D/quantized-*`.

Matched one-device, 20-update P3 cylinder pilots quantify the optional modes:
FP32/TF32 halve resident sweep-map bytes (6,426,432 to 3,213,216), but their
reconstruction coefficients fail the predeclared pointwise comparison bounds.
The conservative fields pass those bounds. Maximum coefficient differences
are 3.366e-8 (FP32) and 1.459e-4 (TF32); these short pilots establish neither a
converged residual floor nor a useful whole-solver speedup. Keep these modes
explicit experiments, retain the failed comparisons, and do not weaken the
FP64 acceptance gate. The plan repository records the bounds and full metrics.

## CPU threading and device leaf map

The earlier inventory is a chronological implementation record; its original
"pending" entries are not the current source status. These are the implemented
leaf families behind the shared operator scheduling interfaces:

| Family | Device leaf | CPU threading boundary |
| --- | --- | --- |
| Auxiliary means/faces/cell sums | `Discretization/CUDA/FiniteVolumeAuxiliaryKernels.cu` | Independent row work and private point workers |
| Boundary values/forces | `FiniteVolumeBoundaryKernels.cu`, `BoundaryForceKernels.cu` | Point rows; force reduction uses private contributions |
| RHS and source/rotation | `FiniteVolumeRHSKernels.cu`, `_Faces.hpp`, `_Reconstruction.hpp`, `FiniteVolumeSourceKernels.cu` | Cell rows with private physics/scratch; no MPI in workers |
| Gradients/characteristics/LVR/PP | `FiniteVolumeGradientKernels.cu`, `FiniteVolumeReconstructionKernels.cu`, `FiniteVolumeLimitedVRKernels.cu`, `FiniteVolumePositivityKernels.cu` | Row selection plus explicit integer/statistics merging |
| Implicit blocks/actions/relaxation | `Implicit/CUDA/FiniteVolumeJacobianKernels*`, `FrozenJacobianKernels*`, `LinearBlockKernels.cu`, `LinearDiagonalKernels.cu` | Independent rows; CPU SGS alone respects ordered mesh subpartitions and frozen cross-partition input |
| State, time, nonlinear and FAS algebra | `State/CUDA/*`, `Core/CUDA/TimeStageKernels.cu`, `Implicit/CUDA/NonlinearKernels*`, `Multigrid/CUDA/PolynomialMultigridKernels.cu` | `OMP::ForRows`, status-aware rows and private sum buffers |
| Interface sampling/publication | CFV `Operators/CUDA/DistributedInterfaceSamplingKernels.cu`; Cauchy `Interface/CUDA/InterfacePacketKernels.cu` | Query/receiver rows; communication remains outside kernels |
| Output field conversion | `IO/CUDA/OutputFieldsKernels.cu` | CPU private point workers before snapshot serialization |
| CFV metrics/reconstruction setup | Host setup by design; placed immutable arrays are kernel inputs | `ForSetupRows` in `CFV/Prepare/FiniteVolumePlan*`, `GradientPlan.cpp`, `PolynomialSamplingPlan.cpp` |
| CFV iteration/sampling/limiting | `CFV/Operators/CUDA/{ReconstructionDevice,GradientKernels_CUDA,PolynomialSamplingKernels_CUDA,LimitedVRKernels_CUDA,CharacteristicKernels_CUDA}.*` | Jacobi independent rows; CPU SOR uses configured subpartitions |

Helpers operating on one point, one small matrix or one metadata descriptor do
not spawn their own OpenMP teams. Their caller owns the row partition. This
avoids nested parallelism and preserves the OMP=1 arithmetic anchor. MPI calls,
leases and ownership transitions remain in the operator/controller layers.
