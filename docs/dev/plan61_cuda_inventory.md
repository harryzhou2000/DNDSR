# Plan 61 CUDA implementation inventory

Baseline: `c1700f6d`, branch `codex/cauchy-cuda`. Updated 2026-11-13 (Asia/Shanghai).

For the current source-linked execution boundaries, optional CPU-only API
limits and evidence qualifications, see [the coverage review](plan61_cuda_coverage.md).
Historical pending statements below describe their dated checkpoints.

**Phase A foundations and Phase B spatial gates passed at their recorded snapshots; Phase C is in its grouped build/validation round.** The initial header catalog
below is a navigation aid. The recursive source-linked call/leaf audit is still
open; header enumeration alone does not establish complete hot-path coverage.
Every numerical helper reached by supported flow5/SA/2EQ AoS perfect-gas paths
must execute on CUDA. CPU OMP=1 remains the mathematical anchor.

The source checkpoints below are chronological; later integration entries
supersede earlier pending lists. Current source includes the managed spatial
schedule, generic CPU mixture characteristics and experimental TF32 Jacobi.
The full build passed; focused CPU checks passed all 15 selected entries across
the initial run and the corrected drain-contract rerun. Cauchy Python operator
checks passed 2/2 at np=1. The later B correction gate passed all twelve selected
CUDA executables across reruns, including the two-device spatial schedule;
the RANS point mismatch and CPU gradient lifetime defect were resolved.
Phase C's first runs exposed queued-entry and immutable-stream lifetime defects,
now corrected: the full build, 16 selected CPU entries, device implicit/time/FAS
checks and four two-device entries pass across recorded reruns. Remaining
fixture/checks and the recursive coverage audit are open; final application
qualification remains open. Later entries retain exact
checkpoint/evidence references; earlier failure records are historical.

### State and analytic-error reduction source additions (unbuilt)

`Discretization/CUDA/FiniteVolumeStateKernels.*` and its `_Profile.cu` definition
provide owned-mean admissibility, local CFL minima and analytic flow-error
summaries. They use caller-owned partial/result arrays and CUB block reductions
with an explicit non-atomic tree, followed by a final reduction launch on the
same stream. No floating-point atomic sums or host scalar returns are used.
Required failure status remains present even with diagnostic counters disabled.

The flow-error leaf consumes prepared maximum-order physical quadrature in the
CPU oracle's order, independently of the residual quadrature choice. It samples
the shared analytic profiles in world coordinates and rotates momentum back to
local axes. L1/L2-squared/Linf contributions and volume are rank-local, leaving
MPI and normalization at the reporting boundary. The production geometry cache
and solver dispatch remain pending; a standalone numerical launcher is not
end-to-end device support.

Focused test sources compare against CPU reductions (empty/one/multiple blocks)
and the actual CPU error oracle on 2D periodic quads and 3D hexes, with moving
frames and combined flow/scalar rows. Repeated queued reductions check exact
same-device repeatability without requiring bitwise equality to serial CPU
summation. These sources have not been compiled or run in the current B batch.

### Quantized sweep reference (source only)

`ReconstructionResidencyOptions::matrixPrecision=FP32` explicitly converts the
completed FP64 inverse products, mean-injection vectors and boundary-injection
maps at device placement. The device owns only those FP32 copies; secondary
WBAP maps and periodic rotations remain FP64. FP64 remains the default.
`ReconstructionDeviceKernel::FP32` selects SIMT FP32 dense-product accumulation
with FP64 relaxation/state accumulation. Ordinary and limited Jacobi share the
same shape-specialized arithmetic, including alpha-one neighbor exclusion.
Existing FP64 tensor kernels reject quantized maps instead of dereferencing
missing FP64 storage. Python placement exposes the same explicit choice.

This is the arithmetic/storage baseline for the planned quantized MMA trial,
not an implementation or performance claim for FP32/TF32 tensor cores. That
trial remains pending. Test sources compare CPU sweeps on a rotated 2D sector
and 3D hex mesh, including core/dependent selection and LVR penalties; they also
check half-sized map allocations with no duplicate FP64 device maps. No B build,
accuracy acceptance, or speed assessment has run yet.

### Phase B LVR source checkpoint (unbuilt)

CFV `LimitedVRPlan` now has retained queued CPU-inline/worker and CUDA seed /
neighbor-maximum operators. Shared cell formulas preserve face-incidence
multiplicity, boundary exclusion, history retention and saturation. Preparation
allocates the incidence buffers before OMP row fill. The caller still owns every
halo exchange and must order it before dependent work; core reads owned only.

CUDA limited Jacobi preserves the CPU relaxation formula and its exact
alpha-one neighbor/BC exclusion. A separate device penalty validation pass
precedes the sweep; the existing sticky workspace status guards publication.
The launch projects a compact sweep-only descriptor instead of carrying unused
WBAP tables. No per-iteration allocation or host synchronization is introduced.

Cauchy physical LVR CUDA leaves provide O2 target PP, scalar preparation,
per-face sensors, deterministic cell gather, base merge and history/penalty
publication. Setup publishes a local cell-face incidence CSR. These leaves
preserve the current pure-flow5 LVR physical adapter; they do not imply a new
RANS LVR scheme. Device operator/workspace wiring is still pending.

New `CFV/test_LimitedVRCUDA.cu` and `Cauchy/test_LimitedVRCUDA.cu` sources cover
CPU-oracle comparisons, queued sensor-to-gather dependencies, periodic incidences,
core ghost isolation and the fixed-target exclusion. They have **not run**.
The single grouped B build/test round remains pending until spatial editing is
complete; neither kernel presence nor test source constitutes solver acceptance.

### Queued characteristic publication (source editing, unbuilt)

The normal schedule is one compute stream:
`produce matrices -> WBAP(order P..1)`. Same-stream ordering
establishes readiness without a host wait or CUDA event; cross-stream producers
need an explicit event dependency. Frozen CPU-finalized transforms retain their
identity/completed-version checks. Device-produced transforms retain identity,
shape and residency checks but do not interpret a pending writer's completed
version as readiness.

`BuildQueuedFaceTransform` provides that explicit publication contract for modal
and gradient WBAP. The producer supplies finite inverse matrices.
`SubmitValidateFaceTransform` is an **opt-in numerical diagnostic**, not a
mandatory publication or ordering operation. Cauchy defaults
`numerics.characteristicDiagnostics=false`; CPU `FinalizeFaceTransform` normally
publishes shape/identity/version without reading matrix payloads. Neither queued
nor synchronous limiter APIs automatically run inverse checks. When enabled,
check once per refreshed matrices, not per WBAP order. Scheduled leases prove
access ordering, not numerical matrix validity. The existing physical producer's
near-singular componentwise fallback remains a distinct numerical policy.

Opt-in device diagnostic failure guards limiter output; the controller checks status
after the segment. Gradient workspaces now have the same deferred status contract.
Test sources enqueue producers, validation, multiple WBAP passes and publication
without an intermediate drain, compare each pass with CPU, and check one invalid
producer. These are not yet executed.

Cauchy's physical CUDA producer now has source for analytic flow-five L/R
matrices (no numerical inversion) and bounded shared host/device point helpers.
It uses Roe averages of cell means, never reconstructed face states, and restores
canonical parent-zero periodic orientation even for a side-one-owned incidence.
The stencil optionally publishes a canonical-face incidence map; unused faces
receive identity. CPU still uses its existing analytic right basis plus Eigen
inverse as the oracle. The existing near-singular componentwise fallback is
preserved separately from optional CFV diagnostic scans. Focused test sources
cover 5/6/7-wide inputs, periodic side-one and physical/unused faces, the fallback
and one invalid-state status. Production operator wiring and execution remain
pending. Mean auxiliaries could supply pressure for enthalpy in future, but are
not required or reused by this producer; their cached sound speed is not a Roe
sound speed and cannot simply be averaged.

## Execution and ownership contract

Shared solver/time policies call operator facades. Facades retain fields,
geometry, metadata and scheduled access, build compact views, then dispatch
C++ host launchers. Launchers validate whole-array Device tags and the total
argument budget before issuing CUDA kernels. Device leaves have no ownership,
lease acquisition, MPI, exception recovery or host physics service access.
Whole-array tags do not extend lifetimes or establish producer readiness.

## Cauchy CPU family inventory

The entries below locate the numerical interfaces to port/share. Status is
explicitly pending until the individual dispatch and its transitive helpers
have been audited and validated. Source/setup-only entries are distinguished
from required iterative work; excluded CPU algorithms are not deleted.

| Source interface | Numerical entries / responsibilities | Batch | Current disposition |
| --- | --- | --- | --- |
| [Cauchy/Boundary/CPU/RANSWallKernels.hpp](../../src/Cauchy/Boundary/CPU/RANSWallKernels.hpp) | RANSWallTarget | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Core/CPU/TimeStageKernels.hpp](../../src/Cauchy/Core/CPU/TimeStageKernels.hpp) | UpdateExplicitBlock, MeasureExplicitBlend, UpdateExplicitFlow | C | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeAuxiliaryKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeAuxiliaryKernels.hpp) | PrepareAuxiliaryMeans, PrepareAuxiliaryFaces, ReduceAuxiliaryCells | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp) | FillBoundaryTraces, CopyBoundaryRows, FillPrimitiveBoundary, ValidateInterfacePacketRows, AccumulateBoundaryForce, ValidateInterfaceAliasRows | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeGradientKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeGradientKernels.hpp) | LimitPhysicalGradients, LimitPositivePhysicalGradients, FillPrimitiveMeans | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp) | LimitLVRTarget, FillLVRScalars, BuildLVRBranches, MergeLVRBase, PublishLVRPenalty | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp) | PointScratch, mean linearization, Gather/RotatePoint and point preparation helpers | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp) | ValidateReconstructionMeans, MarkAdmissibleDegrees, TruncateReferenceModes, SelectReferenceDegree, CompressPolynomial, MarkPolynomialRecovery, RecoverPolynomial, VerifyReconstructedPoints, PolynomialPositivityStatistics, LimitPhysicalStepRows | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp) | EvaluateFlowFlux, AddFlowRotationSource, EvaluateEquationFlux | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeReconstructionKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeReconstructionKernels.hpp) | PrepareFaceCharacteristicMatrices | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp) | State rotation, direct-FV displacement, physical/interface boundary sampling | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp) | AddMeanSourceJacobian, AddQuadratureSourceJacobian, AddVolumeSources, ReplicateSourceValues, AddEquationRotationSource | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp](../../src/Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp) | InitializeFlowProfile, MeasureFlowError, AdmissibleMeans, MinimumStableStep | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp](../../src/Cauchy/Implicit/CPU/FiniteVolumeJacobianKernels.hpp) | RotateState, GridVelocity, PrepareCompressibleFluxDerivative, TurbulentNormalFluxDerivative, PrepareAdvectedScalarFluxDerivative, AddEquationJacobianPoint, EvaluateEquationJacobianBlock, ApplyEquationJacobianBlock, EvaluateFlowJacobianBlock, ApplyFlowJacobianBlock, PrepareFlowJacobian, FreezeFlowJacobian, InitializeEquationJacobian, CompleteEquationJacobian | C | Required device implementation; CPU OMP audit pending |
| [Cauchy/Implicit/CPU/JacobianKernels.hpp](../../src/Cauchy/Implicit/CPU/JacobianKernels.hpp) | AddFrozenFluxBlock6, PrepareCompressibleFluxDerivative, PrepareAdvectedScalarFluxDerivative | C | Required device implementation; CPU OMP audit pending |
| [Cauchy/Implicit/CPU/LinearBlockKernels.hpp](../../src/Cauchy/Implicit/CPU/LinearBlockKernels.hpp) | ApplyLinearRows, JacobiLinearRows, LowerSGSRows, UpperSGSRows, FillLinearFactor | C | Apply/Jacobi required; lower/upper SGS and factor fill excluded from CUDA |
| [Cauchy/Implicit/CPU/LinearProblemKernels.hpp](../../src/Cauchy/Implicit/CPU/LinearProblemKernels.hpp) | InvertLinearDiagonal | C | Required device implementation; CPU OMP audit pending |
| [Cauchy/Implicit/CPU/NonlinearKernels.hpp](../../src/Cauchy/Implicit/CPU/NonlinearKernels.hpp) | ScaleRateByVolume, NormalizeRateByVolume, AssembleDefectRows, AssembleEquationRows, AssembleEquationAndRateRows; ApplyCorrection, ValidateImplicitMeans, AssembleDefect, PreparePseudoTime, ApplyAgreedRHSAlpha | C | Required device implementation; CPU OMP audit pending |
| [Cauchy/Interface/CPU/InterfacePacketKernels.hpp](../../src/Cauchy/Interface/CPU/InterfacePacketKernels.hpp) | FillInterfacePacket | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/Interface/CPU/InterfaceSamplingKernels.hpp](../../src/Cauchy/Interface/CPU/InterfaceSamplingKernels.hpp) | BuildInterfaceQueries | B | Geometric query setup may stay host; audit for numerical helper work |
| [Cauchy/Multigrid/CPU/PolynomialMultigridKernels.hpp](../../src/Cauchy/Multigrid/CPU/PolynomialMultigridKernels.hpp) | CheckFrozenComponents, ProjectSelectedCorrection, AddSelectedCorrection, CopyPolynomialLevel, AccumulateFASNorm | C | Required device implementation; CPU OMP audit pending |
| [Cauchy/Operators/Source/CPU/ChemicalSourceKernels.hpp](../../src/Cauchy/Operators/Source/CPU/ChemicalSourceKernels.hpp) | AdvanceChemicalRows, EvaluateChemicalRows | — | Excluded: chemistry/species outside Plan61; preserve CPU implementation |
| [Cauchy/State/CPU/FieldKernels.hpp](../../src/Cauchy/State/CPU/FieldKernels.hpp) | CopyRows, FillRows, RequireFiniteRows, RequireCappedRows, AccumulateMeanScale, FiniteRows, DivideRowsByVolume, AccumulateVolumeNorm, AccumulateReconstructionUpdateNorm, FillConstantRows | B | Required device implementation; CPU OMP audit pending |
| [Cauchy/State/CPU/StatePropertiesKernels.hpp](../../src/Cauchy/State/CPU/StatePropertiesKernels.hpp) | RefreshDerivedRows, AccumulateStateDiagnostics | B | Required device implementation; CPU OMP audit pending |

## Shared/runtime and CFV coverage

| Source | Required follow-through | State |
| --- | --- | --- |
| [Runtime ArrayView](../../src/DNDS/Runtime/Array/ArrayView.hpp) | Device provenance, all projections, ordinary/eager factories and wrapper validation | Source edits in progress; no build/test |
| [Eager Array submission](../../src/DNDS/Runtime/Execution/EagerArray.hpp) | Retained allocation/shape/access and tags through asynchronous launch | Source edits; existing lifecycle reused |
| [DOF algebra](../../src/DNDS/Algebra/DofAlgebra.hpp) | All field combinations/reductions needed by time/linear/pMG paths; reuse existing CUDA kernels | Existing device primitives, residency edits; full Cauchy adapter audit pending |
| [Equation DOF operations](../../src/Cauchy/State/EquationDofAlgebra.hpp) | Remove host-only field arithmetic for supported device layouts | Required in B/C |
| [Solver linear algebra](../../src/Solver/Linear.hpp) | Global GMRES vectors, reductions, Hessenberg/Givens/backsolve; host stopping only | Source call walk pending |
| [CFV plan](../../src/CFV/Plans/FiniteVolumePlan.hpp) | Capability-selected placement; every invoked sweep/limiter/gradient/sampling phase | Existing partial CUDA support; remaining B work |
| [CFV CUDA reconstruction](../../src/CFV/Operators/CUDA/ReconstructionDevice.cu) | Jacobi/WBAP/commit; matching BC, periodicity and layouts | Existing implementation; tag/budget wrapper edits unvalidated |
| [CFV tensor variants](../../src/CFV/Operators/CUDA/ReconstructionTensorCore.cu) | FP64 paths retained; optional post-inversion FP32 matrix experiment separate | Existing FP64; new precision work pending |
| [CFV direct gradients](../../src/CFV/Operators/GradientPlan_Queued.cpp) | GG/LS/GGMP, coefficient conversion, direct WBAP and prescribed reconstruction BC values | Shared CPU/CUDA leaves and queued wrappers edited; B validation and production wiring pending |
| [CFV LVR](../../src/CFV/Operators/LimitedVRPlan_Operators.cpp) | Target/gate/expansion phases; transfers outside kernels | Device work pending |
| [CFV polynomial sampling](../../src/CFV/Operators/PolynomialSamplingPlan_Values.cpp) | Cached/direct values and gradients, periodic transforms | Device work pending |
| [PerfectGas](../../src/Cauchy/Physics/PerfectGas.hpp) | Shared pure host/device math; compact immutable descriptor | Existing helpers; transitive audit pending |
| [Riemann](../../src/Cauchy/Riemann/RiemannFlux.hpp) | Shared header arithmetic and physical setup, gas-agnostic flux contract | General evaluator/setup still out of line; B work |
| [RANS model](../../src/Physics/RANS/Model.hpp) | SA/2EQ source, diffusion, boundary and approximate Jacobian helpers | Transitive CPU/device/OMP audit pending |

## Work and validation groups

- **A validated:** whole-array Device tags and projections; host wrapper checks;
  launch argument budget; import checks; complete inventory; context/placement;
  allocation naming/accounting; CPU/setup OMP; source-built single-device AmgX.
- **B:** every supported spatial/CFV/device physics path and precision option.
- **C:** Jacobi/AmgX/GMRES/pMG/time arithmetic and asynchronous snapshots.
- **D:** final two-local-GPU cylinder and auwd NACA0012 acceptance.

## Foundation validation (group A, 2026-11-13)

Built the selected C++, CUDA and Python targets at j16 with GCC 12, CUDA 12.6,
CUDA-aware OpenMPI 5.0.9 and installed NCCL. Built pinned AmgX v2.5.0 without
MPI. The adapter's 5/6/7-component scalar-CSR checks passed on local device 0.
AmgX's prefix-only RUNPATH was disabled so the application's selected CUDA
toolkit takes precedence; installed DNDS Python libraries retain that toolkit
directory too. The corrected loader trace resolves CUDA libraries to 12.6.

- 24 selected CPU/MPI CTest entries passed (np=1/2/4 as registered).
- Eight further OMP=2 entries passed, including np=2/4 auxiliary/nonlinear/DOF
  checks and reconstruction with empty ranks and frozen subpartition reads.
- Four CUDA entries passed: eager engine, native DOF algebra, AmgX and CFV
  reconstruction. These test existing device facilities and new contracts;
  they do **not** establish a complete Cauchy CUDA solver.
- Rebuilt and installed Python bindings, including EulerP's package dependency:
  24 array tests and 42 eager-engine tests passed at np=1; 24 array tests passed
  on each rank at np=2. No module-level skip substitutes for bridge coverage.

Source corrections in this group include the dependency's required C-API
installation artifact and a named communicator array for CUDA 12.6 parsing of
an existing Geom serial-output template. Mesh/connectivity mathematics are
unchanged. Full logs, configure/build commands and fixture hashes are under
`workspace/plan61-validation/A/` in this checkout.

Spatial CPU OMP completion remains grouped with B; implicit/time/reduction OMP
completion with C. The source-linked transitive catalog above remains open.
Finish each group's source/test-source editing before its build/test round;
end-to-end solver acceptance remains group D.

## Spatial group B source checkpoint (not yet built)

- Riemann numerical arithmetic now lives in `RiemannFluxMath.hpp`, shared by
  CPU and CUDA. Names and throwing validation stay in the host facade.
  `RiemannPreparation.hpp` supplies model-dependent Roe/wave/AUSM inputs;
  thermodynamic recovery has not moved into the gas-agnostic Riemann kernels.
- Perfect-gas point properties, conservative-gradient conversions, total-to-
  static inlet math and the RANS wall target have host/device numerical entries.
- Analytic profile names are resolved to plain parameter records. Rigid-frame
  kernel parameters own rotation/translation/velocity values without a Geom
  object ABI. Perfect-gas physical-boundary math consumes these records and
  distinguishes physical face from mirrored exterior targets. Interface values
  remain packet inputs handled outside this physical-boundary producer.
- `Physics/RANS/PointKernels.hpp` uses a small value-only `PointParameters`
  descriptor and the existing SA/Wilcox/SST formulas. Kernel-facing transport,
  diffusion and source calls return explicit numerical status. Legacy default
  fatal diagnostics remain available; status adapters explicitly disable those
  diagnostics and check the results. RealizableKE remains CPU-only.
- Cauchy's auxiliary, source/Jacobian and RANS viscous-flux CPU leaves now
  consume those statuses, removing the visible DES/SST throwing source guard
  from the numerical path. Existing checked host convenience methods remain.
- `FiniteVolumePlan::Placed` supports metric-only CUDA storage independently
  of VR allocations. Payload placement does not change mesh/entity/halo
  identity. The placed object does not retain source metric arrays. Read scopes
  retain the placed arrays and geometry guard. Stencil field storage also has
  explicit placement and context-aware retained reads, with once-per-wrapper
  residency validation. These facilities are not yet full solver dispatch.
- Test sources cover all 17 Riemann schemes with distinct means and moving
  faces, SA/Wilcox/SST transport/diffusion/source/implicit-diagonal comparisons,
  source-geometry errors, and standalone CUDA metric use after dropping the
  host metric plan. They will run with group B, not as a new validation group.
  Moving-frame boundary comparisons include farfield/inlet, slip/isothermal
  no-slip and analytic/composite DMR conditions against the CPU owner methods.

Still pending in B: spatial CUDA launchers and operator wiring, boundary/PP/
gradient/LVR/sampling/device-cache completion, optional matrix precision and
remaining spatial OMP work. No source checkpoint here is numerical acceptance.

### Sampling and auxiliary source additions (unvalidated)

- `CFV/Plans/PolynomialSamplingPlan.hpp` now exposes queued sampling on the
  caller's engine/stream. CPU worker and CUDA dispatch share that entry. The
  sampler snapshots only query IDs, required basis cache or direct-evaluation
  geometry/moments, plus mesh/space identity; it no longer pins the complete VR
  matrix owner. Cached-only placement has no point/geometry payload to copy.
- `CFV/CPU/PolynomialSamplingKernels*.cpp` contains independent-row generic and
  bounded cached CPU arithmetic, using the canonical OMP row helper. The CUDA
  launcher in `Operators/CUDA/PolynomialSamplingKernels_CUDA.cu` selects cached
  versus direct Taylor evaluation once and uses bounded point scratch. Device
  direct derivatives use canonical Geom mode ordering through `TaylorBasis`.
- `Cauchy/Discretization/WallDistanceSamples.*` has explicit immutable placement
  and resident read views. It retains mesh-generation validity, not a second
  wall-distance provider or the source host quadrature buffers.
- `Cauchy/Discretization/CUDA/FiniteVolumeAuxiliaryKernels.*` provides mean,
  face and cell-reduction host launchers for combined AoS flow5/SA/2EQ. The
  parameter projection in `PerfectGasKernelPoint.*` preserves the equation's
  viscosity enablement. Periodic vector/gradient rotation is conditional, and
  physical boundaries keep CPU auxiliary mean-state semantics while interface
  boundaries consume supplied receiver-frame packet values. These launchers
  are **not yet connected to the production auxiliary operator**.
- Numerical failures have explicit row-status outputs. Residency/shape and
  aggregate CUDA argument budget are checked before launch; leaves contain no
  leases, MPI, host copies or exception-based recovery. Optional face outputs
  and scratch still follow the existing auxiliary bundle contract.
- `EagerEngine::RetainUntilSynchronize` pins thread-sensitive host owners to the
  controller's next drain, independently of scheduled Array access. The queued
  sampler uses it for its mesh guard; it adds no event per kernel. See the
  runtime caller contract for destruction and ordering responsibilities.

Added test sources (not executed): P1/P3 sampling in 2D/3D under no/partial/full
basis caching, owned/ghost queries and CPU-worker/CUDA comparison; controller
retirement after inline/worker work; CUDA auxiliary means versus the generic CPU
flow5/SA/Wilcox/SST property path; a rotated-neighbor/physical/interface face
stencil followed by cell reduction with an independent acoustic oracle.
Group B still requires source completion, grouped compilation and numerical
validation. No end-to-end CUDA capability or timing result is claimed.

Resident VR face-basis placement now exposes `PointBasisView()` for cached or
direct evaluation, with an explicit derivative-order request (default one).
It does not implicitly request all FV metric tables. Direct inputs are copied
only when cache coverage is insufficient. The source generation stamp is now
retained independently of host payloads, closing an invalidation gap when
`retainHostPayloads=false`. Added test source compares cache-off/value-only/
value-gradient placement against CPU face-basis evaluation and checks the
retained buffer names and invalidation after host-plan release. This follow-up
is also unbuilt and belongs to the same group B validation.

### Direct-gradient source additions (unvalidated)

- `GradientPlan_Placement.cpp` places immutable GG/LS/GGMP incidence/rotation
  tables and shares a caller-supplied placed FV metric plan. CPU boundary
  coordinates are not retained by the placed plan. Setup uses prefix allocation
  and independent per-cell OMP fills, preserving incidence and boundary order.
- `CPU/GradientKernels.cpp` and `Operators/CUDA/GradientKernels_CUDA.cu` supply
  evaluation, linear coefficient conversion/extraction and one-level physical
  gradient WBAP. Shared `Math/GradientEvaluation.hpp`, `GradientLimiting.hpp`
  and `WBAPScalar.hpp` keep point arithmetic free of leases and exceptions.
- Queued operator wrappers retain geometry and Array leases, validate device
  tags once, and declare ghosts only outside Core. Reusable workspace holds
  immutable periodic-component metadata and optional CPU limiter scratch.
  CUDA selects combined 5/6/7-component paths with bounded thread scratch.
- Partial characteristic ranges and scalar remainder are preserved; count zero
  reads no matrix payload. The current transform placement takes finalized CPU
  matrices. Device-generated transform publication remains production work.
- Python overload bindings now distinguish synchronous/queued gradient and
  polynomial sampling calls. Tests added for CPU-worker real halo overlap,
  AoS/SoA/tiled CPU fields, 2D/3D device gradient conversion, rotated momentum,
  ghost-only extraction, activation and disabled/flow5 characteristics. They
  have **not run**; no new build/test group has started.

### Boundary producer source additions (unvalidated)

`Cauchy/Discretization/CUDA/FiniteVolumeBoundaryKernels.*` adds C++ launchers
for physical/interface reconstruction targets, mean-based primitive-gradient
targets and boundary row copies. Whole-array residency, profile/shape and
aggregate launch arguments are checked before submission. Point failures use
per-output-row status; no kernel performs communication, allocations or host
fallback. Parameters/condition registries are explicit resident metadata rows.

`Boundary/PerfectGasBoundary.hpp` now provides combined flow5/SA/two-equation
state production, including total-inlet RANS primitives and the same wall
omega/isothermal reflection as the generic CPU operator. Reconstruction
targets preserve optional mean fallback for temporary unlimited VR traces and
the SA wall compile-time option. Interface rows bypass physical BC arithmetic
and copy the supplied complete receiver-frame state. The polynomial basis view
supports cached/direct evaluation without mesh-owner access inside a kernel.

Added test source compares the actual CUDA launchers against generic CPU
boundary production for flow5, SA, Wilcox and SST; face/mirrored targets,
primitive targets, row maps, nonconstant polynomial input, a thermal mean
fallback, isothermal wall and total-condition inlet, plus interface packets.
These additions remain unbuilt. Production workspace placement/dispatch and
the remaining RHS, PP, LVR and characteristic producers are still open in B.

### Face RHS and cell-gather source additions (unvalidated)

`Discretization/CUDA/FiniteVolumeRHSKernels.*` now provides specialized 5/6/7
component face integration followed by a separate canonical cell gather.
Physical/interface states, optional mean acoustics and stencil inputs, all
Riemann selections, conservative or primitive viscous gradients, RANS transport,
wall forces, periodic component/gradient rotation and flow5 low flux outputs
have explicit operands. Per-face numerical status and fallback counts avoid
hot-loop atomics. Core calls can receive no ghost views.

The large immutable geometry descriptor is published once into device storage;
the enclosing operator must retain its nested geometry/cache scopes and the
publication until completion. Whole-array residency and launch-argument budgets
are checked by host wrappers. No launch allocates, stages field payloads or
performs communication. Production operator integration is still pending.

The test source compares actual CUDA face/gather launchers with the generic CPU
RHS on a two-owned/one-ghost incidence fixture, including periodic rotation,
two-point faces, wall/interface rows, separate viscous coefficients, P0/direct
gradients/P3 cached/P3 direct basis, and flow5/SA/Wilcox/SST. This is numerical
kernel coverage, not a mesh-builder or end-to-end solver test. It has not run.
The optimized CPU flow5 route has separate viscosity conventions (notably its
interface diffusion and unlimited viscous point values); generic CPU OMP=1 is
the agreed Plan61 oracle. Do not infer equivalence of those other conventions
from this fixture. Group B remains source-only and the full CUDA capability
remains disabled.

### Volume source and positivity/gradient source additions (unvalidated)

- `Discretization/CUDA/FiniteVolumeSourceKernels.*`: quadrature RANS residuals,
  mean/quadrature positive destruction diagonal approximations (diagonal vector
  or full-block destinations), directional-value replication, and the moving
  component-axis flow source. The source uses shared RANS point arithmetic and
  preserves model scalar repair at quadrature points. Cell-local integration
  commits only after every point succeeds. Mean thermodynamic linearization
  cache output remains separate work in the implicit integration batch.
- `Discretization/CUDA/FiniteVolumePositivityKernels.*`: modal degree selection,
  reference truncation, flow/RANS compression, endpoint recovery/verification,
  and physical-dt forecasts. Thermal formulas remain shared. Per-cell status,
  beta and flags leave reductions and controller decisions outside the leaves.
  The physical-dt bound never compresses an explicit per-cell update.
- `Discretization/CUDA/FiniteVolumeGradientKernels.*`: periodic-aware Barth
  limiting, direct-gradient thermal/RANS positivity and primitive means. These
  are Cauchy physics operations; WBAP remains a CFV operator. Direct PP samples
  are evaluated from bounded per-cell gradient scratch, with no quadrature-count
  limit on a local sample array.
- Source review fixed the RHS publication's initially absent read lease to use
  an explicit optional scope; the retained device descriptor still owns no
  nested field allocation implicitly.

Test sources compare RANS residual/destruction outputs against the generic CPU
source kernels, cover source replication/rotation, modal PP/reference/recovery,
physical-dt scaling independent of RANS bounds, and direct-gradient Barth/PP
and primitive means. One invalid-mean case checks explicit numerical status.
All remain **unbuilt/unrun**, and production placement/queued dispatch is still
pending. Dynamic characteristic production, LVR, state/reduction leaves,
workspace integration, remaining OMP and optional quantized CFV matrices are
still part of batch B; no CUDA solver acceptance is claimed.

### B placement integration checkpoint (source-only)

`FiniteVolume_Placement.cpp` explicitly places immutable spatial publications
while retaining one source geometry generation. Host mesh/control metadata
remain, but replaced numerical buffers do not retain CPU duplicates.
`FiniteVolumeInternal_SourceQuadrature.cpp` prepares source quadrature once;
workspaces share immutable offsets, weights and samplers instead of rebuilding
host geometry. Its independent filling loops use the existing setup OpenMP
policy after serial shape-cache preparation. Mutable source/gradient, boundary,
face-flux and polynomial-PP operands allocate on the plan's device. CPU row-local
direct-gradient scratch is not allocated for a device plan.

Placed CFV plans now allocate boundary values without a staging copy. Native
synchronous DOF algebra selects the field device and drains its owned stream;
queued hot paths retain caller streams. Neither API silently transfers fields.
Focused source tests cover P0/P1/P3 placement and native CUDA convenience calls.
**No B build/test has run yet.** Remaining spatial work includes complete
workspace initialization, queued operator dispatch, production error-profile
sampling, and quantized MMA evaluation. Characteristics, LVR, state/error
reductions and the FP32 SIMT baseline have source implementations, but are not
yet qualified as a complete solver path. C/D remain pending.

### B queued auxiliary and boundary integration (source-only)

Runtime now exposes shared `MutationEpoch` provenance at write issuance;
completed-write `Version` and readiness remain separate. Auxiliary preparation
uses this to retain/check queued inputs, executes core and halo-dependent CUDA
passes with optional output arrays, and publishes after the caller's drain and
one reduced numerical-status readback. Its CPU queued overload retains the
existing numerical preparation scope. Full spatial-schedule integration and
the two-equation mean-gradient producer remain unfinished.

`FiniteVolume_BoundaryQueued.cpp` retains mesh/metric/basis and immutable policy
scopes, submits managed means/coefficients/packets and output/status operands,
and preserves physical-trace versus mirrored VR targets. Time and frame are
per-launch POD values, not mutable resident metadata. Gradient and primitive
boundary targets remain distinct. A persistent device status accumulator keeps
earlier failures across repeated boundary updates, with explicit segment reset
and post-drain checking; successful later updates cannot erase them. Reduction
uses a bounded two-launch tree without floating-point atomics.

Test sources cover queued auxiliary preparation with a real halo transfer,
optional caches/outputs and split/unsplit calls; repeated P3 boundary updates
with cached/direct basis and primitive targets; and sticky status reduction,
including empty inputs. They are **not yet built or run**. Spatial workspace
construction resolves CUDA placement without enabling the still-incomplete
solver backend; the remaining operator/schedule integration is required before
the grouped B build/test and any CUDA production acceptance.

### B managed flux and gradient integration (source-only)

`FiniteVolume_RHSQueued.cpp` now builds a retained resident geometry descriptor
once per workspace. Mutable means, coefficients/physical gradients, optional
auxiliary inputs, face outputs and cell residuals are declared Runtime operands.
Core submissions omit ghost reads; the same face/cell leaves serve split and
unsplit regions. Time/frame are per-call POD arguments, while physics metadata
stay immutable. A sticky status accumulator is checked after complete region
submission and the caller's drain. Source terms remain a separate operator.

`FiniteVolume_GradientQueued.cpp` submits CFV gradient evaluation through its
backend-neutral entry, with reusable per-layout workspaces. Primitive means
dispatch to CPU/CUDA leaves. Direct-gradient Barth and positivity device
submissions retain their geometry and use explicit output/status arrays. Their
failures fold into the same preparation accumulator as boundary evaluation;
primitive/PP scratch allocates only when that option is active. Equation scratch
allocation now preserves the prototype device without changing arithmetic
dispatch or initializing payloads implicitly.

The managed-flux test source compares generic CPU P0/P1/P3 fluxes against CUDA,
including mean/coefficient/gradient transfers, core/dependent scheduling,
primitive viscous gradients and direct Barth/PP preparation. It is **unbuilt**.
Rotating-frame periodic cases are intentionally included as a numerical check
of face-oriented gathering, not accepted as equivalent without execution.

The common `EvaluateImpl` transition is still required: CPU callbacks currently
acquire some immediate leases at execution, so replacing them with a queued
consumer alone would conflict with pending reservations. Complete the producer
contract as well as CUDA dispatch. Other open items include source/mean-gradient
integration, modal PP/LVR/characteristics scheduling, gradient-PP statistics,
state/error reporting, OMP coverage and the quantized MMA experiment. No B
build/test or full CUDA solver acceptance is claimed by this checkpoint.

The next source increment adds Runtime `SubmitArrayBatch`, retaining the normal
Array operand/access contract for dynamically grouped fields. Cauchy's common
`Workspace::SubmitBoundary` entry now dispatches to a queued CPU producer or
the CUDA producer. CPU means, coefficients, optional interface values and all
boundary targets are declared operands; geometry, thermal metadata and the
point evaluator are retained through completion. This entry supports split CPU
equation fields without a Cauchy-specific lease implementation. The old
synchronous entry remains while the full `EvaluateImpl` schedule is migrated.
Batch lifetime/failure and CPU inline/worker boundary comparison test sources
are present, still awaiting the grouped B validation.

### B source sampling and RANS addition (source-only)

`FiniteVolume_SourceQueued.cpp` submits source sampling through the same CFV
entry on CPU and CUDA, with direct mean/gradient replication dispatched inside
the operator. Its device volume-source adapter adds rigid-frame rotation and
RANS quadrature sources using retained complete geometry views and explicit
mutable Runtime operands. RANS numerical failure joins the preparation status
segment; provenance is published only after the caller's drain. The direct
gradient input must already be prepared; this operator hides no halo exchange.

RANS source wrapper bounds now accept owned+ghost geometry publications while
means/status/output remain owned-only. The former exact-owned-row requirement
would reject valid distributed wall and length caches. Source tests now use
extra geometry rows; a managed source test compares direct SA and P3 SST
sampling/addition with the CPU source operator. These tests remain unrun.
Mean-gradient workspaces can now allocate at plan residency and retain CFV
execution metadata; their scheduling/publication migration is still pending.
P0/mean-only RANS source gradients share the already prepared field; no second
allocation/copy is needed. Pure P0 also allocates no unused modal viscous-gradient
field when unlimited viscosity is selected.

`FiniteVolume_RHSHostQueued.cpp` provides the common RHS submission entry and
retains the existing fixed-flow/generic CPU arithmetic. Runtime batches declare
segmented inputs, gradients, auxiliary values and outputs; the callback holds no
workspace borrow. Small diagnostic records publish after execution. Derived
cache columns are immutable metadata, with payload reservations and completed
publication validation kept separate. Added P3 CPU inline/worker comparison
test source remains unrun.

The next shared-schedule prerequisite is CPU auxiliary preparation: its current
constructor takes immediate mean/gradient/packet reads and seven output writes.
Queued RHS cannot safely reserve those outputs until they retire. Migrate
request assembly, derived refresh/publication and deferred packet binding
together to scheduled operands, while sharing the existing pure auxiliary
kernels. A partial output-only change or a new inter-operator drain is not the
intended completion of the common schedule.

### B mean-gradient/auxiliary schedule migration (source-only)

The two-equation preparation entry now submits one sequence on every backend:
mean exchange, boundary production, CFV core/dependent gradients, gradient
exchange, auxiliary core/dependent work, final join and publication. Dispatch
is confined to the boundary, CFV and auxiliary operators. The retired immediate
mean-gradient boundary wrapper has no remaining call sites.

Mean-gradient provenance has an issued-epoch form for consumers within that
ordered segment and a completed-version form for ordinary callers. Recording
an epoch grants no access or readiness; the Runtime reservations/events remain
mandatory. The completed publication is created only after the join and device
boundary-status check. A focused SST test source exercises this same entry on
CPU inline, CPU worker and CUDA with overlap on/off and real halo transfers.
CPU auxiliary payload/derived-field migration is part of this source batch;
these changes remain unbuilt/unrun until the grouped B gate.

CPU `AuxiliaryPreparation` now selects immediate or queued access at its first
`Apply`; `Begin` retains handles and immutable geometry only. Both modes share
request construction and numerical kernels. Queued derived refresh reserves
its fields and publishes metadata after CPU retirement on the same lane;
chemistry workers are acquired during execution. Queued cache hits may still
advance conservative write generations. Failed allocations retain Runtime's
existing replacement requirement, with no new solver exception recovery.

Boundary provenance checks are separate from immediate access acquisition.
A queued consumer must still receive a known packet publication before being
submitted. The existing deferred synchronous path is preserved. Wiring a
controller submission ahead of an as-yet-unknown interface publication requires
a stable packet/publication dependency handle. The subsequent source edits
below implement that contract without adding a host drain or another solver loop.

### B shared spatial integration (source-only, not validated)

`EvaluateImpl` now calls managed boundary, auxiliary, gradient/projection,
reconstruction-copy, RHS and volume-source submissions directly. The numerical
CPU leaves remain the same. `FiniteVolume_SourceHostQueued.cpp` retains generic
CPU equation/source support; `FiniteVolume_DirectGradientQueued.cpp` separates
queued projections from completed provenance. Rotation's optional low-order
residual is an explicit output. Optional CPU point evaluators acquire chemistry
workers during execution, avoiding controller-time starvation of earlier work.

Interface receiver mapping now reserves writes at submission. Pending packet
copies retain a one-round publication slot and payload allocations; completion
includes mapping, followed by validation/publication. `QueuedEvaluationDependency`
carries those handles before the callback runs. Access-check policy follows
the receiver, including mixed-domain configurations. Existing distributed tests
now retain pending handles across publication and exercise inline/worker and
mixed checking policies. Source tests are unbuilt/unrun.

The subsequent B source batch replaces the CPU-only spatial admission with
stream/array placement checks. This is not yet a validated CUDA solver claim:
the production driver and linear/time-integration path remain later work.

### B queued reconstruction, limiting and source integration (unbuilt/unrun)

- `CFV/CPU/ReconstructionKernels.*` shares the existing CPU VR, limited-VR,
  WBAP and commit mathematics between synchronous and queued wrappers. Queued
  wrappers reserve actual payload operands at submission; callbacks no longer
  reacquire immediate mutable leases. SOR ordering and general CPU layouts remain.
- `FiniteVolume_CharacteristicsQueued.cpp` declares matrix writes and mean/aux
  reads. CPU mixtures consume cached thermodynamics; specialized CUDA computes
  perfect-gas thermodynamics inline. Inverse-consistency checks remain diagnostic
  only. Integer fallback counts use a bounded deterministic reduction.
- `FiniteVolume_GradientHostQueued.cpp` and `FiniteVolume_PositivityQueued.cpp`
  supply the common direct-gradient and modal-PP operator entries. Modal PP keeps
  the all-good and recovery-any observation points and existing recovery bound;
  it does not issue an unconditional fixed number of recovery steps.
- `FiniteVolume_ReconstructionNorm.cpp` computes local norms on the execution
  backend. Only the two scalars become host-visible at the configured stopping
  interval; MPI reduction and the host decision remain outside the queued segment.
- `FiniteVolume_LimitedVRExecution.cpp` now queues target conversion, WBAP,
  target PP, scalar/face sensors, CFV gate expansion and history/penalty writes.
  CPU keeps the original face loop; CUDA uses independent face sensors plus
  deterministic owned-cell gathering. Target and gate exchanges retain their
  producer-prefix ordering. LVR history/scratch is allocated at plan residency.
- Correction preparation uses queued auxiliary/source sampling; the public
  sampling refresh now validates placement instead of requiring host fields.
  Finalized reconstruction and optional halo checks publish after completion.

`test_SpatialScheduleCUDA.cu` adds a production-entry comparison on distributed
physical-boundary hex meshes: generic CPU inline, CPU worker, and CUDA, repeated
evaluations with P3 WBAP, P3 LVR, direct-gradient SA, and P3 SST. The source also
covers reconstruction norm observations. Together with the focused CFV queued
chain and existing primitive tests, this is the intended B validation, not
evidence of a completed test run.

Mean-gradient auxiliary preparation now has a private enqueue/finalize split.
The standalone public API uses the same segment and joins before returning;
the FV residual/correction schedule reuses its already ordered mean halo and
publishes at the enclosing final join. This removes the intermediate drain and
duplicate exchange. Stable pending interface buffers can feed that segment.
Mean-gradient boundary chemistry workers are acquired during execution.
`RefreshCorrectionSource` now accepts the caller's engine/stream and submits the
same sampling operator before publishing; its nonlinear caller is updated.

### B source consolidation: timing, algebra, OMP and quantized MMA

The common orchestration sequence is established. Remaining queued work is
operator contract/backend completion, not another CPU/GPU solver loop.
`FiniteVolume_Profiling.cpp` now records optional CPU lane intervals with queued
start/end markers and merges them after completion. These intervals can include
scheduling gaps; they are not exclusive kernel cycle timings. CUDA reports only
the enclosing host total here. Fail-closed diagnostic extraction now targets the
queued RHS/source wrappers; source generation succeeded, but compilation and the
diagnostic executable remain unrun. Old campaigns are untouched.

`EquationDofAlgebra.cpp` adds owned-only queued copy/fill/scale/AXPY/combination.
All physical operands are reserved together, including CPU segmented fields and
cross-layout inputs. CUDA dispatch requires combined AoS and delegates to DNDS
algebra. Operation captures use numeric aliases so a worker does not become the
owner of MPI resources. Focused CPU test source chains operations across two
ordered lanes, destroys caller temporary handles before drain, and checks one
late-output reservation conflict. The spatial CPU-inline/worker/CUDA test now
advances the means by a small forward step between residual evaluations; it
checks evolved state as well as residual/coefficient agreement. This is a
producer/halo/consumer test, not an implicit driver or temporal accuracy gate.

CPU generic boundary/RHS/source kernels accept private evaluator/scratch slots;
perfect-gas/RANS operators create the slots before OMP work. Chemistry retains
one leased worker until bounded team reservation is supported. Gradient limiting
and PP use disjoint per-worker scratch windows. LVR CPU now uses face sensors
followed by deterministic cell gathering, sharing the GPU decomposition; the old
scatter path remains a test oracle. Independent initialization, admissibility
and stable-step scans also have OMP paths. Error norms and generic boundary-force
reporting accept caller-owned per-worker scalar slots and merge in fixed slot
order. Reconstruction norms reuse managed two-column partial storage. Later
implicit/time kernels still need their remaining OMP work; this is not a claim
of complete CPU coverage.

`ReconstructionDeviceKernels_TF32.cu` adds the explicit experimental
`TF32TensorCore` selection for quantized unlimited Jacobi. It rounds FP32 map and
coefficient tiles to TF32, accumulates MMA products in FP32, and adds relaxed
increments to FP64 state. Mean/BC terms keep the existing FP32-map/FP64-state
arithmetic. The wrapper uses compact sweep POD arguments and explicit device
capability admission. Limited Jacobi still uses the FP32 SIMT path. Test and
benchmark sources select TF32 separately and record its error against FP64;
neither accuracy nor performance has been accepted yet.

All changes in this subsection are **unbuilt/unrun source edits**. Remaining B
work is source/inventory closure followed by the grouped build/test gate. Full
driver/linear/ODE and output integration remains Phase C, and final application
acceptance remains Phase D.

The spatial test has an explicit np=2 registration using the existing
one-rank-per-GPU launcher, in addition to its single-process registration.
Actual distributed coverage requires the np=2 result; the serial CTest entry
alone is not evidence for halo/communication correctness.

### B validation entry

The grouped CPU/CUDA/Python build has started in `build-plan61` at `-j16`.
An initial CMake aggregate-target check caught missing dependencies for the two
new CFV CUDA tests; this was corrected before successful regeneration. Logs and
source-state records live in `workspace/plan61-validation/B/`. No B runtime
acceptance is claimed until the focused tests finish.

### B runtime correction batch

The grouped build succeeded. Focused CPU entries passed after correcting the
expected drain counts for queued preparation; installed Cauchy Python operators
passed 2 tests and Runtime Python engine passed 42. Device Riemann/boundary,
Cauchy characteristic and LVR checks passed, but other CUDA suites remain open.
Unused host-only Eigen::seq construction inside shared RANS helpers caused
silent device early return; remove those declarations, retain fixed-block math,
and seed the point-test status to prove publication. CFV host BindHalo now uses
its metric mapping instead of dereferencing device placement. RANS test setup
and the combined-buffer rotation-source wrapper were corrected. Rebuilding and
rerunning this batch is still Phase B, not a new validation group. Full details
and preserved failed logs are in the plan61 status document and workspace/B.

The focused correction reruns now pass all twelve selected CUDA kernel
executables, plus the common spatial schedule at np1 and np2 with distinct
local GPUs. OMP4 spatial/RHS comparisons also pass. The queued-versus-sync
gradient check exposed and corrected a dangling Eigen displacement expression;
the np2 run corrected an owned-versus-owned+ghost wall-metric extent check.
Evidence: `workspace/plan61-validation/B/cuda-final-tests-20261113T192146+0800/`
and `B/cuda-omp4-20261113T192407+0800/`. These close the tested spatial schedule,
not the still-open implicit/time/pMG/output or final application gates.

### C source batch: sparse storage and first linear leaves

`Implicit/LinearPattern.*` and `Implicit/JacobianGeometry.*` now provide explicit
setup placement and context-aware read scopes. DNDS `BlockMatrixStorage` owns
its offsets as Runtime data and exposes a retained device-callable
`BlockMatrixArrayView`; its CPU Eigen adapter is preserved.

`Implicit/CUDA/LinearBlockKernels.hpp` / `.cu` add assembled matrix action and
ordinary/zero-start Jacobi for 5/6/7 combined AoS components. Selected core rows
need no ghost input. Whole-array residency/shape checks occur at the host
wrapper; no numerical atomics or host waits occur in these leaves.
`test/cpp/Cauchy/test_LinearBlockCUDA.cu` contains CPU-oracle and split/unsplit
comparisons with variable sparse row lengths and remote columns. **Source only:
not built or tested yet.** Managed linear integration, evaluated actions,
inversion, AmgX, GMRES, pMG/time and output remain open in Phase C.

`Implicit/LinearProblem_Queued.cpp` now retains managed numerical operands and
symbolic views for Field/EquationDofs actions. `LinearSolver.cpp` uses those
entries in the existing exchange/core/dependent sequence; the optional second
local Jacobi sweep uses queued DNDS halo copy/fill. Inverse freshness checks
submission mutation epochs. CPU matrix action/Jacobi and diagonal inversion
have independent-row OMP work with private scratch/counters. The delayed-owner
and pending-diagonal-write test source is in `test_LinearProblem.cpp`. All these
C additions still await the grouped build/test; no full implicit CUDA gate is
claimed. Evaluated CUDA actions and device inversion are still pending.

CUDA inverse source foundations are now present: DNDS
`Algebra/BatchedSVD.hpp/.cpp` isolates the cuSOLVER FP64 provider; Cauchy
`Implicit/CUDA/LinearDiagonalKernels.hpp/.cu` implements scalar inversion,
row scaling and SVD recovery with explicit per-cell numerical status. The
approved initial CUDA full-block policy uses SVD for every block with the
native epsilon-relative filter, while CPU retains guarded LU/SVD. Provider
workspace and inverse publication integration/tests are still pending; these
leaves have not been built or executed.

`Implicit/LinearProblem_Inverse.cpp` now owns the reusable inverse workspace and
managed scale/provider/recovery/status submissions. `LinearSolver.cpp` observes
the compact result at its preparation boundary before consuming the inverse.
`test_LinearInverseCUDA.cu` compares all inverse columns against CPU for 5/6/7,
scalar/full modes, healthy/rank-one blocks and empty ranks, plus one rejected
invalid diagonal. **Source-only, unbuilt/unrun.** No full-driver CUDA acceptance
is inferred from this wiring.

### C source batch: frozen Jacobian leaves (not yet validated)

`Implicit/CUDA/FrozenJacobianKernels.hpp/.cu` now provides assembled
off-diagonal block filling and direct evaluated matrix/Jacobi actions. Both
consume the same independent mean/pressure/pressure-derivative snapshot and
two-level CSR geometry. Rotated periodic images stay separate contributions;
component selection is applied in the stored equation frame, after composing
the physical derivative with the rotation. Identity faces skip rotation.
Scalar dissipation supports flow5/SA/2EQ, including the two-equation rho*k
normal-stress derivative; flow5 also supports the existing Roe matrix option
with prepared contact/acoustic magnitudes. Per-row numerical statuses replace
device exceptions or silent characteristic fallbacks.

`Implicit/CUDA/FiniteVolumeJacobianKernels.hpp/.cu` freezes owned+ghost means
with perfect-gas pressure derivatives and initializes diagonal mass and rigid
frame terms. RANS destruction and face/boundary contributions are separate
additive operators. `Boundary/BoundaryJacobian.hpp` shares the existing CPU
approximate wall reflection formula as host/device arithmetic; it does not
change prescribed/open or thermal boundary derivative policy.

`test_FrozenJacobianCUDA.cu` adds existing-CPU-block comparisons, direct
action/Jacobi checks, core launches without ghost descriptors, repeated images
of one neighbor, selected momentum components, moving frames and widths 5/6/7.
It also checks diagonal initialization. These are **source-only tests**. The
full CUDA boundary/face producer, frozen-evaluator publication and production
linear integration remain unfinished. No C compilation or execution has been
performed, and the full CUDA driver gate remains closed.

The next source edit adds
`Implicit/CUDA/FiniteVolumeJacobianKernels_Faces.cu`: centered scalar/Roe face
dissipation, mean-based physical/interface exterior states, the approximate
wall chain rule, self-periodic diagonal contributions and final component
projection. RANS no-slip exterior derivatives remain `-I`; these are not the
exact thermal/property derivatives. `eulerScalar` skips physical BC evaluation
in the scalar-dissipation diagonal, while Roe preserves its wall derivative.
Roe magnitudes are produced per incidence for subsequent evaluated actions.

Assembly, evaluated actions and diagonal face preparation now share the
point arithmetic in `FrozenJacobianKernels_Point.hpp`. Failed evaluated rows
write explicit status and NaN output, so deferred observation cannot mistake
old finite scratch values for a successful result. The focused test source now
also compares centered wall/open/self-periodic diagonals against CPU flux and
boundary producers, for scalar/physical/Roe policies and diagonal-entry/full
storage where supported. **Still unbuilt/unrun.** Managed device snapshot
publication, production preparation/scheduling and linear-solver wiring remain
open; these kernel additions do not enable the full driver.

The managed evaluated-action layer is now source-edited in
`Implicit/FrozenJacobianOperator.*`, `LinearProblem_Queued.cpp` and
`LinearProblem_ActionStatus.cpp`. Its geometry/column agreement is checked once
and shared by numeric publications. Eager callbacks retain the immutable
publication and managed operands. Context-owned action status is sticky across
the segment, explicitly reset/observed, and reduced to one result at the
solver's final numerical decision boundary. CPU/assembled operators do not
allocate this extra workspace. The focused test source includes the managed
`LinearProblemContext -> FrozenJacobianOperator -> CUDA` entry in addition to
bare-kernel comparisons. **Not built/run.** The production FV preparation
operator still needs to construct this device publication; full driver and
application gates remain open.

### Phase C linear orchestration and provider connection (source only)

The resident FV Jacobian producer is now connected through
`FiniteVolume_DeviceJacobian.cpp` and `FiniteVolumeDeviceJacobian.*`, with one
explicit status/publication boundary. Its spatial schedule test now includes
production matrix preparation and assembled/evaluated actions.

`LinearSolver` uses the same controller for CPU inline/worker and CUDA Jacobi
or GMRES. `LinearSolverAlgebra.hpp` binds vector operations to the compute lane;
`Solver/StateGMRES.hpp` accepts that per-instance adapter. Linear transfers
import the compute prefix before posting. Host Arnoldi/stopping decisions
remain explicit. `DNDS/Algebra/LocalProducts.*` provides reserved weighted
products and finite reductions, with deterministic device trees.

`LocalAMGPreconditioner.*` and `CUDA/LocalAMGKernels.*` connect the single-device
AmgX provider to an assembled rank-owned principal matrix. DNDS retains global
communication/operator semantics. Compact AoS vectors are borrowed; padded
rows use reusable device packing. Provider calls drain the engine first and
are intentionally not stream-native/capture-ready.

`test_LinearSolverCUDA.cu` adds known-solution distributed solve and reduction
checks. **All these Phase C changes remain unbuilt/unrun.** The next unfinished
parts are nonlinear update/time/pMG/output integration and the grouped C gate.

### Phase C nonlinear operator connection (source only)

`Implicit/NonlinearOperators.*` now separates managed submission/observation
from the controller and the CPU/CUDA arithmetic leaves. The solver's ordinary
and pMG single-correction paths call the same operators for defect assembly,
pseudo-time alpha, final increment repair, volume/rate conversion and mean
metrics. CPU worker, inline and CUDA use the same calls. Host-returning methods
drain at explicit convergence/acceptance boundaries; pointwise Submit methods
retain managed operands without an implicit numerical MPI operation.

`CUDA/NonlinearKernels.*` implements bounded deterministic summaries and the
perfect-gas 5/6/7 pseudo-time/correction leaves. The correction follows the CPU
policy: whole-increment thermal forecast, final thermal line search, separate
RANS scalar repair, then controller validation before mean commit. It is not
an explicit physical-time PP policy. `LinearScaling.hpp` shares the final MPI
and component normalization after backend-local mean statistics.

The spatial-schedule test source now also exercises managed nonlinear
operators and compares CPU inline/worker/CUDA with one analytical defect
check and a critical output-alias rejection. **This expanded test has not run;
old Phase B evidence applies only to its earlier source snapshot.** No Phase C
build/test has started. ODE/pMG, output, remaining OMP and full-driver gates
remain open.

### Current launch mappings (source inspection; not a performance ranking)

All mappings below describe work assignment, not SM ownership. CUDA may place
multiple thread blocks on an SM. Grid-stride loops reuse blocks/threads across
more entities than the launch grid. Specialized reconstruction selections are
explicit; they do not imply an automatic measured winner for every shape.

| Kernel | Work assignment | Notes |
|---|---|---|
| CFV Jacobi FP64 / quantized FP32 SIMT | 128-thread block per cell | Component/mode output entries distributed across threads; each thread visits neighbors and inner matrix dimension. |
| CFV FP64 MMA | Warp per cell, four cells/block | Shared matrix tiles and padded 8x8x4 MMA. |
| CFV shared FP64 MMA / async variant | 128-thread block per cell | Four component-tile warps share each neighbor matrix; async variant double-buffers `cp.async` loads. |
| CFV experimental TF32 MMA | Warp per cell, four cells/block | Shared tiles, FP32 product, FP64 coefficient contribution. |
| CFV polynomial WBAP | 128-thread block per cell | Component/mode parallelism, serial neighbor visits, barriers and six bounded global scratch rows per active block. |
| Cached/direct polynomial sampling | Thread per quadrature/sample point | Serial component/modal evaluation; separate cached/on-demand kernel specialization. |
| Cauchy polynomial PP | Thread per cell, 128 threads/block | Serial quadrature scan; separate degree-selection, compression, recovery and verification launches. No cooperative warp-per-cell implementation yet. |
| Gradient reconstruction/WBAP/Barth/PP | Thread per selected cell, 128 threads/block | Serial stencil/component work. |
| RHS flux / assembly | Thread per face / thread per cell, 128 threads/block | Face quadrature loop then ordered cell gather; no FP scatter atomics. |
| Characteristics | Thread per face, 128 threads/block | Analytic L/R matrices. |
| Aux / LVR | Thread per cell or face, 128 threads/block | Mapping follows each pass's output entity; propagation visits neighbors. |
| RANS source / source Jacobian | Thread per cell, 128 threads/block | Serial cell quadrature and source accumulation. |
| Jacobian row assembly/action, block Jacobi | Thread per block row/cell, 128 threads/block | Serial neighbor/small-block arithmetic; Phase C unvalidated. |
| Nonlinear defect/PP update | Thread per cell, 128 threads/block | Deterministic block and final summaries where requested; Phase C unvalidated. |
| Explicit SSPRK update/blend | Thread per cell, 128 threads/block | Flow5 shared flux blend or raw flow plus separate RANS repair; bounded deterministic status/minimum/count summaries. Phase C unvalidated. |
| pMG selected update/frozen check | Thread per cell, 128 threads/block | Independent selected components; bitwise frozen check uses a deterministic first-invalid-row reduction. Phase C unvalidated. |
| Diagonal pseudoinverse | cuSOLVER batched SVD | Library owns internal mapping; pre/postprocessing is thread per cell. |

The PP optimization candidate is warp-per-cell quadrature checks/reduction,
followed by cooperative coefficient scaling. It must preserve the current
degree/base selection and final recovery semantics. This is an unmeasured
candidate, not an implemented mapping or a new acceptance requirement. The
thread-per-face RHS and thread-per-cell source/PP paths can incur high serial
work and register pressure; correctness evidence does not establish optimal
mapping or coalescing.

### Phase C time/history/publication continuation (source only)

`QueuedEquationOperations` binds the generic BDF/DIRK/DITR arithmetic to a
compute lane. Method coefficient/history semantics stay in the ODE module;
whole-array algebra is dispatched by DNDS. `PolynomialMultigridOperations`
similarly supplies selected updates and cross-space identity transfers to
the unchanged FAS recursion. Coarse plans inherit device placement and the
domain context. Level callback/pool release boundaries remain explicit drains.
The reserved DOF pool now supports resident arrays; no asynchronous return of
borrowed scratch is implied.

`Core/TimeStageOperators` owns explicit-stage submission, with CPU and CUDA
leaves and no solver-loop backend branch. The CPU blend reduction uses only
minima, exact integer counts and flags, so its OMP path introduces no floating
sum-order change. Flow/species SSPRK increments remain conservative; only
RANS receives scalar exponential repair. `State/MeanStateCheck` is shared by
domain acceptance and nonlinear validation. Invalid physical candidates are
status results; invalid programming contracts remain host-side failures.

Accepted-state copying now uses `EquationDofOperations::CopyReserved`: retain
the cross-domain pre-acquired scopes, submit on the ordinary engine lane and
drain before publication. Trial/observation copies use queued algebra with
explicit observation boundaries. Quiescent snapshots acquire device-aware
read pins and mark their already-observed completion, so their destruction
does not require a stream owned by a potentially destroyed domain. This is
still a pinning snapshot, not the planned detached asynchronous output copy.

`test_TimeFASCUDA.cu` adds all-method variable-step arithmetic comparisons and
a selected two-level FAS formula oracle. `test_SpatialScheduleCUDA.cu` now also
checks explicit blending/RANS repair and numerical rejection as a status.
**Unbuilt/unrun.** Remaining C source coverage includes reconstruction restore,
full-driver device setup/capability closure, physical-step selection, interface
sampling, diagnostics/output and remaining OMP work. A/B pass records do not
validate these additions; no new build/test group has started.

### Phase C restore, timestep and reduction continuation (source only)

Reconstruction reservation/restore/reset now have context-aware CPU/CUDA
execution in `FiniteVolume_ReconstructionRestore.cpp`; finite/history checks
and coefficient copies retain their scopes through completion. Physical-step
selection dispatches through `Core/TimeStageOperators` on the common engine
lane. Driver construction places finalized metrics and combined fields once;
the full-driver CUDA capability gate remains closed until interfaces and
output/diagnostic paths are complete. CPU nonlinear norm/scale leaves now
have pre-reserved OMP partial sums with a fixed merge order; OMP=1 retains
the canonical order. These are source changes, not validated capabilities.

CFL uses block-local CUB minima followed by unsigned-bit atomic minima for
positive FP64 steps and the first invalid row. A retained immutable device
identity resets the scalar result in stream order before each call. There is
no partial table, 256-block ceiling, or separate final reduction kernel.
Empty partitions produce the identity. Invalid radii are diagnosed separately;
positive finite cell volumes are a setup precondition. This exception to
tree reduction applies to order-independent extrema, not floating sums.

`DNDS/Algebra/LocalProducts_CUDA.cu` now elects the last block using the shared
`DNDS/Device/ReductionCompletion.hpp` acquire/release helper. It completes
the fixed-index sum tree in the same kernel, without grid spinning or FP
atomic adds. The managed workspace owns/reset-orders its completion counter
and partials. Other fused numerical summaries still use their existing
two-launch implementations; conversion is not claimed complete. Integer
counts and extrema may instead use atomics when their overflow/NaN policies
permit. A hardware/library memset or copy used for initialization is still
queued work; removing the explicit tail kernel does not promise a single
device command or a measured speedup.

Focused test source covers empty/single/multiblock CFL, more than 256 blocks,
larger next-call minima, and repeated fixed-tree products. **Unbuilt/unrun**;
they belong to the grouped Phase C gate. Remaining source work includes
interfaces, diagnostics/checkpoint/async output, capability closure and the
remaining OMP/kernel inventory audit.

### Phase C checkpoint ingress and state diagnostics (source only)

`EquationDofOperations::NewVectorOn` and `CopyOwnedTo` preserve logical blocks,
physical layouts and row identity while explicitly allocating/copying on another
device. Only owned values are transferred. `StateArchive::Restore` remaps host
file shards into a host ingress mirror before uploading to resident destinations;
this also covers regrouped archive fields. Cauchy's `WriteStateArray` stages
resident checkpoint payloads explicitly. Runtime array serialization retains
its host-only direct-I/O contract. Exact stage/history snapshots are drained
after their method-owned queued copies and before serialization.

Checkpoint publication now uses retained context-aware scopes and the same
reserved queued copy used by accepted-state publication. Reconstruction and
limited-VR restart payloads are placed on the domain device, with no host loop
dereferencing them. `DomainStateSnapshot::DetachedHost` provides a deep copy of
owned means, optional reconstruction/history and their captured metadata,
including for CPU input. This is the immutable payload foundation, **not yet
the asynchronous writer queue or pinned asynchronous D2H implementation**.

`State/StateDiagnostics` dispatches the existing generic CPU point kernel and
the new perfect-gas CUDA leaf on the domain engine. It returns local mass,
energy, residual-squared and thermodynamic extrema; the domain still owns MPI
reductions and residual normalization. CUDA sums use a fixed tree with the
last-block tail; only a compact summary is downloaded. Boundary-force and
analytic-error reporting and visualization still require integration.

Focused source tests cover deep CPU snapshots, CUDA archive write/read with
padded flow5/SA/2EQ and cross-rank reverse-row remapping, and inline/worker/CUDA
diagnostic agreement with repeatable same-backend reductions. **Unbuilt/unrun.**
No new build/test group, commit, remote job or application run was started.

### Phase C force and analytic-error reporting (source only)

`Discretization/BoundaryForceReduction` now submits the CPU and CUDA force
reduction on the caller's lane. It consumes the actual per-boundary-quadrature
momentum flux written by RHS, preserving zone selection, RANS stress and
viscous traction semantics. Workspace-owned partials and selected-zone metadata
are reused. Only the final three numbers cross to the host; domain aggregation
stays outside the queued segment. The fixed-flow CPU diagnostic accumulator
remains its existing fast path.

`Discretization/FlowErrorOperator` supplies the production analytic-error path
for inline/worker/CUDA execution. Its lazy CUDA cache stores max-order reference
quadrature coordinates/weights, prepared once with count/allocate/fill and OMP
after warming shape caches. It uses the same cell-average analytic profile and
rigid-frame transform as the CPU oracle; optional RANS components do not enter
the flow-five error. Geometry access is managed and validates the metric
generation. The one-shot `FiniteVolume::Error` delegates to the same operator;
Domain retains the operator across reports. GPU errors and forces both use the
last-arriving-block fixed-tree tail and operation-owned completion counters.

Test source adds selected-zone force comparisons including duplicate zones,
empty selection and repeatability, and production error-operator comparisons
across backends. Existing raw analytic-kernel tests now exercise the fused tail.
**Unbuilt/unrun**; these additions await the grouped C gate. Visualization and
asynchronous output, interface sampling, capability closure and the remaining
OMP/source inventory review are still open.

### Phase C interface numerical operators (source only)

`CFV::DistributedInterfaceSamplingPlan` now has one queued donor-evaluation
entry for inline CPU, worker CPU and CUDA. Constant and physical-gradient
sampling use view-only kernels (one CUDA thread per query); VR sampling uses
the existing queued polynomial evaluator. The return payload still uses the
canonical direct Runtime transfer binding. Matching/query routing remain CPU
setup. The synchronous convenience explicitly drains its private engine.

Device-only VR can retain cell moments via the opt-in
`ReconstructionResidencyOptions::arbitraryPointSampling`. Cauchy requests this
for interface configurations. Rebuilding sample points receives explicit FV
metrics and can temporarily download moments/metrics for CPU setup without
retaining or downloading VR matrices or solution fields. Non-interface cached
configurations do not gain an unused moment allocation.

Receiver mapping has moved from `InterfaceCoupling`'s inline numerical wrapper
to `InterfacePacketOperator`, with separate CPU/CUDA leaves and shared pure
frame rotation. The prepared operator snapshots row mappings, validates unique
destinations and component permutations, retains placed metadata and operands,
and preserves packet rows belonging to other connections. CPU query rows use
OMP; CUDA supports the combined AoS contract. This does **not** yet close the
full Cauchy CUDA coupling path: its lane placement, boundary packet allocation,
publication/validation and MPI completion bridge still need integration.

Focused test source covers distributed direct-return sampling on CPU/worker/GPU,
rebuilding sampling from device-only VR, and receiver component permutation,
frame rotation and untouched rows. **Unbuilt/unrun**; the grouped C build/test
gate has not started.

### Phase C interface integration and output production (source only)

The follow-on source edits connect device-aware `InterfaceCoupling` lanes,
resident boundary packets, queued donor sampling, Runtime `SubmitAfter` /
`EnqueueWait` transfer edges and the receiver mapping operator. All connection
transfers are posted before receiver completion is consumed. Repeated refresh
reuses the CPU-prepared matching route. Packet publication checks only selected
interface rows with `MeanStateCheck`, avoiding a full packet download. Its
host-observed publication boundary remains explicit: the full interface path
is not claimed capture-safe or sync-free. Cauchy's selected transport remains
direct MPI; the generic Runtime edge does not itself enable NCCL in Cauchy.

`SubmitArrays` and `SubmitArrayBatch` now retain their kernel capture alongside
operand reservations through retirement, even when a launcher discards its
submission callback after launch. Thus geometry/metadata owners captured by
sampling/mapping survive device execution without accumulating engine-wide
retentions across refreshes. Test source uses a deferred-retirement launcher
to exercise this contract directly.

Focused CUDA test source now includes actual colocated/dislocated interface
refresh and route reuse, masked packet acceptance and frozen packet placement.
These are operator/coupling correctness tests, not a full multidomain flow
accuracy campaign. The public synchronous `FiniteVolume::Admissible` also uses
the resident `MeanStateCheck`; domain MPI acceptance still occurs on the host.

Visualization cell preparation is extracted into `OutputFields`, with
view-only CPU/OpenMP and CUDA leaves. It produces logical conservative columns
plus p,T,u,v,w,Mach and, on CPU mixtures, all mass fractions. CPU point workers
use the nonthrowing property entry; CUDA uses the shared perfect-gas helper.
Status is checked before output is copied to host. The device surface adapter
uses the existing resident boundary-trace producer with finalized coefficients
and frozen interface values, then the same output-property producer. The file
writer only formats prepared host fields; it does not download coefficients
for CPU sampling or recompute device point physics. Boundary records now carry
their already-prepared quadrature weight for output without downloading all
volume metrics. Output fallback is explicit and does not mutate reconstruction.

The output-field test source checks inline/worker/GPU, empty rows, 257 rows and
5/6/7 components against independent perfect-gas values. **None of these Phase C
edits have been built or executed yet.** `git diff --check` passes. Asynchronous
pinned snapshots/writer queue and full output/restart validation remain open,
as do driver capability closure, remaining OMP/inventory review, the grouped C
gate and final Phase D solver campaigns. Synchronous output is not credited as
completion of the asynchronous-output requirement.

### Phase C detached checkpoint and writer foundations (source only)

`Buffer::AllocatePinnedHost` adds zero-initialized portable CUDA page-locked
storage while preserving `Device::Host()` view residency. Allocation records
identify pinned storage and reports include unique pinned-host bytes. There is
no pageable fallback when CUDA is unavailable. Ordinary CopyOperation leases
retain source/destination until completion; this is reusable Runtime support,
not a Cauchy-owned raw allocation.

`EquationSnapshotOptions::pinnedHost` selects that destination policy. All
physical-field copy operations are submitted before observing their completion.
`DomainStateSnapshot::DetachedHost` uses it for device means, reconstruction,
history and LVR payloads; CPU snapshots remain independent ordinary copies.
The source snapshot remains pinned through D2H, so later mutation cannot race
the copy. The detached result may remain alive during subsequent solver work.

`PreparedCheckpoint` separates accepted-step capture from serialization. It
owns the frozen state/control metadata, cell IDs, mesh envelope and shared I/O
communicator, and contains no live Domain reference. Its Write preserves the
v5 schema. The synchronous Domain writer uses this implementation too, retaining
the zero-copy immutable CPU snapshot option; device capture explicitly stages
to host. Internal exact-stage checkpoint writing has not yet been converted
into an asynchronous request.

`OutputQueue` provides one ordered writer, a default capacity of two, and
backpressure before controller-side snapshot preparation. Submitted writers
must own complete immutable inputs. Production failure callbacks must abort
without collective recovery; only UT callbacks may return to inspect a recorded
failure. Drain observes completion; Close joins pending work. Test source covers
ordering/snapshot independence and terminal failure. Additional test source
covers pinned DMA exclusion/accounting and checkpoint ownership after its
source Domain/communicator handle is released.

These facilities are **not yet connected to OutputCoordinator's asynchronous
mode**, and do not establish MPI/HDF5 thread-safety or complete async output.
Pending integration must validate MPI_THREAD_MULTIPLE, isolate I/O collective
ordering, serialize HDF5 access, prepare visualization snapshots, publish
manifests only after success, and drain before synchronous internal/final output
and shutdown. Everything remains unbuilt/unrun until the grouped Phase C gate.

### Phase C asynchronous output integration (source only)

`output.asynchronous=false` is now a registered configuration option.
`AsyncOutput` validates MPI_THREAD_MULTIPLE, creates separate I/O control/domain
contexts once, and shares them across bounded writer requests. Capture runs on
the controller, with no live Domain/Simulation reference escaping into the
writer. `PreparedVisualization` separates numerical preparation from VTKHDF
and surface writing, retaining detached fields and a mesh read. CPU production
already makes a distinct output allocation; no second full cell-field copy is
needed. GPU fields transfer to accounted pinned host storage.

`OutputCoordinator` routes accepted-step/final snapshots through this service
when selected; internal output drains before retaining its synchronous path.
Final output and teardown drain too. The writer preserves direct/multidomain
checkpoint schemas, generation directories, output names and manifest ordering.
Its failure scope and handler abort independently without distributed recovery.
SerializerH5 sessions and Geom's VTKHDF transaction now share an HDF5 lock;
the installed HDF5 reports `Threadsafety: OFF`, making this serialization
necessary. Raw external HDF5/CGNS use and mesh mutation require a prior drain.
The serializer also checks file-close success before output is considered done.

Focused test source exercises detached checkpoint+visualization output while
the foreground mutates its means and performs MPI, then restores the captured
state and checks surface values/completion metadata. Reduced MPI thread support
exercises the explicit rejection instead. **Still unbuilt/unrun**; no async
runtime acceptance is claimed. Remaining Phase C work includes source review
of this integration, capability/inventory closure, OMP review and the grouped
CPU/CUDA/distributed output-restart validation. Phase D runs remain open.

### Phase C placement and capability audit continuation

The uniform-layout `FiniteVolumePlan::BuildField` overload now inherits the
plan's device, matching `BuildField(width)`. Previously its default host
argument placed pMG coarse means on CPU even when the coarse metrics were on
CUDA. Explicit placement remains available for staging, and Python exposes the
same two overloads. The queued interface fixture now checks this allocation
contract on host and device; full pMG execution remains a grouped-test gate.

Async output resource construction and initial reporting now precede callback
installation. A rejected MPI thread capability or failed output initialization
must not leave callbacks borrowing an incompletely constructed coordinator.

The configuration resolver now selects the implemented CUDA profile instead
of rejecting every CUDA domain. It requires combined AoS/perfect gas, supported
SA or k-omega models, Jacobi VR, and Jacobi or GMRES with Jacobi/AmgX. The same
linear checks apply to each resolved coarse-level override; AmgX requires the
compiled provider and assembled storage. Existing CPU and extended-equation
restrictions are retained. This is source-level capability routing, **not
runtime qualification**. Focused configuration tests are added but unrun.

Source tracing confirms the CPU `PrepareFlow5Jacobian` specialization is
downstream of the device dispatch in `PrepareJacobian`, and production mean
scaling uses the queued CPU/CUDA reduction operator. The host-only helper
overloads are not the GPU nonlinear path. Phase C remains unbuilt/unrun;
remaining inventory/OMP review and grouped validation are still required.

The independent Jacobian-initialization loops now use the established OMP row
helpers. Dense mass/rotation initialization writes the strided output directly
instead of sharing a temporary Eigen matrix. Pseudo-time/PP-alpha preparation
now partitions rows with exact count/minimum reductions and deterministic
first-invalid-row status. No FP summation atomics or MPI work enter these
loops. These changes are also unbuilt/unrun. The more involved correction and
thermodynamic Jacobian loops still need their worker/scratch ownership review;
the OMP coverage audit is not closed by these two changes.

### Phase C source batch entering grouped validation

Nonlinear correction now receives operator-owned partition scratch and merges
exact counts/minima after independent row updates. The leaf no longer allocates
species scratch. Generic Jacobian completion has separate joined phases for
mean properties, boundary properties and row assembly, with private evaluator,
point and matrix scratch. Perfect-gas/RANS source Jacobian calls select the
existing parallel worker kernel; mixture calls retain their already-held
single chemistry worker to avoid bounded-pool starvation. Flow5 Roe assembly,
frozen-row copies and ILU template filling also partition independent rows.
Ordered SGS/ILU triangular algorithms are unchanged by these edits.

`test_DomainCUDA.cpp` adds production-domain checks beyond leaf tests: a
nonuniform P3 BDF2/GMRES solve, ESDIRK4/pMG solve, inline/worker/device comparison,
and CUDA checkpoint continuation preserving BDF history and coefficients.
Its distributed entry exposes one local device per rank. The diagnostic
extraction anchor follows the new Jacobian worker argument.

The Phase C source batch is now entering its grouped build/test round. All new
results remain pending; fixes and reruns belong to this same round. Phase D
physical campaigns and final requirement-by-requirement qualification remain
separate gates, including the OMP=1 numerical anchor and OMP scaling reading.

### Phase C first build observations

The grouped j16 build and retries are recorded under
`workspace/plan61-validation/C/`. The Cauchy static library and canonical
`app/cauchy.exe` linked in `build-retry3.log`. The complete target set has not
passed and no Phase C test execution is credited. Corrected the first compiler
findings: dependent-template disambiguation in queued linear access, direct
OptionalRef/ComponentSelection includes, the concrete reconstruction-state
include in the interface test, and doctest's single-argument CAPTURE contract
across the new CUDA tests. Retry4 terminated on the latter test issue; its
corrections await the next build invocation.

A further source audit found a remaining requirement in section 9 of Plan61:
CPU linear SGS does not yet consume mesh subpartition ranges, although VR SOR
does. Close this before Phase C acceptance. Preserve the single-thread scan;
with multiple threads, lower scans read frozen old values across partitions,
and upper scans read the completed frozen lower result across partitions.
Within each partition retain the existing logical order, core/dependent call
contract and distinct output buffers. Carry immutable partition metadata from
FV/mesh setup through the linear operator; do not query mesh ownership or add
lease work inside numerical loops. ILU elimination/triangular ordering stays
unchanged. This gap is not covered by independent-row OMP tests.

The subsequent source correction now carries `LinearSweepPartitions` from
FV mesh setup through the retained linear pattern into CPU SGS. Within-block
logical scans and frozen cross-block reads implement the policy above. A
four-owned/two-ghost dense oracle exercises ordinary/zero-start and optional
middle-exchange split sweeps, with both Field and EquationDofs interfaces and
column-major storage. A separate np2/OMP2 entry checks the threaded policy;
ordinary linear tests retain the OMP1 oracle. Source implementation is complete
for this gap; build and numerical validation remain pending.

Retry5 compiled the SGS implementation and test but failed in two other test
fixtures: dynamic Eigen inverse under nvcc and obsolete raw-communicator
`DofSpace` constructor calls. The inverse oracle now resides in a host `.cpp`
driver calling the same CUDA launch wrappers; time/FAS fixtures retain an
explicit shared communication context. Retry6 continues the same grouped
build round. Source-review notes and the exact dirty source snapshot are in
`workspace/plan61-validation/C/`; numerical tests are not yet credited.

### Phase C runtime corrections and partial validation

Retry7 rebuilt the full targets. All 16 selected CPU entries passed, including
the np2/OMP2 SGS oracle. Device1 linear, time/FAS and full-domain advancement
(BDF2 and ESDIRK4/pMG plus restart) now pass. Four two-device entries passed:
distributed sampling, colocated/dislocated interface refresh, linear solve and
complete domain advancement. Python operator and RANS domain checks passed four
cases at np1 and on each np2 rank after rebuilding/installing the libraries.

The production fixes are queued producer-to-transfer ordering (without immediate
FV-entry lease probes), and stable-context immutable Jacobian read guards that
cannot retain an expired preparation-stream address. Existing failed tests were
retained, then rerun. Remaining spatial fixture branches must supply complete
RANS references; this is still part of the current C correction round. Final
physical campaigns and source-linked coverage closure are not established by
these UT results. Detailed failure/pass logs remain in the C evidence folder.

