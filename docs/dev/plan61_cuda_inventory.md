# Plan 61 CUDA implementation inventory

Baseline: `c1700f6d`, branch `codex/cauchy-cuda`. Updated 2026-11-13 (Asia/Taipei).

**Phase A validated; phase B source editing in progress.** The initial header catalog
below is a navigation aid. The recursive source-linked call/leaf audit is still
open; header enumeration alone does not establish complete hot-path coverage.
Every numerical helper reached by supported flow5/SA/2EQ AoS perfect-gas paths
must execute on CUDA. CPU OMP=1 remains the mathematical anchor.

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
`produce matrices -> validate once -> WBAP(order P..1)`. Same-stream ordering
establishes readiness without a host wait or CUDA event; cross-stream producers
need an explicit event dependency. Frozen CPU-finalized transforms retain their
identity/completed-version checks. Device-produced transforms retain identity,
shape and residency checks but do not interpret a pending writer's completed
version as readiness.

`BuildQueuedFaceTransform` and `SubmitValidateFaceTransform` provide that explicit
publication contract for modal and gradient WBAP. After each production/mutation,
the caller queues one inverse-consistency check into the consuming workspace's
sticky status, then may run multiple limiter passes without repeating the matrix
check. Caller must not mutate matrices between validation and consumption.
Scheduled leases prove access ordering, not that numerical precondition. The
synchronous convenience APIs perform their own validation before their drain.

Device validation failure guards limiter output; the controller checks status
after the segment. Gradient workspaces now have the same deferred status contract.
Test sources enqueue producers, validation, multiple WBAP passes and publication
without an intermediate drain, compare each pass with CPU, and check one invalid
producer. These are not yet executed. The actual Cauchy physical characteristic
producer and production orchestration integration remain pending.

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

