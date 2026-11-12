# Plan 61 CUDA implementation inventory

Baseline: `c1700f6d`, branch `codex/cauchy-cuda`. Updated 2026-11-13 (Asia/Taipei).

**Phase A source work, not validation evidence.** The initial header catalog
below is a navigation aid. The recursive source-linked call/leaf audit is still
open; header enumeration alone does not establish complete hot-path coverage.
Every numerical helper reached by supported flow5/SA/2EQ AoS perfect-gas paths
must execute on CUDA. CPU OMP=1 remains the mathematical anchor.

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
| [CFV direct gradients](../../src/CFV/Operators/GradientPlan_Operators.cpp) | GG/LS/GGMP phases and prescribed reconstruction BC values | Device work pending |
| [CFV LVR](../../src/CFV/Operators/LimitedVRPlan_Operators.cpp) | Target/gate/expansion phases; transfers outside kernels | Device work pending |
| [CFV polynomial sampling](../../src/CFV/Operators/PolynomialSamplingPlan_Values.cpp) | Cached/direct values and gradients, periodic transforms | Device work pending |
| [PerfectGas](../../src/Cauchy/Physics/PerfectGas.hpp) | Shared pure host/device math; compact immutable descriptor | Existing helpers; transitive audit pending |
| [Riemann](../../src/Cauchy/Riemann/RiemannFlux.hpp) | Shared header arithmetic and physical setup, gas-agnostic flux contract | General evaluator/setup still out of line; B work |
| [RANS model](../../src/Physics/RANS/Model.hpp) | SA/2EQ source, diffusion, boundary and approximate Jacobian helpers | Transitive CPU/device/OMP audit pending |

## Work and validation groups

- **A editing:** whole-array Device tags and projections; host wrapper checks;
  launch argument budget; import checks; complete inventory; context/placement;
  allocation naming/accounting; CPU/setup OMP; source-built single-device AmgX.
- **B:** every supported spatial/CFV/device physics path and precision option.
- **C:** Jacobi/AmgX/GMRES/pMG/time arithmetic and asynchronous snapshots.
- **D:** final two-local-GPU cylinder and auwd NACA0012 acceptance.

No build or numerical test has been run for these edits. Finish a group's source
and focused test-source edits before its grouped build/test round. New unit tests
cover contracts concisely; end-to-end acceptance is separate evidence.

