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
remains open. The direct RTX3060 `auwd` port times out, but WSL inventory now
works through `auw`; remote source/runtime staging is still pending. Final
physical gates, quantized-kernel assessment and full completion remain open.
