# Cauchy preliminary infrastructure completion

Implementation checkpoint: `6ff890d6`, branch `codex/refactor-dnds-array-geom`.
The independent plan execution ledger is `dndsr-next-plan/docs/40-cauchy-preliminary-execution.md`;
its `experiments/cauchy-preliminary/` directory holds compact numerical/timing
rows, manifests and hashes. This phase builds infrastructure, not Cauchy itself.

## Delivered boundaries

- Canonical DNDS/Geom/CFV responsibility folders and opt-in historical include
  overlays; legacy qualified names and Python surfaces remain available.
- Type-oriented Runtime Array storage with Table, ColMajorTable and tiled column
  tables; separate padding/alignment, stride-aware/static/checked views and the
  centralized Eigen row adapter. Native algebra, direct MPI, host archives,
  Python and DLPack use actual physical offsets. Device I/O still rejects.
- CPU CFV accepts physical U/URec layouts without changing prepared internal
  arrays. Gradient setup/evaluation are outside its public header. All six moved
  method bodies match the reference except configurable access-bounds policy.
- Explicit resident mesh/metric/basis/operator publications, retained source
  generations and device views; CPU-prepared CUDA Jacobi/WBAP with separate
  output, bounded scratch, core/dependent lists, periodic/characteristic data,
  axisymmetry and activation/error status. GPU SOR rejects explicitly.
- Explicit [device-packed transfers](../architecture/DevicePackedTransfer.md):
  CUDA gather, contiguous schema-typed CUDA-aware MPI and scatter/reduction,
  with nonblocking Begin and MPI-thread progress. Direct remains default.
  [Device reconstruction](../architecture/DeviceReconstruction.md) documents
  eager ordering/lifetime, capabilities and Python shared ownership.

## Completed validation

| Gate | Result |
|---|---|
| Full DNDS/Geom/CFV/Euler CTest |330/330 pass,461.73s |
| Runtime/overlap after DevicePacked |75/75 pass |
| CUDA VR, one GPU |5 cases,349,287 assertions |
| CUDA VR, two GPUs |5 cases/rank,171,138 /178,168 assertions |
| DevicePacked standalone |6 cases,34,114 assertions at np1 and each np2 rank |
| DevicePacked kernel sanitizers |memcheck/racecheck/synccheck:0 errors/hazards |
| Rebuilt/installed Python+Torch |68 pass15 expected skips, serial and each np2 rank |
| Explicit GPU layout conversion |232 assertions, memcheck0 errors |
| Matched allocated-node CPU benchmark |8 cases×5 trials, np1/2/4/32, all numerical oracles pass |
| Real GPU direct/packed subset |8 cases×3 layouts×3 trials, all complete and numerically pass |

The GPU suite includes nonzero fields, all physical table families, rotated
periodic vector blocks, nonidentity characteristic transformations, axisymmetry,
3D19-mode/37-component reconstruction, empty ranks, activation and malformed
workspace rejection. Communication tests include CSR/padded/zero rows, POD
extent/alignment/omitted bytes, collisions, transient/persistent requests,
collective rollback, retained handles and gated multistream overlap.

Two bugs found by the expanded coverage were fixed: host face/edge extraction
uses an explicitly laid-out functor across GCC/NVCC template instantiations;
format conversion selects addresses from host shape metadata instead of reading
CUDA CSR offsets on the host. The initial gated fixture's module-loading stall
was retained as failed evidence; the final warmed fixture requires independent
core completion while packing remains pending.

## Performance and limits

At262,144 cells/np32 on one exclusive BSSCA node, degree2 native/legacy preparation
is1.453/3.095s, halo-inclusive Jacobi29.457/40.323ms and WBAP73.377/106.324ms.
This CPU checkpoint was `aa6817b3`, before later device/ABI guards and source-only
gradient separation. It establishes legacy comparability, not exact overhead
against the previous native implementation.

On two local A100s at65,536 cells, degree2/21 components, direct versus packed
column communication is556.331/0.311ms, integrated iteration560.906/2.659ms;
tiled is557.052/0.305ms and560.205/2.767ms. AoS integrated is2.368/2.402ms.
These are sequential baseline measurements with CPU compilation concurrent;
GPUs were otherwise idle. Packing removes the observed fragmented-MPI cost;
it does not prove GPUDirect or optimal kernel mapping. The earlier direct-only
matrix stopped partway; its incomplete campaign is distinguished in the ledger.

The raw sanitizer reports CUDA713 in OpenMPI5.0.9 finalization's host-memory
unregister, after numerical tests pass. Kernel sanitizer runs disable API-error
reporting and are clean; they do not validate that external cleanup. No MPI
source/environment workaround or sanitizer suppression is installed.

GPU gradients/sampling/linear actions, parallel SOR, tree meshes, graph capture,
automatic eager stream-dependency checking and FP64 tensor-core kernels remain
explicit future capabilities. Current sm80 machine code has zero DMMA/HMMA and
ordinary DADD/DMUL/DFMA. Callers still order cross-stream access, progress MPI,
retain native streams and observe errors at completion boundaries.
