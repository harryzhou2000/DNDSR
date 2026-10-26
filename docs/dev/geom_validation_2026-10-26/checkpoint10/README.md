# Native Geom ownership retirement and completion gates

The refactor removes mesh-owned legacy ArrayPair payloads and the legacy
construction/publication switch. `UnstructuredMesh` owns immutable native state;
file input, build, topology, mutation, transitions, restart and output use Runtime
Arrays. Legacy DNDS Arrays remain supported for solver fields. Python retains
readonly pair spellings and explicit managed geometry edits.

This record extends checkpoints 6–9. The original 23-topic brief and architectural
scope are unchanged. Final validation results and source/binary manifests are
recorded here; historical failed and interrupted runs remain distinguishable.

## Concrete changes

1. Remove the old mesh storage, serial reader/partition chain, state-checked legacy
   DSL, mutable raw adjacency APIs and duplicate native publication/capture paths.
   Keep the semantic C++ facade, standalone legacy field/array utilities, one-way
   historical primary-wire decoder and output-only SerialRW adapter.
2. Retain native state in managed read scopes, including through scope moves and
   placement/cache-handle replacement. Hide unrequested relation groups. Preserve
   explicit local/global conversion against canonical entity orders; compute
   publications remain local and preserve unavailable/absence encodings.
3. Refresh existing Face/Edge topology using sparse exact keys with persistent
   node IDs, periodic image occurrences, vertex roles and edge incidence. Hashes
   route records; complete keys establish identity. Preserve surviving IDs and
   namespace/high-water marks, allocate new keys above the mark, and retain the
   explicit destroy/reconstruct fresh-ID policy. Remap ordered named relations
   and ancestry, re-sort set variants, and reject noninferable changed paths.
   Reject cracked high-order faces even when their full node multisets match.
4. Keep partition's source field associations by completing only missing faces
   before placement. An explicit topology refresh may change face ownership;
   partition's returned source transfers must begin at the original source order.
5. Reuse global traversal payloads and cache local-to-global conversion once per
   immutable registered relation. No global-entity-sized lookup/scratch table is
   introduced. Add independent high-valence node gather/SUM/scatter, cell/edge
   bridge, frame/order and scratch-memory scenarios.
6. Use exact sparse/high numeric IDs in legacy solver-field restart routing.
   Resolve fixed and dynamic vector widths from stored matrix metadata. Exercise
   actual Euler H5 save/reload, changed ownership, selected-variable import,
   duplicate/missing IDs and collective error/retry behavior.
7. Migrate the original unit-test purposes and Python setup demos. Preserve the
   full 958,994-cell mixed Ball2 gate, exact ordered incidence/periodic/geometry
   oracles, meaningful CPU reference paths and CUDA numerical comparisons.

## Requirement-to-evidence map

| Requirement | Direct evidence |
|---|---|
| GN0 relation definitions | `test_NativeConnectivity`, original connectivity/ghost/interpolation tests, `test_NativeMeshScenarios`: direct/inverse/composed any/all/exact relations coexist; actual traversal and periodic frame matching |
| GN1 native rows/access | `test_NativeMeshConstruction`, snapshot/publication/facade tests, Python pair/POD tests; retained immutable arrays and shape metadata; no legacy mesh payload fields remain |
| GN2 independent construction | Native builder, CGNS/OpenFOAM/archive fixtures and standalone Runtime inputs, including independently owned entities and empty ranks |
| GN3 sparse closure | Remote valence-eight owned-node star; distinct Hex8 Node–Cell–Node and Node–Edge–Node paths; multilayer/periodic closure; source allocation audit and measured C++ traversal scratch |
| GN4 identities/transitions | Full ordered records for all five families, field transfers, rollback, high-water/nonreuse, exact-key refresh, named/provenance remap and deliberate 64-bit hash collision |
| GN5 mesh workflows | Native elevation/bisection/extraction, geometry/affine transforms, wall distance, smoothing, current serial/distributed output and matching/changed-NP archives |
| GN6 consumers | CFV and Euler numerical/field/output/restart tests; EulerP independent volume reference; native CUDA mesh/CFV kernels; Python helper and headless demo checks |
| GN7 independent execution | Runtime-input and file/restart paths, inline/worker CPU and two-GPU direct-MPI overlap before/after permutation, redistribution and full restart |

`test_NativeMeshScenarios` overrides all scalar/array/aligned/sized/nothrow C++
allocation forms in its dedicated executable. A known 1 MiB vector verifies local
interception. Only requested bytes allocated on the phase's calling thread count;
counter epochs survive frees outside that phase. With a fixed observed closure,
unrelated remote triangles grow 1 → 128 → 4096. Measured warmed four-hop traversal
peaks remain 7,780 bytes at np2, 8,148 at np4 and 8,884 at np8. Retained mesh-state
Runtime buffers remain 2,000 bytes. These are **not** whole-process peaks: registry
caches, setup outside the phase, C malloc, MPI internals, CUDA and other threads
are excluded. Source inspection separately covers builder, exact-key routing,
inverse and remapping allocations. High-valence or hash-server work can itself
be large; the bound is local requested/served incidence plus rank metadata.

## Validation record

The first broad ownership-retirement run passed 179/187 registrations. All eight
Pipeline/Reorder failures were migrated-fixture errors: global VTK offsets,
incomplete custom descriptors, omitted identity transfer entries and a missing
face prerequisite. Corrected registrations pass at MPI 1/2/4/8. A repeated
validating-reader acquisition inside a large test loop was moved outside the
loop; only the owned test processes were interrupted, and mesh sizes stayed intact.

The first Python run similarly preserved obsolete legacy failure expectations.
Native staged callback failures leave the source valid; readonly POD references
require an explicit mutable snapshot. Corrected tests exercise rollback/retry and
readonly rejection. A subsequent changed-NP archive test caught the partition
source-association issue described above; preserve that failure and its focused
and matrix retries. A C++ regression moves face ownership independently and checks
source field bindings and exact ordered records.

Final matrices are recorded in the closure logs/manifests. CPU and CUDA Python
profiles share one installed package and run sequentially. Each profile rebuilds
and installs all four bindings, then verifies eight actual ELF build IDs and
`Runtime.cuda_enabled`. Two-rank CUDA Python starts one process per physical GPU
using the retained rank-to-UUID launcher. CTest discovery and syntax checks are
never counted as runtime passes. No production-scale accuracy/speedup claim follows
from these unit/mock solver scenarios.

The full closure Geom run passes **191/191** registrations at MPI 1/2/4/8,
including the complete mixed Ball2 Pipeline and index gates. The additional
partition source-order case and final frame-matching scenario pass separately
at all four sizes. CPU Python passes **45 plus two documented skips per rank**
at MPI 1/2/4/8. CUDA Python passes **46 plus one documented skip per rank** at
MPI 1/2, with exactly one rank per physical GPU. Both profiles verify all eight
extension/shared-library IDs. Four CUDA CTest gates pass: mesh views, CFV kernels,
and two-rank direct-MPI overlap and entity-field transfer. The final relinked
DNDS/CFV/Euler run passes **81/81** registrations, including actual sparse/high-ID
Euler H5 restarts and static/dynamic selective variable import.

## Explicit limits

- Node-centered **topology/closure and numerical gather/scatter** are covered;
  a node-centered dual-volume metric/discretization is outside this phase.
- Topology mutation, interpolation-matrix construction and external CPU physics
  remain host operations. GPU topology mutation, stream-ordered MPI and CUDA
  graph capture are not claimed. Overlap gates use direct CUDA-aware MPI and
  explicitly observed active transfer/read-lease intervals.
- Legacy Euler solution checkpoints match numeric Cell IDs without a mesh
  namespace check. Callers must pair them with the same physical mesh. Full
  native mesh archives retain namespace and topology; they do not retrofit the
  old solution-checkpoint format.
- The full sanitizer run remains **nonzero**: duplicate CUB kernel registration
  and OpenMPI CUDA finalization error713. Ordinary CUDA tests pass, but are not a
  sanitizer-clean result. Checkpoint7 contains the independent two-TU/RDC CUB
  reproducer. `mpi-cuda-finalize-reproducer/` reproduces the identical finalize
  error without DNDSR; CUDA-before-MPI is a clean diagnostic control. No API-error
  suppression, project RDC removal, installed MPI change or service change was
  used. The executable is omitted; exact source, commands, outputs and hashes
  suffice to rebuild it.
- Raw borrowed C++ views, unmanaged writes, foreign streams and communicator
  lifetime retain their documented caller responsibilities. No universal race
  detector is claimed. Serial file ingress/output and replicated wall/smoothing
  spatial datasets have their separately documented scaling limits.

Evidence text removes terminal ANSI and escapes NUL bytes; original/recorded
hashes are retained where copied. Build products and unrelated draft files are
excluded from commits.
