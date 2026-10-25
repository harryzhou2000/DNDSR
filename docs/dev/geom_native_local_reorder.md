# Native local cell reorder

`NativeMeshTransition::ReorderLocalCells(source, mpi, nParts, nPartsInner,
ghostLayers)` is collective and returns a detached native state plus an explicit
`PermutationTransfer` for every published family. All entity ownership stays on
the same rank. Cell owned rows follow the permutation; other families keep their
numeric owned order. All fixed and named relation references, element metadata,
periodic frames, persistent IDs, wall vectors and ancestry travel through the
existing native transition implementation. No legacy mesh is allocated.

The permutation is computed by the shared `ComputeCellPermutation` algorithm:
Metis partitioning, optional inner partitioning, RCM, then private-before-outer
sorting within each first-level part. Exact face neighbors form the partition
graph; full node-neighbor `CellCells` rows determine whether a cell touches a
ghost. A remote contact at only one vertex still makes a cell outer. Prepared
native facial topology is required, including locally available cells on faces
incident to owned cells. Generic primary-only input is rejected rather than
approximating face adjacency.

The new state retains the mesh namespace and persistent ID high-water marks.
Rebuilt entity map objects describe the destination order, including families
whose numeric row order did not change. Existing fields and routes remain
associated with the source publication. Callers may retain the source and its
read views, explicitly transfer external fields using the returned maps, and
construct a new facade with `UnstructuredMesh::FromNative`.
The result's `sourceSpaces` records the exact input association for each family;
validate fields against those descriptors before applying numeric transfer maps.

The public facade `ReorderLocalCells` also dispatches native authority to this
implementation. This entry is setup-only: it rejects active managed readers,
publishes after preparation succeeds, and invalidates old field/map associations
through layout/distribution generations. It does not silently move external
solver fields. Its old legacy implementation remains available for compatibility.

Local partition boundaries and their filtered owned face-neighbor cache are
installed on the newly prepared state. `WithLocalCellSetup` retains its stricter
contract of filling metadata for already established boundaries; it cannot
silently repartition an existing setup.

Partition counts follow the existing public normalization to at least one.
Overflow and inconsistent collective options are rejected before transfer.
Empty ranks participate in validation, routing and publication with empty maps.
The focused target `geom_test_native_mesh_local_reorder` checks legacy permutation
parity, corner-only ghost contacts, inner parts, empty owners, topology and payload
identity, retained source reads, explicit external-field transfer, invalid options
and facade version invalidation. All three test cases passed at MPI sizes 1, 2, 4
and 8 in the CPU profile; the final run is recorded in
`/tmp/geom-native-local-reorder-third-tests.log`.

The small graph cases exposed two defects in the shared serial helper. Metis's
contiguous-part requirement is now disabled for disconnected input graphs while
connected inputs retain the previous option. The temporary insertion counters
are sized by the partition count, including when there are more parts than
cells. Direct disconnected, empty and singleton tests cover these cases.

The input-only `EntityReorderMap`, `FollowSpec` and `ReorderInput` declarations
live in `MeshReorderTypes.hpp`. That neutral header includes no legacy mesh,
array or transformer. `ReorderPlan.hpp` keeps source compatibility by including
it; native transitions include the neutral declarations directly. The native
transition result still exposes `PermutationTransfer`, whose compatibility
header retains legacy row-transfer templates.

`NativeMeshState::ArrayBytes()` reports rank-local retained Runtime array
allocation capacity across all published placements, named relations, persistent
IDs and ancestry. It includes value buffers and host/device row metadata, counts
shared allocation identities once, and includes the full backing owner of an
imported subspan. It excludes STL containers, mappings, access controllers and
route caches. The facade's `getArrayBytes()` preserves its global MPI sum result.
The Runtime allocation test and native construction tests passed in the CPU
profile, including MPI sizes 1, 2, 4 and 8 for construction. The exact native
oracle covers imported backing capacity, shared fixed/named arrays, distinct
values sharing CSR metadata, persistent-ID/ancestry aliases and retained state
copies. Logs are `/tmp/geom-native-memory-runtime-tests.log` and
`/tmp/geom-native-memory-construction-final-tests.log`. The CUDA allocation case
passes on physical GPU 1 (one case, three assertions), and the full CUDA mesh
facade suite passes there (11 cases, 807 assertions). This includes the exact
native host/device allocation delta and retained asynchronous read lifetime.
The final logs are `/tmp/geom-native-memory-runtime-cuda-gpu1.log` and
`/tmp/geom-native-memory-full-facade-cuda-gpu1.log`. These are single-GPU checks;
they do not establish fresh two-GPU communication coverage.
