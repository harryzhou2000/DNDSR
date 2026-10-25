# Native periodic construction: memory and frame audit

Read-only source audit of the current working tree, 2026-10-26. This is a
storage/call-path review, not a measured large-mesh scaling result. Focused
periodic/connectivity tests passed at MPI 1/2/4/8 in
`/tmp/geom-native-periodic-second-tests.log`; they use small independent fixtures.
Storage follows local requested/served incidence volume plus O(P) rank metadata,
including temporary copies and relation/path multiplicity. Genuine local support
can cover the global mesh; no owned-only memory bound is claimed.

1. No production allocation indexed by the global entity count or persistent-ID
   high-water mark was found in the periodic builder, shared-node closure,
   topology completion or assembler. `NumGlobal()` is used for validity/range
   checks. Distributed entity offsets and peer route inventories require O(P)
   storage per rank. IDs remain payload; a sparse/high persistent ID does not
   create an array extending to that ID.

2. Primary inverse support is routed to canonical node owners, carrying only
   incidence contributions (`NativeConnectivity::Inverse`). Node owners retain
   the complete incident support of their owned nodes. Ghost discovery and
   `BuildNativeCellClosure` request only reachable parent/node rows; face/edge
   interpolation enumerates the resulting local closure. Intermediate rows and
   publication copies scale with owned/queried topology incidences, rather than
   an unconditional global table. Several simultaneous copies exist during
   setup; this audit does not claim zero-copy topology construction.

3. High valence is a material limit. A canonical hub node can have support from
   every cell, so its owner and all requesters can legitimately obtain an entire
   mesh through a complete node star. Periodic image collapse can increase this
   effect. The guarantee is absence of unconditional global-sized indexing;
   there is no bound on a requested closure independent of the mesh topology.

4. `PeriodicBoundaryRelations` stores and fetches each local
   `(boundary, candidate-cell)` occurrence separately. If K is their count, its
   request/candidate metadata is O(K), and cone/frame payload is O(K times the
   candidate cell width). The same cell can be fetched more than once for
   different boundaries. In a pathological all-canonical-one-node input this
   can approach `B_local * C_incident`; candidate discovery and setup costs can
   therefore be quadratic in genuine local support. A future optimization can
   fetch unique candidate-cell globals once and retain per-boundary indices into
   that table. This is not an existing optimization.
   `Runtime::RowTransferPlan` rejects request extents outside its MPI integer
   limits. Exact frame comparison inspects at most eight masks over a bounded
   element subentity; its temporary storage does not depend on global counts.

5. `NativeConnectivity::Evaluate` retains one reachable set per compiled tree
   node and re-encodes owned relation rows per hop. Prefix sharing prevents
   duplicate traversal prefixes, but distinct paths/publication variants can
   retain overlapping sets and payload copies. No cache/memory complexity
   stronger than the reachable incidence volume is asserted.

6. Builder frame agreement broadcasts 60 binary64 components and compares their
   values exactly. Assembler includes all 60 components in its rank-agreed
   schedule using `std::hexfloat`, not `std::to_string` or default decimal
   precision. Hexadecimal formatting preserves the binary64 significand. A
   local standalone probe distinguished adjacent doubles at normal, subnormal,
   minimum-normal and near-maximum scales. Both entry paths reject nonfinite
   components and non-rigid rotation matrices before topology communication.
   The existing three-bit, uniform-XOR frame model remains the contract; this
   does not introduce a general noncommuting transform-group representation.

7. Remaining validation limits: no new large-scale peak-RSS or CUDA periodic
   construction measurement was performed for this audit. CGNS periodic pairing,
   Python periodic builder arguments and periodic refinement/extraction remain
   outside the implemented construction slice. The earlier negative matcher
   oracle failure is preserved in `/tmp/geom-native-periodic-first-tests.log`;
   that pattern was transform-equivalent and was corrected before the passing
   matrix.

Source anchors at audit time:

| Path | Anchor | Relevant storage |
| --- | --- | --- |
| Boundary candidate fetch | `src/Geom/Mesh/NativeMeshBuilder.cpp:122`, `PeriodicBoundaryRelations` | One request/payload row per boundary-candidate incidence. |
| Shared-node closure | `src/Geom/Mesh/NativeMeshClosure.cpp:8`, `BuildNativeCellClosure` | Reachable parent cones and aligned element/frame companions. |
| Ghost traversal | `src/Geom/Mesh/NativeConnectivity.cpp:113`, `Fetch`; `:132`, `Evaluate` | Owned row copies per hop and reachable tree-node sets. |
| Inverse support | `src/Geom/Mesh/MeshConnectivity_Native.cpp:152`, `Inverse` | Owner-directed incidence batches and owned-node support. |
| Rank metadata | `src/Geom/Mesh/Mesh_EntitySpace.cpp:28`, `CreateOwned` | Rank counts/offsets and explicitly requested ghost maps. |
| Exact frame comparison | `src/Geom/Mesh/NativePeriodicMatch.hpp:15`, `NativePeriodicFrame` | Eight masks over each small node-image pattern. |
| Frame agreement | `src/Geom/Mesh/NativeMeshBuilder.cpp:480`; `src/Geom/Mesh/NativeMeshAssembler.cpp:232` | Fixed 60-component frame data; hexadecimal schedule representation. |
