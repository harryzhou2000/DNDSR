# Geom facade and transition checkpoint — 2026-10-24

Implementation checkout: `DNDSR-array-geom`, branch
`codex/refactor-dnds-array-geom`. Approved work order: the plan repository's
`docs/17-geom-facade-migration.md` at `c531ccc`. Array baseline: `232df1a0`.
Implementation checkpoints: `19f726b5`, `92e14290`; closing test additions:
`3258e82f`. The bounded GE0–GE6 phase is validated with the compatibility
limits below; this is not a complete native rewrite of every Geom workflow.

## Delivered boundary

| Package | Implemented result |
|---|---|
| GE0 | Caller, mutator, cache and compatibility inventories; distributed face lookup reproduction and closing tests. |
| GE1 | Explicit legacy host storage bridge, canonical EntitySpaces, retained read scopes and allocation-free host/CUDA kernel queries. Legacy copying and read-only device views remain compatible. |
| GE2 | CFV/Euler semantic queries and field setup, cache dependency checks, managed geometry setup, EulerP native snapshot adapter, Python read/edit scopes and headless overset callers. Numerical formulas remain in their existing operators. |
| GE3 | Private native authority for primary/facial/edge groups; current-authority relation readers; native inverse/composition, GhostSpec and interpolation builders, shared entity routes, ordered periodic companions and collective schedule checks. |
| GE4 | Detached mesh/field preparation and coordinated publication, stable original IDs, preserved/rebuilt faces and edges, typed EntityField association, current-state export and coherent primary-mesh/four-history checkpoint composition. |
| GE5 | Real-mesh neighbor stencil with independent original-ID oracle, CPU inline/worker overlap and two-GPU direct MPI; execution after permutation, redistribution and actual field/mesh restart. |
| GE6 | Managed mutation/version contracts, explicit native authority, stale-cache/field/binding rejection, ownership diagrams and recorded compatibility exceptions. |

The same query semantics work with legacy or native backing. Native publication
retains private values, row metadata and canonical maps; raw legacy workspace
edits do not change it. A managed read owns leases outside the hot loop. The plain
view and its row spans borrow pointers and do not acquire ownership or locks per
element. `DeviceView()` is directly callable in CUDA code, with bounded inline
coordinate outputs. Existing `Mesh_DeviceView.hpp`, `ArrayPair.hpp`, and
`ArrayAdjacency.hpp` are unchanged from the Array baseline.

Transitions prepare destinations and all registered fields before publication.
Old retained readers continue to see their old state; new typed operations reject
wrong mesh/kind/order and obsolete bindings. `EntityField` supplies this stronger
association around `Runtime::Array`; generic Arrays and raw registrations still
require the caller to establish semantic row correspondence. The existing eager
engine remains responsible for launch/completion; callers supply cross-stream
dependencies. No second Geom scheduler was introduced.

## Compatibility limits

1. Legacy arrays remain available and are the default setup backing. Raw public
   arrays, metadata and device-mirror methods retain their phase-barrier contract.
   They are not retrospectively made lease-safe. Native authority is explicit.
2. Existing full-mesh import, ghost/interpolation setup and transition rebuilds
   still reuse validated legacy workspace algorithms before native publication.
   The new native DSL is a separate real builder implementation; it does not
   silently replace every legacy setup algorithm.
3. Managed native edits support owned coordinates, same-type cell `ElemInfo`
   changes and neighbor permutations. Arbitrary topology changes, native periodic
   frame transforms and general shape edits require a separately prepared mesh.
   Passive legacy elevation/bisection requires the legacy workspace.
4. Detached transitions reject parent/extraction and elevated-prefix inventories
   lacking an explicit transfer policy. Supported volume meshes preserve or
   rebuild faces/edges. Native serial-reader and reduced-boundary output reject;
   current distributed volume output uses a synchronous detached export.
5. The primary mesh codec remains the existing Serializer format. Runtime fields
   can be saved under a jointly retained snapshot, as tested with four histories;
   this is not an atomic filesystem transaction or a complete solver restart.
   Auxiliary rebuild/drop policies are explicit.
6. Python managed rows retain their scope and metadata; native CUDA debug access
   synchronizes or returns an explicit snapshot. Legacy Python properties remain
   compatibility escapes. There is no Python `EntityField` binding in this phase;
   Python Runtime Arrays retain their documented untyped association contract.
7. Orchestration and mesh publication are serialized. MPI process failure,
   arbitrary mismatched user collectives and universal allocation-failure recovery
   are outside the contract. Correctness tests do not establish performance gains,
   GPUDirect transport or complete Euler feature parity.

## Validation

| Gate | Result |
|---|---|
| Broad CPU CTest, MPI 1/2/4/8 where registered | **172/172**: DNDS 55, Geom 91, CFV 9, Euler 17. |
| Closing CPU native setup/restart/typed field checks | **12/12** at MPI 1/2/4/8, after rebuilding the `3258e82f` test additions. |
| CUDA Geom query/overlap/field gates | **3/3**, including actual checkpoint restart and two distinct GPUs for direct MPI. |
| Legacy CFV CUDA transfer | **1/1**. |
| Installed CPU Python, MPI 1/2/4 | **54 passed, one CUDA-only skip per rank**. Overset cases execute. |
| Installed CUDA Python, MPI 2 | **55 passed per rank, no skips**. Separate EulerP smoke explicitly uses `isCuda=True` and exits successfully. |
| Native CUDA query memcheck | **9 selected cases, 768 assertions, zero errors** under the limited configuration below. |
| Representative 2EQ/EX consumer compilation | **8 syntax checks passed**; command and source-hash manifest retained. |

The broad run validates the formatted production implementation at `92e14290`.
The subsequent `3258e82f` commit changes tests only: the closing CPU run and CUDA
rebuild execute its added restart and multi-parent-edge cases. These are distinct
runs, not a fictitious single 184-test invocation. The final source manifest pins
`3258e82f`; all syntax-check production hashes still match. The installed Python
extension **and implementation-library** ELF IDs match their selected build;
`Runtime.cuda_enabled` is checked for each profile. The checkout is left with
the verified CUDA Python profile installed.

Logs and exact commands are in [the evidence index](geom_validation_2026-10-24/README.md#final-formatted-source-regression).
The [source/binary manifest](geom_validation_2026-10-24/final-source-and-binaries.json)
and profile JSON files identify the tested artifacts. Historical failures remain
in the same directory and are not relabeled as successful evidence. No failing
required test remains at this checkpoint; this is not a claim of universal bug
freedom or exhaustive coverage of all legacy physics/options.

The sanitizer boundary is deliberately narrower than the ordinary CUDA tests:
the recorded clean native query run disables MPI's CUDA accelerator component
and excludes the legacy parity registration case. Normal two-GPU direct-MPI
tests exercise communication. Neither their successful assertions nor the
query-only sanitizer result proves sanitizer-clean GPU-aware MPI shutdown.

The next solver architecture, tree backend, overset coupling, AMG integration and
physics expansion remain separate work. This checkpoint changes the mesh access
and transition foundation while retaining the existing solver families.
