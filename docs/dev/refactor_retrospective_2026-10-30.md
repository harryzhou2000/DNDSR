# Array/Geom retrospective audit, 2026-10-30

Follow-up: [the accepted fixes](accepted_audit_fixes_2026-10-30.md) resolve the
field-layout and compiled-program issues described below. This document retains
the initial findings and reproducer evidence; it is not the current defect list.

Scope: `codex/refactor-dnds-array-geom` at `f8fff305745d29d8b7d450dbfb0e46a5c92a1b78`
against its `dev/harry` merge base `4e9c3acb339f3d4c2196db6e0f57067aa719c98c`.
The source/test inventory spans 526 changed files. This is a semantic review of
the complete refactor's subsystems and their boundaries, with focused source
inspection and regression experiments, not a claim that every possible input
or every changed line has independent test coverage. The new solver, sliding
coupling and tree mesh are not implemented by this audit.

## Review map

| Boundary | Source inspected | Main acceptance question |
| --- | --- | --- |
| Storage and shape | Runtime Array, Buffer, Metadata, RowBuilder, AllocationResource | Do aliases retain allocation and shape, and can unpublished shape escape? |
| Execution | EagerEngine, EagerArray, CUDAExecution, Version, MPIProgress | Do commands retain accesses through actual completion and surface failures? |
| Communication | RowTransfer, DynamicRowTransfer, schema/binding APIs | Are source/destination order, collision, layout, lifetime and collective failures separated? |
| I/O and Python | ArrayIO, Runtime binding, Array binding, DLPack binding | Is one schema used, with correct ownership and collective preflight? |
| Fields/algebra | DofField, DofAlgebra, EntityField, FieldTransfer | Does row identity remain distinct from component/shape compatibility? |
| Mesh connectivity | EntitySpace, MeshAdjacency, AdjacencyRegistry, ghost compiler/evaluator/assembler | Are branches distinct until marked endpoints are collected; are published ghosts canonical? |
| Mesh lifetime | State, Snapshot, MeshAccess, mutations, wall-distance publication | Do derived caches depend on the generations actually changed? |
| Consumers | FV/VR settings, geometry, reconstruction, limiter and DOF factory | What can be retained mathematically without retaining legacy field ownership? |
| Tests/docs | Native/legacy purpose ledger, connectivity/field/CFV tests, source and caller contracts | Is the oracle independent, and is its claimed scope accurate? |

The owned-only construction pipeline and final caller-specified ghost assembly
remain intact. No optimization, field transfer, or interface plan in this audit
bypasses the DSL, adjacency objects, ghost tree or EntityGhostMapping facilities.

## Local fixes

1. **Static DOF component count was not enforced on imported/replaced Arrays.**
   `Runtime::Array<T, L>` is a compatibility alias of a storage-oriented handle;
   `L` is not an ownership invariant. `DofField<double, 5>` accepted two width-six
   Arrays because validation compared them only with each other. The new check
   explicitly enforces `Components`, before any algebra writes. The regression
   also tests mutable handle replacement and verifies that failed Fill leaves
   the original payload unchanged.

2. **Missing dynamic-transfer source could dereference null before collective
   rejection.** `DynamicRowTransfer` now checks the shared source inside its
   existing collective preflight. One rank's missing source rejects all ranks,
   retains valid sources, and permits a subsequent transfer with the same route.

3. **Wall-dependent VR bases could survive wall-distance replacement.**
   `BuildNodeWallDist` publishes Values, Interface and Allocation changes, while
   the reconstruction stamp previously watched geometry/topology/order only.
   Its supposed wall identity helper returned an empty vector unconditionally.
   Remove that ineffective check and include Values in the base dependency mask
   when wall distance is used. Geometric metrics and wall-independent VR remain
   valid. The regression rebuilds wall distance without moving the mesh, requires
   rejection of stale bases/coefficients, then rebuilds successfully. This is a
   conservative dependency: future per-companion generations could avoid
   invalidating wall-dependent bases on unrelated mesh-value publications.

4. **An unpublished shape could split collective checkpoint control flow.**
   C++ and Python now share `Detail::IOPublishedStorage`: pin the published
   storage/shape, agree availability, then enter the existing direct writer.
   Moved handles and rank-local exclusive edits fail before checkpoint mutation.
   Python also agrees on dtype/class lookup success before collective reads;
   an invalid object dtype on one rank cannot leave peers entering file I/O.
   Tests check that no rejected group/payload appears, then commit/restore the
   shape and write/read successfully. This does not turn multiple fields into a
   transaction or recover arbitrary HDF5/MPI process failures.

5. **CSR draft allocation arithmetic had unchecked additions.**
   RowBuilder now checks slab-padding additions and the signed logical extent
   contract before allocation. Near-SIZE_MAX byte-row requests previously could
   wrap reserve sizing. Tests reject such requests without changing existing
   draft rows and then commit ordinary rows successfully. No enormous allocation
   is attempted by the test.

These changes are at validation/setup boundaries. They add no per-element lease
checks, host payload staging, global mesh-sized scratch, or numerical algorithm
changes. CUDA kernels and native mesh construction algorithms are unchanged.

## Decisions before the next feature implementation

1. **Field row-policy ownership needs one consistent design (correctness issue).**
   `Geom/Transition/FieldTransfer.hpp::RegisterArray<T,L>` replaces an inferred
   destination with `Dense(rows, L::Default().Width())` when L is supplied.
   For TableRows/CSRRows/PaddedRows the default width is zero; FixedRows<N,S>
   loses S; StaticMaxRows confuses capacity with width. EntityField still carries
   L and checks it when rebinding, unlike the new Array handle. Existing tests
   principally use generic EntityField or fixed width with default stride, so
   they do not establish all typed field migration cases. Payload checks usually
   reject the wrong shape rather than silently copy it, but the public operation
   is not complete for these policies.

   Recommendation: make `EntityField<T>` storage-oriented too, with `View<L>()`
   selected on access. Migration accepts a separate explicit destination layout
   policy: compact/inferred by default, or preserve table stride/padded capacity
   when requested and collectively compatible. Keep a temporary typed wrapper
   only for source compatibility. Decide whether preserving physical capacity is
   a default promise or an opt-in; arbitrary rank-local capacities need an agreed
   rule. Do not silently advertise every typed field policy as supported today.

   A standalone two-row local-permutation probe at np1 and np2 reproduces this:

   | Source access policy | Migration | Subsequent matching typed access |
   | --- | --- | --- |
   | TableRows, width 2, stride 4 | Rejected | Not reached |
   | FixedRows<2,4> | Succeeds but destination stride becomes 2 | Rejected |
   | CSRRows, lengths 2/1 | Rejected | Not reached |
   | PaddedRows, lengths 2/1, capacity 4 | Rejected | Not reached |
   | StaticMaxRows<4>, lengths 2/1 | Rejected | Not reached |

   The diagnostic deliberately reports failures; its successful process exit is
   not a passing migration UT. Source/build/run logs are retained as
   `field-policy-probe*` in the audit data directory. The initial diagnostic
   compile used a nonexistent RowNumber handle accessor; corrected RowCount
   builds produced the results above. The production issue remains unresolved
   pending R1's API/layout policy decision, not hidden behind an expected-failure
   assertion in the normal suite.

2. **Compiled ghost programs should be immutable (interface safety).**
   `CompiledGhostTree` exposes roots and a separately copied BFS execution table.
   Agreement and required-adjacency discovery walk roots; execution walks levels.
   A caller can modify one representation without the other, defeating agreement
   or producing invalid parent/node indices. Normal `compile(spec)` output is
   coherent and existing branch tests exercise that path. The public comment now
   explicitly forbids post-compile edits; this does not enforce immutability.

   Recommendation: private representation, factory compilation, const diagnostic
   accessors and one canonical program signature. Recompile changed GhostSpec.
   This is preferable to adding a complete structural audit to every Evaluate.
   It changes direct-inspection call sites and needs a deliberate API migration.

3. **Consolidate field associations without merging numerical and mesh layers.**
   Runtime::Field/Exchange is an early compatibility prototype, Algebra::DofField
   owns owned/ghost algebra rows and a DofSpace, and Geom::EntityField associates
   storage with EntitySpace and transition publication. These are different
   responsibilities but too many similarly named entry points for a new solver.
   Keep generic storage/execution and numerical algebra in DNDS, association in
   Geom, and domain/history management in the solver. Provide an explicit
   EntitySpace-to-DofSpace adapter with shared identity/order, rather than copying
   counts or introducing a fourth field owner. Retire the prototype gradually.

4. **Small dependency cleanup should precede adding interface services.**
   Connectivity currently includes the field-transfer header for collective local
   error agreement/alias selection helpers. Extract a narrow DNDS MPI utility,
   keeping field migration in Transition; preserve collective failure behavior.
   Split large transition/registry implementation files by responsibility, not
   by introducing a second algorithm. Keep readable named local stages where
   collective order matters. A wholesale style rewrite would obscure this audit.

   Native-prefixed builders/services are still intentionally retained by the
   published source-organization contract. Rename them together later, with
   forwarding aliases, rather than mixing another set of partial names now.
   `Interpolate` in connectivity means topological subentity construction; the
   new numerical API should say `Sample`/`PointInterpolation`, never reuse that
   name for an unrelated operation.

## Test honesty and remaining evidence limits

The new DOF and wall-distance tests were first run against unmodified production
code at this audit's starting revision. They failed respectively 4/5 and 2/6
assertions, demonstrating actual missing checks rather than tests mirroring the
fix. The null-source and extreme-size cases were not run against known unsafe
old paths; their regressions exercise collective rejection/recovery and bounded
allocation behavior on the corrected code.

The existing CFV detached-publication test samples this implementation before
and after restoring a snapshot. That is a useful metamorphic test of retained
state, not an independent numerical oracle. Its comment now states this. CFV
polynomial/reference fixtures, exact connectivity expectations, independent
periodic-frame checks and the 47-purpose legacy test ledger remain separate
evidence. Finite-only assertions in smoothing tests cannot establish legacy
coordinate parity; the previously recorded RBF difference remains explicit.

Do not equate a CTest registration with an assertion count, or a passing np1
early-return path with multi-rank coverage. New solver, distributed sliding,
tree specialization and comprehensive CUDA sanitizer acceptance remain untested
features. The previous np32 performance acceptance applies to its recorded
revision; no new performance result is claimed here.

Current validation logs are under ignored
`data/retrospective-audit-20261030/`. No generated binary or large raw log belongs
in this document.

| Check | Result | Scope |
| --- | --- | --- |
| Rebuilt native DOF, transient transfer, native I/O, RowBuilder and CFV reconstruction | 17/17 CTest registrations passed, zero skipped | Registered MPI 1/2/4/8 variants; RowBuilder is a local test |
| Rebuilt CUDA Array, eager engine and DOF algebra | 3/3 CTest registrations passed | Ordinary CUDA execution; no sanitizer or new multi-GPU claim |
| Four rebuilt/installed Python bindings and test bridge; Array, bridge, RowBuilder and engine tests | np1: 63 passed, 21 skipped; np2: 63 passed, 21 skipped on each rank | All skips require the CUDA Python profile, unavailable in this CPU binding build |
| Field-policy diagnostic | Reproduced R1 at np1 and np2 | Deliberate defect probe, not an acceptance pass |
| Plan validator | Passed; original brief unchanged | Document links/provenance only |

The first Python launch used a missing checkout-local `venv/bin/python` and ran
no tests. The successful retry used the interpreter actually configured by CMake,
`/home/harry/projects/DNDSR/venv/bin/python`, after building and installing all four
extensions plus the bridge. Failed launch logs are retained separately. The
diagnostic build typo described above likewise produced no runtime evidence.
The final Python dtype-read guard was followed by another complete binding build/
install and the same np1/np2 tests; `final-cpu-py*` logs retain those final results.

Comments now identify Runtime::Field as a compatibility prototype and document
CompiledGhostTree immutability and typed field migration limits. The caller
contract distinguishes optional checked views from the absence of per-element
lease checking. These are documentation corrections, not stronger eager-order
validation than the implementation provides.

## Next CFV and interface work

The standalone plan repository records the implementation order and numerical
acceptance gates in `docs/27-retrospective-cfv-interface-plan.md`. Recommended
direction: a new compositional CFV frontend on Runtime storage, extracting the
existing mathematics and keeping old VR as a reference/legacy frontend. Geom
supplies boundary geometry, search, correspondence and stencil membership;
CFV supplies reconstruction and high-order point sampling. Conservative flux
coupling, ALE/GCL and FSI are later, separately specified operations.
