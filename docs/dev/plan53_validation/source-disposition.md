# Plan53 final source disposition

Reviewed audit implementation through `4decc781`, descended from refactor pin
`aec8d0e4`. This is a targeted review of the changed interfaces and their
callers, not a claim that every line of the solver has been audited. The
independently advancing implementation checkout and MG/IV jobs are excluded.

## Responsibilities and hot-path cost

1. **Runtime owns storage, access and transport.** Metadata-only address
   predicates replace higher-level raw pointer arithmetic without a lease or
   synchronization. CPU/device kernel views remain borrowed pointer/stride
   descriptors; no shared-owner lookup or lease check is added per element.
   Remaining LegacyPointer calls are confined to legacy allocator compatibility,
   retained immutable device shape buffers, unpublished RowBuilder slabs,
   persistent MPI address registration and private transport scratch. Backend
   code retains their actual owners. Independently imported foreign aliases
   still need one shared access controller: address-overlap detection cannot
   make separate controllers coherent.
2. **CommunicationContext owns optional safety policy.** Prepared starts omit
   safety-only agreement in off mode, but still acquire/check local accesses.
   Disabled-mode rejection aborts the communicator; it does not promise a
   recoverable rank-local exception. Route counts, schema/control selection,
   ghost-tree schedules and file transaction protocols remain required.
   Numerical norms, timestep decisions and candidate acceptance stay collective.
   Array handles do not silently carry a mesh communication policy: explicit
   contexts cross the Geom/Runtime boundary. Restored checkpoint maps default
   to checked preparation unless explicitly rebound at a drained boundary.
   Final review fixed companion BeginRows restoring default-on checks in its
   isolated namespace, and its unconditional source safety reduction. The
   existing policy case passes np1/2/4 with simultaneous companion transfers.
3. **TransferGroup shares launch reservation only.** It uses existing member
   bindings and normal operation completion, with rollback before posting on
   enabled validation failure. It does not duplicate route compilation,
   flatten fields, change tags or coalesce messages. CPU persistent Direct and
   HostPacked only; scheduled/checked-stream and device/provider modes reject.
   Group and member operations must be serialized by the caller. This contract
   is stated both in the header and RuntimeCallerContract; the per-binding
   mutex alone is not a concurrent-start guarantee. Separate starts remain
   production behavior after the bounded E4/E5 assessment.
4. **Geom owns discovery, identity and publication.** AdjacencyRegistry retains
   input read scopes; transient closure rows and the final EntityGhostMapping
   remain explicit. Its cached fetch is one last frontier per descriptor, not
   proof that two branches or a final union have identical ghosts. Exact
   IDs/order and retained generations gate reuse. Ownership is not weakened
   to reduce acquisitions. The narrow-read E7 retain recommendation leaves a
   future requirement-aware capture optimization inside this same boundary.
5. **CFV owns reconstruction mathematics and prepared capabilities.** Optional
   tables and face-system scratch follow the requested operation dependencies.
   Raw matrices are assembled independently of SVD-filtered inverses. Missing
   operations reject at entry, before output acquisition/queueing; cache policy
   remains independent. Geometry is pinned and version-validated before use.
   Requirements are not checked in the cell/quadrature hot loop.
6. **Cauchy owns physical orchestration.** Shared auxiliary reduction and
   evaluated-stencil traversal helpers do not own EOS workers, live fields,
   communication or nonlinear state. Exact identity dispatch is one immutable
   plan/geometry flag, not one added field per quadrature point. Any rotated
   incidence keeps the plan on the general path. Sources, moving frames and
   pressure derivatives retain their original numerical producers. Nonlinear
   solvers borrow a stable engine; pMG retains a finest facade by value.
   Copies share mutable implementation and are not independent solvers.

## Tests, consolidation and readability

Aggregate dependencies now build every registered Cauchy executable, and split
SOR compares the actually committed coefficients. Lazy mesh fixtures avoid
importing unrelated large data for filtered tests. The 3D routine fixture and
retained large golden/convergence fixture have separate tier labels; assertions,
goldens and old integration registrations remain. Selected routine np1/2 tests
retain np4/8 in extended coverage; empty-owner, subset-writer, communicator and
multidomain high-rank contracts stay routine. No claimed wall-time saving follows
from labels alone.

Private helpers consolidate exact auxiliary reduction, adjacency facade keys,
local products and geometry-only evaluated stencil construction. Existing
generic nonlinear and pFAS orchestration are retained; no replacement generic
framework or new top-level library was introduced. The source Jacobian's
reconstructed chemistry/RANS work is not replaced by a mean-state shortcut.

Manual review included Runtime Prepare/Post/completion/rollback, CFV capability
construction and entry checks, registry fetch/publication, and flow5/equation
Jacobian and nonlinear adapters. Dense multiline callback bodies remain in
some numerical and setup implementations. Their phase names, ownership
comments and exception boundaries are retained rather than splitting live
lease/collective scopes solely for formatting. The local Cauchy binding
formatter override is idempotent and lexical-token preserving; parent numerical
formatting is unchanged. This closes the bounded readability treatment, not a
repository-wide style cleanup. The effective shared Git hook currently resolves
to the same tracked script bytes as this checkout; partial-stage behavior has
an isolated tooling regression. No shared hook was rewritten by this audit.

## Acceptance limits

The execution ledger records focused Runtime/Geom/CFV/Cauchy and installed
Python gates, original/current Euler bit comparisons, CPU ASan/UBSan ownership
checks and matched E2–E7 observations separately. Earlier source pins are
retained; a later routine pass does not relabel old measurements as current.
CPU sanitizer coverage excludes leak detection, distributed MPI and CUDA.
No fresh CUDA/NCCL runtime or capture qualification is claimed here.

The E4/E5 mock comparison establishes actual transfer correctness, ownership
and useful scheduling overlap under an equal two-core budget. It does **not**
complete E5's composed-RHS net-benefit/staging-time acceptance. Coalescing,
early-ready-field delay, cross-node executor behavior and a production worker
remain unmeasured, so no automatic executor adoption is made. Keep these limits
visible when reconciling plan52 rather than marking the full package complete.
