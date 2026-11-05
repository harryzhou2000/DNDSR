# Plan 53 audit execution

## Source and ownership boundary

This ledger executes plan-repository commit
`06ae603cfd5171c55f3f045ec6b760fbbdc35ef7`, reading
`docs/53-refactor-tip-plan-reconciliation.md` before its coordinated plans 52,
54 and `audit-2026-11-04-programming-euler-performance.md`.
Implementation is confined to `codex/cauchy-audit-fix` in
`workspace/worktrees/DNDSR-cauchy-audit-fix`, with paired `build-audit`.
The other agent's implementation checkout and MG/IV experiments are excluded.

The branch begins at refactor pin `aec8d0e4194aa3ab39758ead263e0929fd1b9135`.
Existing audit commit `7f60eda6` fixes shared SST production and independent
RANS reconstruction positivity, with seven passing focused registrations.
It does **not** close the programming audit, UT consolidation or E1–E7.
The common MPI initializer already checks the provided thread level; that
resolved finding will not be implemented a second time.

## Execution queue

1. Immediate correctness: aggregate target completeness, split-SOR publication,
   MPI initialization return handling, SINGLE/worker rejection and retained
   nonlinear/pMG lifetime contracts. Validate with existing bounded fixtures;
   no new solver trajectory.
2. Private helper consolidation: identical auxiliary reductions and mesh
   adjacency descriptors first; keep leases, EOS and published mesh ownership
   in their current layers. Address unnecessary raw buffer queries and selected
   readability issues without changing kernel access cost.
3. Test organization: separate routine contracts from integration/benchmarks,
   avoid repeated solves and eager large-mesh construction, retain meaningful
   MPI sizes and give serial Python tests one CI owner. Preserve independent
   numerical oracles and all expensive tests as runnable coverage.
4. E1: optional safety-preflight policy reaches Runtime, default on, with
   required protocol collectives and rank-asymmetric failure semantics intact.
5. E2: capability-selected CFV setup storage, compatibility default, explicit
   operator requirements and no recovery of raw matrices from truncated inverses.
6. E3–E7: bounded, matched end-of-goal assessments of coefficient publication,
   halo starts, MPI executor choices, approximate-Jacobian actions and Geom
   acquisition. Record retain/change recommendations, including inconclusive
   results. Performance claims require actual preparation/application costs and
   consistent numerical acceptance. No MG/IV campaign is part of this queue.
7. Shared Euler seams: small independent before/after checks for chemistry,
   transport, source Jacobians, species repair, SA and QCR. Sanitizer ownership
   checking is a separate bounded job, not another full test matrix.

## Current changes and acceptance

The immediate patch adds both missing Cauchy aggregate dependencies, explicitly
commits split-SOR scratch before comparison, checks MPI initialization status,
and rejects CPU workers with actual `MPI_THREAD_SINGLE` support before domain
engine construction. The configuration guard has a bounded capability test;
it does not introduce a communication executor or lower MPI requirements.

pMG retains the finest solver facade by value, sharing its implementation.
Both solver families therefore tolerate moving/destroying the original facade;
their callbacks target a stable hierarchy member. Engine borrowing remains
explicit across every solver/hierarchy copy and queued operation. The existing
flow5 and equation fixtures now exercise expiration of the original facade;
copies remain serialized, not independent solvers.

Safety changes were committed as `01803350`; private helper and buffer metadata
changes as `de8d66e5`. The latter shares exact auxiliary reduction arithmetic
and mesh facade descriptors, adds metadata-only alias/alignment queries and uses
held operation views for transient direct MPI posting. Persistent datatype
bindings and private staging/immutable device metadata retain their justified
raw-pointer access. Host/device kernel views and their indexing cost are unchanged.

## Validation and test organization

Release CPU, GCC 13, Open MPI 5.0.9, Cantera enabled, CUDA disabled, canonical
external installation, paired `build-audit`, eight compiler workers. CTest uses
OMP2 unless its frontend explicitly sets OMP1. These are local correctness and
test-cost observations, not exclusive-node matched performance assessments.

1. Initial configuration/split-SOR checks: 5/5 registrations pass, np1/2/4/8
   split-SOR plus configuration (1.50 s). Publication occurs after both split
   regions before coefficients are compared.
2. Fresh `cauchy_unit_tests` builds both previously missing executables, including
   equation FV and reduced-order SA. Fifty bounded Cauchy entries pass, including
   np1/2 restart/nonlinear/auxiliary/physical operators and frontend (113.82 s).
   Lifetime fixtures include expired/moved original finest-solver facades.
3. Ten Runtime array/transfer/DOF/native-mesh-transition entries pass (5.90 s),
   including np4. Buffer metadata checks include separate foreign imports,
   partial/adjacent/empty ranges and active-writer exclusion.
4. Tooling: 20 tests pass (final run 3.009 s), including transitive aggregate coverage and
   orphan rejection. All 457 configured C++ registrations have module/tier
   labels; 16 retain the large legacy integration fixtures. Four new routine
   registrations use the same 3D executable with a generated 64-cell mesh.
5. After migrating the touched Geom/3D entry points to the DNDS MPI initializer,
   the final split-SOR/3D batch passes 10/10 registrations (100.27 s). The
   consolidated 32^3 CFV integration passes np1/2 at 59.56/38.03 s; the
   generated routine contract passes np1/2/4/8 at 0.32/0.26/0.31/0.38 s.
   All six original golden
   results, P1 stopping checks and P2/P1 ordering remain; no shared mutable
   cache between test cases. These costs are not before/after speedup evidence.
6. Both Geom executables support `--count` without the NACA/Ball2 fixtures
   present. Selected UniformSquare subcases pass at np2 (pipeline cell count
   and nodal global/local round trip), demonstrating that filtering avoids
   unrelated imports. The independent topology-incidence case passes on NACA
   at np2 in both ordinary one-pass mode and the opt-in three-repeat benchmark
   target. Full NACA/Ball2 integration has not been rerun in this batch; it
   remains registered.

The initial wider Cauchy run had 46 passes and two checkpoint-fixture failures:
the isolated tree lacked `IV10_20.cgns`, and its rank-subset writer failure
stranded nonwriters. That local test invocation was terminated. The test now
checks fixture availability collectively before splitting writers. Copying the
canonical small fixture yields the successful 50-entry rerun above. Its SHA256
is `682bf5b2ed1edf9515cfd5003ea5748cd9b7ccaf66a18fa349e6225e608b9a0b`.
These fixture/setup failures are not hidden or counted as numerical passes.

Repeated topology timing is separated from its one-pass correctness test and
has an opt-in benchmark target. CI serial Python coverage has one owner;
explicit multirank Runtime Python tests remain. This CPU batch does not claim
Python extension runtime, CUDA or sanitizer acceptance, nor a whole-suite cost
reduction. No other implementation checkout or MG/IV work was used.

## Still pending

E1's remaining cold mesh-construction propagation and E2 implementation remain
open, as do the bounded
E3–E7 measurements, independent frozen Euler seam comparisons, selected scoped
readability work, per-suite high-rank rationale/curation and the sanitizer
ownership pass. Formatter default discovery now includes Cauchy and Physics;
no repository-wide formatting was performed. Pending queue entries are not
completed merely because neighboring fixes or test labels exist.

## E1: prepared-transfer and FV policy

Runtime communication contexts now agree an immutable `collectivePreflight`
policy at construction, enabled by default. Prepared ordinary, scheduled and
checked MPI starts inherit it; bindings also agree the selected effective mode
in their existing control reduction. This prevents choosing individually valid
contexts with inconsistent policies across ranks. The policy is stored in the
binding, without a shared-owner lookup per launch. Changing policy requires a
drained, coordinated preparation boundary.

Disabled mode retains local access/readiness/version validation and aborts the
communicator on rejected local preparation/launch instead of permitting an
unsafe rank-asymmetric recovery. Required route/count/schema/control exchanges
and numerical reductions remain. Dynamic transfer, the lightweight Exchange
facade and explicit EntitySpace route preparation accept/inherit the context.
The generic mechanism is shared; required NCCL setup agreement remains required.

CFV metrics own the policy context, use it for local preparation/binding and
pass it to the entity-space route builder. VR local preparation inherits it.
Cauchy's existing `implicit.collectiveChecks` now reaches these halos; no
second overlapping Cauchy switch was added. Scheduled-access checking remains
independent. Historical phase-benchmark descriptions retain their old pinned
meaning rather than relabeling earlier measurements.

The completed 10-entry CPU batch passes in 20.03 s. Real MPI profiling-interface
counts prove **0** launch reductions when disabled, versus **1** for ordinary/
scheduled and **3** for checked starts when enabled. The compact fixture covers
table/padded/CSR payloads, untouched padding, persistent/transient requests,
empty owners, and reversed/split communicators at np1/2/4. A bounded two-rank
subprocess confirms explicit off-mode abort. Constructor/binding mode mismatch
checks reject collectively and permit a subsequent valid transfer. Existing
transient and split-SOR checks pass. Cauchy's P3/WBAP RHS is bit-identical between
on/off modes with checked CPU workers at np1/2; this is policy regression parity,
not a new independent discretization oracle or a speedup measurement.

The isolated cache initially selected host Python 3.14. Before runtime Python
validation it was corrected to `/home/harry/projects/DNDSR/venv/bin/python`
(3.12.3), matching headers/library, then the DNDS binding was rebuilt and installed
into this checkout. A direct installed-component two-rank smoke checks the
default/override property, native write leases and a strided direct payload;
see [reproducer](plan53_validation/runtime_policy_smoke.py). This deliberately
bypasses the Python package's eager imports of the other modules; it does not
qualify the complete Python facade or CFV Python module. The first direct smoke
incorrectly assumed the facade's slice monkeypatch was present; using the native
write-lease API fixes that harness mismatch. No library defect or data corruption
was inferred from it.

Cold primary/topology/assembler preparation still has separate local-stage
agreements outside the explicit EntitySpace context. Their policy propagation
remains an E1 follow-up; the work above is not a claim that every mesh setup
agreement has been removed. No CUDA acceptance, E2 storage-profile completion,
or E3–E7 matched performance result is claimed.
