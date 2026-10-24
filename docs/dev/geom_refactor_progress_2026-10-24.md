# Geom refactor progress

This retains the development ledger for the approved GE0–GE6 plan. The final
[checkpoint report](geom_refactor_completion_2026-10-24.md) supersedes the
intermediate validation status below. The bounded phase is validated with its
explicit legacy setup/output/edit compatibility restrictions.
The implementation checkout is `DNDSR-array-geom`, branch
`codex/refactor-dnds-array-geom`, starting from `232df1a0`. The published plan
checkpoint is `dndsr-next-plan` commit `c531ccc`.

The table below records the pre-final checkpoint; every listed closing gate has
now passed. Final coverage is CPU 172/172 plus 12 closing checks, Geom CUDA 3/3,
legacy CFV CUDA 1/1, and verified CPU/CUDA Python profiles. Native 3D four-parent
edge interpolation and actual restored-field overlap were added during the final
audit. See the checkpoint report for counts, source revisions and limitations.

| Package | Implemented at the intermediate checkpoint | Closing work, subsequently validated |
|---|---|---|
| GE0 | Caller/mutator inventory, pre-change baseline, reproduced distributed face-target lookup failures | Keep the ledger synchronized with final migration exceptions |
| GE1 | Legacy host Buffer bridge, entity descriptors, retained mesh scopes, directly callable CPU/CUDA views, private native publication with cloned canonical maps | Final bridge and publication regression gates |
| GE2 | CFV/Euler operator scopes and field adapters; native-authority numerical/output oracles; managed Python read/edit and overset paths; EulerP native snapshot adapter; metric/reconstruction/wall-cache dependencies | Final installed Python profiles and explicit legacy setup/output/benchmark allowlist |
| GE3 | Native inverse/composition, GhostSpec traversal and ghost-space publication; shared local interpolation enumeration with native owner routing and ordered PBI outputs; current native/legacy facade input adapters | Final registry schedule agreement negatives and full legacy/native setup regression |
| GE4 | Detached mesh/field publication; preserved/rebuilt faces and edges; collective source-preserving failures; coherent mesh/four-history checkpoint; EntityField mesh/order binding and checked routes; native destination publication before binding fields | Final rank-local stale association, POD and CUDA field migration tests |
| GE5 | Real CGNS stencil oracle with inline/worker overlap and two-GPU direct MPI; local permutation and detached redistribution followed by native publication; CUDA query/lifetime tests | Re-run final typed association and setup paths with the same execution matrix |
| GE6 | Private native authority and current-state edits; legacy mutators reject native mode; caller contracts, ownership diagrams and failure evidence | Complete remaining gates, final source/binary manifest and release checkpoint |

The legacy mesh device-view files are reference implementations. Their existing
CUDA functionality is preserved. The new view is separately compiled/executed
inside device kernels; the reference parity test uses existing const generic
array views where legacy const adjacency convenience instantiation is defective.
That limitation is recorded rather than used to claim the legacy CUDA path is
generally broken.

`geom_validation_2026-10-24/` retains intermediate failures as well as successes.
Historical `integration4-geom-tests.txt` records 10 focused passes after the native snapshot
and mutation work: MPI 1/2 relation, native snapshot, mutation and exact restart
geometry cases, plus the CUDA query/const parity test and two-GPU direct MPI
overlap. These are not a substitute for the full post-change suites. Earlier
68-test regression output used pre-mutation binaries and retained four failures
from the now-corrected unpublished empty-CSR test fixture. Do not combine those
logs into a fictitious single all-pass run.

Intermediate integration 18 passed 75 CPU Geom registrations excluding the new
edit suite, all 17 Euler registrations, and the two CUDA facade/direct-MPI gates.
The edit suite's global inverse-index setup issue was corrected and integration
19 passed its MPI 1/2/4/8 registrations. CPU Python integration 19 passed 54 tests
at MPI 1 and 2 with one explicit CUDA-only skip per rank; the MPI-4 attempt raced
new test edits against earlier installed binaries and is not final evidence.
Integration 20 passed the new coherent mesh/four-history checkpoint at MPI
1/2/4/8. Integration 21 rebuilds native mutation rejection and malformed
self-periodic incidence checks, then runs the broader suites. These scoped
results do not close the outstanding GE3/GE4 gates above.

Integration 24 passes all eight focused CPU native-connectivity and EntityField
registrations at MPI 1/2/4/8. The setup tests now include intermediate ghost-layer
collection, current-authority input after raw workspace corruption, periodic
distributed physical-coordinate/frame checks and empty ranks. The field tests
pair retained old mesh/field reads with new field publication, check exact
histories and halo values, and reject stale bindings. Additional rank-local
schedule/association negatives and the dedicated two-GPU field test are the final
focused checks before the broad final regression.

Source-level review found two silent-mismatch risks, now addressed in source:
legacy mutators could select an older detached workspace after a native edit,
and duplicated self-periodic face slots could carry identical side bits. Native
mutation now requires explicit authority selection; query publication validates
opposite self-periodic side incidence. Another review finding, same-sized fields
used with an old retained entity order, is addressed by the typed EntityField
binding and its stale-association tests. Final validation uses rebuilt binaries
and verified installed Python profiles; source presence and syntax checks alone
are not execution evidence.
