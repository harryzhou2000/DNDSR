# Geom refactor progress

This is a progress ledger for the approved GE0–GE6 plan, not a completion report.
The implementation checkout is `DNDSR-array-geom`, branch
`codex/refactor-dnds-array-geom`, starting from `232df1a0`. The published plan
checkpoint is `dndsr-next-plan` commit `c531ccc`.

| Package | Implemented in the current worktree | Still required for closure |
|---|---|---|
| GE0 | Caller/mutator inventory, pre-change baseline, reproduced distributed face-target lookup failures | Keep the ledger synchronized with final migration exceptions |
| GE1 | Legacy host Buffer bridge, immutable entity descriptors, retained mesh read scopes, plain CPU/CUDA query view, detached native snapshot | Integrate the native query publication behind the mesh facade |
| GE2 | CFV/Euler operator-level scopes and field setup adapters, managed Python read/coordinate paths, EulerP native mesh snapshot adapter, metric/reconstruction dependencies | Close or explicitly constrain helper/output/legacy setup exceptions; fresh consumer and Python regression gates |
| GE3 | Native relation readers and inverse/composition output using Runtime row builders, independent native halo routes, native snapshot storage | Authoritative managed storage and entity maps; complete connectivity/ghost builder adaptation and publication |
| GE4 | Target-map coverage fixes, field-bundle preparation, duplicate/association/order checks, mutation capability and invalidation, exact checkpoint query test | Detached mesh-plus-field publication with source-preserving failures; complete parent/edge/cache/I/O inventory |
| GE5 | Real CGNS stencil oracle, CPU inline/worker overlap, two-GPU direct datatype MPI, CUDA query parity, retained completion tests | Re-run the same scenarios against final managed storage and detached transition; final negative-case audit |
| GE6 | Caller contracts, ownership diagrams, validation logs | Final private managed authority, compatibility boundaries, complete regression matrix, source/binary manifest and release checkpoint |

The legacy mesh device-view files are reference implementations. Their existing
CUDA functionality is preserved. The new view is separately compiled/executed
inside device kernels; the reference parity test uses existing const generic
array views where legacy const adjacency convenience instantiation is defective.
That limitation is recorded rather than used to claim the legacy CUDA path is
generally broken.

`geom_validation_2026-10-24/` retains intermediate failures as well as successes.
`integration4-geom-tests.txt` records 10 focused passes after the native snapshot
and mutation work: MPI 1/2 relation, native snapshot, mutation and exact restart
geometry cases, plus the CUDA query/const parity test and two-GPU direct MPI
overlap. These are not a substitute for the full post-change suites. Earlier
68-test regression output used pre-mutation binaries and retained four failures
from the now-corrected unpublished empty-CSR test fixture. Do not combine those
logs into a fictitious single all-pass run.
