# Runtime caller contract and production-shaped probes, 2026-10-22

This follow-up documents obligations that remain with the caller in fully eager
execution, adds production-shaped probes, and includes the previously reviewed
[schema/overlap audit](array_runtime_scenarios_2026-10-22.md) in the same authorized
checkpoint. The implementation base is `244b390c4992d7336fd34e8de0c67f5a1651a136`
on `codex/refactor-dnds-array-geom`. Legacy Euler, CFV, Geom and DNDS::Array APIs
are unchanged.

## Findings and changes

1. **Caller obligations are now discoverable at the public boundary.**
   [RuntimeCallerContract](../architecture/RuntimeCallerContract.md) specifies
   cross-stream ordering, raw-view lifetime/bounds, final release of shared
   handles, worker-pool progress, MPI participation/thread rules, schema promises,
   migration rebinding, shape publication and checkpoint consistency. It is linked
   from the architecture guides. C++ Doxygen comments and Python class/method
   docstrings repeat the obligations where they matter, rather than relying on
   this conversation or test source as the specification.
2. **Shared ownership does not establish current-mesh identity or dependency order.**
   Tests retain an MPI binding and Python slice, reset the original handles, and
   verify the old binding still writes the old allocation while replacement values
   remain untouched. Binding replacement shapes to an old payload is rejected.
   Existing deterministic missing-join tests demonstrate stale results without a
   data race; eager mode still deliberately provides no dependency proof.
3. **Field bundles and reverse assembly now have repeated, full-value oracles.**
   New native scenarios gather state and IDs through one shared route, compute
   ghost residual contributions, and reverse-reduce into owned rows, including
   duplicate contributions and a preserved residual seed. Four epochs cover
   persistent/transient requests and direct/host-packed transports. A separate
   test migrates uneven state/ID/ragged-reconstruction bundles through an empty
   rank, reverses the reconstruction migration, and rejects old stamped bindings
   after publication of a new distribution epoch. Different physical strides,
   empty rows and rank/row/component/epoch values expose stale or permuted data.
4. **Dynamic CUDA lifetimes now cover a multilevel control pattern.**
   Three/four/five-level mock restriction/prolongation cycles alternate compute
   streams, allocate temporaries asynchronously behind a gate, and give a third
   stream an independent diagnostic read. Non-final and final shared-handle
   releases occur on different streams with explicit joins. Three cycles use
   different input values without intermediate host synchronization, retaining
   each cycle's output so later cycles cannot hide earlier stale values. Every component is
   checked against a closed-form geometric sum, including odd level sizes and
   padded strides. This probes a multigrid-like lifetime pattern; it is not a FAS
   or CFD numerical validation.
5. **Checkpoint timing is tested from Python on every engine profile.**
   An unfinished stage rejects synchronous view reads, exclusive shape editing,
   and checkpoint snapshots. After the stage join, a parallel HDF5 checkpoint
   restores every value. A retained debug view continues to show the previous
   state after the parent handle is reset. This connects Python/native ownership,
   asynchronous stages and the existing Serializer rather than testing them only
   in isolation.
6. **Fixed: arithmetic schemas disagreed across native/Python boundaries.**
   The preceding type-tag correction exposed incompatible names: native schemas
   used RTTI names while Python used dtype names. A new mixed-language MPI test
   reproduced three failing combinations out of four on each rank, rejecting
   valid payloads/reducers. Supported fixed-width arithmetic schemas now use the
   same names in both interfaces; validation still rejects equal-width wrong
   types. The final test also alternates native/Python schema factories across
   ranks and checks every reduced component. See the
   [before-fix reproduction](array_contract_validation_2026-10-22/schema-bridge-before.txt).
   This is an in-memory transfer-schema change; checkpoint schemas/formats are
   unchanged. Explicit application POD schemas remain caller descriptions.

## Validation

The final validation results and source/binary hashes are recorded in
[array_contract_validation_2026-10-22](array_contract_validation_2026-10-22/).
The earlier audit directory remains a historical snapshot; its source manifest is
not rewritten to match this follow-up.

| Final validation | Result |
|---|---|
| Native DNDS + legacy Geom reorder, CPU | 58/58 CTest entries; MPI np=1/2/4/8 |
| Native DNDS + legacy Geom reorder, CUDA | 64/64 CTest entries; MPI np=1/2/4/8 |
| Python Runtime, CPU, np=1/2/4 | 62 passed, 23 CUDA-only skips per process |
| Python Runtime, CUDA, np=1/2/4 | 85 passed per process, no skips |
| Python DNDS/Geom regression | CPU: 80 passed/30 skips; CUDA: 100 passed/10 skips |
| CUDA engine memcheck | 11 cases, 61,593 assertions, zero errors |
| CUDA staged MPI memcheck, np=2 | 10 cases per rank, 280,152/281,426 assertions, zero errors |

Final results use the formatted source and the unified arithmetic schema names.
The decisive logs have `ctest-*-final`, `*-memcheck-final`, `python-*-np*` and
`python-regression-*` names. `profile-cpu-formatted.txt` and `profile-cuda.txt`
verify the final installed builds. The evidence README distinguishes exploratory
failures, earlier passes and final validation; none of the intentional reproducer
failures count as passing validation. The first checkpoint test used direct
`numpy()` on a restored CSR Array; the test was corrected to use the supported
equal-width slice snapshot, and the final full matrices passed. No production
I/O change was necessary.

The environment uses GCC 13.3, OpenMPI 5.0.9, the project Python 3.12 venv,
OMP_NUM_THREADS=2, and CUDA 13.2 on physical A100 device 1 exposed as logical 0.
Both profiles rebuild the native tests, legacy Geom reorder target, four production
Python modules and the native Python bridge. Installed generated libraries are
preserved outside their install destinations before switching profiles; the
[ELF/import verifier](array_scenario_validation_2026-10-22/verify_python_profile.py)
must pass before Python results count. This avoids same-second installation
timestamps silently retaining the wrong CPU/CUDA module.

The previous audit provides build/test commands. The new suite cases are registered
through the existing targets; no new dependency or test registration is required.
Compute Sanitizer checks native CUDA and staged MPI. Its MPI invocation disables
OpenMPI's accelerator component only to avoid the previously isolated vendor
finalization probe; normal CUDA tests retain the normal MPI configuration.

These are correctness and lifetime tests on one host/GPU. They do not establish
multinode transport, multiple GPU execution, speedup, graph capture or Euler
numerical accuracy. Actual MPI/core and mock device/core rendezvous tests establish
concurrent execution, not useful-work efficiency. Bare pointers, false untagged
POD schemas, unchecked stencil bounds and deliberately omitted joins remain
documented caller responsibilities; no undefined-behavior corruption test is used.
