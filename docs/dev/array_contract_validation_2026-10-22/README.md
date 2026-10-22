# Evidence index

See [the audit report](../array_runtime_contract_audit_2026-10-22.md) and
[caller contract](../../architecture/RuntimeCallerContract.md).

Final validation:

- `ctest-cpu-final.txt`, `ctest-cuda-final.txt`: complete native DNDS and legacy
  Geom reorder matrices, including MPI np=1/2/4/8.
- `python-{cpu,cuda}-np{1,2,4}.txt`: complete Runtime suites; summaries appear
  once per MPI process. CPU expects 62 passes/23 CUDA skips; CUDA expects 85 passes.
- `python-regression-{cpu,cuda}.txt`: broader DNDS/Geom Python regression.
- `cuda-{engine,mpi}-memcheck-final.txt`: final device/multilevel and staged MPI
  memory checking. MPI uses the previously documented OpenMPI accelerator
  workaround; this does not test direct device MPI under the sanitizer.
- `profile-cpu-formatted.txt`, `profile-cuda.txt`: final ELF/import/CUDA-capability
  verification after rebuilding and installing all production extensions.
- `build-cpu-formatted.txt`, `build-cuda-final.txt`: final successful builds.
- `python-docstrings.txt`: inspection of the installed API's caller guidance.
- `source.json`: tested source and binary hashes, environment and test summaries.

Exploratory and historical logs:

- `schema-bridge-before.txt` intentionally fails three mixed-language combinations
  per rank before arithmetic schema names were unified. The final suite adds two
  additional mixed-rank factory cases and passes all six combinations.
- `python-boundary-before-schema-fix.txt` is an exploratory run of the preceding
  compiled profile, not final-source validation. It finds a test-author error:
  direct `numpy()` is unavailable for CSR arrays. The test now uses the equal-width
  slice snapshot. This is not classified as a runtime defect.
- Build/install/profile logs with `before` or intermediate `final` names show
  intermediate builds. The final CPU profile is the later `formatted` profile.
- Native and memcheck logs without `final` precede the final per-cycle CUDA oracle
  expansion and schema-name correction. They are superseded by the final logs.
- `profile-backup-*.txt` identifies preserved generated installed binaries before
  profile switching. No source files were moved or discarded.
- `format.txt` records the repository formatting script before final validation.

The earlier `array_scenario_validation_2026-10-22` directory is a separate
historical snapshot. Its source/binary hashes have not been rewritten.
