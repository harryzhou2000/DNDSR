# Cauchy timing and Caliper profiling

Use the canonical `cauchy_solver` executable (`app/cauchy.exe`) together with
optional Caliper annotations (`-DDNDS_USE_CALIPER=ON` at configure time and
`DNDS_TIMELINE=1` at runtime) to measure solver phases, implicit solves, and
communication overlap on real production configurations.

```sh
cmake --build build -t cauchy_solver -j
OMP_NUM_THREADS=1 DNDS_TIMELINE=1 mpiexec -n 32 ./build/app/cauchy.exe \
  --config path/to/case.json
```

Run timings on an exclusive allocated node with fixed thread affinity and
warmup/steady iteration counts separated cleanly. Small local runs validate
instrumentation wiring; they are not performance evidence.

### Observing solver phases and CPU/GPU halo overlap

With Caliper enabled at build time and `DNDS_TIMELINE=1`, `TimelineScope`
annotations record solver, linear-solver, reconstruction, and discretization
regions on the executing thread or stream:

- `Cauchy/CPU/FluxRHS`, `VolumeSource`, `RotationSource`, `LinearApply`,
  `Jacobi`, and `CFV/CPU/ReconstructionSweep` record their row selection as an
  integer attribute at both boundaries (`0` = all, `1` = core, `2` =
  halo-dependent). Auxiliary preparation records its own phase enumeration.
- `Cauchy/Linear/InvertDiagonal` and related linear-solver scopes separate
  preconditioner setup from Krylov iterations.

Compare worker intervals with MPI/NCCL observations on the **same rank**.
A successful post and a later observed completion give an upper bound on the
time a request was outstanding. For a receive, a singleton `MPI_Testall`
returning false also gives a lower bound from posting to the start of that
test. Use an unsplit (`overlap = false`) run as a control and compare final
fields by persistent cell ID.
