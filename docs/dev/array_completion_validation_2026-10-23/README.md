# Array completion evidence

See [the completion report](../array_runtime_completion_2026-10-23.md) for scope,
results and limits. Logs remove ANSI color and embedded NUL characters for
readability. Initial/singleton diagnostic runs are explicitly distinguished from
the final suites. No generated binaries or the venv are committed.

Environment setup used only the refactor checkout:

```sh
python3 -m venv venv-interop
venv-interop/bin/python -m pip install --no-cache-dir torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
venv-interop/bin/python -m pip install --no-cache-dir numpy==2.4.6 mpi4py==4.1.2 pytest pytest-mpi pytest-timeout h5py scipy
```

`interop-requirements.txt` records resolved versions. The optional environment
occupies 5.6 GB on this host. Do not install these into the original DNDSR venv.

Build each profile's `dnds_unit_tests`, four production pybind modules and
`runtime_array_bridge_test`; install the Python component before importing it.
Use `docs/dev/array_scenario_validation_2026-10-22/verify_python_profile.py BUILD`
to compare installed ELF build IDs and CUDA capability. Profiles share the install
tree; build/install/test sequentially, and do not trust install timestamps alone.

```sh
ctest --test-dir build-refactor -R '^(dnds_|geom_mesh_reorder)' --output-on-failure
venv-interop/bin/python docs/dev/array_completion_validation_2026-10-23/run_cuda_native.py build-refactor-cuda

export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/python:$PWD/build-refactor-cuda/test/cpp"
export DNDS_REQUIRE_TORCH=1
mpirun -np 2 venv-interop/bin/python test/demos/one_rank_per_gpu.py \
  venv-interop/bin/python -m pytest test/DNDS/test_runtime_*.py test/demos/test_runtime_solver.py --with-mpi -q -rs

compute-sanitizer --tool memcheck --error-exitcode 1 \
  build-refactor-cuda/test/cpp/dnds_test_runtime_cuda_engine --no-colors
OMPI_MCA_accelerator=null mpirun -np 1 venv-interop/bin/python test/demos/one_rank_per_gpu.py \
  compute-sanitizer --tool memcheck --error-exitcode 1 --launch-timeout 30 \
  venv-interop/bin/python -m pytest test/DNDS/test_runtime_dlpack.py --with-mpi -q
```

The last command validates tensor sharing only; the disabled MPI accelerator
means it is not a direct-device communication sanitizer test. The failed
singleton attach diagnostic and successful Torch-only probe are retained.
`destructor-before.txt` records the bounded deadlock reproduction before the fix.
`source-and-binaries.json` pins the baseline plus tested worktree sources/binaries;
it does not claim the baseline commit alone contains these changes.
