# Installed DNDS communication-policy smoke

This checks the freshly built native DNDS Python component without importing
the package's Geom/CFV/EulerP modules. It is not full Python-facade acceptance.
The native API uses explicit write leases; the package adds debugging slices.

From the audit checkout, after configuring Python to the project venv:

```sh
cmake --build build-audit --target dnds_pybind11 -j8
cmake --install build-audit/src/DNDS --component py
```

Use that interpreter and the MPI implementation selected by CMake. For the
recorded local check (Python 3.12.3, Open MPI 5.0.9):

```sh
LD_LIBRARY_PATH=/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/lib:/home/harry/projects/DNDSR/external/cfd_externals/install/lib \
PYTHONPATH="$PWD/python/DNDSR/DNDS/_ext" OMP_NUM_THREADS=1 \
timeout 30s /mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin/mpiexec --oversubscribe -np 2 \
  /home/harry/projects/DNDSR/venv/bin/python docs/dev/plan53_validation/runtime_policy_smoke.py
```

Both ranks passed: default-on and explicit-off policy properties, a native
write lease, reordered row requests and a direct strided payload match an
independent NumPy expectation. No Torch installation or other checkout is used.
