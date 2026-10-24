#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/../../.."
profile=$1
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/python"
mpi=/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin/mpirun
py="$PWD/venv-interop/bin/python"
files=(test/Geom/test_basic_geom.py test/Geom/test_mesh_facade.py test/Geom/test_mesh_edit.py test/Geom/test_overset_facade.py test/CFV/test_basic_cfv.py test/CFV/test_basic_fv.py test/CFV/test_fv_correctness.py test/CFV/test_vr_correctness.py test/EulerP/test_basic_eulerP.py)
if [[ $profile == cpu ]]; then
  build=build-refactor
else
  build=build-refactor-cuda
fi
# CMake install uses timestamps; force profile-specific artifacts to be newer
# than any installed files from the other profile before the actual install.
for entry in DNDS:dnds Geom:geom CFV:cfv EulerP:eulerP; do
  directory=${entry%%:*}; name=${entry##*:}
  touch "$build/src/$directory/"${name}_pybind11*.so "$build/src/$directory/lib${name}_shared.so"
done
cmake --install "$build" --component py > "/tmp/geom-final-install-$profile.txt" 2>&1
"$py" "$script_dir/final-verify-profile.py" "$build" "$profile" > "/tmp/geom-final-profile-$profile.json" 2>&1
if [[ $profile == cpu ]]; then
  for ranks in 1 2 4; do
    "$mpi" -np "$ranks" "$py" -m pytest -q -ra "${files[@]}" > "/tmp/geom-final-python-cpu-np$ranks.txt" 2>&1
  done
else
  "$mpi" -np 2 "$py" test/demos/one_rank_per_gpu.py "$py" -m pytest -q -ra "${files[@]}" > /tmp/geom-final-python-cuda-np2.txt 2>&1
  "$mpi" -np 2 "$py" test/demos/one_rank_per_gpu.py "$py" -c 'import runpy; from DNDSR import DNDS; mpi = DNDS.MPIInfo(); mpi.setWorld(); runpy.run_path("test/EulerP/test_basic_eulerP.py")["test_basic_eulerP"](mpi, isCuda=True)' > /tmp/geom-final-eulerP-cuda-np2.txt 2>&1
fi
