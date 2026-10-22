"""Run with the project Python and PYTHONPATH after installing a build profile.

Read-only verification: ELF build IDs survive the install-time RPATH rewrite.
Fail before running tests if installed modules belong to another build.
"""

import importlib
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[3]
build = pathlib.Path(sys.argv[1]).resolve()
cache = (build / "CMakeCache.txt").read_text()
expected_cuda = re.search(r"^DNDS_USE_CUDA:BOOL=(.*)$", cache, re.M)[1] == "ON"


def build_id(filename):
    notes = subprocess.check_output(["readelf", "-n", str(filename)], text=True)
    return re.search(r"Build ID: (\w+)", notes)[1]


modules = []
for component, target in [("DNDS", "dnds"), ("Geom", "geom"), ("CFV", "cfv"), ("EulerP", "eulerP")]:
    original, = (build / "src" / component).glob(f"{target}_pybind11*.so")
    installed = root / "python" / "DNDSR" / component / "_ext" / original.name
    assert build_id(original) == build_id(installed), f"Wrong installed profile: {installed}"
    modules.append((f"DNDSR.{component}._ext.{target}_pybind11", installed))
    print(component, build_id(installed))

original = build / "src/DNDS/libdnds_shared.so"
installed = root / "python/DNDSR/_lib/libdnds_shared.so"
assert build_id(original) == build_id(installed), "Wrong installed libdnds_shared"
for name, installed in modules:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve() == installed.resolve(), module.__file__
from DNDSR.DNDS import Runtime as R

assert R.cuda_enabled == expected_cuda, (R.cuda_enabled, expected_cuda)
print("Verified profile:", build.name, "CUDA:", R.cuda_enabled)
