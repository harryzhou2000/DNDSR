"""Run configured DNDS native tests at np=1/2 with a distinct GPU per rank.

Uses CTest's actual commands, environments and working directories. Does not
change test definitions or run np=4/8 on the two-GPU validation machine.
Usage: python run_cuda_native.py /absolute/path/to/build-refactor-cuda
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

build = Path(sys.argv[1]).resolve()
root = build.parent
tests = json.loads(subprocess.check_output(
    ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True))["tests"]
wrapper = root / "test/demos/one_rank_per_gpu.py"
passed = 0
for test in tests:
    name = test["name"]
    if not name.startswith("dnds_") or re.search(r"_np(4|8)$", name):
        continue
    props = {p["name"]: p["value"] for p in test["properties"]}
    env = os.environ.copy()
    for entry in props.get("ENVIRONMENT", []):
        key, value = entry.split("=", 1)
        env[key] = value
    command = test["command"]
    if re.search(r"_np[12]$", name):
        command = command[:-1] + [sys.executable, str(wrapper), command[-1]]
    else:
        command = [sys.executable, str(wrapper)] + command
    print("TEST", name, command, flush=True)
    subprocess.run(command, check=True, env=env,
                   cwd=props.get("WORKING_DIRECTORY", str(build)),
                   timeout=max(120, props.get("TIMEOUT", 120)))
    passed += 1
print(f"PASS: {passed} configured native CUDA-build tests, MPI np=1/2 only", flush=True)
