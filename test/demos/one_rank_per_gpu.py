#!/usr/bin/env python3
"""Launch one local MPI rank per visible GPU, before importing CUDA or MPI.

Usage: mpirun -np 2 python test/demos/one_rank_per_gpu.py <command> [args...]
Fails on oversubscription. Honors an existing CUDA_VISIBLE_DEVICES subset.
"""
import os
import subprocess
import sys

rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("MPI_LOCALRANKID", "0")))
uuids = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible is not None:
    uuids = [uuids[int(value)] if value.isdigit() else value for value in visible.split(",") if value]
if rank >= len(uuids) or len(sys.argv) < 2:
    sys.exit("one_rank_per_gpu: insufficient GPUs or missing command")
selected = uuids[rank]
os.environ["CUDA_VISIBLE_DEVICES"] = selected
os.environ["DNDS_TEST_GPU_UUID"] = selected
print(f"local rank {rank}: {selected} as CUDA device 0", flush=True)
os.execvp(sys.argv[1], sys.argv[1:])
