#!/usr/bin/env python3
"""Run bounded, fresh-process E2 capability/storage probes; no solver campaign."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--np", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    binary, launcher = args.binary.resolve(), args.mpiexec.resolve()
    workspace = args.workspace.resolve()
    assert args.np in (1, 2, 4), "bounded memory probe, not a scaling campaign"
    assert workspace.is_relative_to(repo / "workspace"), "keep evidence in this source checkout"
    assert subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=workspace, text=True).strip() == str(workspace)
    output = workspace / args.run_id
    output.mkdir()  # Never overwrite or silently resume prior evidence.
    raw = repo / "data" / "plan53-targeted-assessment" / args.run_id
    raw.mkdir(parents=True)
    shapes = [(2, 16), (3, 4)] if args.pilot else [(2, 128), (3, 16)]
    trials = 1 if args.pilot else 2
    cases = []
    for trial in range(trials):
        profiles = ("full", "sweeps", "wbap") if trial % 2 == 0 else ("wbap", "sweeps", "full")
        for dimension, width in shapes:
            for profile in profiles:
                cases.append({"trial": trial, "dimension": dimension, "width": width, "degree": 3, "profile": profile, "status": "unrun"})
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    tracked_env = {key: environment.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DNDS_DISABLE_ASYNC_MPI", "LD_LIBRARY_PATH")}
    relevant = ["test/cpp/CFV/benchmark_ReconstructionStorage.cpp", "test/cpp/CMakeLists.txt", "scripts/benchmark/plan53_storage.py", "build-audit/CMakeCache.txt"]
    manifest = {
        "purpose": "E2 storage and isolated RSS; local timing is observational, no matched speedup claim",
        "pilot": args.pilot, "runner": socket.gethostname(), "np": args.np,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True),
        "source_hashes": {name: sha256(repo / name) for name in relevant},
        "binary": str(binary), "binary_sha256": sha256(binary), "mpiexec": str(launcher),
        "environment": tracked_env, "cpu_affinity_available": sorted(os.sched_getaffinity(0)),
        "rank_binding": "Open MPI --bind-to core; OMP1", "working_directory": str(repo),
        "cpu_topology": subprocess.check_output(["lscpu", "-J"], text=True),
        "per_row_timeout_seconds": 120, "raw_directory": str(raw), "cases": cases,
    }
    record = output / "manifest.json"
    write_json(record, manifest)
    print(f"Manifest: {record}\nRaw logs: {raw}", flush=True)
    for ordinal, case in enumerate(cases):
        command = [str(launcher), "--bind-to", "core", "-np", str(args.np), str(binary), str(case["dimension"]), str(case["width"]), str(case["degree"]), case["profile"]]
        log = raw / f"row-{ordinal:02d}.log"
        case.update(command=command, log=str(log), status="running")
        write_json(record, manifest)
        start = time.monotonic()
        with log.open("w") as stream:
            try:
                result = subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=120, check=False)
                case.update(exit_status=result.returncode, status="executed" if result.returncode == 0 else "failed")
            except subprocess.TimeoutExpired:
                case.update(status="failed", reason="row exceeded 120 seconds")
        case.update(wall_seconds=time.monotonic() - start, log_sha256=sha256(log))
        if case["status"] == "executed":
            records = [line.removeprefix("PLAN53_RESULT ") for line in log.read_text().splitlines() if line.startswith("PLAN53_RESULT ")]
            if len(records) == 1:
                case["ranks"] = json.loads(records[0])
            else:
                case.update(status="failed", reason="missing or duplicate result record")
        write_json(record, manifest)
        print(f"row {ordinal}: {case['status']} ({case['wall_seconds']:.3f}s)", flush=True)
        if case["status"] == "failed":
            raise RuntimeError(f"Stopped at row {ordinal}; see {log}; later rows remain unrun")

    # Cross-process checks, not a second full plan in the same allocator arena.
    for trial in range(trials):
        for dimension, width in shapes:
            group = {case["profile"]: case for case in cases if case["trial"] == trial and case["dimension"] == dimension}
            for rank in range(args.np):
                full, sweep, wbap = [group[name]["ranks"][rank] for name in ("full", "sweeps", "wbap")]
                assert all(row["rank"] == rank and row["width"] == width and row["np"] == args.np for row in (full, sweep, wbap))
                assert full["sweep_digest"] == sweep["sweep_digest"] == wbap["sweep_digest"], "profile changed reconstruction output"
                assert full["limiter_digest"] == wbap["limiter_digest"], "profile changed WBAP output"
                for row in (sweep, wbap):
                    omitted = {"localMatrix", "localInverse", "rawMatrixB", "rawVectorB", "rawBoundaryInjection"}
                    assert all(not table["allocated"] for table in row["tables"] if table["name"] in omitted)
                    if row["owned_cells"]:
                        assert row["retained_allocation_bytes"] < full["retained_allocation_bytes"]
    for case in cases:
        case["status"] = "verified"
    manifest.update(status="verified", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(record, manifest)
    print(f"Verified {len(cases)} rows; {record}", flush=True)


if __name__ == "__main__":
    main()
