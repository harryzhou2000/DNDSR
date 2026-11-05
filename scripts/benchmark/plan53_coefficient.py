#!/usr/bin/env python3
"""Bounded E3 coefficient-publication attribution; not a full-RHS speedup test."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--expected-source")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--candidate", action="store_true", help="bounded follow-up of native whole-array copy, not another full matrix")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    binary, launcher = args.binary.resolve(), args.mpiexec.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if args.expected_source and head != args.expected_source:
        raise ValueError("assessment source differs from requested commit")
    if not args.pilot and (not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1):
        raise ValueError("matched E3 matrix requires one allocated node")
    rows = []
    if args.pilot:
        rows = [{"np": 2, "width": 16, "degree": degree, "grouping": grouping, "layout": layout}
                for degree, grouping, layout in ((3, "flow5", "aos"), (2, "split5+12", "soa"))]
    elif args.candidate:
        rows = [{"np": np, "width": width, "degree": degree, "grouping": grouping, "layout": layout}
                for np, width, degree, grouping, layout in
                ((32, 32, 3, "flow5", "aos"), (32, 400, 3, "flow5", "aos"),
                 (32, 400, 2, "split5+12", "aos"), (32, 400, 3, "split5+12", "aos"),
                 (8, 200, 3, "split5+12", "aos"), (32, 400, 3, "split5+12", "soa"))]
    else:
        for np, width in ((32, 32), (32, 400), (8, 16), (8, 200)):
            for degree in (2, 3):
                for grouping in ("flow5", "split5+12"):
                    rows.append({"np": np, "width": width, "degree": degree, "grouping": grouping, "layout": "aos"})
        # One alternate supported layout, only in the larger split/P3 cases.
        rows += [{"np": np, "width": width, "degree": 3, "grouping": "split5+12", "layout": "soa"}
                 for np, width in ((32, 400), (8, 200))]
    for row in rows:
        row["status"] = "unrun"
    repetitions, trials = (100, 2) if args.pilot else (500, 4)
    args.records.mkdir(parents=True, exist_ok=False)
    args.raw.mkdir(parents=True, exist_ok=False)
    manifest = {
        "purpose": __doc__, "pilot": args.pilot, "candidate": args.candidate, "host": socket.gethostname(), "source": str(source), "revision": head,
        "source_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True),
        "binary": str(binary), "binary_sha256": sha(binary), "cache_sha256": sha(binary.parents[2] / "CMakeCache.txt"),
        "source_sha256": {name: sha(source / name) for name in
                          ("test/cpp/CFV/benchmark_CoefficientPublication.cpp", "test/cpp/Cauchy/CartesianFixture.hpp",
                           "scripts/benchmark/plan53_coefficient.py")},
        "environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DNDS_DISABLE_ASYNC_MPI", "UCX_TLS", "UCX_NET_DEVICES")},
        "job": os.environ.get("SLURM_JOB_ID"), "node_list": os.environ.get("SLURM_NODELIST"),
        "controller_affinity": sorted(os.sched_getaffinity(0)), "rows": rows, "status": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "row_timeout_seconds": 300,
    }
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("E3 uses OMP1 and OPENBLAS1")

    def save():
        temporary = args.records / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.records / "manifest.json")

    print(f"E3 records: {args.records.resolve()}\nE3 raw logs: {args.raw.resolve()}", flush=True)
    save()
    for ordinal, row in enumerate(rows):
        log = args.raw / f"row-{ordinal:02d}.log"
        command = [str(launcher), "--bind-to", "core", "--map-by", "core", "--report-bindings", "-np", str(row["np"]),
                   str(binary), str(row["width"]), str(row["degree"]), row["grouping"], row["layout"], str(repetitions), str(trials)]
        row.update(status="running", command=command, log=str(log.resolve()))
        save()
        start = time.monotonic()
        try:
            with log.open("w") as stream:
                process = subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
            row["exit_status"] = process.returncode
            if process.returncode:
                raise ValueError("MPI process failed")
            prefix = "PLAN53_COEFFICIENT_COPY "
            results = [json.loads(line.removeprefix(prefix)) for line in log.read_text().splitlines() if line.startswith(prefix)]
            if len(results) != trials or {r["trial"] for r in results} != set(range(trials)):
                raise ValueError("missing/duplicate trial records")
            for result in results:
                if any(result[name] != row[name] for name in ("np", "width", "degree", "grouping", "layout")):
                    raise ValueError("benchmark row does not match command")
                if (result["preflight"] or not result["retained_binding_payload_exact"] or not result["mean_generations_unchanged"]
                        or not result["coefficient_allocation_retained"] or result["mpi_thread_provided"] < 1):
                    raise ValueError("binding/input/MPI contract failed")
                expected = {"production_commit", "bare_row_copy", "native_array_copy", "sweep_without_commit", "wbap_without_commit"}
                if result["phases"].keys() != expected:
                    raise ValueError("phase inventory differs")
                for phase in result["phases"].values():
                    if not math.isfinite(phase["seconds"]) or phase["seconds"] <= 0 or phase["calls_per_batch"] != (1 if row["grouping"] == "flow5" else 2):
                        raise ValueError("invalid phase timing/work count")
                modes = (row["degree"] + 1) * (row["degree"] + 2) // 2 - 1
                for bound in ("min", "max"):
                    expected_bytes = 16 * result[f"owned_{bound}"] * modes * (5 if row["grouping"] == "flow5" else 17)
                    if result[f"logical_read_write_bytes_{bound}"] != expected_bytes:
                        raise ValueError("logical byte count differs")
            row.update(status="verified", trials=results)
        except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as error:
            row.update(status="failed", reason=str(error))
        row.update(wall_seconds=time.monotonic() - start, log_sha256=sha(log))
        save()
        print(f"E3 row {ordinal}: {row['status']}, {row['wall_seconds']:.3f}s", flush=True)
        if row["status"] != "verified":
            manifest["status"] = "failed"
            save()
            raise RuntimeError(f"stopped at {log}; later rows remain unrun")
    manifest.update(status="verified", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
