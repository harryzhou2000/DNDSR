#!/usr/bin/env python3
"""E3 composed flow5/split-mixture RHS publication fractions, with timer-off controls."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--profile", choices=("flow5", "split-mixture"), default="flow5")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if revision != args.expected_source or status:
        raise ValueError("source differs from clean requested commit")
    if not args.pilot and (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NUM_NODES") != "1"):
        raise ValueError("matched copy-RHS assessment requires one allocated node")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("copy-RHS assessment uses OMP1 and OPENBLAS1")
    repetitions, trials = (1, 2) if args.pilot else (5, 3)
    cases = ((2, 12),) if args.pilot else ((32, 32), (32, 400), (8, 200))
    rows = [{"np": np, "width": width, "limiter": limiter, "profile": args.profile, "status": "unrun"}
            for np, width in cases for limiter in ("none", "WBAP")]
    binary = args.binary.resolve()
    args.records.mkdir(parents=True, exist_ok=False)
    manifest = {"purpose": __doc__, "revision": revision, "source_status": status,
                "binary": str(binary), "binary_sha256": sha(binary), "cache_sha256": sha(binary.parents[2] / "CMakeCache.txt"),
                "source_sha256": {name: sha(root / name) for name in ("test/cpp/Cauchy/benchmark_CoefficientRHSCost.cpp", "scripts/benchmark/plan53_copy_rhs.py")},
                "job": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURM_NODELIST"), "pilot": args.pilot,
                "rows": rows, "status": "running", "controller_affinity": sorted(os.sched_getaffinity(0)),
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DNDS_DISABLE_ASYNC_MPI", "UCX_TLS", "UCX_NET_DEVICES")}}
    if args.profile == "split-mixture":
        mechanism = Path(os.environ["DNDS_MECH_PATH"]) / "h2o2.yaml"
        manifest["mechanism"] = dict(path=str(mechanism.resolve()), sha256=sha(mechanism))

    def save():
        temporary = args.records / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.records / "manifest.json")

    print(f"E3 RHS records/raw logs: {args.records.resolve()}", flush=True)
    save()
    for ordinal, row in enumerate(rows):
        log = args.records / f"row-{ordinal:02d}.log"
        command = [str(args.mpiexec.resolve()), "--bind-to", "core", "--map-by", "core", "--report-bindings", "-np", str(row["np"]),
                   str(binary), str(row["width"]), row["limiter"], str(repetitions), str(trials), row["profile"]]
        row.update(command=command, log=str(log.resolve()), status="running")
        save()
        try:
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
            row["exit_status"] = result.returncode
            if result.returncode:
                raise ValueError("MPI process failed")
            prefix = "PLAN53_COPY_RHS "
            records = [json.loads(line[len(prefix):]) for line in log.read_text().splitlines() if line.startswith(prefix)]
            expected_records = {(trial, flag) for trial in range(trials) for flag in (False, True)}
            if len(records) != 2 * trials or {(r["trial"], r["instrumented"]) for r in records} != expected_records:
                raise ValueError("missing/duplicate timer-on/off trial records")
            expected_calls = 5 + (3 if row["limiter"] == "WBAP" else 0)
            for record in records:
                fields, components = (1, 5) if row["profile"] == "flow5" else (2, 14)
                if (any(record[key] != row[key] for key in ("np", "width", "limiter", "profile"))
                        or record["physical_fields"] != fields or record["components"] != components
                        or record["degree"] != 3 or record["repetitions"] != repetitions):
                    raise ValueError("record identity/work differs from command")
                if not record["finite"] or not record["mean_generation_unchanged"] or record["provided"] < 1:
                    raise ValueError("numerical/generation/MPI contract failed")
                if not math.isfinite(record["rhs_seconds"]) or record["rhs_seconds"] <= 0:
                    raise ValueError("invalid RHS timing")
                if (not all(math.isfinite(record[key]) for key in
                            ("publication_seconds", "local_fraction_min", "local_fraction_max"))
                        or not 0 <= record["local_fraction_min"] <= record["local_fraction_max"] <= 1):
                    raise ValueError("invalid publication time/fraction")
                calls = repetitions * expected_calls * fields if record["instrumented"] else 0
                logical_bytes = repetitions * expected_calls * 16 * 9 * components if record["instrumented"] else 0
                owned_max = (row["width"] ** 2 + row["np"] - 1) // row["np"]
                if (record["publication_calls_max"] != calls
                        or record["publication_logical_bytes_max"] != logical_bytes * owned_max):
                    raise ValueError("actual publication count/bytes differ from fixed-sweep P3 contract")
                if (record["publication_seconds"] > 0) != record["instrumented"]:
                    raise ValueError("timer mode was not honored")
            row.update(status="verified", trials=records, log_sha256=sha(log))
            print(f"verified {ordinal + 1}/{len(rows)} np{row['np']} width{row['width']} {row['limiter']}", flush=True)
            save()
        except Exception as error:
            row.update(status="failed", error=str(error))
            manifest["status"] = "failed"
            save()
            raise
    manifest.update(status="verified", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
