#!/usr/bin/env python3
"""E6 same-allocation frozen-Jacobian comparison; no physical trajectory claim."""

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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    # Provenance belongs beside the caller's frozen name, including when the
    # pilot aliases an existing binary. Hashing still follows the executable.
    binaries = {name: getattr(args, name).absolute() for name in ("baseline", "candidate")}
    provenance = {}
    for name, binary in binaries.items():
        sidecar = binary.with_suffix(".json")
        provenance[name] = json.loads(sidecar.read_text())
        if provenance[name]["binary_sha256"] != sha(binary) or provenance[name]["source_status"]:
            raise ValueError("frozen binary hash/source-status mismatch")
    if not args.pilot and (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NUM_NODES") != "1"):
        raise ValueError("matched E6 comparison requires one allocated node")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("E6 uses OMP1 and OPENBLAS1")
    # Opposite candidate/control order between workloads, and opposite storage
    # order per version. Each invocation has isolated allocator high-water.
    cases = ((2, 12),) if args.pilot else ((32, 32), (32, 400), (8, 200))
    repetitions, trials = (1, 1) if args.pilot else (3, 3)
    rows = []
    for workload, (np, width) in enumerate(cases):
        versions = ("baseline", "candidate") if workload % 2 == 0 else ("candidate", "baseline")
        for version_index, version in enumerate(versions):
            modes = ("assembled", "evaluated") if (workload + version_index) % 2 == 0 else ("evaluated", "assembled")
            for mode in modes:
                rows.append({"np": np, "width": width, "version": version, "mode": mode, "status": "unrun"})
    args.records.mkdir(parents=True, exist_ok=False)
    manifest = {"purpose": __doc__, "job": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURM_NODELIST"),
                "pilot": args.pilot, "provenance": provenance, "rows": rows, "status": "running",
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DNDS_DISABLE_ASYNC_MPI", "UCX_TLS", "UCX_NET_DEVICES")},
                "controller_affinity": sorted(os.sched_getaffinity(0))}

    def save():
        temporary = args.records / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.records / "manifest.json")

    print(f"E6 records/raw logs: {args.records.resolve()}", flush=True)
    save()
    expected_phases = {"auxiliary", "prepare_warm", "inverse", "matrix_action", "sgs_pair_local",
                       "prepare_inverse_pairs_1", "prepare_inverse_pairs_5", "prepare_inverse_pairs_20", "prepare_inverse_solve5"}
    for ordinal, row in enumerate(rows):
        log = args.records / f"row-{ordinal:02d}.log"
        command = [str(args.mpiexec.resolve()), "--bind-to", "core", "--map-by", "core", "--report-bindings", "-np", str(row["np"]),
                   str(binaries[row["version"]]), str(row["width"]), row["mode"], str(repetitions), str(trials)]
        row.update(command=command, status="running", log=str(log.resolve()))
        save()
        try:
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
            row["exit_status"] = result.returncode
            if result.returncode:
                raise ValueError("MPI process failed")
            prefix, verified_prefix = "PLAN53_JACOBIAN ", "PLAN53_JACOBIAN_VERIFIED "
            records = [json.loads(line[len(prefix):]) for line in log.read_text().splitlines() if line.startswith(prefix)]
            verified = [json.loads(line[len(verified_prefix):]) for line in log.read_text().splitlines() if line.startswith(verified_prefix)]
            if len(records) != trials or {record["trial"] for record in records} != set(range(trials)):
                raise ValueError("missing/duplicate trial record")
            if verified != [{"action": True, "sgs": True, "state_generation_unchanged": True}]:
                raise ValueError("missing terminal complete-field oracle")
            for record in records:
                if any(record[key] != row[key] for key in ("np", "width", "mode")) or record["degree"] != 3:
                    raise ValueError("trial identity differs from command")
                if record["provided"] < 1 or not record["state_generation_unchanged"] or record["completed_solve_iterations"] != 5 * repetitions:
                    raise ValueError("MPI/generation/completed-work contract failed")
                if record["repetitions"] != repetitions or record["phases"].keys() != expected_phases:
                    raise ValueError("phase/repetition inventory differs")
                if any(not math.isfinite(value) or value <= 0 for value in record["phases"].values()):
                    raise ValueError("invalid phase timing")
                if not math.isfinite(record["first_preparation"]) or record["first_preparation"] <= 0:
                    raise ValueError("invalid first preparation")
                if any(not isinstance(record[key], int) or record[key] <= 0 for key in ("rss_before_problem", "rss_after_work", "peak_rss")):
                    raise ValueError("invalid memory observation")
            row.update(status="verified", trials=records, log_sha256=sha(log))
            print(f"verified row {ordinal + 1}/{len(rows)}: {row['version']}/{row['mode']} np{row['np']} width{row['width']}", flush=True)
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
