#!/usr/bin/env python3
"""E4/E5 bounded same-allocation Runtime scheduling assessment, not solver speedup."""

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
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    if revision != args.expected_source or status:
        raise ValueError("scheduling assessment requires clean pinned source")
    if not args.pilot and (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NUM_NODES") != "1"):
        raise ValueError("matched scheduling assessment requires one allocated node")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("scheduling assessment uses OMP1 and OPENBLAS1")
    rows = []

    def add(np, owned, halo, modes, transport, schedule, preflight, work, mpi_level):
        rows.append(dict(np=np, owned=owned, halo=halo, modes=modes, transport=transport,
                         schedule=schedule, preflight=preflight, work=work, mpi_level=mpi_level, status="unrun"))

    if args.pilot:
        for schedule in ("separate", "grouped", "compute-worker", "mpi-worker"):
            add(2, 64, 8, 9, "packed", schedule, schedule in ("separate", "grouped"),
                int("worker" in schedule), "serialized")
    else:
        # Isolate grouping/preflight without changing messages or core work.
        for case, (owned, halo, modes) in enumerate(((32, 8, 1), (5000, 256, 9))):
            for transport in ("direct", "packed"):
                for preflight in (True, False):
                    for schedule in (("separate", "grouped") if case == 0 else ("grouped", "separate")):
                        add(32, owned, halo, modes, transport, schedule, preflight, 0, "serialized")
        # Equal reserved CPU budget: two physical cores/rank for every control.
        for transport in ("direct", "packed"):
            for schedule in ("separate", "compute-worker", "mpi-worker"):
                add(32, 5000, 256, 9, transport, schedule, False, 2, "serialized")
            # Current application MULTIPLE versus caller SERIALIZED control.
            add(32, 5000, 256, 9, transport, "separate", False, 2, "multiple")
    repetitions, trials = (2, 1) if args.pilot else (10, 3)
    binary = args.binary.resolve()
    args.records.mkdir(parents=True, exist_ok=False)
    manifest = dict(purpose=__doc__, revision=revision, source_status=status,
                    binary=str(binary), binary_sha256=sha(binary), cache_sha256=sha(binary.parents[2] / "CMakeCache.txt"),
                    source_sha256={name: sha(source / name) for name in
                                   ("test/cpp/DNDS/benchmark_TransferScheduling.cpp", "scripts/benchmark/plan53_transfer_scheduling.py")},
                    rows=rows, status="running", pilot=args.pilot,
                    job=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURM_NODELIST"),
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

    def save():
        temporary = args.records / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.records / "manifest.json")

    print(f"E4/E5 records and raw logs: {args.records.resolve()}", flush=True)
    save()
    for ordinal, row in enumerate(rows):
        log = args.records / f"row-{ordinal:02d}.log"
        command = [str(args.mpiexec.resolve()), "--bind-to", "core", "--map-by", "slot:PE=2", "--report-bindings", "-np", str(row["np"]),
                   str(binary), str(row["owned"]), str(row["halo"]), str(row["modes"]), row["transport"], row["schedule"],
                   str(int(row["preflight"])), str(row["work"]), str(repetitions), str(trials)]
        environment = dict(os.environ, DNDS_DISABLE_ASYNC_MPI="1" if row["mpi_level"] == "serialized" else "0")
        row.update(command=command, mpi_environment=environment["DNDS_DISABLE_ASYNC_MPI"], status="running", log=str(log.resolve()))
        save()
        try:
            with log.open("w") as stream:
                result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=180, check=False)
            row["exit_status"] = result.returncode
            if result.returncode:
                raise ValueError("MPI process failed")
            prefix = "PLAN53_TRANSFER "
            records = [json.loads(line[len(prefix):]) for line in log.read_text().splitlines() if line.startswith(prefix)]
            if len(records) != trials or {r["trial"] for r in records} != set(range(trials)):
                raise ValueError("missing/duplicate trial records")
            for record in records:
                identity = dict(np=row["np"], rows=row["owned"], halo=row["halo"], modes=row["modes"], transport=row["transport"],
                                schedule=row["schedule"], preflight=row["preflight"], core_work=row["work"], repetitions=repetitions)
                if any(record[key] != value for key, value in identity.items()):
                    raise ValueError("record differs from intended workload")
                if record["provided"] < (2 if row["mpi_level"] == "serialized" else 3):
                    raise ValueError("insufficient MPI thread support")
                if not record["all_oracles"] or not record["source_generations_unchanged"]:
                    raise ValueError("payload/core/generation oracle failed")
                placements = record["placements"]
                if len(placements) != 8 * row["np"]:
                    raise ValueError("missing per-rank CPU/NUMA provenance")
                cores = [(placements[i + a], placements[i + b]) for i in range(0, len(placements), 8) for a, b in ((2, 3), (5, 6))]
                if len(set(cores)) != len(cores):
                    raise ValueError("comparison oversubscribes physical cores")
                times = record["max_seconds"]
                if len(times) != 14 or any(not math.isfinite(value) or value < 0 for value in times.values()) or times["mock_stage"] <= 0:
                    raise ValueError("invalid/incomplete measured phases")
                if not math.isfinite(record["setup_seconds_rank0"]) or record["setup_seconds_rank0"] <= 0:
                    raise ValueError("missing setup observation")
                expected_agreements = (1 if row["schedule"] == "grouped" else 2) * repetitions if row["preflight"] else 0
                counts = record["counts"]
                if counts[0] != expected_agreements or counts[3] != counts[4] or counts[3] <= 0:
                    raise ValueError("agreement/request count mismatch")
                if (counts[1] != counts[2] or (counts[1] > 0) != (row["transport"] == "packed")):
                    raise ValueError("pack/unpack dispatch mismatch")
                if record["receive_payload_bytes"] <= 0 or (record["explicit_staging_copy_bytes"] > 0) != (row["transport"] == "packed"):
                    raise ValueError("payload/staging accounting mismatch")
            row.update(status="verified", trials=records, log_sha256=sha(log))
            print(f"verified {ordinal + 1}/{len(rows)} {row['transport']}/{row['schedule']} rows{row['owned']} checks{row['preflight']}", flush=True)
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
