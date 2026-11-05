#!/usr/bin/env python3
"""Bounded E5 actual-RHS and E6 coupled nonlinear-stage completion assessment."""

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


def verify(row, records, repetitions, trials):
    fields, components = (1, 5) if row["physics"] == "flow5" else (2, 14)
    if row["kind"] == "rhs":
        if len(records) != 2 * trials or {(r["trial"], r["instrumented"]) for r in records} != {
                (i, flag) for i in range(trials) for flag in (False, True)}:
            raise ValueError("incomplete paired RHS records")
        for record in records:
            expected = dict(np=row["np"], width=row["width"], profile=row["physics"], limiter=row["limiter"],
                            mode=row["mode"], transport=row["transport"], repetitions=repetitions,
                            degree=3, fields=fields, components=components)
            if any(record[key] != value for key, value in expected.items()):
                raise ValueError("RHS identity differs from command")
            if not record["full_field_oracle"] or not record["unchanged_means"] or record["provided"] < 2:
                raise ValueError("RHS numerical/source/MPI contract failed")
            placements = record["placements"]
            if len(placements) != 8 * row["np"]:
                raise ValueError("missing affinity provenance")
            physical = [(placements[i + a], placements[i + b]) for i in range(0, len(placements), 8)
                        for a, b in ((2, 3), (5, 6))]
            if len(set(physical)) != len(physical):
                raise ValueError("comparison oversubscribes physical cores")
            if not math.isfinite(record["rhs_seconds"]) or record["rhs_seconds"] <= 0:
                raise ValueError("invalid RHS timer")
            if not record["instrumented"]:
                if record["trace_max_per_rhs"] is not None or record["timeline_rank0"] is not None:
                    raise ValueError("timer-off mode retained instrumentation")
                continue
            trace = record["trace_max_per_rhs"]
            starts = fields * (7 + (3 if row["limiter"] == "WBAP" else 0))
            if row["mode"] == "production":
                if trace["starts"] != 0 or trace["vr_core_calls"] != 5 * fields or record["timeline_rank0"]:
                    raise ValueError("original production baseline unexpectedly substituted payloads")
                continue
            if trace["starts"] != starts or trace["vr_core_calls"] != 5 * fields:
                raise ValueError("fixed work count differs")
            if any(not math.isfinite(v) or v < 0 for v in trace.values()):
                raise ValueError("invalid trace phase")
            packed = row["transport"] == "packed"
            if trace["staging_calls"] != (3 * starts if packed else 0) or (trace["staging_bytes"] > 0) != packed:
                raise ValueError("staging dispatch/accounting differs")
            if (trace["pack_calls"] > 0) != packed or trace["pack_calls"] != trace["unpack_calls"]:
                raise ValueError("pack dispatch differs")
            if trace["receive_bytes"] <= 0 or trace["requests"] <= 0:
                raise ValueError("missing actual payload/requests")
            timeline = record["timeline_rank0"]
            if len(timeline) != starts * repetitions:
                raise ValueError("missing producer/communication timestamps")
            for event in timeline:
                if not (0 <= event["producer"] <= event["enqueue"] <= event["start"] <= event["first_post"]
                        <= event["received"] <= event["ghost_ready"]):
                    raise ValueError("unordered transfer timestamp")
    else:
        if len(records) != trials or {r["trial"] for r in records} != set(range(trials)):
            raise ValueError("incomplete nonlinear records")
        for record in records:
            expected = dict(np=row["np"], width=row["width"], physics=row["physics"], storage=row["storage"],
                            repetitions=repetitions, fields=fields, components=components, degree=3,
                            iterations_per_solve=1, linear_iterations_per_solve=5, residual_evaluations_per_solve=2)
            if any(record[key] != value for key, value in expected.items()):
                raise ValueError("nonlinear work/identity differs")
            if not all(record[k] for k in ("repeat_full_field_equal", "source_unchanged", "reference_unchanged")):
                raise ValueError("nonlinear input/repetition oracle failed")
            if not 0 <= record["energy_ratio_rank0"] < 1 or record["provided"] < 2:
                raise ValueError("nonlinear defect or MPI contract failed")
            if not math.isfinite(record["wall_seconds"]) or record["wall_seconds"] <= 0:
                raise ValueError("invalid nonlinear call time")
            if len(record["phases_max_seconds"]) != 8 or any(not math.isfinite(v) or v < 0 for v in record["phases_max_seconds"].values()):
                raise ValueError("invalid nonlinear profile")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    if revision != args.expected_source or status:
        raise ValueError("assessment requires clean pinned source")
    if not args.pilot and (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NUM_NODES") != "1"):
        raise ValueError("matched assessment requires one allocated node")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("assessment requires OMP1/OPENBLAS1")
    np = 2 if args.pilot else 32
    repetitions, trials = (1, 1) if args.pilot else (2, 3)
    rows = []
    workloads = ((12 if args.pilot else 32, "flow5", "none"),
                 (12 if args.pilot else 400, "flow5", "none"),
                 (12 if args.pilot else 400, "split-mixture", "WBAP"))
    # No broad matrix: one small control and the existing two large physics
    # profiles. Opposite placement order across workloads, every row serial.
    for ordinal, (width, physics, limiter) in enumerate(workloads):
        if args.pilot and ordinal == 1:
            continue
        rows.append(dict(kind="rhs", np=np, width=width, physics=physics, limiter=limiter,
                         transport="direct", mode="production", status="unrun"))
        for transport in ("direct", "packed"):
            for mode in (("caller", "compute-worker", "mpi-worker") if ordinal % 2 == 0 else
                         ("mpi-worker", "compute-worker", "caller")):
                rows.append(dict(kind="rhs", np=np, width=width, physics=physics, limiter=limiter,
                                 transport=transport, mode=mode, status="unrun"))
    for physics in ("flow5", "split-mixture"):
        for storage in (("assembled", "evaluated") if physics == "flow5" else ("evaluated", "assembled")):
            rows.append(dict(kind="stage", np=np, width=12 if args.pilot else 400, physics=physics, storage=storage, status="unrun"))
    binaries = {kind: (args.build / "test/cpp" / name).resolve() for kind, name in
                (("rhs", "cauchy_benchmark_rhs_execution"), ("stage", "cauchy_benchmark_nonlinear_stage"))}
    mechanism = Path(os.environ["DNDS_MECH_PATH"]) / "h2o2.yaml"
    manifest = dict(purpose=__doc__, revision=revision, source_status=status, rows=rows,
                    binaries={k: dict(path=str(p), sha256=sha(p)) for k, p in binaries.items()},
                    cache_sha256=sha(args.build / "CMakeCache.txt"),
                    source_sha256={str(p.relative_to(source)): sha(p) for p in
                                   [source / "test/cpp/Cauchy" / name for name in
                                    ("RHSBenchmarkFixture.hpp", "RHSExecutionBenchmark.hpp", "RHSExecutionBenchmark.cpp",
                                     "benchmark_RHSExecution.cpp", "benchmark_NonlinearStage.cpp")]},
                    mechanism=dict(path=str(mechanism.resolve()), sha256=sha(mechanism)), status="running",
                    job=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURM_NODELIST"), pilot=args.pilot,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    args.records.mkdir(parents=True, exist_ok=False)

    def save():
        temporary = args.records / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.records / "manifest.json")

    print(f"E5/E6 records: {args.records.resolve()}", flush=True)
    save()
    for ordinal, row in enumerate(rows):
        command = [str(args.mpiexec.resolve()), "--bind-to", "core", "--map-by", "slot:PE=2", "--report-bindings", "-np", str(np),
                   str(binaries[row["kind"]]), str(row["width"]), row["physics"]]
        if row["kind"] == "rhs":
            command += [row["limiter"], row["mode"], row["transport"]]
        else:
            command += [row["storage"]]
        command += [str(repetitions), str(trials)]
        log = args.records / f"row-{ordinal:02d}.log"
        row.update(command=command, status="running", log=str(log.resolve()))
        save()
        try:
            with log.open("w") as out:
                result = subprocess.run(command, env=dict(os.environ, DNDS_DISABLE_ASYNC_MPI="1"),
                                        stdout=out, stderr=subprocess.STDOUT, timeout=180, check=False)
            row["exit_status"] = result.returncode
            if result.returncode:
                raise ValueError("MPI process failed")
            prefix = "PLAN53_RHS_EXECUTION " if row["kind"] == "rhs" else "PLAN53_NONLINEAR "
            records = [json.loads(l[len(prefix):]) for l in log.read_text().splitlines() if l.startswith(prefix)]
            verify(row, records, repetitions, trials)
            if row["kind"] == "stage":
                verified = [json.loads(l.removeprefix("PLAN53_NONLINEAR_VERIFIED ")) for l in log.read_text().splitlines()
                            if l.startswith("PLAN53_NONLINEAR_VERIFIED ")]
                if len(verified) != 1 or not verified[0]["full_field"] or not math.isfinite(verified[0]["max_absolute"]):
                    raise ValueError("missing independent storage-mode complete-field oracle")
                row["terminal_oracle"] = verified[0]
            row.update(status="verified", trials=records, log_sha256=sha(log))
            save()
            print(f"verified {ordinal + 1}/{len(rows)} {row['kind']} {row['physics']}", flush=True)
        except Exception as error:
            row.update(status="failed", error=str(error))
            manifest["status"] = "failed"
            save()
            raise
    manifest.update(status="verified", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
