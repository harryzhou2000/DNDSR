#!/usr/bin/env python3
"""Validate and preserve compact evidence from an isolated E2 storage probe."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    encoded = args.manifest.read_bytes()
    data = json.loads(encoded)
    cases = data["cases"]
    assert data["status"] == "verified" and len(cases) == 12
    assert all(case["status"] == "verified" and case["exit_status"] == 0 for case in cases)
    expected = {(trial, dim, profile) for trial in (0, 1) for dim in (2, 3) for profile in ("full", "sweeps", "wbap")}
    assert {(case["trial"], case["dimension"], case["profile"]) for case in cases} == expected
    omitted = {"localMatrix", "localInverse", "rawMatrixB", "rawVectorB", "rawBoundaryInjection"}
    for case in cases:
        assert len(case["ranks"]) == data["np"]
        log = Path(case["log"]).read_bytes()
        assert digest(log) == case["log_sha256"]
        raw_rows = [json.loads(line.removeprefix("PLAN53_RESULT ")) for line in log.decode().splitlines() if line.startswith("PLAN53_RESULT ")]
        assert len(raw_rows) == 1 and raw_rows[0] == case["ranks"]
        for rank, row in enumerate(case["ranks"]):
            assert row["rank"] == rank and row["np"] == data["np"]
            assert all(row[key] == case[key] for key in ("dimension", "width", "degree", "profile"))
            assert row["owned_cells"] == case["width"] ** case["dimension"] // data["np"]
            assert row["cpu_affinity"] and row["mpi_provided_thread_level"] == 3
            tables = {table["name"]: table for table in row["tables"]}
            assert omitted <= tables.keys(), "missing table reports cannot count as absent allocations"
            if case["profile"] != "full":
                assert all(not tables[name]["allocated"] and tables[name]["allocation_bytes"] == 0 for name in omitted)
            assert tables["secondary"]["allocated"] == (case["profile"] != "sweeps")
    for trial in (0, 1):
        for dim in (2, 3):
            group = {case["profile"]: case for case in cases if case["trial"] == trial and case["dimension"] == dim}
            for rank in range(data["np"]):
                rows = [group[name]["ranks"][rank] for name in ("full", "sweeps", "wbap")]
                assert len({row["sweep_digest"] for row in rows}) == 1
                assert rows[0]["limiter_digest"] == rows[2]["limiter_digest"]
    rows = []
    measurements = ("retained_allocation_bytes", "retained_payload_bytes", "runtime_scratch_bytes", "runtime_peak_bytes", "rss_before_bytes", "rss_after_bytes", "hwm_before_bytes", "hwm_after_bytes", "prepare_seconds")
    for dim in (2, 3):
        for profile in ("full", "sweeps", "wbap"):
            group = [case for case in cases if case["dimension"] == dim and case["profile"] == profile]
            row = {"dimension": dim, "width": group[0]["width"], "degree": 3, "profile": profile, "owned_cells_per_rank": group[0]["ranks"][0]["owned_cells"]}
            for key in measurements:
                values = [max(rank[key] for rank in case["ranks"]) for case in group]
                row[key] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
            rows.append(row)
    summary = {"source_head": data["source_head"], "binary_sha256": data["binary_sha256"], "manifest_sha256": digest(encoded), "runner": data["runner"], "np": data["np"], "timing_limit": "local shared-host observations; no matched speedup conclusion", "rows": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "manifest.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        for ordinal, case in enumerate(cases):
            payload = Path(case["log"]).read_bytes()
            member = tarfile.TarInfo(f"row-{ordinal:02d}.log")
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    (args.output / "raw-logs.tar.gz").write_bytes(gzip.compress(archive.getvalue(), mtime=0))
    # Preserve measured bytes before the commit hook formats source. Do not
    # rebuild merely to replace the measured revision with formatted spelling.
    source = io.BytesIO()
    repo = Path(__file__).resolve().parents[2]
    with tarfile.open(fileobj=source, mode="w") as bundle:
        for name, expected_hash in data["source_hashes"].items():
            if name.startswith("build-audit/"):
                continue  # Cache hash/options remain in the manifest/report.
            payload = (repo / name).read_bytes()
            assert digest(payload) == expected_hash, f"measured source changed: {name}"
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    (args.output / "measured-sources.tar.gz").write_bytes(gzip.compress(source.getvalue(), mtime=0))
    print(json.dumps({"verified_rows": len(cases), "output": str(args.output), "manifest_sha256": digest(encoded)}, indent=2))


if __name__ == "__main__":
    main()
