"""Bounded executable integration: CLI, segmented history, output and restart.

Uses only the Python standard library. CTest launches this controller outside
MPI; its child solver processes use the configured MPI launcher and rank count.
This is a functionality check, not an IV accuracy or performance experiment.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mpi", required=True)
    parser.add_argument("--np", type=int, required=True)
    args = parser.parse_args()

    def run(*arguments, success=True):
        result = subprocess.run(
            [args.mpi, "-np", str(args.np), str(args.exe), *map(str, arguments)],
            cwd=args.source, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60,
        )
        assert (result.returncode == 0) == success, result.stdout
        return result.stdout

    assert "--overwrite_key" in run("--help")
    with tempfile.TemporaryDirectory(prefix="cauchy-frontend-") as directory:
        folder = Path(directory)
        # Independent tiny input, not a rewrite of any maintained case.
        config = folder / "case.json"
        config.write_text(json.dumps({
            "defaults": {
                "mesh": {"file": "data/mesh/IV10_20.cgns", "dimension": 2,
                         "periodicLengths": [10, 10, 0],
                         "boundaryIds": {"PERIODIC_1": -1, "PERIODIC_1_DONOR": -4,
                                         "PERIODIC_2": -2, "PERIODIC_2_DONOR": -5}},
                "initialPrimitive": [1, 1, 1, 0, 1],
                "initialProfile": {"kind": "isentropicVortex", "strength": 1},
                "numerics": {"reconstruction": "constant", "limiter": "none"},
            },
            "domains": [{"name": "fluid", "overrides": {"execution": {"overlap": True}}}],
            "time": {"method": "ssprk3", "steps": 2, "dt": 1e-5},
            "output": {"prefix": str(folder / "first"), "interval": 0,
                       "historyInterval": 1, "historySegmentRows": 2},
        }))
        resolved = json.loads(run(config, "--check-config", "-k", "/time/dt", "-v", "2e-5",
                                  "-k", "/domains/0/settings/execution/overlap", "-v", "false"))
        assert resolved["time"]["dt"] == 2e-5
        assert not resolved["domains"][0]["settings"]["execution"]["overlap"]
        assert "paired" in run(config, "--check-config", "-k", "/time/dt", success=False)
        run(config)
        first = json.loads((folder / "first_summary.json").read_text())
        assert first["steps"] == 2 and math.isclose(first["time"], 2e-5)
        assert first["status"] == "target-reached"
        rows = []
        header = None
        for segment in (folder / "first_history.csv", folder / "first_history_1.csv"):
            with segment.open() as stream:
                reader = csv.DictReader(stream)
                if header is None:
                    header = reader.fieldnames
                assert reader.fieldnames == header and len(header) == 36
                data = list(reader)
                assert len(data) == 2
                rows.extend(data)
        assert [int(row["step"]) for row in rows] == [0, 1, 2, 2]
        assert all(None not in row and all(math.isfinite(float(value)) for value in row.values()) for row in rows)
        assert (folder / "first_final.vtkhdf").exists()
        run(config, "--restart", folder / "first_final.h5", "-k", "/output/prefix",
            "-v", json.dumps(str(folder / "resumed")), "-k", "/time/steps", "-v", "1")
        resumed = json.loads((folder / "resumed_summary.json").read_text())
        assert resumed["steps"] == 3 and math.isclose(resumed["time"], 3e-5)
        assert resumed["status"] == "target-reached"
    print(f"Cauchy frontend np={args.np}: CLI, history rotation, output and restart passed")


if __name__ == "__main__":
    main()
