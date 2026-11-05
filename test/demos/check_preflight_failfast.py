"""Bounded MPI failure check: reject a missing safety agreement by aborting."""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", required=True)
    parser.add_argument("--exe", required=True)
    args = parser.parse_args()
    result = subprocess.run(
        [args.mpi, "-np", "2", args.exe, "--preflight-off-failure"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15,
    )
    if result.returncode == 0 or "collective preflight disabled; aborting" not in result.stdout:
        raise RuntimeError(f"Expected explicit preflight-off MPI abort, got {result.returncode}:\n{result.stdout}")


if __name__ == "__main__":
    main()
