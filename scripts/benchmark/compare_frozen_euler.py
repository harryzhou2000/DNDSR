#!/usr/bin/env python3
"""Compare independently built Euler probes by persistent ID, without a solver run.

The default gate is exact bit equality. Explicit tolerances apply only to
numeric outputs; frozen input, geometry and metadata must always match exactly.
Bit differences, absolute error, component-scaled relative error and RMS error
remain visible even when an explicitly requested numeric tolerance accepts them.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


MATRICES = {"source_jacobian_bits", "no_reactive_jacobian_bits", "cell_jacobian_bits"}
EXACT = {"u_bits", "rec_bits", "volume_bits"}


def decode(bits):
    value = struct.unpack("=d", struct.pack("=Q", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite probe payload")
    return value


def read_probe(prefix):
    """Require the complete rank inventory and unique persistent owned IDs."""
    files = sorted(prefix.parent.glob(prefix.name + ".rank*.json"))
    if not files:
        raise ValueError(f"missing probe: {prefix}")
    rows, ranks, metadata, hashes = {}, set(), None, {}
    for path in files:
        raw = path.read_bytes()
        record = json.loads(raw)
        current = {key: record[key] for key in ("model", "np", "polynomial", "energy_convention", "scales")}
        if metadata is None:
            metadata = current
        if current != metadata or record["rank"] in ranks or not record["input_bits_preserved"]:
            raise ValueError("rank metadata or input preservation mismatch")
        ranks.add(record["rank"])
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for row in record["cells"]:
            if row["id"] in rows:
                raise ValueError("duplicate persistent owned ID")
            rows[row["id"]] = row
    if ranks != set(range(metadata["np"])):
        raise ValueError("missing or extra rank records")
    if not rows:
        raise ValueError("empty global probe")
    return metadata, rows, hashes


def compare(old_prefix, new_prefix, atol=0.0, rtol=0.0):
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("finite nonnegative comparison tolerances required")
    old_meta, old, old_hashes = read_probe(old_prefix)
    new_meta, new, new_hashes = read_probe(new_prefix)
    if old_meta != new_meta or old.keys() != new.keys():
        raise ValueError("metadata or persistent cell inventory differs")
    fields = old[min(old)].keys() - {"id"}
    if any(row.keys() - {"id"} != fields for row in [*old.values(), *new.values()]):
        raise ValueError("probe field inventory differs")
    width = len(old[min(old)]["u_bits"])
    modes, remainder = divmod(len(old[min(old)]["rec_bits"]), width)
    if not modes or remainder:
        raise ValueError("coefficient shape is not modes by components")
    report = {"metadata": old_meta, "cells": len(old), "variables": width, "modes": modes,
              "atol": atol, "rtol": rtol, "sha256": {"old": old_hashes, "new": new_hashes}, "fields": {}}
    accepted = True
    for key in sorted(fields):
        groups = {}
        for cell_id in sorted(old):
            left, right = old[cell_id][key], new[cell_id][key]
            left = left if isinstance(left, list) else [left]
            right = right if isinstance(right, list) else [right]
            if len(left) != len(right):
                raise ValueError(f"row width differs for {key}")
            expected = width
            if key in MATRICES:
                expected = width * width
            elif key == "rec_bits":
                expected = modes * width
            elif key == "properties_bits":
                expected = 6
            elif key in {"volume_bits", "dt_bits"}:
                expected = 1
            # NS_EX stores only N-1 species equations but returns all N diffusivities.
            if key == "diffusion_bits":
                expected = width - 4
            if len(left) != expected:
                raise ValueError(f"unexpected field shape: {key} ({len(left)} != {expected})")
            for ordinal, (a, b) in enumerate(zip(left, right)):
                component = ordinal // modes if key == "rec_bits" else ordinal % width if key in MATRICES else ordinal
                groups.setdefault(component, []).append((decode(a), decode(b), a != b))
        summaries = []
        for component, pairs in sorted(groups.items()):
            scale = max(max(abs(a), abs(b)) for a, b, _ in pairs)
            errors = [abs(a - b) for a, b, _ in pairs]
            bit_differences = sum(different for _, _, different in pairs)
            exact = key in EXACT or (atol == 0 and rtol == 0)
            passed = bit_differences == 0 if exact else all(error <= atol + rtol * scale for error in errors)
            accepted &= passed
            summaries.append({"component": component, "values": len(pairs), "bit_differences": bit_differences,
                              "max_abs": max(errors), "scale": scale, "max_scaled_relative": max(errors) / scale if scale else 0.0,
                              "rms_abs": math.hypot(*errors) / math.sqrt(len(errors)), "passed": passed})
        report["fields"][key] = summaries
    report["passed"] = accepted
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_prefix", type=Path)
    parser.add_argument("new_prefix", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    result = compare(args.old_prefix, args.new_prefix, args.atol, args.rtol)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['cells']} cells, {len(result['fields'])} fields; {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
