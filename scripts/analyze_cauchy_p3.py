#!/usr/bin/env python3
"""Inspect Cauchy VTKHDF/CSV observables without treating startup as acceptance.

Requires numpy/h5py. Writes compact JSON and profiles beside an explicitly
selected result path; ordinary VTKHDF data stays in the solver data tree.
Cell coordinates are physical centroids of output geometry, not persistent IDs.
Surface samples carry persistent cell IDs for independent association.
"""
import argparse
import csv
import json
import hashlib
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, help="Output prefix ending in _final")
    parser.add_argument("--kind", choices=("vortex", "inviscid", "supersonic", "viscous", "dmr"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology-reference", type=Path,
                        help="Explicit offsets reference, after exact Points/Connectivity/Types equality; never repairs the source file")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(args.prefix) + ".vtkhdf") as stream:
        grid = stream["VTKHDF"]
        points = grid["Points"][()]
        connectivity = grid["Connectivity"][()]
        reference_record = None
        if args.topology_reference is None:
            offsets = grid["Offsets"][()]
        else:
            with h5py.File(args.topology_reference) as reference:
                other = reference["VTKHDF"]
                for name, values in (("Points", points), ("Connectivity", connectivity), ("Types", grid["Types"][()])):
                    if not np.array_equal(values, other[name][()]):
                        raise ValueError(f"topology reference differs in {name}")
                offsets = other["Offsets"][()]
            with args.topology_reference.open("rb") as source:
                reference_hash = hashlib.file_digest(source, "sha256").hexdigest()
            reference_record = dict(path=str(args.topology_reference),
                                    sha256=reference_hash,
                                    exact_checks=["Points", "Connectivity", "Types"],
                                    limitation="Offsets borrowed explicitly; source visualization remains invalid")
        fields = {key: grid["CellData"][key][()] for key in grid["CellData"]}
        # Native VTKHDF stores one global offsets sequence. Assert rather than
        # silently interpreting a partition-local representation incorrectly.
        count = len(fields["rho"])
        if offsets.size != count + 1 or offsets[0] != 0 or offsets[-1] != connectivity.size:
            raise ValueError("VTKHDF offsets require partition-aware normalization")
        centers = np.vstack([points[connectivity[offsets[i]:offsets[i + 1]]].mean(axis=0) for i in range(count)])
        result = dict(cells=count, bounds=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                      finite={key: bool(np.isfinite(value).all()) for key, value in fields.items()},
                      ranges={key: [float(np.min(value)), float(np.max(value))] for key, value in fields.items()})
        if reference_record is not None:
            result["topology_reference"] = reference_record
        if args.kind == "supersonic":
            # Narrow band near the upstream symmetry line; midpoint threshold
            # is a recorded diagnostic, not a grid-independent shock oracle.
            band = (np.abs(centers[:, 1]) < .025) & (centers[:, 0] < -.5)
            indices = np.flatnonzero(band)
            indices = indices[np.argsort(centers[indices, 0])]
            crossing = indices[fields["rho"][indices] >= 3]
            if crossing.size:
                shock_x = float(centers[crossing[0], 0])
                result.update(shock_density_threshold=3, shock_x_estimate=shock_x, standoff_estimate=-.5 - shock_x)
            with args.output.with_suffix(".centerline.csv").open("w") as target:
                writer = csv.writer(target)
                writer.writerow(["x", "y", "rho", "p", "Mach"])
                for i in indices:
                    writer.writerow([*centers[i, :2], fields["rho"][i], fields["p"][i], fields["Mach"][i]])
        if args.kind == "viscous":
            near_wake = (centers[:, 0] > .5) & (centers[:, 0] < 10) & (np.abs(centers[:, 1]) < .1)
            result["near_wake_negative_u_cells"] = int(np.sum(near_wake & (fields["u"] < 0)))
        if args.kind == "dmr":
            result["geometry_cell_types"] = {str(int(k)): int(v) for k, v in zip(*np.unique(grid["Types"][()], return_counts=True))}
            result["mach_stem_band_rho_range"] = [float(fields["rho"][(centers[:, 0] < 1) & (centers[:, 1] < .2)].min()),
                                                 float(fields["rho"][(centers[:, 0] < 1) & (centers[:, 1] < .2)].max())]
    surface_files = sorted(args.prefix.parent.glob(args.prefix.name + "_surface_rank*.csv"))
    samples = []
    for file in surface_files:
        with file.open() as stream:
            samples.extend(csv.DictReader(stream))
    if args.kind == "inviscid":
        samples = [sample for sample in samples if int(sample["zone"]) == 2]
        for sample in samples:
            theta = np.arctan2(float(sample["y"]), float(sample["x"]))
            sample["theta"] = float(theta)
            sample["Cp"] = (float(sample["p"]) - 71.42857142857143) / .5
            sample["Cp_potential"] = float(1 - 4 * np.sin(theta) ** 2)
        if samples:
            weight = np.array([float(sample["weight"]) for sample in samples])
            error = np.array([sample["Cp"] - sample["Cp_potential"] for sample in samples])
            result["Cp_potential_weighted_L2"] = float(np.sqrt(np.sum(weight * error ** 2) / np.sum(weight)))
            samples.sort(key=lambda row: row["theta"])
            with args.output.with_suffix(".cp.csv").open("w") as target:
                writer = csv.DictWriter(target, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
    result["surface_quadrature_samples"] = len(samples)
    result["numerically_qualified"] = False  # join with horizon/residual/reference evidence
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
