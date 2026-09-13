#!/usr/bin/env python3
"""Generate a stationary Cantera-profile ExprTk overlay for premixed flames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--closure-species", required=True)
    parser.add_argument("--front-x", type=float, default=0.005)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def column_name(field_names: list[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    raise KeyError(f"missing one of {candidates}")


def read_profile(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0] is None:
        raise ValueError(f"empty profile: {path}")
    fields = list(rows[0])
    aliases = {
        "x": column_name(fields, "z_m", "x_m"),
        "temperature": column_name(fields, "T_K", "temperature_K"),
        "pressure": column_name(fields, "p_Pa", "pressure_Pa"),
        "velocity": column_name(fields, "velocity_m_per_s", "velocity_m_s"),
    }
    data = {
        logical: np.asarray([float(row[name]) for row in rows], dtype=float)
        for logical, name in aliases.items()
    }
    species = [field.removeprefix("Y_")
               for field in fields if field.startswith("Y_")]
    for species_name in species:
        data[f"Y_{species_name}"] = np.asarray(
            [float(row[f"Y_{species_name}"]) for row in rows], dtype=float
        )
    if not np.all(np.diff(data["x"]) > 0.0):
        raise ValueError("profile coordinates must be strictly increasing")
    return data, species


def midpoint(x_values: np.ndarray, temperature: np.ndarray) -> float:
    target = 0.5 * (temperature[0] + temperature[-1])
    crossings = np.flatnonzero(
        (temperature[:-1] - target) * (temperature[1:] - target) <= 0.0
    )
    if crossings.size != 1:
        raise RuntimeError(
            f"expected one temperature midpoint, got {crossings.size}")
    index = int(crossings[0])
    fraction = (target - temperature[index]) / \
        (temperature[index + 1] - temperature[index])
    return float(x_values[index] + fraction * (x_values[index + 1] - x_values[index]))


def selected_indices(size: int, maximum: int) -> np.ndarray:
    if maximum < 4:
        raise ValueError("max-points must be at least four")
    if size <= maximum:
        return np.arange(size, dtype=int)
    return np.unique(np.rint(np.linspace(0, size - 1, maximum)).astype(int))


def piecewise_expression(x_values: np.ndarray, values: np.ndarray) -> str:
    expression = format(float(values[-1]), ".17g")
    for index in range(len(x_values) - 2, -1, -1):
        slope = (values[index + 1] - values[index]) / \
            (x_values[index + 1] - x_values[index])
        segment = f"{values[index]:.17g}+({slope:.17g})*(x[0]-({x_values[index]:.17g}))"
        expression = f"if(x[0]<={x_values[index + 1]:.17g},{segment},{expression})"
    return f"if(x[0]<={x_values[0]:.17g},{values[0]:.17g},{expression})"


def main() -> None:
    args = parse_args()
    data, species = read_profile(args.profile)
    if args.closure_species not in species:
        raise KeyError(
            f"closure species {args.closure_species!r} is absent from profile")
    retained_species = [
        name for name in species if name != args.closure_species]
    source_indices = selected_indices(data["x"].size, args.max_points)
    mapped_x = args.front_x + data["x"] - \
        midpoint(data["x"], data["temperature"])
    source_indices = source_indices[np.argsort(mapped_x[source_indices])]
    x_values = mapped_x[source_indices]
    state_fields = [
        data["temperature"],
        data["velocity"],
        np.zeros_like(data["temperature"]),
        np.zeros_like(data["temperature"]),
        data["pressure"],
        *(np.maximum(data[f"Y_{name}"], 0.0) for name in retained_species),
    ]
    exprs = ["inRegion := 1;"]
    for state_index, values in enumerate(state_fields):
        expression = piecewise_expression(x_values, values[source_indices])
        if state_index >= 5:
            expression = f"max({expression},0.0)"
        exprs.append(f"UExprtk[{state_index}] := {expression};")
    exprs.append("0")
    payload = {
        "profile": str(args.profile),
        "profile_sha256": __import__("hashlib").file_digest(args.profile.open("rb"), "sha256").hexdigest(),
        "closure_species": args.closure_species,
        "transported_species": retained_species,
        "front_x_m": args.front_x,
        "selected_points": int(source_indices.size),
        "fresh_state": {
            "temperature_K": float(data["temperature"][0]),
            "pressure_Pa": float(data["pressure"][0]),
            "velocity_m_s": float(data["velocity"][0]),
        },
        "burned_state": {
            "temperature_K": float(data["temperature"][-1]),
            "pressure_Pa": float(data["pressure"][-1]),
            "velocity_m_s": float(data["velocity"][-1]),
        },
        "overlay_key": "/eulerSettings/exprtkInitializers/0/exprs",
        "exprs": exprs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(
        {key: value for key, value in payload.items() if key != "exprs"}, indent=2))


if __name__ == "__main__":
    main()
