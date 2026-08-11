"""Load and transform DNDSR CSV convergence histories."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d

ArrayLike = Sequence[float] | np.ndarray
LogData = Mapping[str, ArrayLike]


@dataclass(frozen=True)
class ReachResult:
    """First logged iteration and wall time satisfying a residual threshold."""

    iteration: float
    wall_time: float

    @property
    def reached(self) -> bool:
        return np.isfinite(self.iteration) and np.isfinite(self.wall_time)


def load_dndsr_log(path: str | Path) -> dict[str, np.ndarray]:
    """Load a DNDSR comma-separated ``*.log`` file as numeric columns."""

    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"DNDSR log has no header: {path}")
        columns: dict[str, list[float]] = {
            name: [] for name in reader.fieldnames if name
        }
        for row_number, row in enumerate(reader, start=2):
            for name in columns:
                value = row.get(name)
                if value is None or value == "":
                    raise ValueError(
                        f"missing value for {name!r} at {path}:{row_number}"
                    )
                try:
                    columns[name].append(float(value))
                except ValueError as exc:
                    raise ValueError(
                        f"non-numeric value for {name!r} at {path}:{row_number}: "
                        f"{value!r}"
                    ) from exc
    return {name: np.asarray(values) for name, values in columns.items()}


def windowed_std(values: ArrayLike, window_size: int) -> np.ndarray:
    """Return the notebook's reflected moving standard-deviation series."""

    values_array = np.asarray(values, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size >= values_array.size:
        raise ValueError("window_size must be smaller than the input series")
    mean = uniform_filter1d(values_array, size=window_size, mode="reflect")
    mean_sq = uniform_filter1d(
        values_array**2, size=window_size, mode="reflect")
    return np.sqrt(mean_sq - mean**2)[:-window_size]


def compute_residual_maxima(
    runs: Sequence[LogData],
    *,
    prefix: str = "res",
    absolute: bool = False,
) -> dict[str, float]:
    """Compute one full-history maximum per residual column across all runs.

    ``absolute=False`` reproduces the MGTest0012 notebook exactly. Set it to
    ``True`` when signed residual-like data must use a maximum magnitude.
    """

    maxima: dict[str, float] = {}
    for run in runs:
        for name, values in run.items():
            if not name.startswith(prefix):
                continue
            array = np.asarray(values, dtype=float)
            if array.size == 0:
                continue
            candidate = np.nanmax(np.abs(array) if absolute else array)
            maxima[name] = max(maxima.get(name, -np.inf), float(candidate))
    return maxima


def _trim(values: np.ndarray, drop_last: bool) -> np.ndarray:
    return values[:-1] if drop_last else values


def prepare_series(
    data: LogData,
    *,
    x_key: str = "tWall",
    y_key: str = "res0",
    residual_max: float | Mapping[str, float] | None = None,
    std_window: int = 0,
    residual_smooth_window: int = 20,
    drop_last: bool = True,
    offset_wall_time: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare one curve with the historical notebook's operation order."""

    if x_key not in data or y_key not in data:
        missing = [name for name in (x_key, y_key) if name not in data]
        raise KeyError(f"missing log column(s): {', '.join(missing)}")

    x_values = _trim(np.asarray(data[x_key], dtype=float), drop_last).copy()
    y_values = _trim(np.asarray(data[y_key], dtype=float), drop_last).copy()
    if x_values.size != y_values.size:
        raise ValueError(f"{x_key} and {y_key} have different lengths")

    if x_key == "tWall" and offset_wall_time:
        if x_values.size < 2:
            raise ValueError("at least two wall-time samples are required")
        x_values = x_values - x_values[0] + x_values[1] - x_values[0]

    if std_window > 0:
        y_values = windowed_std(y_values, std_window)
        x_values = x_values[: y_values.size]

    if y_key.startswith("res"):
        if residual_max is None:
            raise ValueError(f"residual_max is required for {y_key}")
        denominator = (
            residual_max[y_key]
            if isinstance(residual_max, Mapping)
            else residual_max
        )
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("residual_max must be finite and positive")
        y_values /= denominator
        if residual_smooth_window > 1:
            y_values = uniform_filter1d(
                y_values, size=residual_smooth_window, mode="reflect"
            )

    return x_values, y_values


def first_threshold_reach(
    data: LogData,
    threshold: float,
    *,
    residual_key: str = "res0",
    residual_max: float | Mapping[str, float],
    iteration_key: str = "iterAll",
    wall_time_key: str = "tWall",
    drop_last: bool = True,
    offset_wall_time: bool = True,
) -> ReachResult:
    """Return the first unsmoothed, uninterpolated normalized crossing."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    denominator = (
        residual_max[residual_key]
        if isinstance(residual_max, Mapping)
        else residual_max
    )
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("residual_max must be finite and positive")

    residual = _trim(np.asarray(data[residual_key], dtype=float), drop_last)
    iterations = _trim(np.asarray(data[iteration_key], dtype=float), drop_last)
    wall_time = _trim(np.asarray(
        data[wall_time_key], dtype=float), drop_last).copy()
    if not (residual.size == iterations.size == wall_time.size):
        raise ValueError("threshold columns have different lengths")
    if offset_wall_time:
        if wall_time.size < 2:
            raise ValueError("at least two wall-time samples are required")
        wall_time = wall_time - wall_time[0] + wall_time[1] - wall_time[0]

    reached = residual / denominator <= threshold
    if not np.any(reached):
        return ReachResult(float("inf"), float("inf"))
    return ReachResult(float(np.min(iterations[reached])), float(np.min(wall_time[reached])))
