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


def normalize_residual(
    values: ArrayLike,
    residual_key: str,
    residual_maxima: float | Mapping[str, float],
) -> np.ndarray:
    """Normalize a residual with an externally computed ensemble maximum.

    The denominator must come from the complete comparison ensemble. This
    helper intentionally does not offer per-series maximum normalization.
    """

    denominator = (
        residual_maxima[residual_key]
        if isinstance(residual_maxima, Mapping)
        else residual_maxima
    )
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("residual maximum must be finite and positive")
    return np.asarray(values, dtype=float) / denominator


def _trim(values: np.ndarray, drop_last: bool) -> np.ndarray:
    return values[:-1] if drop_last else values


def historical_wall_time(values: ArrayLike) -> np.ndarray:
    """Apply the historical MGTest0012 wall-time origin convention.

    The first retained point is placed at one measured first-step duration:
    ``t - t[0] + (t[1] - t[0])``.
    """

    return startup_corrected_wall_time(values)


def startup_corrected_wall_time(
    values: ArrayLike,
    *,
    retain_first_step: bool = True,
) -> np.ndarray:
    """Subtract per-run startup time from cumulative wall-clock samples.

    ``retain_first_step=True`` reproduces the old MGTest0012 plots: subtract
    the first timestamp, then add the first measured step duration so the
    first point is not placed at zero. Set it to ``False`` for a strict
    zero-origin series.
    """

    wall_time = np.asarray(values, dtype=float).copy()
    if wall_time.size == 0:
        raise ValueError("at least one wall-time sample is required")
    corrected = wall_time - wall_time[0]
    if retain_first_step:
        if wall_time.size < 2:
            raise ValueError("at least two wall-time samples are required")
        corrected += wall_time[1] - wall_time[0]
    return corrected


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
    truncate_residual_at: float | None = None,
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
        x_values = startup_corrected_wall_time(x_values)

    if truncate_residual_at is not None:
        if not y_key.startswith("res"):
            raise ValueError("truncate_residual_at requires a res* y column")
        if truncate_residual_at < 0:
            raise ValueError("truncate_residual_at must be non-negative")
        if residual_max is None:
            raise ValueError(f"residual_max is required for {y_key}")
        normalized_raw = normalize_residual(y_values, y_key, residual_max)
        crossings = np.flatnonzero(normalized_raw <= truncate_residual_at)
        if crossings.size:
            stop = int(crossings[0]) + 1
            x_values = x_values[:stop]
            y_values = y_values[:stop]

    if std_window > 0:
        y_values = windowed_std(y_values, std_window)
        x_values = x_values[: y_values.size]

    if y_key.startswith("res"):
        if residual_max is None:
            raise ValueError(f"residual_max is required for {y_key}")
        y_values = normalize_residual(y_values, y_key, residual_max)
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
    residual = _trim(np.asarray(data[residual_key], dtype=float), drop_last)
    iterations = _trim(np.asarray(data[iteration_key], dtype=float), drop_last)
    wall_time = _trim(np.asarray(
        data[wall_time_key], dtype=float), drop_last).copy()
    if not (residual.size == iterations.size == wall_time.size):
        raise ValueError("threshold columns have different lengths")
    if offset_wall_time:
        wall_time = startup_corrected_wall_time(wall_time)

    reached = normalize_residual(
        residual, residual_key, residual_max) <= threshold
    if not np.any(reached):
        return ReachResult(float("inf"), float("inf"))
    first_reached = int(np.flatnonzero(reached)[0])
    return ReachResult(
        float(iterations[first_reached]),
        float(wall_time[first_reached]),
    )
