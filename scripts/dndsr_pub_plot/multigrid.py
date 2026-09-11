"""Naming helpers for DNDSR p-multigrid benchmark histories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MultigridRunName:
    """Parsed base solver and coarse-level smoother counts."""

    raw: str
    base_solver: str
    levels: tuple[tuple[int, str], ...]

    def padded_counts(self, length: int = 2) -> tuple[int, ...]:
        counts = tuple(count for count, _ in self.levels)
        return (counts + (0,) * length)[:length]


def parse_multigrid_run_name(name: str | Path) -> MultigridRunName:
    """Parse names such as ``x1-gmres5x1ilu-2ilu-4ilu_.log``."""

    raw = Path(name).name
    stem = raw
    for suffix in ("_.log", "-stdout.txt", ".log"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if re.match(r"x\d+-", stem):
        stem = stem[3:]
    parts = stem.split("-")
    if not parts or not parts[0]:
        raise ValueError(f"invalid multigrid run name: {raw!r}")
    levels: list[tuple[int, str]] = []
    for part in parts[1:]:
        match = re.fullmatch(r"(\d+)(.+)", part)
        if match is None:
            raise ValueError(f"invalid multigrid level {part!r} in {raw!r}")
        levels.append((int(match.group(1)), match.group(2)))
    return MultigridRunName(raw=stem, base_solver=parts[0], levels=tuple(levels))


def old_plot_selector(run: MultigridRunName) -> bool:
    """Reproduce the old notebook's sparse selection for readable figures."""

    counts = tuple(count for count, _ in run.levels)
    if len(counts) == 1 and counts[0] == 8:
        return True
    return all(count < 8 for count in counts)
