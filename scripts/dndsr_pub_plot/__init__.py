"""Publication plotting helpers for DNDSR convergence histories."""

from .multigrid import (
    MultigridRunName,
    old_plot_selector,
    parse_multigrid_run_name,
)

from .series import (
    ReachResult,
    compute_residual_maxima,
    first_threshold_reach,
    load_dndsr_log,
    historical_wall_time,
    normalize_residual,
    prepare_series,
    startup_corrected_wall_time,
    windowed_std,
)
from .style import (
    DEFAULT_MARKERS,
    DEFAULT_STYLE,
    PublicationStyle,
    apply_publication_style,
    create_figure,
    finalize_axes,
    map_solver_name,
    plot_one,
    save_figure,
)

__all__ = [
    "DEFAULT_MARKERS",
    "DEFAULT_STYLE",
    "MultigridRunName",
    "PublicationStyle",
    "ReachResult",
    "apply_publication_style",
    "compute_residual_maxima",
    "create_figure",
    "finalize_axes",
    "first_threshold_reach",
    "load_dndsr_log",
    "map_solver_name",
    "historical_wall_time",
    "normalize_residual",
    "old_plot_selector",
    "plot_one",
    "parse_multigrid_run_name",
    "prepare_series",
    "startup_corrected_wall_time",
    "save_figure",
    "windowed_std",
]
