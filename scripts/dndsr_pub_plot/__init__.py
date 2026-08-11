"""Publication plotting helpers for DNDSR convergence histories."""

from .series import (
    ReachResult,
    compute_residual_maxima,
    first_threshold_reach,
    load_dndsr_log,
    prepare_series,
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
    "PublicationStyle",
    "ReachResult",
    "apply_publication_style",
    "compute_residual_maxima",
    "create_figure",
    "finalize_axes",
    "first_threshold_reach",
    "load_dndsr_log",
    "map_solver_name",
    "plot_one",
    "prepare_series",
    "save_figure",
    "windowed_std",
]
