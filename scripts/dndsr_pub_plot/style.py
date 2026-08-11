"""Publication style and plotting primitives matching MGTest0012."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt

from .series import LogData, prepare_series

DEFAULT_MARKERS = (
    ".",
    "s",
    "o",
    "v",
    "^",
    "<",
    ">",
    "8",
    "p",
    "*",
    "h",
    "H",
    "D",
    "d",
    "P",
    "X",
)


@dataclass(frozen=True)
class PublicationStyle:
    """Controllable form of the historical notebook's plotting globals."""

    styles: tuple[str, ...] = ("science",)
    font_size: float = 12
    figure_size: tuple[float, float] = (6, 4.5)
    dpi: int = 120
    output_format: str = "pdf"
    line_width: float = 0.5
    marker_size: float = 5
    marker_every: int = 400
    markers: tuple[str, ...] = DEFAULT_MARKERS
    color_sequence: str = "tab10"
    residual_smooth_window: int = 20
    legend_font_size: float = 9
    legend_kwargs: Mapping[str, Any] = field(
        default_factory=lambda: {
            "frameon": True,
            "edgecolor": "none",
            "framealpha": 0.5,
        }
    )
    grid_alpha: float = 0.3


DEFAULT_STYLE = PublicationStyle()


def apply_publication_style(
    style: PublicationStyle = DEFAULT_STYLE,
    *,
    use_scienceplots: bool = True,
) -> None:
    """Register SciencePlots, apply its style, and set the notebook font size."""

    if use_scienceplots:
        try:
            import scienceplots  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SciencePlots is required for the DNDSR publication style; "
                "install requirements.txt or pass use_scienceplots=False"
            ) from exc
        plt.style.use(list(style.styles))
    matplotlib.rc("font", size=style.font_size)


def create_figure(
    *,
    style: PublicationStyle = DEFAULT_STYLE,
    figure_id: int | None = None,
    clear: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create the notebook's 6 by 4.5 inch, 120 dpi single-panel figure."""

    figure = plt.figure(
        num=figure_id,
        figsize=style.figure_size,
        dpi=style.dpi,
    )
    if clear:
        figure.clear()
    return figure, figure.add_subplot(1, 1, 1)


def map_solver_name(name: str) -> str:
    """Apply the notebook's solver-name replacements in the same order."""

    for old, new in (("ilu", "ILU"), ("gmres5x1", "GMRES+"), ("lusgs", "LUSGS")):
        name = name.replace(old, new)
    return name


def plot_one(
    axes: matplotlib.axes.Axes,
    data: LogData,
    label: str,
    *,
    plot_index: int = 0,
    x_key: str = "tWall",
    y_key: str = "res0",
    residual_max: float | Mapping[str, float] | None = None,
    std_window: int = 0,
    drop_last: bool = True,
    offset_wall_time: bool = True,
    style: PublicationStyle = DEFAULT_STYLE,
    **plot_kwargs: Any,
) -> matplotlib.lines.Line2D:
    """Plot one DNDSR history using the notebook's line and marker cycle."""

    x_values, y_values = prepare_series(
        data,
        x_key=x_key,
        y_key=y_key,
        residual_max=residual_max,
        std_window=std_window,
        residual_smooth_window=style.residual_smooth_window,
        drop_last=drop_last,
        offset_wall_time=offset_wall_time,
    )
    colors = matplotlib.color_sequences[style.color_sequence]
    defaults = {
        "label": label,
        "lw": style.line_width,
        "marker": style.markers[plot_index % len(style.markers)],
        "markevery": style.marker_every,
        "markersize": style.marker_size,
        "markeredgewidth": style.line_width,
        "markerfacecolor": "none",
        "color": colors[plot_index % len(colors)],
    }
    defaults.update(plot_kwargs)
    return axes.plot(x_values, y_values, **defaults)[0]


def finalize_axes(
    axes: matplotlib.axes.Axes,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xlabel: str = "t",
    ylabel: str = "res0",
    title: str | None = None,
    log_y: bool = True,
    legend: bool = True,
    legend_kwargs: Mapping[str, Any] | None = None,
    style: PublicationStyle = DEFAULT_STYLE,
) -> matplotlib.legend.Legend | None:
    """Apply scale, legend, limits, labels, title, and grid in notebook order."""

    if log_y:
        axes.set_yscale("log")
    legend_artist = None
    if legend:
        options = dict(style.legend_kwargs)
        if legend_kwargs is not None:
            options.update(legend_kwargs)
        legend_artist = axes.legend(fontsize=style.legend_font_size, **options)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)
    axes.grid(visible=True, which="both", alpha=style.grid_alpha)
    return legend_artist


def save_figure(
    figure: matplotlib.figure.Figure,
    output: str | Path,
    *,
    output_format: str | None = None,
    style: PublicationStyle = DEFAULT_STYLE,
    create_parent: bool = True,
    **savefig_kwargs: Any,
) -> Path:
    """Save a figure with the notebook's PDF default and explicit format."""

    selected_format = output_format or style.output_format
    output_path = Path(output)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(f".{selected_format}")
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format=selected_format, **savefig_kwargs)
    return output_path
