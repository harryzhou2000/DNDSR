"""Publication style and plotting primitives for DNDSR figures."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import matplotlib.pyplot as plt

from .series import LogData, prepare_series

DEFAULT_MARKERS = (
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
    "v",
    "<",
    ">",
    "p",
    "h",
)

A4_PORTRAIT = (8.2677, 11.6929)
A4_LANDSCAPE = (11.6929, 8.2677)
ARTICLE_PORTRAIT = (4.0, 6.0)
ARTICLE_LANDSCAPE = (6.0, 4.0)
COLORBLIND_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
LINE_STYLES = ("-", "--", "-.", ":")


@dataclass(frozen=True)
class PublicationStyle:
    """Controllable form of the historical notebook's plotting globals."""

    styles: tuple[str, ...] = ("science",)
    font_size: float = 10
    figure_size: tuple[float, float] = ARTICLE_LANDSCAPE
    dpi: int = 300
    output_format: str = "pdf"
    line_width: float = 1.2
    marker_size: float = 4.5
    marker_every: int | None = None
    target_markers: int = 16
    minimum_markers: int = 4
    maximum_markers: int = 40
    markers: tuple[str, ...] = DEFAULT_MARKERS
    colors: tuple[str, ...] = COLORBLIND_COLORS
    line_styles: tuple[str, ...] = LINE_STYLES
    residual_smooth_window: int = 20
    legend_font_size: float = 8.5
    legend_kwargs: Mapping[str, Any] = field(
        default_factory=lambda: {
            "frameon": True,
            "edgecolor": "none",
            "framealpha": 0.5,
        }
    )
    grid_alpha: float = 0.22


DEFAULT_STYLE = PublicationStyle()


def apply_publication_style(
    style: PublicationStyle = DEFAULT_STYLE,
    *,
    use_scienceplots: bool = True,
) -> None:
    """Apply a serif, math-aware publication style with point-sized text."""

    if use_scienceplots:
        try:
            import scienceplots  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SciencePlots is required for the DNDSR publication style; "
                "install requirements.txt or pass use_scienceplots=False"
            ) from exc
        plt.style.use(list(style.styles))
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "font.size": style.font_size,
            "mathtext.fontset": "stix",
            "axes.titlesize": style.font_size + 1,
            "axes.labelsize": style.font_size,
            "xtick.labelsize": style.font_size - 1.5,
            "ytick.labelsize": style.font_size - 1.5,
            "legend.fontsize": style.legend_font_size,
            "lines.linewidth": style.line_width,
            "savefig.dpi": style.dpi,
        }
    )


def marker_every_for_count(
    sample_count: int,
    *,
    target_markers: int = 16,
    minimum_markers: int = 4,
    maximum_markers: int = 40,
) -> int:
    """Choose a stride giving 4--40 markers when a line has enough samples."""

    if sample_count <= 0:
        return 1
    visible = min(max(target_markers, minimum_markers),
                  maximum_markers, sample_count)
    return max(1, math.ceil(sample_count / visible))


def series_encoding(
    plot_index: int,
    sample_count: int,
    *,
    style: PublicationStyle = DEFAULT_STYLE,
) -> dict[str, Any]:
    """Return redundant color, marker, and line-style encoding for one series."""

    marker_every = style.marker_every
    if marker_every is None:
        marker_every = marker_every_for_count(
            sample_count,
            target_markers=style.target_markers,
            minimum_markers=style.minimum_markers,
            maximum_markers=style.maximum_markers,
        )
    return {
        "color": style.colors[plot_index % len(style.colors)],
        "linestyle": style.line_styles[plot_index % len(style.line_styles)],
        "marker": style.markers[plot_index % len(style.markers)],
        "markevery": marker_every,
        "markersize": style.marker_size,
        "markeredgewidth": max(0.6, 0.7 * style.line_width),
        "markerfacecolor": "white",
    }


def create_figure(
    *,
    style: PublicationStyle = DEFAULT_STYLE,
    figure_id: int | None = None,
    clear: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a single-panel figure using the selected physical page size."""

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
    truncate_residual_at: float | None = None,
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
        truncate_residual_at=truncate_residual_at,
    )
    defaults = {
        "label": label,
        "lw": style.line_width,
        **series_encoding(plot_index, len(x_values), style=style),
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
    options = {"bbox_inches": "tight", "facecolor": "white"}
    options.update(savefig_kwargs)
    figure.savefig(output_path, format=selected_format, **options)
    return output_path
