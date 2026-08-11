"""Command-line entry point for standardized DNDSR history plots."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .series import load_dndsr_log
from .style import (
    DEFAULT_STYLE,
    apply_publication_style,
    create_figure,
    finalize_axes,
    plot_one,
    save_figure,
)


def _run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "run must have a non-empty label and path")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot DNDSR CSV histories with the MGTest0012 publication style"
    )
    parser.add_argument("--run", action="append",
                        required=True, type=_run_spec)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--x", default="iterAll", dest="x_key")
    parser.add_argument("--y", default="res0", dest="y_key")
    parser.add_argument("--residual-max", type=float)
    parser.add_argument("--xlabel", default="Iteration")
    parser.add_argument("--ylabel", default=r"$L^1$ density residual")
    parser.add_argument("--title")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--figsize", nargs=2, type=float,
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=DEFAULT_STYLE.dpi)
    parser.add_argument("--format", default=DEFAULT_STYLE.output_format)
    parser.add_argument("--marker-every", type=int,
                        default=DEFAULT_STYLE.marker_every)
    parser.add_argument(
        "--residual-smooth-window",
        type=int,
        default=DEFAULT_STYLE.residual_smooth_window,
    )
    parser.add_argument("--std-window", type=int, default=0)
    parser.add_argument("--linear-y", action="store_true")
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--raw-wall-time", action="store_true")
    parser.add_argument("--no-science-style", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.y_key.startswith("res") and args.residual_max is None:
        raise SystemExit(
            "--residual-max is required when plotting a res* column")

    style = replace(
        DEFAULT_STYLE,
        figure_size=tuple(
            args.figsize) if args.figsize else DEFAULT_STYLE.figure_size,
        dpi=args.dpi,
        output_format=args.format,
        marker_every=args.marker_every,
        residual_smooth_window=args.residual_smooth_window,
    )
    apply_publication_style(style, use_scienceplots=not args.no_science_style)
    figure, axes = create_figure(style=style)
    for plot_index, (label, path) in enumerate(args.run):
        plot_one(
            axes,
            load_dndsr_log(path),
            label,
            plot_index=plot_index,
            x_key=args.x_key,
            y_key=args.y_key,
            residual_max=args.residual_max,
            std_window=args.std_window,
            drop_last=not args.keep_last,
            offset_wall_time=not args.raw_wall_time,
            style=style,
        )
    finalize_axes(
        axes,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        log_y=not args.linear_y,
        style=style,
    )
    save_figure(figure, args.output, output_format=args.format, style=style)
    return 0
