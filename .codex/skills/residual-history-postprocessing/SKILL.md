---
name: residual-history-postprocessing
description: Post-process and compare CFD convergence or residual-history logs, especially DNDSR CSV `*_.log` multigrid sweeps. Use when Codex must download run artifacts, preserve one global residual normalization across an ensemble, remove per-run startup cost from wall time, reproduce historical MGTest0012-style plots, compute convergence-threshold times or iterations, or export CSV and LaTeX benchmark tables.
---

# Residual History Postprocessing

Build reproducible script-based analyses from immutable run artifacts. Prefer
the repository's `scripts/dndsr_pub_plot` package when it is available.

## Workflow

1. Define the comparison ensemble before computing statistics. Include every
   condition whose normalized residuals will be compared, even if plots are
   later split by Mach number, solver family, or multigrid depth.
2. Copy only required artifacts into a dedicated workspace. Keep raw histories
   separate from generated plots and tables. Record source host/path and the
   selected filenames.
3. Load all histories; validate required columns, aligned lengths, finite
   values, monotonic iteration and wall time, and sufficient samples. Then
   compute one maximum per residual column across the complete ensemble. Never
   compute a separate denominator per run or plot subset.
4. Match the chosen legacy normalization convention explicitly. The old
   MGTest0012 notebook scans full parsed histories for global maxima, then
   excludes the final sample from displayed curves and threshold tables.
5. Correct wall time independently for each run. For exact old-plot behavior,
   use `t - t[0] + (t[1] - t[0])`: this subtracts startup while retaining one
   estimated first-step duration. Record the formula in the analysis manifest.
6. Use raw, unsmoothed normalized residuals for threshold detection and tables.
   Report the first logged point at or below the threshold; do not interpolate.
7. Apply smoothing only to display curves. Reproduce visual details from the
   reference plot or notebook rather than inventing a new style. When plots
   must stop at convergence, locate the first crossing in the raw normalized
   series, include that sample, truncate, and only then apply display smoothing.
8. Export a machine-readable summary, publication tables, figures, and a
   manifest containing run membership, denominators, timing convention, and
   threshold.

## DNDSR API

Use these helpers from `scripts.dndsr_pub_plot`:

- `load_dndsr_log`: parse numeric CSV history columns.
- `compute_residual_maxima`: compute ensemble-wide `res*` denominators.
- `normalize_residual`: normalize only with an externally supplied ensemble
  maximum.
- `startup_corrected_wall_time`: remove per-run startup time.
- `first_threshold_reach`: find an exact logged threshold crossing.
- `parse_multigrid_run_name`: recover base solver and smoother counts.
- `plot_one` and related style helpers: reproduce MGTest0012 figures.

When working outside DNDSR, implement the same invariants locally and include
focused validation checks that distinguish global from per-run normalization.
Within DNDSR, do not add pytest tests solely for `scripts/` or `workspace/`;
use project-venv CLI smoke checks or standalone validation commands instead.

## Visual contract

For `PrintLogErrMGTest-2-1.ipynb` compatibility, read
[`references/mgtest0012-visual-contract.md`](references/mgtest0012-visual-contract.md).
Compare at least one generated plot with an old reference PDF or SVG when one
is available. Verify canvas size, typography, line and marker treatment,
legend, axes, grid, smoothing, and curve selection.

## Validation

Before reporting completion:

- Check two histories having different maxima and prove both use the larger
  ensemble denominator.
- Check startup correction with known timestamps.
- Confirm threshold tables use unsmoothed data.
- Confirm manifests contain every analyzed run and the exact denominator.
- Record whether the global maximum scan includes or excludes terminal samples.
- Reject missing, truncated, non-finite, or non-monotonic histories.
- Run the analysis with the project virtual environment.
- Inspect a rendered figure, not only its file existence.
