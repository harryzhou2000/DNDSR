# DNDSR publication plots

`dndsr_pub_plot` turns the plotting conventions from
`PrintLogErrMGTest-2-1.ipynb` into a reusable package. Its defaults retain the
notebook's SciencePlots style, 12-point font, 6 by 4.5 inch figure, 120 dpi,
PDF output, line/marker cycle, legend, grid, residual smoothing, wall-time
offset, and final-sample exclusion.

From the repository root:

```bash
python -m scripts.dndsr_pub_plot \
  --run 'Static=path/to/static_.log' \
  --run 'Residual CFL=path/to/residual_.log' \
  --x iterAll --y res0 --residual-max 26.193664859 \
  --xlim 0 10000 --output comparison_residual.pdf
```

For CFL histories, select `--y CFLNow --ylabel CFL --linear-y`. Every notebook
default has a corresponding API field on `PublicationStyle` or CLI option.
The Python API also exposes raw cross-run residual maxima and exact threshold
crossings; neither operation smooths or interpolates data.
