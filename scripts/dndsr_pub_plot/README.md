# DNDSR publication plots

`dndsr_pub_plot` turns the plotting conventions from
`PrintLogErrMGTest-2-1.ipynb` into a reusable package. Its current publication
defaults use a 6 by 4 inch article figure sized for placement on an A4 page,
serif text and STIX mathematics, 300 dpi, colorblind-safe colors, redundant
line/marker encodings, and an automatic
marker stride that displays 4--40 markers per sufficiently sampled line. The
default markers are hollow, so they do not obscure the plotted curves.
historical residual smoothing, per-run startup correction, and final-sample
exclusion remain available.

From the repository root:

```bash
python -m scripts.dndsr_pub_plot \
  --run 'Static=path/to/static_.log' \
  --run 'Residual CFL=path/to/residual_.log' \
  --x iterAll --y res0 \
  --xlim 0 10000 --output comparison_residual.pdf
```

For every `res*` plot, the CLI computes one maximum over all supplied runs and
uses it for every curve. `--residual-max` can pin an externally recorded global
denominator, but normalization is never performed independently per run.
Use `--truncate-residual-at 1e-5` to stop each displayed curve at the first
raw, unsmoothed normalized crossing; the crossing sample remains visible.

For CFL histories, select `--y CFLNow --ylabel CFL --linear-y`. Every notebook
default has a corresponding API field on `PublicationStyle` or CLI option.
The Python API also exposes raw cross-run residual maxima and exact threshold
crossings; neither operation smooths or interpolates data.
