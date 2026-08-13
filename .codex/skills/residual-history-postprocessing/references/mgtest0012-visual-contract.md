# MGTest0012 visual contract

Match `PrintLogErrMGTest-2-1.ipynb` with these defaults:

- Matplotlib with the SciencePlots `science` style.
- Font size 12.
- Single-panel figure, 6 by 4.5 inches, 120 dpi.
- PDF output unless the user requests another format.
- Line width 0.5 points.
- Marker sequence: `.`, `s`, `o`, `v`, `^`, `<`, `>`, `8`, `p`, `*`, `h`,
  `H`, `D`, `d`, `P`, `X`.
- Mark every 400 samples; marker size 5; marker edge width 0.5; hollow marker
  face.
- Matplotlib `tab10` color sequence.
- Residual display smoothing: reflected `uniform_filter1d`, window 20.
- Logarithmic y-axis for residual plots.
- Legend font size 9, frame enabled, no edge, alpha 0.5.
- Major and minor grid enabled with alpha 0.3.
- Density residual label: `$L^1$ density residual`.
- Wall-time label: `$t_{wall}$`; iteration label: `Iteration`.
- Compute global residual maxima from full parsed histories, then drop the final
  sample from displayed curves and threshold tables.
- Correct wall time with `t - t[0] + (t[1] - t[0])`.
- Normalize each residual column by one maximum computed over the complete
  comparison ensemble.

The old notebook's readable-plot selection excludes multigrid sequences with
any smoother count of 8 or more, except the single-coarse-level count-8 case.
Keep all runs in exported tables even when figures use this sparse selection.
