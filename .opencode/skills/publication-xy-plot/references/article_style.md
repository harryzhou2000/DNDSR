# Article-Style XY Figures

## Physical Figure Scale

Use physical sizes because a large bitmap can still produce unreadably small
text when fitted to a page.

- A4 page: 8.2677 by 11.6929 inches in portrait, or 11.6929 by 8.2677 inches
  in landscape.
- Typical landscape figure: 6 by 4 inches.
- Typical portrait figure: 4 by 6 inches.
- Readable 2 by 2 composite: about 6 by 5.5 inches.
- Tick labels and legends: normally 8--9.5 pt; never below 7.5 pt.
- Axis labels: normally 9--10.5 pt.
- Subplot titles: normally 10--11.5 pt.
- Figure-level title, when necessary: normally 11.5--13 pt.
- Line width: normally 1.0--1.6 pt.
- Marker diameter: normally 4--7 pt with at least 0.6 pt edge width.
- A scientific panel should normally remain at least 2.4 inches wide and
  1.7 inches high after margins and legends are included.

If these sizes do not fit, split the figure. Do not reduce all fonts to rescue
a dense grid. Treat page, figure, and panel geometry separately. A4 is the
containing page, not the default figure canvas; modest page-layout resizing is
acceptable.

## DNDSR Module

From the DNDSR root:

```python
from scripts.dndsr_pub_plot import (
    ARTICLE_LANDSCAPE,
    PublicationStyle,
    apply_publication_style,
    save_figure,
    series_encoding,
)

style = PublicationStyle(
    styles=("science", "no-latex"),
    figure_size=ARTICLE_LANDSCAPE,
    dpi=300,
    output_format="png",
    font_size=10,
    target_markers=16,
)
apply_publication_style(style)

for series_index, (label, x_values, y_values) in enumerate(series):
    axis.plot(
        x_values,
        y_values,
        label=label,
        **series_encoding(series_index, len(x_values), style=style),
    )

save_figure(figure, output, output_format="png", style=style)
```

`series_encoding` uses the DNDSR colorblind-safe palette plus distinct markers
and line styles. Its automatic stride targets 16 markers and bounds the visible
count to 4--40 whenever the line has at least four samples.

## Labels and Layout

- Use serif text and STIX mathematical symbols consistently.
- Prefer `$\rho$`, `$T$`, `$p$`, `$Y_{\mathrm{H_2}}$`, `$\Delta t$`, and
  `$1-\chi$` over plain-text names.
- Put units in brackets once, such as `$x$ [mm]` or `$p$ [bar]`.
- Use typeset powers of ten rather than strings such as `2e-05`.
- Let subplot titles state only the changing case, method, or time step.
- Put shared mechanism, mesh, source snapshot, and selector details in the
  caption or report prose.
- Prefer one shared legend outside the axes. If a legend is long, allocate a
  dedicated row or column rather than shrinking the data panels.
- Use a light major grid only when it helps quantitative reading. Avoid dense
  major-plus-minor grids and avoid grids for profile panels where curves and
  ticks suffice.

## Output Audit

1. Place the PDF at its intended size on an A4 page and check all symbols.
2. Decode the PNG and confirm its dimensions and color mode.
3. Inspect line overlap in grayscale or by shape: color alone must not carry
   the series identity.
4. Count visible markers on representative long and short lines.
5. Check that legends, annotations, and uncertainty bands do not obscure data.
