---
name: publication-xy-plot
description: Create or revise Matplotlib x-y figures, line charts, convergence histories, and one-dimensional profiles for colorblind-safe, print-readable publication output. Use when plotting series or auditing plot readability; do not use for spatial VTK/PyVista fields.
---

# Publication XY Plot

Use the canonical DNDSR helpers in `scripts/dndsr_pub_plot`, especially
`PublicationStyle`, `apply_publication_style`, `series_encoding`,
`marker_every_for_count`, and `save_figure`.

## Required Decisions

1. Distinguish the A4 page, the inserted figure, and each scientific panel.
   A4 portrait is 8.2677 by 11.6929 inches; landscape reverses those values.
   Design at the figure's printed insertion size. Use 6 by 4 inches for a
   typical landscape plot, 4 by 6 inches for portrait, and about 6 by 5.5
   inches for a readable 2 by 2 composite. Minor resizing on an A4 page is
   acceptable.
2. Split a dense grid when its scientific panels would become smaller than
   about 2.4 by 1.7 inches; prefer at most a 2 by 2 grid.
3. Encode every compared series redundantly by color, marker, and line style.
4. Use 4--40 visible markers per sufficiently sampled line, normally 12--20.
5. Keep titles concise: put only the control that differs between panels in a
   subplot title and move shared context to the caption or report prose.
6. Use mathematical symbols and units in axis labels. Avoid code identifiers,
   raw scientific-notation strings, and prose-heavy legends.
7. Put shared legends outside the data axes when they would cover curves.
8. Save a vector PDF and a 300 dpi PNG from the same source.

Read [references/article_style.md](references/article_style.md) for physical
font and marker sizes, colorblind-safe encodings, grid and legend choices, and
a minimal DNDSR API example.

## Validation

Inspect the PNG at original resolution and the PDF at its intended 4 by 6 or
6 by 4 inch placement on an A4 page. Check that series remain distinguishable
without color, no text overlaps data, and symbols remain legible at normal page
zoom.
