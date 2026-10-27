# Static reconstruction mask

`implicitReconstructionControl.reconstructionMask` fixes selected cells to piecewise constant
reconstruction. A binary value of **1 means zero reconstruction** and 0 means
ordinary reconstruction. All polynomial modes and all solution variables
are suppressed in selected cells; conservative cell averages are unchanged.
This is independent of LVR and can be combined with conventional limiters.

```json
{
  "implicitReconstructionControl": {
    "reconstructionMask": {
      "expression": "x[0] > 24.5",
      "boundaryNames": ["Outlet"],
      "boundaryIDs": []
    }
  }
}
```

The selectors form a **union**:

- `expression` is an ExprTk scalar expression. A finite nonzero result selects
  a cell. `x[0]`, `x[1]`, `x[2]` are its physical volume barycentre coordinates
  **after configured mesh transformations**, in solver mesh units. In 2D,
  `x[2]` is zero. No flow variables or time are inputs. An empty string
  disables this selector. Parse errors and nonfinite values are errors.
- `boundaryNames` lists exact boundary **zone names**, as registered by
  `bcSettings` or the default mesh BC-name mapping. Any directly incident cell
  is selected. Unknown names and periodic names are errors.
- `boundaryIDs` lists internal positive external-BC **face-zone IDs**, as
  assigned by the boundary handler, not `BCIn`/`BCFar` type enum values and
  not arbitrary original mesh file IDs. It provides the same direct incidence
  selection as names. A valid ID absent from the mesh simply selects no cells.

There is no automatic neighbour expansion. Use a coordinate band to cover
more than the immediately adjacent boundary cells. The example's `Outlet`
must match the actual case name, and 24.5 must be replaced with a coordinate
appropriate for that case. Expressions are not evaluated in units of body
diameter unless the mesh is already nondimensionalized that way.

The solver owns the binary double array, cached selected indices, selector
settings, and setup logic. Reconstruction and evaluator routines receive
a read-only `ReconstructionMaskView` as an explicit function argument.
Neither object retains a mask or an internal reference between calls.
Omitting the optional argument leaves that call unconstrained. There is
no mask setter or binding/detachment lifecycle on VR.

The mask is built once during setup and exchanged to ghosts once. It is
recomputed from the current mesh and config on a restart, so it need not be
stored in the checkpoint. Empty selectors leave mask storage unallocated.
No mask halo exchange or ExprTk evaluation occurs during time/reconstruction
iterations. Selected indices are cached for sparse projection. Initial
restart coefficients, each VR Jacobi/SOR update (including LVR), GG gradients
and conversions, and final conventional-limiter results obey the mask.
Krylov reconstruction uses the constraint row $u_i^{\mathrm{rec}}=0$ and
its identity derivative rather than a singular zero row.

The constraint applies to the polynomial and to the solver's direct GG
reconstruction path; it does not disable viscous face-normal mean-jump
corrections or modify fluxes, boundary data, means, or the mesh. A static
first-order band is a stabilization experiment, not evidence that an outlet
boundary condition is physically appropriate or that a run is stable.

Add `"reconstructionMask"` to `dataIOControl.outCellScalarNames` and/or
`outBndScalarNames` to inspect the binary field in ordinary cell/boundary
VTKHDF output. It reports zero when the feature is disabled.
