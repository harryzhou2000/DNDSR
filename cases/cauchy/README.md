# Native Cauchy P3 cases

Run from the source root with `build/app/cauchy.exe cases/cauchy/<case>.json`.
Equation, EOS and transport selection live inside each config. Validate without
running with `--check-config`. These use the native DNDS/Geom/CFV solver and
SSPRK3 in the retained initial templates; they do not wrap legacy Euler.
Native implicit BDF1/2, Trapz, ESDIRK2/3/4 and DITR are available through
`time.method` and root `implicit` controls. The current implicit qualification
controls/results are kept separately in the independent plan repository at
`experiments/cauchy-implicit/`; the cancelled explicit viscous/DMR campaign is
not resumed merely by the continued presence of its template here.

| Case | Physical specification | Requested evidence |
| --- | --- | --- |
| `isentropic_vortex_20/40/80` | beta=5, periodic [0,10]^2, t=1, P3, no limiter or face fallback | Volume-average density L1/L2/Linf, mesh/time refinement, conservation |
| `cylinder_inviscid_subsonic` | Upper half, D=1, U=1, rho=1, M=.1 | Steady residual, Cp, pressure drag |
| `cylinder_supersonic_front` | Front half, D=1, U=1, rho=1, M=20, characteristic 3WBAP | Detached bow shock, stand-off, pressure |
| `cylinder_re20` | Full D=1, U=1, rho=1, mu=.05, M=.2, adiabatic no-slip | Steady Cd, wake, residual |
| `cylinder_re200` | Full D=1, U=1, rho=1, mu=.005, M=.2, small transverse wake seed | Transient through t=200, several late CL/CD cycles, St |
| `dmr_triangular_3wbap` | Mach-10, angle=60 deg, x0=1/6, UniformDM240U triangles, t=.25 | Fully active characteristic descending 3WBAP, shock/reflection morphology |

P3 means cubic reconstruction (nominal fourth-order spatial accuracy). The
vortex uses integrated cell-average initialization and analytic error, not
pointwise centroid values. `reconstructionTolerance` is the relative update
criterion; inspect `vr_relative_update` to detect an insufficient sweep cap.
Its time step also needs refinement before attributing the measured slope to
space alone.

The Mach20 and DMR cases enable `positivity` with `temperatureFloor=1e-10`.
The bound is on internal energy at Tmin, not on a substituted perfect-gas
gamma inside a Riemann solver. Reconstruction uses one mean-preserving beta
over face and volume quadrature points. SSPRK uses a shared conservative
high/low flux blend when needed; nonfinite data are errors, and an inadmissible
low-order candidate requires a smaller dt. History includes temperature minimum
and beta/alpha activity. The vortex cases keep PP disabled.

`CylinderInvisHalfA`: `wall` is the curved body, `WALL_INVIS` is the flat
symmetry plane. `CylinderB1` has a radius-.5 body and radius-200 farfield;
its smallest radial layer is about .002. The explicit diffusion and acoustic
stability constraints can make a one-hour run incomplete. A capped result is
not a steady solution or a verified vortex street. The compatibility default
`numerics.viscousGradient="primitive"` uses cell primitive gradients. Select
`"polynomial"` explicitly for cached polynomial derivatives and Euler's normal
face-jump construction; the [viscous operator contract](../../docs/architecture/CauchyImplicit.md#polynomial-viscous-operator)
describes filtered/unlimited selection and boundary behavior. P3 inviscid
reconstruction alone does not establish fourth-order viscous accuracy.

The supersonic front mesh includes exit faces at x=0 in `FAR`. The farfield
condition respects supersonic inflow/outflow and the subsonic incoming component
count, instead of clamping an outflow to the freestream.

Output is a compact accepted-state history, native persistent-ID restart,
parallel VTKHDF means and rank-local boundary quadrature CSVs. Histories/output
use separate reconstruction scratch so cadence does not affect the trajectory.
The C++ frontend uses DNDS MPI initialization/finalization and `argparse`:

```sh
# Invoke from the checkout root for these case-relative mesh paths.
mpirun -np 4 build-refactor/app/cauchy.exe cases/cauchy/isentropic_vortex_20.json \
  -k /time/method -v '"esdirk4"' -k /time/dt -v 0.01
```

`--check-config` prints the resolved configuration without mesh construction;
`--restart <file>` resumes accepted state/history. They are mutually exclusive.
Repeated `-k/--overwrite_key` and `-v/--overwrite_value` arguments pair in order,
using JSON pointers into the resolved configuration and JSON values. They apply
after profiles/defaults/domain settings, before final validation. For example,
`-k /domains/0/settings/execution/overlap -v false` overrides that domain's
file settings. Only existing resolved keys can be targeted; `--check-config`
shows their paths.
Quote string values as JSON, as above. Unknown or mistyped settings fail the
same validation as file input. MPI cleanup runs on help, config-only and error
paths as well as normal execution.

History uses `DNDS::CsvLog`: `output.historySegmentRows` defaults to 10000.
Segments are `<prefix>_history.csv`, `<prefix>_history_1.csv`, etc., each with
the same header. Readers must concatenate segments in numeric order. As with
the existing writer, rotation can create an empty trailing segment at an exact
row limit. Use a fresh output prefix for a resumed run to preserve old evidence.

Restart permits changed MPI ownership/order and polynomial degree. It omits
reconstruction by default; missing or incompatible coefficients cold-start at
zero. Set `output.checkpointReconstruction=true` for compatible finite-sweep
continuation. BDF2 includes its accepted previous state/timestep. File metadata
and physical mesh records validate interpretation before accepted publication.
Body force is the integrated wall momentum flux; Cd/Cl use `.5*rho*U^2*D`.
For the upper half body, pressure drag must be doubled for a full-body Cd.

The dedicated campaign enforces aggregate <=256 ranks, OMP=1, one hour per
ordinary row and eight hours for DMR. A graceful wall/step cap returns 2;
process timeout returns 124. Both require further work before numerical
qualification. Acceptance records live in the independent plan repository at
`experiments/cauchy-p3/`.
