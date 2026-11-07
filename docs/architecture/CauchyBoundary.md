# Cauchy boundary conventions and Euler correspondence

Boundary physics is resolved by `Cauchy/Boundary/BoundarySystem`; FV preparation,
RHS and implicit operators consume the same resolved condition. EOS evaluation
stays outside Riemann solvers. Boundary quadrature arrays contain explicit
targets, not callbacks, and interface packets remain externally provided values.

## Three different states

Do not interchange the exterior, the conservative gradient trace and a physical
wall value:

| Consumer | Input/convention |
|---|---|
| VR boundary equation | Euler exterior state, evaluated from the current reconstructed point |
| Convective numerical flux | Interior and Euler exterior states |
| Direct conservative gradients | `(interior + exterior)/2`, equivalent to Euler's half exterior jump over a mirrored centroid distance |
| Polynomial viscous flux | Average conservative state, average conservative gradient plus the exterior jump divided by Euler's prepared GRP distance |
| `noRsOnWall=true` | Replace both convective states by the physical no-slip wall state after forming the viscous jump gradient; use that wall state for flux thermodynamics |
| Mean auxiliary spectral preparation | Duplicate the owned mean at a physical boundary, as Euler `EvaluateDt` does; interface input is a separate extension |
| Exact `ExteriorJacobian` | Derivative of the physical exterior map with branch frozen |
| Approximate Roe/LU-SGS BC derivative | Euler `linMode=1`: homogeneous wall reflection before thermal/scalar adjustments, zero for open/prescribed boundaries |

`FaceValue` describes a physical wall value. It is useful for diagnostics and
the `noRsOnWall` option, but is not the default VR target or conservative-gradient
trace. In particular, removing wall-normal momentum while retaining pressure
changes kinetic/total energy; averaging the reflected conservative pair retains
the pair's energy instead. This distinction matters on the coarse pMG levels.

## Wall states

For interior `rho,u,p`, prescribed wall velocity `uw`, and outward unit normal n:

- Slip exterior: `uR = u - 2*((u-uw).n)*n`; retain rho, pressure, composition and
  internal energy. No viscous flux is evaluated on a slip boundary.
- Adiabatic no-slip exterior: `uR = 2*uw-u`; retain rho, pressure and composition.
  Heat conduction through the wall is disabled; species diffusion is impermeable.
- Isothermal no-slip exterior: prescribe `TR=Tw`, `uR=2*uw-u`, `pR=p` and unchanged
  mass fractions. The EOS determines rhoR and total energy. For a perfect/ideal
  mixture gas, `rhoR=p/(R(Y)*Tw)`. This is applied to VR and convection as well
  as the viscous exterior, matching Euler's `generateBV_ViscousWall`.

There is no extrapolated negative ghost temperature and no unit fallback.
For an isothermal wall, the arithmetic mean of the two conservative states
need not itself have temperature Tw; this is Euler's default flux linearization.
The `noRsOnWall` option instead uses the physical wall state for thermodynamics.
`noGRPOnWall` suppresses the polynomial jump penalty while retaining the gradient.
Neither option changes the prescribed wall temperature or silently changes dt.

RANS follows the active Euler policy: SA's `USE_FIX_ZERO_SA_NUT_AT_WALL` zero
target is honored in reconstruction and RHS. Two-equation wall values use
the cell mean and nodal-distance-derived mean distance. Molecular viscosity is
evaluated at the wall temperature for isothermal walls and the cell mean for
adiabatic walls. Isothermal primitive/conservative conversion rescales SA/k
with density; the prescribed omega/epsilon expression reflects the original
interior conservative second scalar, matching the legacy ordering.

Approved exception: Euler's source-gradient GG loop reuses a mutable state copy
across faces; its zero-SA wall fix can therefore change later face increments.
Cauchy retains immutable cell means and applies the zero-SA target independently
per face. It does not reproduce that cross-face mutation. This preserves the
explicit operator contract and avoids face-order-dependent source gradients.

Rigid-frame boundary inputs use the documented Cauchy `wallVelocityFrame`
choice and absolute flow velocity represented in mesh axes. These explicit
kinematics retain the moving-frame extension; they do not copy Euler's known
limitations for arbitrary moving/deforming walls. Mesh geometry remains in its
mesh-bound reference frame.

## Far field and open boundaries

Let `s=(u-vgrid).n`, with interior sound speed a. Euler's strict branch boundaries
are preserved, including the sonic equalities:

| Branch | Exterior |
|---|---|
| `s > a` | Entire interior state |
| `0 < s <= a` | Interior rho/velocity/composition, prescribed pressure |
| `-a < s <= 0` | Prescribed rho/velocity/composition, interior pressure |
| `s <= -a` | Entire prescribed state |

Species and RANS follow the same incoming/outgoing selection. Energy is rebuilt
through the equation EOS; no perfect-gas equivalent gamma enters the Riemann API.
Farfield, extrapolation and analytic physical boundaries have zero viscous flux,
matching Euler's active wall/open-boundary dispatch. Extrapolation returns the
interior exterior state; its approximate implicit BC derivative is nevertheless
zero under Euler's `linMode=1` convention.

## Scope and validation

The shared conventions apply to the connected flow5 and logical RANS/species
profiles. `viscousGradient="polynomial"` is the Euler conservative-gradient
baseline; `"primitive"` retains the existing alternative interior discretization
while sharing wall states and boundary flux suppression. This is not a promise
that the two gradient algorithms produce identical solutions.

Unconnected Euler BC families (anchor/profile pressure outlets, total-condition
inlets and additional built-in special profiles) remain explicit unsupported
configurations, not aliases of a loosely similar boundary. Inter-domain traces
are Cauchy extensions and stay prescribed point values rather than reflected
ghosts. No BC routine performs MPI or silently refreshes a halo.

## Explicit interface packet

`FiniteVolume::BuildBoundaryInputs()` allocates two small boundary tables in
receiver mesh axes. `flow` follows `BoundaryPoints()` and contains incoming
flow5 values at RHS/VR quadrature points. `faceFlow` follows `BoundaryFaces()`
and contains independently sampled flow5 values at maintained face centers;
these descriptors have `quadrature=-1`. Approximate Roe/BC Jacobians read
`faceFlow`, without selecting a quadrature row or averaging quadrature states.
Physical BCs compute their states directly and need no incoming rows.

The coupling controller owns both sampling and transfer. It batches face-center
and quadrature queries into the same distributed match/sample operation, then
routes each result to its explicit packet table and rotates momentum into
receiver axes. It does not change the mesh reference geometry. A domain does
not communicate either table independently.

Call `BeginBoundaryUpdate`, fill both interface sets, then publish once with
their shared physical time and donor epoch. Missing required rows fail at
publication. Mutating either table invalidates the publication, including its
aliases on compatible pMG levels. Layout/geometry matching and both allocation
versions are checked before reuse. The caller must release write views before
publishing or launching readers; the packet does not perform hidden refreshes.

Focused tests cover prescribed isothermal pressure/temperature/velocity, exact
physical versus approximate implicit derivatives, sonic farfield branches,
flow5/logical residual agreement, mixture wall conduction/species isolation,
RANS wall targets and distributed pMG. Physical endpoint/timing claims require
their separate source/config/binary receipts in the plan repository.
