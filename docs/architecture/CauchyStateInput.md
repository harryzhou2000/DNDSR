# Physical state input

Cauchy uses one setup-only resolver for `initialState`, prescribed boundary
`state`, and `referenceState`. The common syntax and unit conversion live in
`Physics/StateInput`; the Cauchy adapter binds equation component ordering and
uses its EOS evaluator. No Euler evaluator is linked into the production path.

## Representations and units

```json
{
  "equation": {
    "referenceScales": {"rho0": 1.225, "U0": 340, "L0": 2, "T0": 288.15}
  },
  "initialState": {
    "type": "primRhoT_phy",
    "state": [0.98, 68, 0, 0, 1400]
  }
}
```

This five-component example requires a species-free equation. Mixture/RANS
inputs append **all** their independent components. Vector order is flow5,
RANS, then transported species in mechanism order, excluding the final dependent
species. Geometry dimension does not remove the third velocity component.

| Type | First five entries | Appended entries |
|---|---|---|
| `cons` | rho, rhou, rhov, rhow, rhoE | conservative RANS, rhoY |
| `consSensible` | same, with sensible total energy | conservative RANS, rhoY |
| `primRhoP` | rho, u, v, w, p | specific RANS, Y |
| `primRhoT` | rho, u, v, w, T | specific RANS, Y |
| `primTP` | T, u, v, w, p | specific RANS, Y |

All five accept `_phy` for physical input. Without it, input is already in
solver units. A bare array means `consSensible`, as in Euler. Unknown tags,
wrong widths, absent values and nonfinite values fail; no state entries are
filled from unit defaults.

When applying layered configuration or CLI overrides, change `type` and `state`
together: changing only the type deliberately reinterprets the existing vector.

`referenceScales` specifies the **unit system**, not a physical far-field
state. Its object requires all four positive values. Null retains the existing
`equation.mixture` scale fields and their explicit schema defaults. Conflicting
nondefault old/new scale specifications fail. Choosing scales equal to one is
allowed; it never supplies an omitted density to a state.

Conversions use rho0, U0, rho0*U0² and T0 for density, velocity, pressure/energy
density and temperature. Conservative momentum uses rho0*U0. Primitive RANS
scales preserve Euler: SA is already nondimensional, k uses U0², omega U0/L0,
epsilon U0³/L0. Conservative RANS adds rho0. Species Y is dimensionless and
rhoY uses rho0. The configured perfect-gas gas constant remains in solver
units; its physical conversion factor is U0²/T0.

Perfect-gas molecular parameters are also supplied in code units, independently
of a tagged state's `_phy` suffix. `viscosityModel` selects `constant`,
`sutherland`, or `constantKinematic`. `viscosity` is the reference coefficient
scaled by rho0*U0*L0; Sutherland additionally takes
`viscosityReferenceTemperature=TRef_phys/T0` and
`sutherlandConstant=C_phys/T0`. The constant-kinematic option preserves Euler's
muModel=2 convention, `mu_code=viscosity*rho_code`, using the actual point
density. No model obtains density from the unit scales or reference state.
Mixtures use material transport and reject unused perfect-gas law parameters.

For mixtures, sensible energy adds the species-weighted internal-energy
reference supplied by the existing chemical backend. It is not merely a
constant heat-capacity conversion or an arbitrary formation-energy shift.
Primitive species use Euler's clipped/normalized composition. Conservative
species remain unchanged; EOS and sensible-reference queries use repaired
composition, as Euler does. Domain admissibility remains a separate check.

## Ownership and setup

`ResolvePhysicalState` consumes one explicit chemistry-worker lease and returns
owned conservative values, primitive rho/u/v/w/p, temperature and named scalar
adapters. It has no MPI, hidden global worker or runtime-array dependency.
`ResolvePhysicalStateInputs` returns a preparation snapshot containing a
normalized settings copy and separately owned conservative initial/reference
vectors. Input configuration contains no hidden derived-state cache. Rebuild
the snapshot after changing inputs. Domain performs
this work inside its collective setup-error guard before constructing FV or PP
workspaces. Original tagged inputs remain in the root configuration/provenance.

Initial conservative data is filled directly, with only the configured initial
rigid-frame momentum rotation. This avoids losing supplied total energy through
a primitive round trip. Prescribed BCs receive resolved pressure/composition
through the existing boundary producer. RANS PP receives the resolved reference
rho and conservative RANS values; primitive reference floors divide by that rho.
No unit-density, scale-density, initial-state or boundary-state fallback exists.

Null tagged fields retain the existing Cauchy primitive/named-scalar inputs.
Tagged input rejects nondefault legacy fields at the same site; schema-default
legacy fields are inactive. A complete `referenceState` conflicts with the old
partial `numerics.ransState.referenceConservative` record. Direct low-level FV
or boundary construction must normalize tagged inputs first; otherwise it fails
instead of silently ignoring them.

Tagged initial states require the constant background profile. Spatial overrides
below can then produce a nonuniform field. This implementation does not change
wall-temperature/velocity input units or introduce new BC laws.

## Spatial initial fields

`initialField` contains `boxInitializers`, `planeInitializers` and
`exprtkInitializers`. They apply in that order, and in list order within each
category; later matching entries override earlier point values. The background
is the resolved `initialState` or the existing constant initial inputs. Combining
these overrides with a nonconstant built-in `initialProfile` fails explicitly.

```json
{
  "initialField": {
    "boxInitializers": [{
      "x0": 0, "x1": 0.4, "y0": -1, "y1": 1, "z0": -1, "z1": 1,
      "v": {"type": "primRhoP", "state": [2, 0, 0, 0, 1]}
    }],
    "planeInitializers": [{
      "a": 1, "b": 0, "c": 0, "h": -0.6,
      "v": {"type": "primRhoP", "state": [1, 0, 0, 0, 2]}
    }],
    "exprtkInitializers": [{
      "stateType": "primRhoT",
      "exprs": ["inRegion := 1; UExprtk[4] := 1 + 0.1*x[0]; 0;"]
    }]
  }
}
```

This example is flow5 only. Every box/plane `v` requires a complete state,
including RANS/species when active. Bounds are strict: `x0<x<x1`, similarly for
y and z. Even 2D configurations must give a nonempty z interval containing their
geometry. The plane test is **`a*x+b*y+c*z+h>0`**, matching Euler's implemented
sign (some legacy descriptions instead say `dot(normal,x)>=h`). Zero normals
and invalid bounds are errors.

Expressions receive `x[3]` in **ground-frame code-length units**, `t=0` in code
time, and a rank-local `iCell` for diagnostics. Do not use `iCell` to define a
partition-independent field. Velocity is absolute in ground axes. `stateType`
selects only the state representation/units, never coordinates or time. Thus
`primRhoT_phy` receives physical temperature but `x` still uses code coordinates.
All ten state types are supported. `UExprtk` is fully seeded from the current
point state in the selected representation; `inRegion` resets to zero before
each call. Set it nonzero to accept the edited vector, and return zero. The
converted state must be finite with positive physical density and temperature;
missing or zero density is never repaired by a unit fallback.

The spatial field is evaluated at the highest supported cell quadrature and
**conservative values** are volume-averaged. It does not average temperature or
primitive velocity and then convert. Momentum is finally rotated into mesh axes.
This deliberately differs from Euler's centroid selection for boxes/planes and
its per-expression cell-mean reseeding: Cauchy composes all overrides at a point
before averaging. A cut discontinuity has quadrature accuracy, not an exact
geometric cut-volume calculation. Smooth polynomial tests verify the conservative
integral independently.

`PhysicalStateConverter` owns one reusable EOS worker. `InitialFieldEvaluator`
owns that converter and compiled ExprTk programs for the entire serial setup
pass. Concurrent callers must create separate evaluators/workers. No worker is
acquired per point and no expression state is global. The convenience
`ResolvePhysicalState` uses the same converter. `FiniteVolume::InitializeField`
accepts all native CPU DOF layouts and writes owned rows without exchanging
ghosts. It is a setup operation: if point evaluation fails, output may be
partially filled and must be discarded. Domain construction enforces that rule
through its collective error guard and final admissibility check.

These are initial-field programs, not time-dependent BC expressions. Prescribed
BC tagged states and existing analytic profiles retain their existing scope.
