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

Tagged initial states currently require the constant profile. Ordered box/plane
and expression initialization are separate follow-up work. This implementation
does not change wall-temperature/velocity input units or introduce new BC laws.
