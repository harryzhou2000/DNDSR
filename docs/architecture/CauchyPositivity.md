# Cauchy positivity policy

The implicit CPU policy preserves Euler's implemented numerical sequence. A thermal
criterion and the components multiplied by its factor are different concepts.
RANS positivity does not determine the thermal factor, but Euler applies that
factor to the entire nonlinear increment/RHS. Reconstruction has a different
scope. Do not replace these policies with a common generic limiter.

## State and energy constraints

Write the conservative state as `U=(rho,m,rhoE,z,r)`, where `z_k=rho*Y_k`
contains independent species densities and `r` contains RANS variables. Logical
block locations are resolved separately from physical AoS/SoA/AoSoA storage.
The implicit last species is `z_last=rho-sum(z_k)`.

For fixed reference temperature `T*`, the raw energy reference is affine:

```text
B_T*(U) = rho*e_last(T*) + sum_k z_k*[e_k(T*) - e_last(T*)]
Q_T*(U) = rhoE - dot(m,m)/(2*rho) - B_T*(U)
```

Require `rho >= densityFloor` and both prepared `Q >= floor` constraints.
For a perfect gas these are the pressure/internal-energy floor and the
temperature floor. For mixtures they are the base-reference sensible-energy
margin and the energy at `max(temperatureFloor, material.BaseTemperature())`.
No equivalent gamma is used. The equation owner prepares the species energies;
the numerical view performs arithmetic without chemistry callbacks.

With positive density and the **raw affine reference**, Q is concave along a
line in conservative state space. Its endpoint chord supplies Euler's
scheme-one admissible compression estimate. Species must move with flow along
that line: independently changing composition can violate the energy bound
even when the flow increment is zero. The accepted design retains common
flow/species thermal compression rather than introducing a species backoff.

Property evaluation can use a clipped/normalized temporary composition. That
reference is not globally affine; the same concavity argument is not a proof
for that repaired map. The implementation therefore also checks the repaired
reference. Reconstruction can fall back to its validated base polynomial;
final increment repair has bounded pull-up followed by validation. An exhausted
or nonfinite repair is an explicit failure, not a successful PP update.

## Operations and their scopes

RHS FPP means **RHS forecasted positivity preserving**. Its factor can be
applied to either the implicit solve defect or the pseudo-step; it is not a
selection of the spatial RHS algorithm. Current configuration separates the
three operators:

```json
{
  "time": {
    "positivity": {"enabled": false, "relax": 0.8, "scale": 1, "increaseLimit": 2}
  },
  "defaults": {
    "implicit": {
      "positivity": {
        "rhsFPPMode": "rhs", "rhsFPPReduction": "cell",
        "rhsFPPScale": 1, "rhsFPPRelax": 0.5,
        "incrementRelax": 1, "compressionPower": 1
      },
      "multigrid": {"coarseRHSFPP": false}
    },
    "numerics": {
      "positivity": true, "positivityReconstruction": "Euler",
      "positivityCompressToMean": true, "positivityVolumePoints": true,
      "positivityBetaPower": 1, "positivityBetaCutoff": 0.001
    }
  }
}
```

Domain settings can override the `defaults` sections. Physical dt PP has its
own enable flag; `numerics.positivity` currently gates reconstruction and
implicit thermal PP. `rhsFPPMode` selects none, complete-defect scaling (`rhs`)
or scalar pseudo-step compression (`dTau`). `rhsFPPScale` multiplies only the
trial increment; `rhsFPPRelax` sets its Euler relaxed density/energy threshold.
`incrementRelax` separately controls the post-linear correction.

`rhsFPPReduction` preserves Euler's factor sharing. For original cell factors
`[1,.8,.3]`, `cell` applies `[1,.8,.3]`; `minimumLimited` applies `[1,.3,.3]`;
`minimumAll` applies `[.3,.3,.3]`. The minimum spans the domain communicator.
The trigger count records original limited cells. Even cell mode retains the
required global diagnostic statistics; disabled RHS FPP performs neither its
factor calculation nor these reductions. This scope does not select a new
spatial flux blend or a global increment-repair method.

The former development keys `rhsMode`, `rhsReduction`, `rhsScale`, `rhsRelax`
and `coarseRHSPositivity` are renamed to the RHS-FPP keys above. Unknown config
keys fail explicitly. Archived run configs retain their original source/schema
provenance; their immutable records are not rewritten.

| Operation | Factor calculation | Application |
| --- | --- | --- |
| Reconstruction beta | Density, sensible energy and temperature at the configured cell/face/source quadrature points | Flow5 and species coefficient difference from the selected reference polynomial; not RANS |
| Implicit RHS compression | Thermal trial `U + rhsFPPScale*dTau*defect/V` | Entire solve defect, as Euler's `cres *= alpha` |
| Implicit dTau compression | Same thermal trial | Replace scalar dTau by `max(alpha,smallReal)*dTau`; no second component-wise RHS scaling |
| Implicit increment compression | Thermal trial with the selected nonlinear correction | Entire correction, including RANS and species, before final repair |
| Implicit final increment repair | Density and thermal bound of the proposed update | Entire correction, then separate RANS repair, addition, and species mean repair |
| Physical dt restriction | Thermal trial with `dt*scale*R/V` | A scalar global timestep restriction; no independent RANS/species positivity scale |

Coarse pMG defaults follow Euler's omission of the RHS/dTau limiter:
`implicit.multigrid.coarseRHSFPP=false` forces coarse rhsFPPMode to none
after level overrides. True enables the inherited/per-level RHS/dTau policy.
This switch does not disable finest RHS/dTau limiting, reconstruction beta or
final increment compression/repair. It changes numerical policy, independently
of optional collective diagnostics and MPI launch validation.

Thermal factors depend on species through energy, even though there is no
additional minimum over individual species or RANS positivity ratios. A scalar
time restriction naturally affects all coupled equations. CPU physical-dt PP
also accepts the logical extended state; it uses the same mixture reference
energies and `compress=1`, with no independent species/RANS positivity minimum.
This is a timestep proposal, not a proof that every explicit stage is positive.
Extended explicit flux blending remains unavailable.

## Explicit stage acceptance and retry

SSPRK3 uses the uncompressed conservative residual for flow5 and species in each
Shu-Osher stage. It never applies cellwise thermal increment compression or
species mean repair to make that stage acceptable. The user permits Euler's
separate RANS exponential increment repair and model scalar mean bounds; those
act only on the RANS block and never rescale flow5/species. Reconstruction
limiting and temporary property-state validation remain spatial operations.
The raw species candidate must satisfy nonnegative independent densities and
their sum no greater than rho (the implicit last species is the remainder).
Failing this check rejects the trial; no clipping/normalization of accepted
explicit species values is allowed.

The established perfect-gas flow5 path may blend low/high residuals with one
globally shared factor. Both sides of a face use the same factor, retaining
flux cancellation. Extended equations currently have no such flux-blend adapter:
an inadmissible stage rejects the complete physical-step trial. `numerics.positivity`
still selects their reconstruction PP; it does not enable increment repair.

For each cell, form the complete low/high SSPRK stage candidates with the
same history weights and dt. `AdmissibleFraction(low, high)` bisects their
conservative-state segment for 56 iterations against density and thermal
interior bounds, retaining a small numerical safety margin. If every high
candidate is admissible, alpha is one. Otherwise the low candidates must be
admissible everywhere and alpha is the minimum local fraction over all cells
and MPI ranks. Apply that single alpha to the residuals, not a per-cell factor
to updates. The search re-evaluates state predicates, never the face fluxes.

**Deferred direction:** replace the domain-global factor with a conservative
local flux-correction scheme. A face must use one shared correction coefficient
for its two adjacent cells, including across MPI partitions, and satisfy both
cells' coupled density/thermal/species constraints. The admissible low-order
update, source and boundary treatment, and multi-stage guarantee need a separate
design and validation. No local algorithm is implemented here; independently
compressing cell increments is not a substitute. The global blend remains a
simple baseline until that explicit PP work is undertaken.

The caller retries from the accepted state with smaller dt. `Simulation::Advance`
discards every domain's failed trial without changing accepted values, time or
step count. The production C++ frontend halves dt up to `time.maxRetries`
(default zero, meaning report the rejection). Direct Domain/Python callers
perform the same discard/retry explicitly; use `DiscardStep()` after a failed
`PrepareStep()` and supply the reduced dt. The next trial restores the accepted
reconstruction guess and restarts all three stages. Backend/allocation failures
can poison private scratch and require rebuilding the Domain; they are not
ordinary positivity rejections. No timestep restriction proves every later
stage admissible.

Physical Strang splitting also accepts SSPRK3 transport: chemical half-step,
source-free SSPRK3 transport, then chemical half-step. The external CVODE
reactor retains its separate Euler-compatible species-output convention; it
does not apply a compressed increment to the explicit transport stages. Overall
Strang composition is second order regardless of the transport method's order.

The extended trial uses `EquationMeanWorkspace`, separate from the normal VR
workspace, over the same FV plans and CPU engine. Its coefficients stay zero,
VR/WBAP/rec-PP are skipped, and viscous face fluxes are ignored. Configured
chemical/RANS volume sources and frame sources remain present, following
Euler's mean-trial sequence. RANS source/wall producers still receive actual
molecular properties and geometry; ignoring viscosity does not disable sources.
No accepted owned values, warm polynomial, physical clock or history changes.

`FiniteVolume::LimitPhysicalStep` reads ready, nonaliased owned logical rows;
a mean ghost transfer may be active. Its caller retains the matching frozen
thermal-policy buffers and coordinates the same model/policy on domain ranks.
The wrapper validates schema/mapping and finite increments before the existing
global minimum/count reductions. Domain lazily owns the independent trial
workspace/residual and applies growth capping from the last committed dt.

## Implicit compression details

The relaxed density target is
`rhoTarget = relax*densityFloor + (1-relax)*rho`. Density compression is applied
first. For a decreasing sensible energy the relaxed energy target is
`floor + (1-relax)*(Q_old-floor)`. After the scheme-one energy estimate, Euler's
additional compression is `(alphaRho*alphaEnergy)^(compress*power)*(1-1e-5)`
when the unpowered factor is below one. Internal/increment control uses
`compress=2`; physical-dt control uses `compress=1`.

Implicit final increment repair uses the unrelaxed bound, the legacy `.99` safety factor
and, if needed, successive `.9` pull-ups with a finite iteration cap. Afterwards
RANS uses the model-specific exponential negative-update repair (and the SA
floor where enabled), then its mean bounds. Species means are repaired against
the accepted density. The combined state is validated after these repairs.
Scalar repairs must not silently change a component frozen by a partial solve.

Reconstruction optionally chooses a lower-order admissible reference before
compression. This order fallback can discard higher modes in **all** blocks,
as in Euler; it is distinct from multiplying the retained polynomial by beta.
The unlimited VR warm guess remains unchanged; PP updates the flux polynomial.
Thermal admissibility at sampled points is not a proof of positivity at every
point inside a cell.

## Legacy correspondence and validation limits

- `EulerEvaluator.hxx::EvaluateURecBeta`: reference/order fallback,
  flow/species beta, and separate two-equation scalar coefficient damping.
- `EulerEvaluator.hxx::EvaluateCellRHSAlpha`: relaxed thermal ratio, powers,
  safety factor and global reduction options.
- `EulerSolver.hxx::fsolve/fincrement`: whole-defect scaling or scalar dTau
  reduction, then whole-increment scaling before `AddFixedIncrement`.
- `EulerEvaluator.hpp::CompressInc/AddFixedIncrement`: final density/energy
  repair, scalar RANS repair, addition and species mean repair.

RANS reference selection checks every scalar against the flow density floor,
as Euler's `checkRecBaseGood` does. Compression uses different thresholds:
`smallReal * referencePrimitive[j]`. The current input is
`numerics.ransState.referenceConservative`, containing explicit `rho` and all
named conservative RANS components. Primitive references divide those scalars
by the supplied rho. Missing/invalid rho fails when reconstruction PP needs
the conversion. There is no unit-density or scaling-density fallback. The
conservative omega mean floor does not need density and remains unchanged.
These values are code-unit physical-state references, not the independent
`rho0/U0/L0/T0` nondimensionalization scales.

For two-equation RANS, the minimum scalar base-to-high ratio determines theta;
when limited, Euler multiplies both **entire coefficient columns** by `.9*theta`.
It does not blend only their difference from the reference. This distinction
is intentionally preserved even with a nonconstant reference. Scalar damping
does not lower thermal beta or enter its diagnostic count. During RHS/property
evaluation, the separate component-wise fallback of negative RANS point values
to their cell means remains active. No stronger positivity theorem is claimed
for the whole-coefficient rule with a nonconstant reference.

The SA `USE_NS_SA_NUT_REDUCED_ORDER` switch contributes its scalar ratio to the
flow/species density fraction, without separately damping SA coefficients here,
matching Euler. Its point fallback retains the existing compile-time control.
Without that switch, SA still participates in reference-order quality checks.
A nonnegative scalar mean below the reference-quality threshold can force
complete order fallback; coefficient damping cannot raise a zero mean. An
out-of-range/nonfinite scalar ratio fails explicitly rather than being clamped
to hide incompatible reference thresholds.

The homogeneous mixture fixture demonstrates convergence of a short discrete
reacting step, with selected H2/H2O residuals and actual species evolution. It
does not establish equilibrium, flame accuracy, or robustness for arbitrary
chemistry steps. See the matched physical studies in the active phase plan.
