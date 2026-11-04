# Cauchy positivity policy

The CPU policy preserves Euler's implemented numerical sequence. A thermal
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

| Operation | Factor calculation | Application |
| --- | --- | --- |
| Reconstruction beta | Density, sensible energy and temperature at the configured cell/face/source quadrature points | Flow5 and species coefficient difference from the selected reference polynomial; not RANS |
| RHS compression | Thermal trial `U + rhsScale*dTau*defect/V` | Entire solve defect, as Euler's `cres *= alpha` |
| dTau compression | Same thermal trial | Replace scalar dTau by `max(alpha,smallReal)*dTau`; no second component-wise RHS scaling |
| Increment compression | Thermal trial with the selected nonlinear correction | Entire correction, including RANS and species, before final repair |
| Final increment repair | Density and thermal bound of the proposed update | Entire correction, then separate RANS repair, addition, and species mean repair |
| Physical dt restriction | Thermal trial with `dt*scale*R/V` | A scalar global timestep restriction; no independent RANS/species positivity scale |

Thermal factors depend on species through energy, even though there is no
additional minimum over individual species or RANS positivity ratios. A scalar
time restriction naturally affects all coupled equations. Extended-state
physical-dt PP remains an explicitly unsupported profile; this row describes
the established flow5 path and the intended extension, not completed support.

The relaxed density target is
`rhoTarget = relax*densityFloor + (1-relax)*rho`. Density compression is applied
first. For a decreasing sensible energy the relaxed energy target is
`floor + (1-relax)*(Q_old-floor)`. After the scheme-one energy estimate, Euler's
additional compression is `(alphaRho*alphaEnergy)^(compress*power)*(1-1e-5)`
when the unpowered factor is below one. Internal/increment control uses
`compress=2`; physical-dt control uses `compress=1`.

Final increment repair uses the unrelaxed bound, the legacy `.99` safety factor
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
