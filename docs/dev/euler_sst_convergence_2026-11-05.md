# SST convergence audit: O2 NACA0012

The working case uses full O2 flow/RANS, Roe_M1 flux, ILU(2) without GMRES,
and density-only convergence. Raw residual magnitudes across different
equations are not comparable: omega has a particularly large wall scale.
The research record is `workspace/sst_convergence_20261105/`.

The maintained input is
[`euler2EQ_config_NACA0012_SST_O2.json`](../../cases/euler2EQ/euler2EQ_config_NACA0012_SST_O2.json).
From the build directory, run:

```sh
OMP_NUM_THREADS=1 mpirun -np 16 ./app/euler2EQ.exe ../cases/euler2EQ/euler2EQ_config_NACA0012_SST_O2.json
```

It records the successful Roe Jacobian/CFL 20 setup, with a 6,000-update
cap and density tolerance of 1e-5. The experiment's 600-second wall limit
was imposed by its launcher, not the solver config. The case writes final
cell and boundary VTK output under `data/out/NACA0012_SST_O2/`.

## Model and numerical details

The implementation retains the original SST vorticity-limited eddy
viscosity, original gamma constants and a k-production upper cap of20.
Omega production uses deviatoric strain, independently of the cap. This is
an existing implementation choice, not a complete SST-2003 implementation.
Changing to SST-2003 requires changing the model consistently, including
strain limiter, production treatment and constants, rather than adjusting
one constant to improve a convergence plot.

`Pk = mu_t * SS:grad(v) - 2/3*rho*k*div(v)` is signed. Its upper cap does
not suppress negative production. The obsolete, immediately overwritten
positive-clamp assignment was removed without changing the source result.
Wall omega is `800*nu/d1^2`, equivalent to the original recommended
`10*6*nu/(beta1*d1^2)` with beta1=.075. No wall constant changed.

Primitive gradients of k/omega use density-gradient corrections. Viscous
flux uses `mu+sigma*mu_t`. The approximate implicit viscous spectral radius
uses a flow-based upper scale; the source diagonal is integrated per cell
and added with the positive negative-source-Jacobian convention. This
approximation freezes diffusion coefficients and omits turbulence/flow
couplings, nonlinear blending and coefficient derivatives.

Two-equation reconstruction currently replaces both scalar reconstructions
with the GG second-order result, independent of `ransForce2nd`. Source
gradients can independently use `ransSource2nd`/`source2nd`; the first keeps
VR flow gradients and the second replaces all source gradients with GG.
Both retain the source quadrature/state selection. Do not confuse those
options with first-order source evaluation.

In this O2 case, `maxOrder=1` and `subs2ndOrder=1` already replace the VR
iteration with direct GG for all components, so increasing its iteration
count is redundant. `subs2ndOrderGGScheme=1` weights reconstruction by
cell-to-face distances, whereas `EvaluateCellSource`'s GG alternative uses
unweighted averaging. The source-gradient experiment therefore changes a
specific averaging convention, not simply the number of reconstruction
updates.

## Optional source diagonal

`eulerSettings.KOmegaSSTConfig.implicitAdditionalSinks` defaults to false.
It changes implicit mode1 (and consequently the mode2 diagonal), never
the RHS. With density, primitive gradients, eddy viscosity and blending
frozen, retain only additional nonnegative diagonal terms:

- k: add `max(2/3*div(v),0)` when the production cap is inactive;
- omega: write cross diffusion as `Q/omega`; add
  `max(Q/omega,0)/(rho*omega)` to `-dS/d(rho*omega)`.

Negative cross diffusion stays explicit. This is a frozen-coefficient
approximation, not a full Jacobian. Finite differences in the far-wall,
pure-RANS limit test both cross-diffusion signs and RHS invariance.
The O2 NACA0012 pilot did not benefit from this option: at6000 updates its
density residual ratio was7.183e-5 versus5.656e-5 with the option off.
It is an opt-in experimental control, not a recommended default.

## Wall perturbation correction

The viscous-wall `linMode=1` branch used to return before negating the SA
or two-equation perturbations. The physical ghost relation is
`qGhost=2*qWall-qInterior`, so its frozen-target perturbation is
`delta(qGhost)=-delta(qInterior)`. The correction affects the optional
Roe Jacobian. It does not change wall RHS values or the default scalar
spectral approximation. SST/Wilcox k/omega finite differences exercise
the public boundary dispatcher at np1 and np2.

The wall target's dependence on the mean thermodynamic state remains
frozen, as do existing approximate flow/isothermal boundary derivatives.
This is not an exact thermodynamic boundary Jacobian.

## Remaining hypotheses

The default source Jacobian lacks the k-omega destruction off-diagonal
`betaStar*k`, which SU2 retains. Our flux Jacobian also treats turbulence
variables as passive scalars, omitting isotropic k-stress derivatives in
momentum/energy. Those are accuracy limitations of the implicit operator,
not demonstrated RHS errors. A full coupled derivative needs separate tests.

The positivity repair may drive k to extremely small values or exact zero;
it only repairs proposed negative updates and has no positive relative
increment limit. The omega mean floor is relative to the far field.
Scaling these safeguards or introducing relative update limits would be
a new numerical control. Do not silently introduce a different RHS floor.

## Primary comparisons

- [Turbulence Modeling Resource SST definitions](https://tmbwg.github.io/turbmodels/sst.html):
  original/SST2003 differences, production limits and wall/far-field values.
- [NASA LAVA numerical controls](https://www.nas.nasa.gov/LAVA/curv_docs/yaml_input_file/):
  explicit/implicit production limits, full/STTS linearization and separate
  positive/negative turbulence update limits. These are solver choices.
- [SU2 turbulence source implementation](https://github.com/su2code/SU2/blob/master/SU2_CFD/include/numerics/turbulent/turb_sources.hpp):
  destruction diagonal plus k-omega coupling; variants select production.
- [OpenFOAM SST source](https://cpp.openfoam.org/v13/kOmegaSSTBase_8C_source.html):
  bounded fields, relaxation and sign-dependent implicit source treatment.
  Its segregated matrix splitting is not our frozen-derivative prescription.

The old OVERFLOW PDF URL redirects to the NASA overview; no numerical
claim here relies on an unverified copy of that guide.

## First matched pilot results

All runs use the same current-code snapshot,20816 cells,16 MPI ranks,
one thread per rank,6000-update cap and600-second wall cap. These are
diagnostics, not converged aerodynamic coefficients. Ratios below use the
solver's running-maximum component normalization, including unlogged early
updates; the analysis also retains absolute residuals.

| Variant | Density ratio | k ratio | omega ratio | Wall seconds | Terminal status |
|---|---:|---:|---:|---:|---|
| Baseline, CFL10 |5.656e-5|.1934|.01499|554.0|Update cap|
| Extra source sinks |7.183e-5|.2195|.01581|555.1|Update cap|
| GG source gradients |3.611e-5|.09319|.03461|548.6|Update cap|
| Roe Jacobian |2.113e-5|.004187|1.480e-5|600.0|Wall cap during finalization|

The Roe run reached6000 updates and wrote both terminal fields before the
wall cap; its boundary flux independently reintegrates to the6000-update
history within1.6e-12. It remains wall-capped in the manifest. None met
the1e-5 density criterion. Its density/k/omega ratios improved by about
2.68/46.2/1013 relative to baseline, with about8% extra time per update.
These compare complete choices of implicit operator, including the corrected
wall branch; they do not isolate the wall correction's effect alone.

The new baseline's6000-update residuals and terminal fields reproduce the
previous current-HEAD O2 result. Lowering final CFL to1 and applying
RANSRelax.25 did not improve convergence per update in the initial pilots.

## Final control check

With the same Roe Jacobian, raising final CFL10->20 (same1000-update ramp)
meets the requested density criterion at update4619 in435.2s: density
ratio9.988e-6, k ratio.002751, omega ratio2.806e-5. No source/RHS option is
changed for this row. All terminal fields are finite, rho,p,T,k,omega are
positive, and the boundary-force/log discrepancy is3.3e-12. Combining Roe
with GG source gradients at CFL10 instead reaches1.588e-5 at6000 updates.

These justify Roe/CFL20 as a better initial O2 solver configuration for
this case, not a universal SST default. The wall-force vertical component
still changes by about1.2% from update4000 to4500. Density-only stopping
does not establish force convergence or tighter turbulence convergence.
Detailed comparison, exact inputs and residual plots are in the study.

Validation:34 RANS kernel cases/334 assertions; all13 evaluator pipeline
cases at np1/np2 for the numerical patch; focused wall finite differences
also cover SA after the later test-only extension. The experimental source
diagonal remains off by default. No new physical SST model choice or
increment limiter was introduced.
