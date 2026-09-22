# Reactive Split Method {#reactive_split_method}

This guide describes the reactive-source integration modes implemented by the
extended Euler solvers and the reaction-region indicator (RRI) used by the mixed
Strang/coupled mode.

@tableofcontents

## Refined v2 chemistry escape (BA-CE6)

`indicatorMode=2` adds the separately selectable BA-CE6 candidate; modes 0 and 1
retain their existing selector definitions. This is an experimental candidate,
not yet a validated accuracy improvement. Define the unsnapped broadA coupled
fraction as @f$f_A@f$, stiffness activity @f$s=\Delta t\Lambda@f$ and ratio
@f$r=b/s@f$. The new local selector is

@f[
E=H(s;s_0,p_s)[1-H(r;r_e,p_r)],\qquad
\chi^*=1-f_A(1-E),\qquad
H(z;z_0,p)=\frac{(z/z_0)^p}{1+(z/z_0)^p}.
@f]

Set the escape gate to zero when the stiffness activity is zero. Apply the
existing endpoint snapping only after combining the gate with the unsnapped
coupled fraction. No additional shock-floor operation is used in this mode:
the broadA score already includes the pressure-jump gate. Spatial passes are
disabled and rejected for mode 2 so that expansion cannot undo chemistry escape.

BA-CE6 requires `chemicalActivityThreshold=10`, `diffusiveActivityThreshold=2`,
`activitySaturationExponent=0.5`, `coupledThreshold=0.005`, `hillExponent=3`,
`switchShape=1`, `shockScale=0.08`, `strangBias=1`, `spatialPasses=0` and
`chiOverride=-1`. Use `coupledSnapTolerance=0.01`, `strangSnapTolerance=0.05`.
The independent gate settings are `escapeStiffnessThreshold=100`,
`escapeStiffnessExponent=2`, `escapeRatioThreshold=1e-6`, `escapeRatioExponent=6`.
Selecting mode 2 alone does not change the baseline activity parameters to broadA.

The diagonal concentration-Jacobian proxy uses Cantera's derivative convention;
it is not a full constant-energy spectral radius or an implicit-failure detector.
Only frozen-state selection evidence currently supports this recipe. The new
a-posteriori campaign must assess evolving flame/ignition accuracy separately.

## Scope and operator convention

The method applies when `eulerSettings.reactiveFlow.enabled` is true and the
selected extended model has a chemical source. Let

$$
\frac{du}{dt}=F(u)+S(u),
$$

where `F` contains the spatial convective and diffusive terms and `S` is the
local chemical source. Denote by `A_R(Delta t)` the configured ODE advance of
the right-hand side `R`, and by `B_S(Delta t)` the constant-volume local
chemical-source advance.

`timeMarchControl.sourceStrangSplitting` selects:

| Value | Method | Physical-step operator |
|---:|---|---|
| `0` | Coupled | `A_(F+S)(Delta t)` |
| `1` | Strang | `B_S(Delta t/2) A_F(Delta t) B_S(Delta t/2)` |
| `2` | Mixed RRI | `B_(chi S)(Delta t/2) A_(F+(1-chi)S)(Delta t) B_(chi S)(Delta t/2)` |

The local switch satisfies `0 <= chi_i <= 1`. The convention is

$$
\chi_i=0\quad\Longrightarrow\quad\text{fully coupled},
\qquad
\chi_i=1\quad\Longrightarrow\quad\text{full Strang splitting}.
$$

The selector is evaluated from cell means at the beginning of a physical step
and then frozen for both source half-steps and every ODE stage of that step.
The solver owns the selector and diagnostic arrays; evaluator calls receive
them explicitly and retain no mixed-splitting state between calls.

`eulerSettings.reactiveSourceScale` is an independent debugging multiplier
`s_R`. Mixed splitting never replaces its meaning. The chemistry terms used by
the two operators are therefore

$$
S_{\mathrm{RHS},i}=s_R(1-\chi_i)S_i,
\qquad
S_{\mathrm{split},i}=s_R\chi_i S_i.
$$

The split fraction affects only the chemical contributor. Body-force,
axisymmetric, rotating-frame, and turbulence source contributors are not
scaled by `chi_i`.

## Constituent diagnostics

### Chemical activity

The dimensionless chemical activity is

$$
a_i=\Delta t_{\mathrm{phys}}r_{\mathrm{chem},i},
$$

with

$$
r_{\mathrm{chem},i}=
\left[
\sum_k\left(\frac{\dot\omega_{k,i}W_k}{\rho_i}\right)^2
+\left(
\frac{|\dot q_i|}{\rho_i c_{v,i}T_{\mathrm{scale},i}}
\right)^2
\right]^{1/2}.
$$

Here `dot(omega)_(k,i)` is the net molar production rate of species `k`,
`W_k` is its molecular weight, and `dot(omega)_(k,i) W_k / rho_i` is its local
mass-fraction rate scale. The volumetric heat-release rate is

$$
\dot q_i=-\sum_k \dot\omega_{k,i}W_k h_{k,i},
$$

where `h_(k,i)` is the species mass-specific enthalpy. The heat-release term is
converted to a temperature-rate scale using the mixture constant-volume heat
capacity `c_(v,i)` and

$$
T_{\mathrm{scale},i}=\max(T_i,T_{\mathrm{floor}}).
$$

`T_scale` is a dimensional temperature, not a logarithm. Dividing
the temperature rate by it produces an inverse-time relative-change scale.

### Diffusive activity

The dimensionless diffusive activity is

$$
b_i=\Delta t_{\mathrm{phys}}
\frac{D_{\max,i}}{L_{\mathrm{grad},i}^{2}},
$$

where `D_(max,i)` is the largest mixture species diffusivity returned by the
chemistry/transport model. `L_(grad,i)` is inferred from neighbor-cell
temperature and active-species mass-fraction changes and is bounded below by
the local cell length. The normalized temperature contribution is equivalent
to a discrete gradient of `ln(T/T_ref)` because the constant
reference temperature disappears under differentiation.

### Shock gate

The pressure-jump sensor and smooth shock gate are

$$
h_i=\max_{j\in\mathcal N(i)}
\frac{|p_j-p_i|}{\max(|p_i|,|p_j|,p_\epsilon)},
\qquad
g_h(h_i)=\frac{1}{1+(h_i/h_0)^4}.
$$

The gate suppresses coupled-region activation near strong pressure jumps. This
allows a detonation shock to remain predominantly Strang-split even when its
chemical activity is large, while smooth diffusion-reaction zones can activate
the coupled method.

## Indicator modes

`indicatorMode=0` selects the original v1 activity-product indicator, while
`indicatorMode=1` selects the v2 diffusion-to-chemistry rate-ratio indicator.
Both modes use the same `chi` convention, shock sensor, endpoint snapping, and
optional spatial expansion.

### v1 activity-product mode

Each activity uses the dimensionless threshold-and-power saturation

$$
\operatorname{sat}(z;z_0,p)=
\frac{(\max(z,0)/z_0)^p}{1+(\max(z,0)/z_0)^p}.
$$

The local coupled score is

$$
C_i=\operatorname{sat}(a_i;a_0,p)
\operatorname{sat}(b_i;b_0,p)g_h(h_i).
$$

The logistic final map uses

$$
f_{c,i}=\frac{1}{1+\exp\left[-(C_i-C_0-w\ln b_s)/w\right]},
\qquad \chi_i=1-f_{c,i},
$$

where `f_c` is the coupled fraction. Its midpoint is
`C_i = C_0 + w ln(b_s)`. The compact-tail Hill alternative uses

$$
f_{c,i}=\frac{C_i^n}{C_i^n+(C_0b_s)^n},
\qquad \chi_i=1-f_{c,i}.
$$

For either map, increasing the positive Strang-bias factor `b_s` requires a
larger score to select the same coupled fraction.

Let `chi_i* = 1 - f_(c,i)` denote the raw result of either switch map. Two
independent endpoint tolerances convert nearly pure cells into exact methods:

$$
\chi_i=
\begin{cases}
0, & \chi_i^*\leq\epsilon_c,\\
1, & 1-\chi_i^*\leq\epsilon_s,\\
\chi_i^*, & \text{otherwise}.
\end{cases}
$$

`coupledSnapTolerance` is `epsilon_c` and `strangSnapTolerance` is
`epsilon_s`; both default to `0.01`, and their sum must be less than one.
Snapping is applied after the local map and after every spatial expansion pass.
A nonnegative `chiOverride` remains an exact override and bypasses snapping.
When `chi_i = 1`, the coupled chemical contributor is bypassed. When
`chi_i = 0`, that cell does not enter the split-source integrator. If all cells
have `chi_i = 0`, the solver also skips the split-source half-step globally.

### v2 rate-ratio mode

The v2 diagnostic first forms the dimensionless chemical-Jacobian activity

$$
s_i=\Delta t_{\mathrm{phys}}\lambda_i,
\qquad
\lambda_i=\max_k\left|\frac{\partial\dot Y_{k,i}}{\partial Y_{k,i}}\right|_{\rho,T}.
$$

Cantera supplies the concentration Jacobian
`partial omega_i / partial C_j`, where `omega_i` is molar production rate. At
fixed density and temperature,

$$
\dot Y_i=\frac{\omega_i W_i}{\rho},
\qquad
C_j=\frac{\rho Y_j}{W_j},
$$

and therefore

$$
\frac{\partial\dot Y_i}{\partial Y_j}
=\frac{W_i}{W_j}\frac{\partial\omega_i}{\partial C_j}.
$$

The molecular-weight factor is exactly one on the diagonal. The implemented
estimate is consequently

$$
\lambda_i=\max_k\left|\frac{\partial\omega_k}{\partial C_k}\right|.
$$

This is a cheap diagonal-rate estimate, not the spectral radius of the complete
chemical Jacobian. In particular, it omits off-diagonal coupling and the
temperature dependence of the constant-volume source.

The local v2 Strang fraction before shock handling is

$$
r_i=\frac{b_i}{\max(s_i,s_\epsilon)},
\qquad
\chi_i^*=\operatorname{sat}(r_i;r_0,p_r).
$$

Because both `b_i` and `s_i` contain the same physical step, `r_i` is
dimensionless and independent of `Delta t`. It compares the estimated
diffusion-gradient rate with this chosen diagonal chemical-Jacobian rate. A
small ratio maps toward coupled integration and a large ratio maps toward
Strang integration. This mapping is an empirically calibrated classifier; a
large ratio must not be described as chemical stiffness dominance.

In v2 the pressure sensor supplies a Strang floor rather than multiplying a
coupled score:

$$
\chi_i\leftarrow
\max\left(\chi_i^*,\operatorname{snap}(1-g_h(h_i))\right).
$$

Thus a resolved strong pressure jump cannot be forced toward coupled
integration by the rate ratio. Endpoint snapping is then applied as described
above. The current experimental defaults are `r_0=0.1` and `p_r=2`; they are
configuration defaults rather than a universal calibration.

## Spatial expansion

When `spatialPasses` is positive, the coupled fraction is expanded through
face-neighbor cells. One pass applies

$$
f_{c,i}\leftarrow
\max\left(f_{c,i},\eta\max_{j\in\mathcal N(i)}f_{c,j}\,g_h(h_i)\right),
$$

where `spatialDecay` is `eta` in `[0,1]`. The shock gate limits propagation into
strong pressure jumps. One `reactiveSplitChi` halo exchange occurs immediately
before each pass; no diagnostic-field exchange is performed.

## Configuration

The following fragment shows every RRI setting and its current default:

```json
{
  "timeMarchControl": {
    "sourceStrangSplitting": 2
  },
  "eulerSettings": {
    "reactiveSplitIndicator": {
      "indicatorMode": 0,
      "chemicalActivityThreshold": 1.0,
      "diffusiveActivityThreshold": 1.0,
      "activitySaturationExponent": 1.0,
      "ratioThreshold": 0.1,
      "ratioExponent": 2.0,
      "coupledThreshold": 0.005,
      "transitionWidth": 0.001,
      "shockScale": 0.08,
      "strangBias": 1.0,
      "switchShape": 1,
      "hillExponent": 3.0,
      "spatialPasses": 0,
      "spatialDecay": 0.65,
      "strangSnapTolerance": 0.01,
      "coupledSnapTolerance": 0.01,
      "chiOverride": -1.0
    }
  }
}
```

`indicatorMode=0` selects v1 and `indicatorMode=1` selects v2. The final
`switchShape`, `coupledThreshold`, `transitionWidth`, `strangBias`, and
`hillExponent` settings apply to v1; `ratioThreshold` and `ratioExponent` apply
to v2. `switchShape=1` selects the default v1 Hill map and `switchShape=0`
selects the v1 logistic alternative. A nonnegative `chiOverride` disables
indicator selection and forces that value of `chi`; a negative value enables
the indicator. The defaults `a_0 = b_0 = p = 1` recover the original
mathematical saturation `z/(1+z)`.

Do not edit `cases/eulerEX_schema.json` or `cases/eulerEX3D_schema.json`
directly. They are generated from the registered C++ configuration by the
corresponding executable's `--emit-schema` option.

## Diagnostic output

Add any of these names to `dataIOControl.outCellScalarNames`:

- `reactiveSplitChi`: applied Strang fraction `chi_i` after spatial expansion.
- `reactiveSplitChemicalStep`: chemical activity `a_i`.
- `reactiveSplitDiffusiveStep`: diffusive activity `b_i`.
- `reactiveSplitShockSensor`: pressure-jump sensor `h_i`.
- `reactiveSplitCoupledScore`: local score `C_i` before spatial expansion.
- `reactiveSplitStiffnessStep`: v2 chemical-Jacobian activity `s_i`; zero in v1.

All six fields are solver-owned `ArrayDOFV<1>` storage. Their owning-cell entries
`[0, mesh->NumCell())` are valid after selector evaluation and are sufficient
for cell-data output. Ghost entries of the five constituent diagnostics are
unspecified. Ghost entries of `reactiveSplitChi` are synchronized only while a
spatial pass needs neighbor values and have no validity guarantee afterward.

## Evaluator communication contract

`EulerEvaluator::UpdateReactiveSplitChi` follows the normal evaluator API
contract: the caller must provide current owning and ghost entries of the input
state array. The function does not pull `u` itself. This is required because the
discrete temperature, species, and pressure differences read neighbor states.
The evaluator receives all output arrays explicitly. Only the supplied
`reactiveSplitChi` halo is communicated, and only for spatial expansion.
Mode 1 requires a supplied `reactiveSplitStiffnessStep` array and fails fast if
it is absent; v1 and override paths write zero to that array when it is supplied.
Chemistry evaluations inside the OpenMP cell loop use the existing per-thread
`ChemicalSource` pool, so stateful Cantera objects are not shared between threads.
