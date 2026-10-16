# Limited variational reconstruction controls

LVR is selected by `limiterControl.useLimiter=true` and
`limiterControl.limiterProcedure="LVR"`. It needs `vfvSettings.maxOrder>1`,
`implicitReconstructionControl.useExplicit=false`, `recLinearScheme=0`,
`dampRecIncDTau=false`, and `limiterControl.preserveLimited=false`.
The reconstruction iteration uses the existing block Jacobi or block GS/SOR
choice (`vfvSettings.SORInstead`). These are reconstruction iterations, not
restrictions on the physical time integrator.

## Named selections

Selectors use scoped C++ enums and case-sensitive JSON strings. Saved/resolved
configuration always writes names. Exact legacy integer inputs are accepted
only for the retained choices below; fractional numbers, booleans, null and
unknown names are rejected.

| JSON field | Named values | Legacy integer aliases | C++ type |
| --- | --- | --- | --- |
| `limiterControl.limiterProcedure` | `"3WBAP"`, `"CWBAP"`, `"LVR"` | `0`, `1`, `2` respectively | `ReconstructionLimiter` |
| `limitedVR.o2ReferenceLimiter` | `"Barth"`, `"WBAP"` | `0`, `1` respectively | `O2ReferenceLimiter` |
| `limitedVR.gateMode` | `"ProductCubic"`, `"PressureExtremum"` | `0`, `8` respectively | `LimitedVRGateMode` |

`ProductCubic` is retained for the original v1 comparisons. `PressureExtremum`
is the supported pressure-extremum correction used by v2 and v3, previously
called mode 8. Both share the same local pressure/compression product gate.
Modes 1–7 were unsuccessful union, rational-tail or neighbour-envelope
experiments. Their implementation and helper functions have been removed;
input selecting them now reports an error. Reproduce those historical runs
with their archived source revision, rather than silently substituting a gate.

The input-only `preset` already uses names (`"v1"`, `"v2"`, `"v3"`). Boolean
branch options remain booleans; halo depths and neighbour counts remain integers.

## Start with a named preset

An optional input-only `limiterControl.limitedVR.preset` selects a documented
starting point. Existing numeric cases using the retained selections keep their
behavior without it.

```json
{
  "limiterControl": {
    "useLimiter": true,
    "limiterProcedure": "LVR",
    "limitedVR": {
      "preset": "v3",
      "alphaMax": 1.0,
      "contactAlphaMax": 1.0
    }
  }
}
```

This example selects WBAP-based Average H2, with shock/contact caps `(1,1)`.
Use `0.5` for both caps for the `(0.5,0.5)` variant. Presets do not change
reconstruction degree, derivative weights, Riemann solver, time stepping or
boundary conditions. Select HQM_OPT or Factorial derivative weights separately.
They also do not enable LVR automatically: the two parent switches above do.

| Preset | Reference and gate | Default branch caps: shock, contact, weak |
| --- | --- | --- |
| `v1` | Barth GG O2; product gate, pressure `0.10/0.22`, compression `0.08/0.16` | `0.5, 0, 0` |
| `v2` | WBAP GG O2; `PressureExtremum`, pressure `0.04/0.16`, compression `0.03/0.13` | `0.5, 0, 0` |
| `v3` | v2 plus Average H2 contact correction: face density gate `0.5/0.8`, cell-mean density support `0.005/0.06`, two correction halo passes | `0.5, 0.5, 0` |

A preset resets the **whole LVR section** to that preset, then applies explicit
keys in the same input layer. Later patch files override earlier files.
A CLI preset is expanded before all ordinary CLI pointer overrides, so those
explicit CLI values win regardless of their position. For example,
`-k /limiterControl/limitedVR/preset -v v3` can replace a long LVR parameter
list. Saved/resolved configuration contains explicit settings with named
selectors and omits `preset`. The generated JSON schemas advertise the named
string selections and all 40 LVR controls for every Euler solver variant.
The input-only preset is expanded before ordinary runtime field parsing.

For example, the reference and gate can also be selected directly:

```json
{
  "limiterControl": {
    "limiterProcedure": "LVR",
    "limitedVR": {
      "o2ReferenceLimiter": "WBAP",
      "gateMode": "PressureExtremum"
    }
  }
}
```

This partial input overrides only those fields. It does not imply the rest of
the v2/v3 preset; use `preset` when that is intended.

## Penalty and update

Let $r_i$ denote the nonconstant reconstruction coefficients in cell $i$.
Let $t_i$ contain the limited GG O2 coefficients in its linear modes and zero
in higher modes. With cell means fixed, ordinary VR has block row
$A_i r_i=\sum_j B_{ij}r_j+b_i$. LVR adds
$\tfrac12\lambda_i(r_i-t_i)^T A_i(r_i-t_i)$, where
$\lambda_i=\alpha_i/(1-\alpha_i)$ for $0\le\alpha_i<1$. Thus

$$
r_i^{k+1}=(1-\omega_i)r_i^k+\omega_i\left[
 (1-\alpha_i)A_i^{-1}\left(\sum_j B_{ij}r_j^{*}+b_i\right)
 +\alpha_i t_i\right].
$$

Here $\omega_i$ is the existing relaxation, and $r_j^{*}$ is the old neighbour
value for Jacobi or the currently available value for GS/SOR. The same cached
$A_i^{-1}$ is used. The limit $\alpha_i=1$ suppresses neighbour/RHS coupling
and relaxes this block toward $t_i$; it does not require an infinite number in
memory. The complete system need not be a globally symmetric least-squares
Hessian; this statement describes the implemented cell-block penalty.

The target and gate depend on cell means, not the unknown high-order
coefficients. They are fixed during each reconstruction solve. Target positivity
uses cell means as bases, compressing its increment to keep face quadrature
states and any required volume states positive. The finalized target supplies
both the ordinary face sensors and the penalty target. The weak extension
instead uses **unlimited** explicit gradients by design.

## Sensor definitions

For positive face pressures $p_L,p_R$, densities $\rho_L,\rho_R$, sound speeds
$c_L,c_R$, and velocities $u_{n,L},u_{n,R}$ along the common left-to-right face
normal, define

$$
 J_p=\frac{2\lvert p_R-p_L\rvert}{p_L+p_R},\qquad
 J_c=\frac{\max(u_{n,L}-u_{n,R},0)}{(c_L+c_R)/2},\qquad
 J_\rho=\frac{2\lvert\rho_R-\rho_L\rvert}{\rho_L+\rho_R}.
$$

A cubic ramp $S(z;z_0,z_1)$ is zero for $z\le z_0$, one for $z\ge z_1$, and
$q^2(3-2q)$ between them, with $q=(z-z_0)/(z_1-z_0)$. The ordinary product
shock response is $\alpha_{s,f}=\alpha_{s,\max}S(J_p;jp0,jp1)S(J_c;jc0,jc1)$.
The independent contact response is
$\alpha_{\rho,f}=\alpha_{\rho,\max}S(J_\rho;j\rho0,j\rho1)$, regardless of
pressure and velocity. Each cell takes the maximum over its interior faces;
final $\alpha_i=\max(\alpha_{s,i},\alpha_{\rho,i},\alpha_{w,i})$.
Physical boundary faces are excluded from the detector's neighbour operations.

`PressureExtremum` additionally detects supported strict extrema in **cell-mean pressure**.
For a strict maximum, its excess $E$ is the difference from the nearest lower
unequal face neighbour, divided by the larger absolute value of the two;
reverse the difference for a minimum. It requires at least the configured number
of nearly equal face neighbours and excludes physical-boundary-adjacent cells.
Let $s$ be the maximum neighbouring base shock fraction divided by its cap.
The correction seed is the branch cap times
$\max[S(E;e_0,e_1)S(s;s_0,s_1),\ r\alpha_{\mathrm{old}}S(s;s_0,s_1)/\alpha_{\max}]$.
Here $e_0,e_1$ are `oscillationExtremumStart/Full`, $s_0,s_1$ are
`oscillationSupportStart/Full`, $r$ is `oscillationRetention`, and
$\alpha_{\mathrm{old}}$ is the previous shock-branch fraction in this cell.
`PressureExtremum` propagates this correction once to immediate face neighbours. v3 applies
the same construction to density, with independently selected mean-density
support and `contactExtremumHaloLayers` maximum passes (H2 means two).
These corrections are supported extremum detectors, not a guarantee that
physical smooth extrema are never limited.

The optional weak branch uses adjacent means and their unrestricted GG
pressure gradients. Set $\Delta p=p_R-p_L$ and
$d_L=\nabla p_L\cdot(x_f-x_L)$, $d_R=\nabla p_R\cdot(x_f-x_R)$, where $x_f$
is the face centroid and $x_L,x_R$ are cell centroids. With
$\epsilon_p=\texttt{weakRelativeFloor}\max(\lvert p_L\rvert,\lvert p_R\rvert)$,

$$
 M=\frac{\lvert\Delta p-d_L+d_R\rvert}
 {\max(\lvert\Delta p\rvert+\lvert d_L\rvert+\lvert d_R\rvert,\epsilon_p)},
 \qquad
 K=\frac{\max(u_{n,L}-u_{n,R},0)\,\overline{\rho c}}{\lvert\Delta p\rvert}.
$$

Here $\overline{\rho c}=(\rho_Lc_L+\rho_Rc_R)/2$.
For compression with $\lvert\Delta p\rvert>\epsilon_p$, the weak response is
$\alpha_{w,\max}S(M;m_0,m_1)S(K;k_0,k_1)$; otherwise it is exactly zero.
A centered uniform-strip GG stencil gives zero mismatch for degree at most two,
but not generally for cubic fields or distorted multidimensional stencils.
The weak branch does not necessarily contain the ordinary shock gate.

## Every LVR control

Defaults below are the raw `LimitedVRSettings` defaults (`v1`). Preset changes
are listed above; all ramp full thresholds must exceed their starts.

| Control | Raw default | Meaning / active dependency |
| --- | --- | --- |
| `alphaMax` | `0.5` | Shock cap $\alpha_{s,\max}$, in `[0,1]`; zero disables shock penalty. |
| `pressureJumpStart`, `pressureJumpFull` | `0.10`, `0.22` | `jp0`, `jp1`: pressure ramp start/full values. |
| `compressionStart`, `compressionFull` | `0.08`, `0.16` | `jc0`, `jc1`: normal-compression ramp start/full values. |
| `o2ReferenceLimiter` | `"Barth"` | `"Barth"` or `"WBAP"` (characteristic). Both build a GG linear target on the high-order mesh/basis. |
| `o2ReferenceUsePP` | `true` | Compress O2 increment toward its cell mean if needed for positivity. |
| `gateMode` | `"ProductCubic"` | `"ProductCubic"` or `"PressureExtremum"`; the latter adds supported pressure extrema and one correction expansion pass. |
| `gateHaloLayers` | `0` | Additional shock-branch face-neighbour maximum passes, after the pressure correction; integer `0..4`. |
| `gateHaloSaturatedOnly` | `false` | In these extra passes, propagate only cells at the shock cap; unsaturated local values remain. |
| `alphaUpdateRelaxation` | `1` | Shock/contact update $\theta\alpha_{\mathrm{fresh}}+(1-\theta)\alpha_{\mathrm{old}}$, $0<\theta\le1$, each RHS evaluation. |
| `oscillationExtremumStart`, `oscillationExtremumFull` | `0.001`, `0.008` | `PressureExtremum` only: normalized pressure-excess ramp $e_0,e_1$. |
| `oscillationSupportStart`, `oscillationSupportFull` | `0.02`, `0.15` | `PressureExtremum` only: adjacent base shock fraction ramp $s_0,s_1$. |
| `oscillationRetention` | `0.95` | Retained fraction $r$ of old shock alpha while adjacent support exists, per RHS evaluation. |
| `oscillationTransverseTolerance` | `1e-8` | Relative equality tolerance for pressure **and** density extrema; scale is the larger absolute neighbour value. |
| `oscillationMinTransverseNeighbours` | `2` | Minimum number of nearly equal face neighbours, integer `1..4`; a neighbour count, not a halo depth. |
| `contactAlphaMax` | `0` | Independent density/contact cap $\alpha_{\rho,\max}$; zero disables this branch. |
| `contactDensityJumpStart`, `contactDensityJumpFull` | `0.20`, `0.80` | `jrho0`, `jrho1`: density-jump ramp start/full values. |
| `contactUseCellMeanJump` | `false` | Use adjacent mean densities for the base contact gate; otherwise use finalized O2 face densities. |
| `contactUseExtremumCorrection` | `false` | Add supported mean-density extrema and retention to the contact gate. |
| `contactExtremumUseCellMeanSupport` | `false` | Use mean-density jump for support only, independently of the base contact gate choice. |
| `contactExtremumStart`, `contactExtremumFull` | `0.001`, `0.008` | Normalized density-extremum ramp, active only with contact correction. |
| `contactSupportStart`, `contactSupportFull` | `0.02`, `0.15` | Adjacent contact-support fraction ramp, active only with contact correction. |
| `contactExtremumRetention` | `0.95` | Fraction of old contact alpha retained while adjacent support exists, per RHS evaluation. |
| `contactExtremumHaloLayers` | `1` | Synchronous face-neighbour maximum passes of the **density correction**, integer `0..4`; v3 uses `2`. |
| `weakAlphaMax` | `0` | Weak-branch cap $\alpha_{w,\max}$; zero skips its gradient communication, pressure-gradient work and halos. |
| `weakMismatchStart`, `weakMismatchFull` | `0.05`, `0.20` | Mismatch $M$ ramp $m_0,m_1$. |
| `weakAcousticStart`, `weakAcousticFull` | `0.10`, `0.50` | Acoustic compression $K$ ramp $k_0,k_1$. |
| `weakRelativeFloor` | `1e-10` | Positive pressure-noise floor multiplier. |
| `weakHaloLayers` | `2` | Synchronous maximum passes on the freshly detected weak branch, integer `0..4`. |
| `weakAlphaUpdateRelaxation` | `1` | Later within-stage weak updates $\theta f+(1-\theta)o$, with fresh $f$ and old $o$. |
| `weakAlphaDecayRetention` | `0` | Alternative weak update $f+q\max(o-f,0)$, $0\le q<1$; immediate rises, decayed falls; nonzero requires weak relaxation `1`. |
| `weakFreezeWithinImplicitStage` | `false` | Retain weak alpha after the first RHS of an implicit stage. Fresh gate at every stage start, including ESDIRK's explicit-first stage; later updates only are frozen/relaxed/decayed. |

Halo depth is graph distance through interior faces. One synchronous maximum
pass copies the largest old value from immediate neighbours; two passes reach
at most two face edges. The pressure correction's built-in one-pass expansion is separate from
`gateHaloLayers`. The pressure/density nearly equal-neighbour requirement is
particularly relevant on triangular meshes; it is not a direction-independent
smoothness theorem.

## Storage, communication and scope

When LVR is not selected, no O2-target or LVR diagnostic DOF layouts or MPI
requests are allocated. Requested diagnostic output returns zero gate/jump
values and `lvrO2Beta=1`. Selected LVR with all three branch caps zero retains
its allocated layout but skips target construction, PP and detector exchanges.

Shock/contact/support/weak alpha components share a four-component DOF array.
Only an enabled, refreshed weak branch needs unlimited gradient ghosts. A
WBAP target executes the linear-order pass only. Target PP omits unused global
statistics reductions; reductions that determine positivity floors remain.
Pressure/contact correction seeds and weak-gate expansions share the existing
four-component branch array. Average H2 with no extra ordinary shock halo uses
three gate pulls: neighbour support, packed correction seeds, then the second
density expansion. An enabled two-layer weak gate shares these rounds; its
unrestricted-gradient prerequisite remains separate. Each correction merges
its unexpanded base gate only after its own expansion finishes. This preserves
the original stencil and keeps `lvrO2Beta` exclusively a PP diagnostic. v2 uses
two gate pulls; deeper configured halos can require more. O2-reference exchanges
remain separate and unchanged. Expansion scratch storage is reused between
rounds.

These controls do not supply a proof of monotonicity, positivity of cell means,
or preservation of arbitrary smooth extrema. The O2 target PP and subsequent
high-order PP protect reconstruction states; flux/time-step PP remains a
separate solver setting. Parameters and accuracy still require the case-specific
comparisons in the limited-variational-reconstruction studies.
