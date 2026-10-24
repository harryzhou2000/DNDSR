# Spalart–Allmaras DES selection and ft2

SA hybrid behavior is selected by **both** `eulerSettings.SADESScale` and
`eulerSettings.SADESMode`. `SADESScale` is the dimensionless coefficient
`C_DES`, not an already constructed length. A value **greater than 100**,
including the effectively infinite default, selects **pure RANS for every
mode**. This preserves the existing way of disabling DES.

For `0 < SADESScale <= 100`, the mode selects:

| SADESMode | Model | LES grid scale |
|---:|---|---|
| 0 | DDES, with wall-region shielding and SA low-Re correction | Maximum cell size |
| 1 | IDDES (default mode) | Wall-distance-dependent IDDES scale |
| 2 | WMLES branch of IDDES | Same scale as IDDES |
| 3 | Original DES97, without shielding or low-Re correction | Maximum cell size |

Modes 0–2 retain their existing numerical selectors; mode 0 was previously
misdocumented as RANS even though it executed DDES. Mode 3 exposes original
DES explicitly. These selectors apply to SA; the existing SST path also uses
`SADESScale` as a cell-size multiplier but does not use `SADESMode`.

## Length construction

Let `d_w` be nearest-wall distance, `h_max` the maximum local cell size, and
`h_wn` the local cell spacing in the wall-normal direction. `h_wn` is a cell
dimension, not a distance from its center to the wall. Following Shur et al.
(2008), equations 4, 7 and 17:

$$
\Delta=\min\{\max(0.15d_w,0.15h_{max},h_{wn}),h_{max}\},
\qquad \ell_{LES}=C_{DES}\,\psi\,\Delta,
$$

$$
\ell_{IDDES}=\widetilde f_d(1+f_e)d_w+(1-\widetilde f_d)\ell_{LES}.
$$

Here `psi` is the SA low-Re correction, `f_d_tilde` is the RANS blend weight,
and `f_e` is the elevating function. WMLES uses the geometric `f_B` blend
instead of `f_d_tilde`. DDES uses `Delta=h_max` and its delaying function:

$$
\ell_{DDES}=d_w-f_d\max(0,d_w-C_{DES}\psi h_{max}).
$$

Original DES97 uses `min(d_w,C_DES*h_max)` without `psi`. In all branches,
the coefficient multiplies the completed geometric scale.

**Current geometry limitation:** the evaluator does not supply `h_wn`.
Its callers therefore use the documented approximation omitting that term;
the shared length helper accepts it when available. No wall-normal spacing
is inferred from wall distance or minimum cell size. This approximation
matches the published geometric scale where
`h_wn <= max(0.15*d_w,0.15*h_max)`.

The corrected source callers pass `C_DES` and `h_max` separately. Earlier
code instead supplied `min(C_DES*h_max,0.15*max(d_w,h_max))` for all active
modes. Consequently active DES results, including existing IDDES inputs,
can change with the correction. Existing cluster binaries and saved results
are unaffected until explicitly rebuilt and relaunched.

## Runtime ft2 switch

```json
{
    "eulerSettings": {
        "SADESScale": 0.65,
        "SADESMode": 1,
        "SAConfig": {
            "useFt2": false
        }
    }
}
```

`SAConfig.useFt2` defaults to **true**, preserving old inputs. When false,
the nonnegative SA branch sets `ft2=0` consistently in production,
destruction, the approximate implicit diagonal, and the DES `psi` formula.
The option applies to RANS as well as DES and replaces the former hard-coded
`SA_USE_FT2_TERM` macro.

The negative-SA continuation retains its recovery coefficient `c_t3=1.2`
regardless of this option. Simply setting that coefficient to zero would
reverse its recovery production for negative modified viscosity. This is a
documented numerical continuation of the nonnegative SA-noft2 model, not a
claim that an unmodified combination of standard SA-neg and SA-noft2 is
supported by the model specification. Source and implicit diagonal use the
same selected policy. Rotation-correction selectors are unchanged.

## References

- Shur et al., [A hybrid RANS-LES approach with delayed-DES and wall-modelled
  LES capabilities](https://doi.org/10.1016/j.ijheatfluidflow.2008.07.001), 2008.
- [Turbulence Modeling Resource: SA, SA-neg, SA-noft2 and SA-R](https://tmbwg.github.io/turbmodels/spalart.html).

Analytic kernel checks cover 2D/3D source and implicit-diagonal behavior with
ft2 on/off, coefficient placement, shielding, the pure-RANS scale override,
and negative-state recovery. The MPI evaluator pipeline also exercises active
IDDES with ft2 disabled. These are implementation checks, not capsule-flow
validation or a demonstration of LES resolution.
