# Euler gradient preparation and explicit reuse

`EvaluateDt` prepares both the unlimited `uGradBufNoLim` and the limited
`uGradBuf`, including both halo exchanges. Its face spectral-radius estimate
reads the limited buffer. Previously it refreshed only the unlimited buffer,
but read the limited buffer left by earlier operations or initialization.
The mismatch originated in commit `e5b00928cc97` (2025-06-15),
**EvaluateDt now does not rely on uRec (for getting gradient)**; modern history
includes it in squash `90490801` (**Initial Release (#1)**).

The limited buffer uses `LimiterUGrad`: Barth limiting plus sensible-energy
positivity protection. `DT_Disable_Gradient_Shock_Limiter` disables only Barth
limiting, matching the direct-FV RHS with its shock limiter disabled. The static
reconstruction mask applies to both gradient buffers. The local limiting pass
overlaps the unlimited-gradient exchange; both exchanges finish before the
face estimate. This adds necessary preparation work to `EvaluateDt`.

## Caller promises

There is no freshness tracker, version check, payload comparison or additional
collective for reuse in Euler. Reuse flags express a definite local producer /
consumer sequence. The caller must guarantee all of the following:

1. The mean field `U` is the same input, including its owned and ghost values.
2. Boundary data, mesh geometry, reconstruction method and mask are unchanged.
3. The relevant gradient buffers have not been overwritten and their halo
   exchanges have completed.
4. For the limited buffer, the shock-limiter mode matches the consumer.
5. For a precise RHS, the boundary evaluation time matches exactly.

The existing `RHS_Direct_2nd_Rec_already_have_uGradBufNoLim` skips the gradient
construction and unlimited exchange, but still limits and exchanges the limited
buffer. The new `RHS_Direct_2nd_Rec_already_have_uGradBuf` skips that remaining
work too; it requires the unlimited reuse flag. Without either flag the RHS
prepares its gradients normally. First-order convection retains its existing
unlimited-gradient copy behavior.

`BuildLimitedVRO2` has analogous `LVR_Already_Have_UGradBufNoLim` and
`LVR_Already_Have_UGradBuf` flags. Limited reuse applies to the Barth O2 reference;
WBAP still performs its polynomial limiter. For LVR estimation only, the caller
may intentionally accept the boundary-time lag of an earlier `EvaluateDt`.
This permission does not extend to a precise direct-FV RHS.

## Solver sequences

The ODE call order is unchanged. Ordinary ODE RHS callbacks recompute by default,
including explicit-first stages and explicit RK stages with changed means.
No reuse permission is retained across callbacks or time steps.

Inside the source-splitting rebuild and pMG restoration, the explicit sequence
`fdtau(U) -> frhs(U, reuseDtGradients=true)` permits LVR reuse. Direct FV also
requires its stage time to equal the estimator time `tSimu`. Boundary-anchor or
profile updates inside `frhs` withdraw the permission. `FixUMaxFilter` is
currently a no-op; if it gains a modifying implementation, this local reuse
permission must be withdrawn after it as well.

Inside a pMG smoothing loop, iterations after the first have the adjacent
`fdtau(uMG1) -> coarse RHS(uMG1)` sequence and can reuse at matching time.
The first projected array and a post-increment recursive RHS instead recompute.
In particular, projecting values into a different array does not establish a
same-input reuse sequence. The two-stage `uPos == 2` branch also defaults to
recomputation unless its caller explicitly provides that sequence.

Within a single `frhs`, LVR may supply the subsequent direct-FV RHS. The caller
finishes any gradient exchanges LVR did not need. A time-lagged LVR estimate is
excluded from precise RHS reuse. The disabled-LVR shortcut does not publish
gradients and grants no reuse.

## Validation

`test/cpp/Euler/test_EulerEvaluator.cpp` compares preparation against a separate
gradient / limiter / exchange sequence after poisoning both internal buffers.
It exercises owned and ghost rows, shock-limited and PP-only gradients,
unlimited-only and both-buffer RHS reuse, and a physical vortex boundary with
changed stage time and unchanged means. A discontinuous fixture ensures the
limited and unlimited buffers differ. A second distributed test compares fresh
and reused P3 LVR O2 targets and gate fields for Barth and WBAP references,
including weak and expanded gate halos. Existing pipeline and mask/ODE tests
exercise the surrounding orchestration.

Validated locally on `dev/harry`:

- Full evaluator pipeline at np=1/2/4: 3/3 CTest entries passed (28.08 s).
- Reactive evaluator at np=1/2/4 and LVR utilities: 4/4 entries passed (1.12 s).
- The two new focused cases independently passed with 12,861 assertions at np=1.
- No-Cantera `euler_library_fast_NS_2D` and `euler_library_fast_NS_2EQ` compiled.

The CTest selections were `^euler_evaluator_pipeline_np[124]$` and
`^euler_(limited_vr|evaluator_reactive_np[124])$`. These are correctness and
regression checks; no setup/iteration performance benchmark was run for this fix.
