# Euler GG boundary targets and fixUL

## Boundary-state ownership

`generateBoundaryValue` accepts a mutable left face state and an immutable cell
mean. With `fixUL=true`, the SA viscous-wall handler sets both left and right
SA components to zero when `USE_FIX_ZERO_SA_NUT_AT_WALL` is enabled. This is an
intentional face-flux boundary condition. The flag does **not** promise that the
left state will be preserved. Pass a fresh local copy, never a stored cell mean.

The high-order VR callback already creates a fresh `ULfixed` for each call and
returns the raw right boundary value. Its difference callback similarly owns
independent left copies. Neither contaminates later faces or stored means. These
VR contracts are unchanged. The RHS's physical flux also intentionally uses the
post-BC left/right pair. Its mean-state boundary call and implicit boundary
linearization use `fixUL=false` and retain their existing semantics.

The source-gradient loop formerly created one mutable cell copy outside the
face loop and passed it as both left state and mean with `fixUL=true`. An SA
wall therefore zeroed the reference used for subsequent faces, making the
gradient depend on traversal order. Other current wall models do not mutate
the left state, but all gradient callbacks must support this contract safely.

## Exact GG formula

Let `M` be the immutable cell mean, `L` the fresh left state after boundary
handling, and `R` the returned right state. The boundary face target is
`(L + R)/2`. To preserve the localized jump formulation, gradient callbacks
return the **effective ghost**

```text
R_effective = R + L - M
(R_effective - M)/2 = (R + L)/2 - M.
```

Thus zero-SA wall targets contribute `-M_SA`, rather than zero or half the
required jump. For a non-mutating BC, `L=M` and the old callback is recovered.
Interior faces continue to use `(R-M)/2`, with periodic state transforms.
Subtracting the constant mean is equivalent to integrating face averages when
the discrete area-normal sum is zero. The jump form avoids summing large
constant face terms and preserves the legacy formula if that closure is only
approximate; it does not add a geometric closure correction.

`EvaluateSourceGradientGG` preserves the legacy **unweighted** arithmetic-mean
formula: face area times the outward normal times the jump, summed and divided
by cell volume. It does not add distance weights, limiting or communication.
The caller supplies current mean halos. Each boundary gets a fresh left copy.

`GetGradientBoundary(t)` applies the effective-ghost adapter to direct FV,
`EvaluateDt` preparation, the LVR O2 reference, degree-one substitution and
forced second-order RANS reconstruction. The CFV gradient algorithms are
unchanged. Method 1 retains its configured interior distance weighting, method
11 remains GGMP and method 2 remains least squares, including their existing
periodicity limitations. This adapter is not used for high-order VR.

The optional `useSourceGradFixGG` correction starts from a reconstructed
polynomial trace `L_before`. Its face contribution is now
`(R + L_after)/2 - L_before`. Retaining `L_before` prevents wall mutation from
silently dropping the left-state part of the correction. The snapshot is only
engaged when this option is enabled.

## Options and timing

- `source2nd`: replace all source gradient columns with legacy unweighted GG.
- `ransSource2nd`: replace only RANS source gradient columns with that GG.
- These options retain reconstructed source states and source quadrature. A
  direct-FV evaluation instead uses its prepared gradient, as before.
- `direct2ndRecMethod`: 1=GG, 11=GGMP, 2=least squares;
  `subs2ndOrderGGScheme` still controls method 1's interior weighting.
- `useSourceGradFixGG`: add the integrated face-average-minus-original-trace
  correction to reconstructed source gradients.

Schema descriptions and C++ docstrings record these distinctions. Stage time
is forwarded unchanged to each boundary call. This correction adds no freshness
tracking, reuse permission or collective operation. The explicit reuse promises
in [the gradient-buffer note](euler_gradient_buffers_2026-11-05.md) still apply.

## EulerP scalar indexing

The cell GG kernel now reads interior scalar values from `iCellOther`, and
boundary scalar values from the face-indexed buffer at `iFace`. Previously both
read `iCell`, unlike the flow-vector path. The boundary-generation stubs in
EulerP are not extended by this fix.

## Focused verification

The evaluator pipeline has two new regression cases on a small wall mesh:

1. An independently assembled SA face-target oracle checks source GG, both
   source option selections, reversed face traversal, unchanged stored means,
   GG/GGMP/least-squares boundary adapters, and the RHS source jump correction
   for both constant and affine reconstructed traces.
2. The actual EulerP host cell kernel checks two nonuniform scalar fields
   against independent neighbour/face-indexed sums and the flow-vector kernel.
   The fixture requires interior neighbours and boundary face IDs different
   from cell IDs; it supplies boundary buffers directly because EulerP boundary
   generation remains unimplemented.

Validated locally on `dev/harry` against base `f5ea928c`:

- Rebuilt `euler_test_evaluator_pipeline`, including its CFV, NS, NS_3D, SA and
  two-equation dependencies.
- Both focused cases passed at np=1/2/4 with `OMP_NUM_THREADS=1`; np=1 checked
  205 assertions.
- Full evaluator pipeline: 3/3 CTest entries passed at np=1/2/4, 24.46 s total,
  using the registered `OMP_NUM_THREADS=2`. All 15 cases passed per rank. The
  existing NS, SA and NS_3D golden values were retained unchanged.

The full selection was `^euler_evaluator_pipeline_np[124]$`; the focused doctest
filter was `-tc=*SA GG*,*EulerP scalar GG*`. EulerP verification runs the actual
shared cell kernel on the host, not its unimplemented BC-generation wrapper.
No CUDA execution, CFD convergence or performance claim is made by these
focused regressions.
