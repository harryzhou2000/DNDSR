# Euler flow beta and RANS reconstruction, 2026-11-05

## Separate spatial and temporal corrections

Reconstruction PP (`EvaluateURecBeta`) selects an admissible flow/species base
and compresses the polynomial toward it. The default
`limiterControl.ppRecLimiterCompressToMean=true` uses zero nonconstant
coefficients (P0) as that base. With the option false, the existing staged
order search is retained. A rejected candidate discards that order's flow
content before trying the next base. Species stay with flow because they
participate in thermal admissibility.

SA and two-equation RANS do not participate in the flow base test or those
coefficient assignments. Their reconstruction is handled independently,
including when the original flow polynomial passes and beta is one.

- For two-equation models, evaluate the original two scalar polynomials on
  the same face and required volume quadratures. Require finite nonnegative
  conservative means. Bound negative point values by a common scalar factor
  about those means, using a zero floor and the existing 0.9 safety factor.
  Tiny positive constant means remain unchanged.
- SA follows `USE_NS_SA_NUT_REDUCED_ORDER`. Without the flag (the normal
  build), negative SA reconstructed values are allowed and scalar coefficients
  are untouched by reconstruction PP. With the flag, use the same independent
  nonnegative scalar compression. This does not change the final mean policy.

For scalar mean q and nonconstant increment d(q,x), compute theta from
`q + theta*d(q,x) >= 0`, then scale all its coefficients by `0.9*theta`
only if theta is below one. The formula and update both use the cell mean
as base. Flow fallback cannot overwrite scalar coefficients, and the flow
compression factors never receive scalar constraints. `betaPP`, `nLim` and
`betaMin` describe flow/species compression and exclude scalar-only repairs.

`EvaluateCellRHSAlpha` and the solver's increment path are intentionally
different: they describe a local temporal evolution. Density/internal-energy
constraints select alpha, and the entire coupled increment is scaled by it.
`RANSRelax` subsequently scales two-equation increments. `CompressInc` and
`AddFixedIncrement` remain final safety repairs, including independent scalar
negativity repair and the configured omega mean floor. These paths are unchanged.

## Defects and history

The first SST source in `3c98578f` (2024-03-25, **SingelBlockApp; SST first
trial**) computed gamma2 but blended gamma1 on both sides. The correction is
`gamma1*F1 + gamma2*(1-F1)`. It preserves the existing SST variant and implicit
sink diagonal; it does not introduce SST-2003. The same correction is published
on `codex/sst-fix` as `4dd845e9`.

`b2464bc9` (2024-03-10) introduced the base-plus-increment reconstruction
scheme and tested RANS values against the flow density threshold. `583863bd`
(2024-03-27) introduced separate two-equation thetaC and removed RANS from
common increment-alpha constraints, but retained the whole-array base check
and rejected-base writes. Later flow-only beta updates did not remove this
earlier coupling. A positive scalar smaller than the density threshold could
therefore fail even at P0 and erase otherwise admissible flow coefficients,
while flow beta still reported one. The old thetaC also derived a bound about
the lower-order base but scaled the whole scalar polynomial about the mean.

The remote `codex/sst-fix` commit `0c8a6f93` diagnoses the two-equation issue
and saves/restores scalar coefficients. This implementation instead makes
the flow writes explicit, includes SA separation, and uses a zero scalar floor.
It does not adopt a reference-state scalar floor capped at half the local mean.

## Focused validation

The simple-shear SST unit test independently checks inner/outer omega production.
One distributed reconstruction fixture uses SA and two-equation P3 fields in
both mean-base and hierarchical modes. Tiny constant scalars and negative
scalar extrapolations must preserve admissible flow exactly. When flow needs
fallback, admissible RANS coefficients must survive. A separate NS evaluator
provides the flow-column/beta oracle. Scalar positivity is checked on face and
volume quadratures for the selected nonnegative policy. Existing Euler and
reactive pipelines cover surrounding code. No Cauchy files are changed.

Bounded steady NACA0012 SA/SST production trials, including source/binary/config
hashes and limits, are recorded separately under
`workspace/euler_pp_sanity_20261105/`. They are code-sanity runs; reaching an
iteration cap is not convergence evidence.
