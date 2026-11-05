# Independent frozen Euler preservation gate

Original source: **4e9c3acb339f3d4c2196db6e0f57067aa719c98c**. Audit physics
execution source: **ba456643bf4c95dae6bd476cb0a4f0e18311e889**, plus the opt-in
probe whose executed bytes are retained here. The original build is an exact
selected Git archive under this audit checkout's independent assessment
workspace, with only an appended probe target in root CMake. It does not use
the current shared Physics implementation. External libraries are shared and
unchanged. The other implementation checkout and MG/IV work are excluded.

## Result and numerical scope

**All eight comparisons are bit-identical**, covering 366,400 recorded values:
NS_EX and NS_SA, np1/2, zero and prescribed nonzero linear reconstruction.
Each case has100 periodic IV cells and nonuniform, admissible mean states.
The same nonunit reference scales rho=1.7, U=230, L=2.3, T=300 apply in both
builds. The reacting case uses the ten-species bundled h2o2 mechanism.

Baseline emits conservative-state bits by persistent cell ID. The current
binary imports them before generating coefficients and evaluating anything.
The comparison requires identical owned-ID inventory, metadata, frozen means,
coefficients and volumes. State-conversion output is also compared separately.
Both probes verify that owned and ghost input bits remain unchanged. There is
no solve, time integration, limiter or PP modification of the frozen inputs.

Outputs include four RHS variants, full source Jacobians, direct cell source
and Jacobian, timestep estimates, temperature/pressure/Cp/Cv, viscosity,
conductivity, species diffusion and mean-species repair. Component comparisons
report bit differences, maximum absolute/component-scaled relative and RMS
errors; **no nonzero tolerance was used**. Nontriviality/filter checks include:

1. Every case has100 nonzero RHS rows. The reactive source and its Jacobian are
   active in all100 cells; suppressing only its Jacobian preserves the RHS,
   while suppressing the reactive source matches suppressing all sources here.
2. Suppressed reactive Jacobians are zero. All100 deliberately perturbed
   species states are repaired to nonnegative species without changing flow5.
3. SA includes34 negative transported-variable states. The direct source and
   Jacobian are nonzero. Nonzero reconstruction activates the RHS source in66
   cells; the zero-gradient periodic fixture has no measurable RHS source.
   That fixture alone is therefore not the SA source acceptance evidence.

SA uses the explicitly common historical version1, including SA-neg property/
diffusion handling. This does not assert that the intentionally revised current
SA-version0 rotation formula equals the old version0 formula. The shared point
transport complement separately checks physical2D/3D QCR with nonzero strain
and vorticity, QCR on/off, zero/nonzero turbulent ratio and adiabatic on/off.
Its16 records match exactly. Independent controls require an active QCR momentum
correction at nonzero ratio, no correction at zero ratio, and an adiabatic
change confined to energy conduction. This is not a full3D mesh experiment.

## Reproduction and provenance

`summary.json` contains the eight-case table and same-source filter oracles.
`component-comparisons.json.gz` retains component-level results and hashes of
all compared rank records. `provenance.json.gz` retains exact commands,
environment, source/binary/cache/config/mesh/mechanism hashes, row exits and
validation histories. `transport-comparison.json` records the point controls.
The executed physics source snapshot is compressed separately because the
commit hook may change its formatting. Raw rank JSON/defaults/logs remain in
`data/plan53-targeted-assessment/frozen-euler/` and the paired workspace.
Generated defaults are not tracked. The two small maintained input patches are
included; their absolute mesh/mechanism/output paths must be adjusted for a new
checkout. Configurations merge over generated defaults; do not pass a partial
patch as a complete Euler configuration.

Build `euler_frozen_physics_probe` in each paired source/build tree; the baseline
root target uses the exact same probe snapshot. The CLI is:

```sh
probe NS_EX config.json old-prefix 1 -
probe NS_EX config.json new-prefix 1 old-prefix
python scripts/benchmark/compare_frozen_euler.py old-prefix new-prefix --output comparison.json
```

Use the same launcher/rank count for each pair (NS_SA selects the other model),
OMP1/OPENBLAS1 and a120s bound per invocation. Compare only after all rank files
exist and both processes exit successfully. The transport probe takes one new
output JSON path and runs serially with a30s bound. Both targets are opt-in and
excluded from ordinary builds/CTest; this gate adds no second source tree to CI.

The first current pilot failed while reading a partial configuration; generating
and merging defaults corrected the harness. A baseline matrix validator then
mistook `rhs_no_reactive_jacobian_bits` for a square matrix. Its actual process
exited0; it was revalidated without rerunning, and the validator failure remains
in history. The first transport baseline build used the newer MPI header path;
an include-path compatibility branch corrected it. Failed/unrun records remain
separate from the successful gates. No physics was changed to obtain equality.

This is a bounded shared-seam preservation check, not a long-time accuracy,
performance, full RANS-model or arbitrary mechanism qualification.
