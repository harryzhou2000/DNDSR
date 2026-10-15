# RRI implementation audit and rebase — 2026-10-15

## Scope and revision

This is a source-level audit of the C++ selector, source splitting, source
Jacobian, ownership, communication, and ODE-stage paths. It is not a new CFD
campaign or a claim that an indicator is accurate on every flow.

- Worktree: `workspace/rri_balance_solver`; branch: `codex/rri-diffusion-balance`.
- Before rebase: `8e0e469e108c9923f723487bf28c5db01fc42b15`.
- Base: `dev/harry`, `0f986c9b3f4280677317bba8b34d5b449adabc5c`.
- After rebase: `9f563c30aca48c9b63e272e9846b6f4f2498c3d6`.
- Original tip retained at `refs/backup/rri-pre-rebase-20261015`.
- Nineteen commits replayed. Conflicts were confined to cluster-skill documents;
  the newer `dev/harry` cluster policy was retained. No C++ conflict occurred.
- `src/Euler`, `test/cpp/Euler`, the reactive-split guide, and the 2D extended
  solver schema are byte-identical to the original feature tip. The rebase
  therefore changes the surrounding base, not the RRI formulas or implementation.
- Main checkout stays on `dev/harry`. This audit does not push rewritten history
  or change the independent RRI research repository.

## Findings and missing coverage

This section records the original audit at the rebased commit. Findings 1 and 2
are addressed by the uncommitted follow-up described at the end of this note.
The original line references below refer to that audit baseline.

1. **P2 — `ignoreSourceTerm` does not disable the split chemistry operator.**
   `EulerEvaluator_EvaluateRHS.hxx:704` suppresses sources in the middle RHS,
   but `EulerSolver.hxx:168` enables physical splitting without that condition,
   and `EulerEvaluator.hxx:2194` only checks the time interval, reactive model,
   source scale and per-cell split weight. With full Strang, chemistry still
   advances despite `ignoreSourceTerm=true`; with mixed splitting, the retained
   chemistry is removed while the split fraction still advances. The combination
   is not rejected at setup. Recommended fix: reject this contradictory setup,
   or consistently disable both source operators. No evidence here says the
   recorded campaigns used this combination.

2. **P2 — native full-Strang output reports an uninitialized-as-a-diagnostic
   value of zero for `reactiveSplitChi`.** All selector arrays start at zero
   (`EulerSolver_Init.hxx:437`); the selector updates only in mixed mode
   (`EulerSolver.hxx:1422`); output reads the array directly
   (`EulerEvaluator_EvaluateDt.hxx:2858`). Thus `sourceStrangSplitting=1`
   does not output its actual all-one split fraction. Mixed mode with
   `chiOverride=1` is not affected. Recommended fix: initialize/output the
   appropriate constant for each integration mode, and distinguish inactive
   sensor diagnostics from measured zeros.

3. **P2 — the tracked 3D extended-solver schema is stale.**
   `cases/eulerEX3D_schema.json` lacks `indicatorMode` and the Balance-filter
   controls, although the shared C++ settings and 2D schema contain them.
   This is a tooling/documentation mismatch, not evidence that the 3D C++
   selector is unavailable. Regenerate it with the solver's schema generator;
   do not hand-edit generated schemas.

4. **Regression coverage is incomplete.** Helper tests cover scalar mappings,
   limits, filters and validation; chemical tests cover directional derivatives
   and source-scale behavior. There is no dedicated solver-level regression
   covering the whole distributed mixed step, per-cell zero/one shortcuts,
   nonchemical-source preservation, frozen weights across staggered stages,
   or the two issues above. The existing reactive evaluator MPI test exercises
   cell-mean species repair, not those contracts. Source tracing found no
   stage-to-stage chi swap or unintended chi mutation, but is not a substitute
   for these end-to-end regressions.

These findings are recorded, not repaired by this rebase.

## The three independent switching levels

### 1. Integration split: one global choice, local weights only in mixed mode

`timeMarchControl.sourceStrangSplitting` selects:

| Value | Middle ODE operator | Two outer half-step operators |
| --- | --- | --- |
| 0 | Full flow plus chemistry | None |
| 1 | Flow without chemistry | Full local chemistry |
| 2 | Flow plus retained chemistry | Local split chemistry |

Let $F$ include transport and any nonchemical sources, $S$ be chemistry,
and $\alpha$ be the independent debugging factor `eulerSettings.reactiveSourceScale`.
The mixed step uses the composition

$$
\mathcal B_{\alpha\chi S}(\Delta t/2)\,
\mathcal A_{F+\alpha(1-\chi)S}(\Delta t)\,
\mathcal B_{\alpha\chi S}(\Delta t/2).
$$

Here the notation denotes successive substeps, not addition of two complete
time integrators. Each cell may have a different weight, but the spatially
coupled middle solve is still one global solve. It does not allocate different
ODE methods to different cells.

- `chi=0`: fully coupled; that cell skips its local chemistry integration.
- `chi=1`: true Strang; chemistry RHS/Jacobian work is skipped in that cell's
  middle solve, while nonchemical sources remain.
- `0<chi<1`: both chemistry paths execute, with complementary weights.
- `reactiveSourceScale` multiplies both portions; it is not replaced by chi.
- Physical splitting is limited to reactive extended Euler physics and
  single-step ODE methods; `useImplicitPP` and simultaneous
  `implicitReconstructionControl.sourceTauSplitting` are rejected.

`sourceTauSplitting` is a separate pseudo-time/source-preconditioning strategy,
not another RRI recipe or physical Strang step.

### 2. Indicator: one recipe per run, evaluated on individual cells

`eulerSettings.reactiveSplitIndicator.indicatorMode` chooses the formula.
It does not select the ODE method. The labels broadA, BA-CE6, B-E2, localR10,
and Kinetic are research/launcher recipes, not C++ enum values or JSON strings.

Define the saturation and shock gate by

$$
H(z;z_0,p)=\frac{(\max(z,0)/z_0)^p}{1+(\max(z,0)/z_0)^p},
\qquad g=\frac{1}{1+(h/h_0)^4}.
$$

Here $h$ is the maximum normalized face-neighbor pressure jump. Write
$M(C)$ for the final coupled-score map: either Hill
$H(C;C_0b_s,n)$, or logistic with midpoint $C_0+w\ln b_s$ and width $w$.
The default shape is Hill. In the table, $f=1-\chi^*$ is the raw coupled
fraction, before endpoint snapping and any enabled spatial passes.

| Mode | Scheme and formula | Active shaping / spatial options |
| --- | --- | --- |
| 0 | Original/default or broadA: $C=H(a;a_0,p)H(b;b_0,p)g$, $f=M(C)$ | Activity thresholds/power, final map, shock gate; optional decaying neighbor expansion |
| 1 | Legacy ratio v2: $\chi^*=\max[H(b/s;r_0,p_r),1-g]$, with each branch snapped before taking the maximum | Ratio map and shock floor; optional legacy neighbor expansion |
| 2 | BA-CE6 family: $f=M(C)(1-E)$, $E=H(s;s_0,p_s)[1-H(b/s;r_e,p_e)]$ | Original score/map plus chemistry escape; no spatial passes; zero stiffness has zero escape |
| 3 | Balance: $C=gad/(1+a)^2$, $f=M(C)$ | Hill map only; optional chain expansion/smoothing before snapping; B-E2 is a filter recipe |
| 4 | localRXX: $f=H(a;a_0,1)H(d/a;r_0,p_r)g$ | Activity midpoint, ratio midpoint/power, shock gate; no final score map or spatial passes |
| 5 | Kinetic: $f=H(B/A;r_0,p_r)$, $A=\|J_s s_{cv}\|$, $B=\|J_s d_{cv}\|$ | Ratio midpoint/power only, then optional endpoint snaps; no activity/shock/spatial gates |

**The ratio direction is not universal:** mode 1 sends increasing transport/
stiffness ratio toward Strang; modes 4 and 5 send increasing transport response
relative to chemical response toward coupled. Changing only `indicatorMode`
does not preserve the meaning of all old thresholds.

For modes 0–2, $a=\Delta t\,r_{chem}$,
$b=\Delta t D_{max}/L_{grad}^2$, and
$s=\Delta t\lambda_{diag}$. The chemical norm combines mass-fraction
production and an enthalpy heat-release temperature proxy. The gradient length
uses cell means and neighbors, with species cutoffs and a cell-length bound.
The stiffness surrogate is the largest absolute species-Jacobian diagonal at
fixed density and temperature, not a full spectral radius or thermal runaway
eigenvalue. These older activity estimates do not include the debugging source
scale, although the actual chemistry stepping does.

Modes 3–4 instead use matched constant-volume species/log-temperature rate
norms, $a=\Delta t\|s_{cv}\|$, $d=\Delta t\|d_{cv}\|$. Their
thermal component uses internal energies. The molecular diffusion estimate
uses conservative two-point species fluxes, mass-flux correction, conduction,
and species-enthalpy transport. These chemical rates include the source scale.

Mode 5 differentiates the same constant-density chemistry in coordinates
$(Y_1,\ldots,Y_{N_s},\ln(T/T_{ref}))$, where $T_{ref}$ is fixed and
does not affect the derivative. `ChemicalSource::constantVolumeSourceResponseNorms`
combines Cantera concentration, temperature and pressure derivatives, the
constant-density pressure chain rule, and the thermal energy/capacity chain
rule. The temperature derivative of mixture heat capacity uses a centered
finite difference. This is a chemical directional-response ratio, **not a
flow/chemistry commutator**. The full species vector includes the inferred last
species; no global spatial Jacobian is assembled. At a fixed snapshot its chi
is timestep independent. Its output activities are $\Delta t^2 A$ and
$\Delta t^2 B$, not the mode-0 activity definitions. Zero $B$, including
$A=B=0$, selects Strang; zero $A$ with positive $B$ selects coupled.

### Recipe values versus defaults

The C++ defaults are mode 0, $a_0=b_0=p=1$, Hill exponent 3,
$C_0=0.005$, $h_0=0.08$, bias 1, no spatial passes, and both snap
tolerances 0.01. **The default is not broadA or Kinetic.**

The checked two-scale launcher uses:

| Recipe | Principal overrides |
| --- | --- |
| broadA | Mode 0; $a_0=10,b_0=2,p=0.5,C_0=0.005,n=3$; Strang snap 0.05, coupled snap 0.01 |
| Balance | Mode 3; $C_0=0.001,n=2$; same snaps |
| B-E2 | Balance plus two max-neighbor expansion passes and one smoothing pass |
| localR10 | Mode 4; $a_0=0.001,r_0=0.1,p_r=4$; coupled snap 0.1, Strang snap 0.05 |
| localR10P2/P3 | localR10 with ratio power 2/3 |
| Kinetic H01P2 | Mode 5; $r_0=0.1,p_r=2$; both snaps zero |

BA-CE6 means the mode-2 broadA/escape family with escape controls
$s_0=100,p_s=2,r_e=10^{-6},p_e=6$; exact case overlays remain the
authoritative record. A recipe name alone is insufficient run provenance.

Global `chiOverride` in [0,1] bypasses indicator evaluation, spatial processing
and snapping; diagnostic scores are zeroed. A negative value enables selection.
Configuration validation still runs, including some common controls unused by
the chosen mode. Normal snapping maps $\chi^*\leq\epsilon_c$ to zero and
$1-\chi^*\leq\epsilon_s$ to one; tolerances must sum to less than one.

### 3. ODE method: independent from the selector

`timeMarchControl.odeCode` selects the middle-step integrator: e.g. 0 is
ESDIRK4, 204 is ESDIRK2, 411 is DITR U2R2, and 412 is DITR U2R1. DITR
solve strategy/preconditioning settings are separate again. RRI does not choose
them dynamically. Multistep modes, including the U3R1 implementation marked
multistep, cannot be combined with physical source splitting.

## State, communication, and code ownership

1. `EulerSolver.hpp` owns the six selector/diagnostic arrays. Evaluator methods
   receive references; the reactor receives an optional **const** chi array.
2. `EulerEvaluator.hxx:1794` computes the selector. Input solution halos must
   already be valid. Diagnostic arrays are valid on owning cells only. Only
   enabled neighbor filters communicate chi; there are no unconditional pulls
   of all diagnostic fields.
3. `EulerSolver.hxx:1422` computes chi once, after physical timestep selection,
   from the pre-first-half-step state. It is frozen across both chemistry halves,
   all middle ODE stages, pseudo-time iterations, and staggered DITR solves.
   Neither the ODE stage state copies nor RHS evaluations overwrite it.
4. `EulerEvaluator_EvaluateRHS.hxx:714` passes the same $1-\chi$ to chemical
   RHS and source-Jacobian construction, including the residual paths used by
   the implicit solver. `SourceTermContributor.hpp` applies this factor only to
   chemical contributions. There is intentionally no derivative of chi in the
   Newton matrix: chi is frozen during that physical step.
5. `EulerEvaluator.hxx:2184` performs constant-volume chemistry, skips individual
   cells with chi zero, preserves density/momentum/total energy, and refreshes
   solution halos after changing species. A global maximum reduction avoids
   both split halves when all cells have chi zero.
6. `ResetFreshStart()` after the split halves invalidates ODE history/cached
   endpoint RHS that would otherwise refer to the pre-chemistry state. This
   includes the staggered two-stage path; chi is not an ODE stage unknown.
7. Output chi is the **weight used for the preceding physical step**, not a fresh
   evaluation on the end-of-step profile. Initial-output diagnostics use the
   configured `dtImplicit` before subsequent CFL clipping. The optional
   `useDtPPLimit` RHS estimate also precedes the new selector update and sees the
   previous selector. These distinctions matter for interpreting snapshots.

## Deliberate limits and follow-up work

- Modes 3–5 estimate molecular diffusion from cell means; they do not use the
  full reconstructed flow residual, convection/compression, a turbulence model,
  or a general nonorthogonal diffusion operator. Boundary faces without another
  cell are skipped, so prescribed thermal/species boundary fluxes are not
  represented in the indicator. Interior orthogonal laminar-strip evidence does
  not establish general boundary-dominated or turbulent-flow validity.
- Local formulas are templated for extended 2D/3D physics. Balance spatial
  filters specifically require a one-dimensional neighbor chain (at most two
  counted non-self neighbor faces), and reject more general connectivity.
  They should not be advertised as general 2D/3D smoothing.
- Mode 5 has stable ratio arithmetic and explicit zero limits, but no
  uncertainty/roundoff gate for small physical rates or nearly equilibrated
  states. Tight thermal inversion helps; it does not prove noise immunity.
- BroadA and Balance do not share identical thermal/transport estimators.
  `reactiveSplitCoupledScore` also changes meaning between modes: it is a raw
  score for 0–3 (with mode-specific use), a raw fraction for 4–5, and never a
  universal final filtered/snapped coupled fraction.
- No adaptive choice of indicator recipe, learned reactor, or new ODE
  integrator is implemented here. Fractional splitting can execute chemistry
  in both paths; choosing it does not itself establish a cost saving.

## Verification

- Built `eulerEX`, `eulerEX3D`, `euler_test_reactive_split_indicator`,
  `euler_test_source_chemical`, `euler_test_physics_properties` and
  `solver_test_ode`, using the worktree's existing `build/`, Cantera enabled,
  and configured `ccache` compiler launcher. Build exited successfully.
- CTest: **4/4 passed** (`euler_reactive_split_indicator`,
  `euler_source_chemical`, `euler_physics_properties`, `solver_ode`).
- Verified `dev/harry` is an ancestor of the rebased branch; the audited Euler
  code/tests match the pre-rebase backup exactly. Main DNDSR and independent
  RRI research repository remain clean. At this initial audit checkpoint only
  this note was uncommitted; the subsequent follow-up is described below.
- No new posteriori CFD cases or distributed mixed-step regression runs are
  part of this audit; successful compilation/unit tests do not close finding 4.

## Implementation delta from dev/harry

The replayed branch is broader than just the latest Kinetic selector: its
committed delta is 49 files, including cases, mechanisms, plotting tools and
skills. The numerical implementation additions are:

1. Solver-owned selector fields, configuration, optional const chi arguments,
   physical chemistry half-steps and history resets in the solver loop.
2. Complementary chemistry-only RHS/Jacobian weighting and exact endpoint
   work skipping; nonchemical source semantics remain unchanged by chi.
3. Six selectable indicator modes, scalar mapping/filter helpers and
   mode-dependent diagnostic output. The newest Kinetic addition supplies the
   constant-volume directional-response API and its derivative tests.
4. Earlier prerequisite fixes retained in the branch: multispecies temperature
   gradient/viscous heat-flux closure and positivity-safe reconstructed point
   output/primitive averaging, with corresponding transport tests and guides.

There is no new ODE integrator in this delta: existing ESDIRK/DITR machinery is
wrapped by the physical splitting sequence. Rebase conflict resolution adds no
numerical changes to these four groups.

## Requested follow-up: named strategies and minimal fixes

The subsequent review authorizes publication and requests schema descriptions;
the earlier decision to leave generated schemas untouched is superseded below.

- Reject physical source splitting with `ignoreSourceTerm=true` during
  configuration validation and again at ODE setup for programmatic callers.
- Initialize the existing chi storage to one for native full Strang and zero
  otherwise. This replaces the existing initialization write; it adds no
  per-step pass, output-time branch or communication.
- Replace integer selector dispatch with `ReactiveSplitIndicatorMode` and its
  maintained JSON string mapping: `Activity`, `StiffnessRatio`,
  `ChemistryEscape`, `TransportBalance`, `MatchedRateRatio`, `ChemicalResponse`.
  No formula changes accompany the conversion. `Activity` remains the default.
- Old integers, unknown names and null fail config validation. Existing archived
  research records are not rewritten; future launcher overlays must translate
  the historical numbers 0–5 to these names. No independent RRI-repo launcher
  changes are included in this source-only follow-up.
- Leave generated schemas untouched as requested. In particular, their current
  integer-mode descriptions are not the new runtime configuration contract.
- Add small enum serialization/invalid-input and split/source-guard fixtures.
  Do not add new CFD matrices or intrusive stage-mutation test machinery.
  The const chi argument/frozen-step contract is unchanged; there is still no
  dedicated whole-integrator mutation regression.

Follow-up verification: both `eulerEX` and `eulerEX3D` build successfully with
ccache. The four selector/chemistry/thermodynamics/ODE CTest targets pass. The
new source-guard fixture passes all eight assertions per rank at np=1 and np=2.
The broader existing reactive evaluator fixture cannot finish in this worktree:
`data/mesh/IV10_10.cgns` is absent here (present in the main checkout), and it
stops on mesh open before the species-repair test. No fixture/data-layout changes
were made to conceal that prerequisite. No new runtime chi-output or whole-step
mutation test was added; the output fix is the single existing initialization
write, with no production accessor or extra test-only integration scaffold.

## Source review map against dev/harry

These anchors refer to the current feature worktree, including the follow-up,
not to `dev/harry`. Review the complete delta with `git diff dev/harry -- src`
from this worktree. This list includes all fifteen changed source files and
separates core RRI work from the earlier supporting transport/output changes.

| Source file and line | What to review |
| --- | --- |
| `src/Euler/ReactiveSplitIndicator.hpp:9` | Named enum, string mapping, settings/validation; scalar formulas, maps, filters and snapping follow |
| `src/Euler/EulerEvaluatorSettings.hpp:455` | Selector settings owned by evaluator configuration; registration at line 803 |
| `src/Euler/EulerSolver.hpp:120` | Solver-owned diagnostic arrays; split control at line 214; cross-field rejection at line 952 |
| `src/Euler/EulerSolver_Init.hxx:437` | Array allocation and constant baseline chi initialization |
| `src/Euler/EulerSolver.hxx:168` | Global split routing and optional chi; step freezing/first half at 1422, second half at 1525, setup rejection at 1968 |
| `src/Euler/EulerEvaluator.hpp:1187` | Reactor interface, explicit output references, selector ownership/halo contract; optional chi also threads through RHS/source APIs |
| `src/Euler/EulerEvaluator.hxx:1794` | Selector dispatch, neighbor transport estimate and filters; constant-volume half-step at 2187 |
| `src/Euler/EulerEvaluator_EvaluateRHS.hxx:704` | Per-cell retained chemistry scale and exact-zero work skipping |
| `src/Euler/EulerEvaluator_EvaluateDt.hxx:1716` | Thread weight into source contributions, including quadrature; diagnostic outputs at 2858; corrected viscous wrapper call at 1216 |
| `src/Euler/SourceTermContributor.hpp:424` | Chemistry-only RHS/Jacobian weighting, independent debugging scale and zero-work guard; auxiliary scale declared at 53 |
| `src/Euler/Chemistry/ChemicalSource.hpp:288` | Chemical stiffness and constant-volume directional-response API contracts |
| `src/Euler/Chemistry/ChemicalSource.cpp:792` | Stiffness surrogate; response chain rule at 819; supporting mass-fraction diffusivity API change at 430 |
| `src/Euler/Physics/PhysicsProperties.hpp:1085` | Supporting multispecies temperature-gradient and viscous-flux correction |
| `src/Euler/Gas.hpp:1895` | Single-species interface contract and lower-level heat-flux handling |
| `src/Euler/EulerSolver_PrintData.hxx:230` | Supporting nodal positivity limiting followed by primitive-state accumulation/averaging |

Lightweight test entry points: `test/cpp/Euler/test_ReactiveSplitIndicator.cpp:7`
and `test/cpp/Euler/test_EulerEvaluatorReactive.cpp:75`. Existing chemistry
directional-difference coverage starts at `test/cpp/Euler/test_SourceChemical.cpp:181`.

## Supporting gradient/output audit and immediate performance review

Scope: current feature source versus `dev/harry`, including the multispecies
transport and nodal-output changes already carried by the branch. These are
source/algebra findings, not profiled timing measurements. Subsequent formatting
can shift the line anchors in the review map; function names remain authoritative.

1. **Gradient correction: no algebra/sign/scaling defect found for admissible
   ideal mixtures.** `reactiveTemperatureGradient` differentiates total internal
   energy, subtracts the kinetic-energy derivative and the composition-energy
   derivative including the dependent last species, then divides by rho cv.
   Species enthalpies/internal energies are in velocity-squared code units;
   species gas constants, temperature and cv use matching reference scales.
   The viscous wrapper disables bare-gas heat conduction and adds the full
   corrected conduction once. The species-flux helper receives zero conductivity,
   so its legacy gas-constant-gradient correction is not applied a second time.
   Stress/viscous work remain delegated to Gas. The mass-fraction diffusivity
   change is consistent with gradients of mass fractions, and the correction
   velocity makes the sum of all species mass fluxes zero.

2. **Gradient scope remains conditional.** The formula assumes positive density,
   ideal-gas species thermodynamics and an admissible composition. Property
   evaluation repairs roundoff excursions to the simplex but does not
   differentiate that repair; therefore it is not an exact derivative of an
   arbitrary nonphysical reconstructed state. The face limiter, not this routine,
   owns that precondition. The frozen-equivalent-gamma pressure gradient still
   exists in the primitive-gradient utility; this change prevents its use as a
   multispecies temperature derivative, rather than claiming to correct every
   consumer of the pressure gradient.

3. **Point PP/output: the intended invariant is preserved.** Each contributing
   reconstructed state is limited toward its cell mean; periodic velocity and
   rotating-frame transformations are applied consistently to both states.
   Finite positive density, nonnegative explicit/dependent species, sensible
   energy and a valid temperature are checked. Bisection retains a known-valid
   state, with the mean as fallback even if no trial is accepted. The owning node
   averages rho/velocity/T/species with positive equal weights, not conservative
   energy. Its positive density/temperature and species simplex are therefore
   retained mathematically; the primitive-to-conservative conversion used for
   pressure/gamma is a forward thermodynamic conversion, not a new UV inversion.
   Output storage is separate from the solution. This is visualization, not a
   conservative nodal discretization. Last-species roundoff is handled by the
   existing conversion sanitation.

4. **Point failsafe is not an arbitrary-invalid-mean recovery mechanism.** An
   invalid mean intentionally throws. Its strict sensible-energy test excludes
   a state exactly at the base-temperature floor. Unsupported EOS and extreme
   density may encounter existing assertion-based thermodynamic guards, which
   `catch(std::exception)` cannot recover. These are limits of the underlying
   state API, not evidence that an admissible flame/detonation reconstruction
   is incorrectly averaged. No general EOS recovery redesign is included.

5. **Immediate avoidable costs removed without changing the methods:** reuse
   the enthalpy vector in place for internal energies (one less per-face heap
   allocation); evaluate Cp once instead of twice per viscous face point; skip
   the exact temperature gradient when conductivity is zero; skip the legacy
   zero-conductivity correction's otherwise needless Rgas/composition work;
   and skip ghost-node reconstruction/PP/EOS work whose results were discarded.
   Ghost cells are still visited when they contribute to an owning node; the
   existing final pull supplies ghost-node output. No additional communication
   or persistent evaluator state is introduced.

6. **Remaining performance opportunities, not silently redesigned:** gradient,
   enthalpy, cv and diffusion property calls still allocate temporary composition
   buffers and repeat property setup; reusing a per-face property bundle would
   require a broader API change. Point output checks the mean per incidence,
   can perform up to 48 UV solves when limiting, and repeats temperature
   recovery during primitive conversion; this costs output time, not every
   physical RHS. Kinetic allocates local Eigen workspaces, obtains the chemical
   derivative matrix and evaluates cv at perturbed temperatures once per cell
   per physical step. Matched transport also evaluates production rates before
   the Kinetic response API repeats them. No global flow Jacobian or dense
   spatial solve is introduced by the indicator. Fractional chi requires both
   integration paths; a cheaper total runtime is not guaranteed.
   The six scalar arrays are also allocated for nonmixed runs (about 48 bytes
   per local/ghost cell before array/communication metadata); avoiding that
   footprint needs an explicit optional-diagnostic storage design, not a hidden
   change to the current output/reference contract.

7. **Configuration documentation is generated from source.** Enum docstrings
   give formulas/report-family names. The registered `indicatorMode` description
   explains every accepted string and distinguishes a formula family from a tuned
   recipe. Both extended-solver schemas are regenerated using `--emit-schema`,
   never manually edited. A small fixture checks that the emitted allowed values
   and description retain all names/report aliases. The heat-flux fixture extends
   the existing gradient finite-difference test with conduction sign, adiabatic
   suppression and zero-conductivity behavior. Full distributed nodal-output
   invariance and timing benchmarks remain outside this lightweight UT scope.

Publication includes the prior guard/chi-output/enum changes. Unit tests are to
be run on the committed source after pushing, as requested; their actual outcome
will be reported in the handoff rather than preclaimed here.
For that run, the missing mesh prerequisite is supplied by an ignored symlink
from this worktree's `data/mesh/IV10_10.cgns` to the existing main-checkout mesh.
No mesh bytes or generated default input are added to the commit.
