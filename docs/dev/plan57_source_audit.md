# Plan 57 second source audit

Compare merge `0e111174` directly with optimized refactor parent `b22e69e4`
and audit parent `c036961f`. The previous work orders are plans 52, 53, 54,
the November 4 programming/Euler audit, and their resolved implementation
[ledger](plan53_audit_execution.md) and [source disposition](plan53_validation/source-disposition.md).
[Plan 57's reciprocal commit ledger](plan57_migration.md) is the inventory.

`git diff --name-only b22e69e4 0e111174 -- src app python test cmake CMakeLists.txt scripts .github`
reports 227 paths. Of these, 126 have exactly the audit parent's contents;
101 differ from both parents. In that same scope, all 16 script changes, the
CMake test-target checker and the CI edit are exactly inherited. There are no
changes under `src/Euler/` or `src/EulerP/`. The shared RANS model changes one
line: the intended SST omega production blend. Whitespace in audit evidence
transcripts is inherited and the raw receipts were left as they were.

The following accounts for the source-side effects of the delta:

| Area | Expected effect and source evidence |
| --- | --- |
| Runtime storage and algebra | `Buffer::{Overlaps,SameAddress,SameRange,IsAligned}` centralize metadata queries; callers no longer retrieve data pointers just to compare them. Physical overlap still does not merge independently imported lease controllers. `ArrayView.hpp` is byte-identical to the optimized refactor, including the static row descriptor ABI. `LocalProducts.hpp` shares arithmetic only; `LinearSolver.cpp` and `EquationDofAlgebra.hpp` keep their separate leases, inverse component weights and dispatch. |
| Runtime transfers and failures | `CommunicationContext` owns the setup policy. `TransferBinding::State` resolves `Inherit` against the selected context once, agrees controls in the existing setup reduction, then uses that effective value for the three portable starts and `TransferGroup`. Groups reserve before posting, reject incompatible policies, and use separate messages and tags. Required route/length/schema exchanges remain. Unchecked launch and active unmatched work are fatal. `DynamicRowTransfer` does no post-Wait error agreement. |
| Geom | The audit context follows owned input, CGNS/OpenFOAM preparation, derived topology, closure/publication, reorder, wall distance, edits, smoothing, output and boundary setup. `RelationAccess.hpp` keeps the independent companion message context with the parent policy. `MeshAdjacencyBindings.hpp` and the small incidence vectors remain shared. Mesh archive and user companion paths keep the refactor's separation of local preparation from collective I/O or callbacks. No new read-scope bypass was added. |
| CFV metrics and capabilities | `FiniteVolumeMetricOptions` replaces the conflicting fourth positional arg. Python supports it plus a keyword-only adapter. `faceMean` remains separately integrated, pinned in host read scopes and uploaded into device metric views. Capabilities still gate setup allocation and public operation entry, independently of basis-cache selection. The optimized value-only and bundled value/gradient sampling arithmetic is retained; its extra changes are formatting and terminal catch handling. |
| Cauchy configuration and ownership | Check MPI initialization and reject workers with SINGLE before engine construction. The solver scope aborts on unexpected distributed errors; local checks can still throw. Timestep numerical rejection now travels through `StepResult` instead of caught exceptions. pMG retains the finest facade by value and keeps the engine borrow contract, coarse smoothers, rate reuse, P1/P0 operator distinctions and diagnostics. |
| Cauchy numerics and hot paths | Reconstruction PP preserves the independent RANS scalar repair, including all-good flow, and leaves the explicit uncompressed SA branch. SST blends gamma1/gamma2, covered by the independent inner/outer shear oracle. Unique-face/signed gathering, staged direct gradients, source quadrature selection and shared owned mean EOS/source preparation are retained. The changes now consolidated in `FiniteVolume.cpp` against the optimized parent are policy/catch changes, not a replacement spatial kernel. |
| Jacobian and linear solve | Both evaluated geometry snapshots use centered incidences, scalar face area and cached column slots; each retains its own numeric point type. Boundary/self contributions stay on the diagonal and repeated face contributions stay distinct. The flow5 shortcut checks exact identity once per plan, while the equation path retains bounded six-component assembly/product and generic selected/rotated fallbacks. Static block inverses, zero-start SGS, optional middle exchange, fixed-count ILU's first direct correction and weighted GMRES are retained. No rejected table-input or per-sample frame-metadata experiment is reintroduced. |
| Test and tooling changes | Both inventories are retained: aggregate dependency validation, committed split-SOR comparison, small 3D routine fixture and large golden integration, lazy Geom fixtures, explicit labels/tiers, and no duplicate serial Python CI owner. New fatal subprocess entries cover the scoped contract; opt-in audit probes and the refactor IV executable remain source-compatible. Its existing jobs/artifacts were not used. The Python follow-up fixes tests that assumed `Collective` or separate fields as defaults; it keeps logical cross-layout and ownership assertions. |

The original E3–E7 choices still hold: row copy into stable coefficient storage;
opt-in CPU transfer grouping; existing production Direct and controller-thread
MPI scheduling; assembled default with direct evaluated still selectable; and
coherent Geom acquisitions. Historical candidate measurements are not reissued
as timings of this merge. The planned GPU copy annotations, production
communication worker, Jacobi extension and pMG pool remain plan 56 work.

The second pass found another test gap: several positive-path C++ and Python
tests ignored the new `StepResult`. Previously a rejection threw and failed
the test; after the merge, two rejected trajectories could accidentally
compare equal. Existing step, restart, pMG, PP and multidomain tests now
require an accepted result before checking or committing the state. Their
numerical-rejection and exceptional-validation tests are separate and
unchanged. This is a test-only correction to the frozen production source.
The checkpoint rerun exposed another old layout assumption: the test used
physical field 1 to block a rank-local species write. The default coupled
layout has one field, so rank 0 threw before peers reached the same checkpoint
call. The test now resolves the logical species block on every rank, then holds
the matching allocation on rank 0 to preserve the intended rejection oracle.
The RANS restart fixture now selects split SoA on the writer and coupled
AoSoA on the reader explicitly. This exercises both a physical grouping
conversion and a layout conversion without treating either layout as default.

The audit also found an extra safety reduction in tracked halo construction.
`DofHalo` now uses the route's setup policy for the local compatibility
check. Ordinary and tracked binding must have identical required MPI counts
with checks off. Both still validate locally; the tracked publication record
and all routing, schema and transport protocols remain. A production catch
around tracked halo retirement or preparation now takes the same fatal path
as the underlying transfer. This small production follow-up supersedes
`0e111174` for the BSSCA candidate.

This audit distinguishes source retention from numerical and performance
acceptance. The affected two-rank C++ and Python checks passed; device 0 passed
the single-device descriptor suite. The additional CFV/Geom/BC/SST gate and
the maintained BSSCA matrices have separate receipts and are not inferred from
this diff.
