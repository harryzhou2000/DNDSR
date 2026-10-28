# Mesh construction semantic audit, 2026-10-28

Status: construction correction validated for the scope below, with the
explicit legacy periodic-frame exception retained. See the
[connectivity contract](../architecture/MeshConnectivity.md). The earlier
[infrastructure finish](infrastructure_finish_2026-10-27.md) acceptance was
withdrawn because a periodic-box rank lacked its required remote boundary ring.

## Source baseline

Legacy baseline: `03891f8b5869017de355468d93294e3eb25f0feb`. The local
`build-euler-legacy-source` is an extracted tree, not an independent Git
checkout; asking Git for HEAD inside it returns the containing checkout's
revision and is not baseline provenance. Its `src/Geom/Mesh/Mesh.cpp` and
`test/cpp/Geom/test_MeshConnectivity.cpp` were content-checked against the
baseline revision. All 294 files in the frozen Geom/DNDS/CFV sources and all
14 frozen Geom test files subsequently matched `git archive` of that revision
exactly; the plan repo records the file-manifest digest. Respective SHA-256
values for the two initial anchors:

```text
4a95bec50329e3caae4140527565ae2e1d1b2e5ea98295ddca1923d030b1cc43
7a865b026acf03b5d2f9f0f1221c1abb1073656c55f9034fd57c25f5a2d8d446
```

## Source-grounded stage comparison

| Stage | Legacy source behavior | Native correction and evidence |
| --- | --- | --- |
| Serial periodic identification | `Mesh_Serial_BuildCell2Cell.cpp::Deduplicate1to1Periodic` removes donor nodes, assigns retained roots consecutive indices and remaps all cell/boundary cones with image bits. | `IO/NativeMeshPeriodicImport.cpp` compacts node IDs and cone indices. Retained coordinates, ordered images, high-water safety and explicit reservations pass the refreshed periodic tests. Exact-ID IV probes confirm O1/O2 behavior on np=1/2/4/8. |
| Owned primary topology | Reader construction establishes owned node/cell/boundary information before compute ghost building. Recovery paths invert node cones, compose node-sharing cell neighborhoods and perform physical boundary checks. | `NativeMeshBuilder::Build` returns owned global arrays. Sparse inverse/support fetches are temporary inputs. All primary records, IDs, slots and companions match the dual-source IV/box probes; independent lattice and triangle oracles check incidence/closure. |
| Partition and local permutation | `Mesh_Helpers.hpp` partitions/distributes primary cells, then optionally refines; preparation reorders cells before faces are enumerated. | `PartitionOwned` and `ReorderOwnedCells` derive temporary exact facial graphs without a persistent Face family, sharing backend/weight policy. `BuildOwned` transfers global references, typed companions, IDs and ancestry. C++ and Python owned setup ordering, reconstruction and fields pass the refreshed matrix; legacy reader/METIS fixture conditions are explicitly retained. |
| First compute ghosts | `BuildGhostPrimary` explicitly compiles `GhostSpec::defaultPrimary`; it includes the boundary-node-boundary traversal. | The primary builder no longer finalizes ghosts. Completed-state compatibility frontends still exist for callers explicitly requesting a state; they must not be used as intermediate stages of a new owned construction pipeline. |
| Faces and edges | `InterpolateFace` runs after cell permutation and obtains complete parent incidence; periodic collaboration checks distinguish physical subentities sharing canonical nodes. | `NativeMeshTopology::Build` consumes owned topology and gets temporary complete Cell→Node→Cell support itself. It returns owned generated families and never assembles. Existing interpolation tests retain mixed elements, high order, physical periodic slots and distributed ownership. |
| Reconstruction during transition | Existing generated families may be deliberately destroyed and regenerated. | The transition now transfers stripped owned topology, reconstructs owned families, then assembles once. Original publication spaces remain the associations reported for external field validation. This avoids treating detached owned-only maps as original field associations. |
| Owned-node support | Preparation converts N2CB to global, calls `BuildGhostN2CB`, then converts it back; every owned node has complete incident rows. | The default policy explicitly includes owned-node stars. No unconditional inverse path may override the disabled-star option. A focused test caught this regression and its assertions were preserved. |
| Boundary ring | A boundary's reached node can be owned remotely. Distributed path evaluation fetches that node's support. | The assembler receives the actual GhostSpec; separate roots never merge by entity kind alone. The independent triangle oracle uses different boundary/node owners and redistributes nodes. The box probe now needs/stores exactly 266/270 remote boundaries, with no missing/extra rows. |
| Global/local representation | Legacy tracked pairs mutate their index state and use wired target maps. | Owned construction uses global references; assembly creates local compute arrays. All 24 legacy index-test purposes/mesh fixtures remain, using detached ordered conversions and checking unchanged live state. Additional tests exercise dense/padded/CSR shapes, empty owners, unavailable encodings, reorder/redistribution and periodic repeated IDs. |
| Periodic boundary slots | Serial CGNS boundary slots and recovered containment are stage dependent. Recovery fills a periodic missing second slot with self and orders two distinct physical candidates using image bits (`Mesh.cpp`, periodic fixup following checked composition). | Legacy-compatible `Bnd2Cell` retains own/self; `Bnd2CellCoupled` explicitly exposes own/partner, separately from physical containment. Both physical cell slots match the exact primary/prepared IV/box probes. |
| Archive restart | Owned saved topology is read in matching or new rank partitions, then requested ghosts are rebuilt. | Owned native/legacy-compatible decoders never assemble intermediate states. Metadata v2 retains full GhostSpec; matching/even/repartition/custom/malformed and v1 fallback tests pass. Explicit cell depth overrides only standard prefixes. |

## Test-purpose comparison

### Explicit ingress limits versus silent legacy failures

Both source conventions use BC main/donor names and caller-provided transforms;
native CGNS ingress explicitly rejects connection-property periodicity instead
of pretending to interpret it. Node/image components remain limited to the
existing three-bit image model. A main/donor match that maps an original node
onto itself is rejected explicitly. In the frozen deduplication code,
`getNewNodeMap` encodes such a donor as `-1 - itself`, and `track` repeatedly
follows that negative link without a cycle guard. This is a legacy hang path,
not evidence that fixed-axis nodes were supported correctly. No deliberately
hanging legacy run is required to establish the source-level cycle. Native
tests exercise rejection; supporting these extra periodic identifications is
outside the current three-bit canonicalization contract.

The three legacy connectivity suites contain 13/17/17 cases. The
[test-purpose correspondence](mesh_connectivity_test_port_2026-10-28.md) lists
all 47 purposes and the explicit replacements for retired containers. Native
case totals differ as safeguards are added; totals are inventory, not proof.

- Four-quad inverse/round-trip/composition/face-sharing cases retain independent
  expected node stars and neighbors. Fixed-width inversion retains absent-cell
  semantics and additionally exercises a padded physical stride.
- UniformSquare_10 inverse/composition cases import the actual file and compare
  against coordinate-derived lattice incidence; their old reference was legacy
  recovery. This is an independent mathematical oracle, not a comparison against
  another call to the same new routine.
- Legacy ConeAdj/SupportAdj/AdjVariant structural tests target retired container
  types. Native registry/access tests are replacements, not literal unchanged
  tests. Their audited replacements check actual availability/lookup, retained
  lease lifetime, positive typed values/companions and invalid construction.
- Ghost compilation retains anchor/hop-prefix checks, collect flags, breadth-first
  levels, required definitions, malformed-path errors and periodic tiled grids.
  The benchmark sizes and rank-dependent selection remain `{32, 100}` plus 500
  for at most four ranks, matching the frozen source despite its older header
  comment claiming different sizes.
- Interpolation must preserve each of the 17 purposes: eight basic element/mixed
  cases; UniformSquare_10 regression; three small periodic face/edge cases; and
  five distributed hex face/edge cases including periodic axes. Matching names
  alone does not establish that every original assertion remains effective.

## Evidence scope

The final dual-source IV10_10 probe compares O1/O2, primary/prepared states
on one, two, four and eight ranks, including ordered cones/image bits, cell neighbors,
face slots, owned-node support, boundary incidence and exact ghost ID sets.
The initial comparison found 40 periodic boundary rows using own/partner in
native versus own/self in legacy. It also found eight native boundary ghosts
per rank where the independent owned-boundary traversal requires four. The
correction separates `Bnd2CellCoupled` and removes overbroad default paths;
explicit wider coverage remains supported and tested. At historical checkpoint v8, exact
owned/ghost records match legacy for O1 on one/two ranks and O2 on one rank.
O2/two-rank records match only after the complete identity bijection described
below; that historical result was explicitly not exact-ID acceptance.

At v8, O2 on two ranks had different node/boundary labels. General native
refinement creates a new namespace and records ancestry. These are not silently
treated as stable IDs. Compare complete exact-coordinate/image/parent
correspondences explicitly, retain identity differences in the report, and
verify every resulting topology/slot/zone value after the bijection. No rounded
coordinate or partial-intersection mapping establishes equivalence.

Source inspection identified the label mismatch: legacy elevation retains
original Node/Boundary IDs and allocates added Node IDs in a separate global
span range. The previous native setup used the new interleaved row numbering
as every node's persistent label. The setup identity policy now retains the
original labels and allocates new labels above the source high-water. General
fresh-namespace elevation still supports fresh labels. The corresponding new
UT includes sparse large IDs, owned-only elevation/bisection and final assembly.

The **final** probes remove that setup identity mismatch: all eight primary
comparisons (O1/O2 × np=1/2/4/8) match exactly, as do prepared O1/O2 on np=1/2.
Prepared np=4 differs only in ten invalid legacy cell-face frame flags per
order, and np=8 only in eight per order. Native incidence/frame violations are
zero. No ID bijection or numerical tolerance is applied to these final results.

The two-rank UP3D_32 probe has 32768 cells (8192 per active CPU core at np=2,
OMP=2). Native and legacy primary records, both physical boundary-cell slots,
owned-node stars and exact ghost-ID sets match. The boundary traversal needs
and stores 266/270 ghosts on ranks 0/1, with zero missing or extra rows.
Prepared records differ in exactly two legacy cell-face transforms: rank 1,
cell 2783 slot 2 stores 0 where the node frames require 4; cell 728 slot 0
stores 0 where they require 1. Among 196608 incidences, the independent image
check finds two legacy violations and zero native violations. Face node images,
cell slots, zones and all other probed records match. This is a reported legacy
defect, not byte identity.

The source cause is legacy `MeshConnectivity_Interpolate.hxx` Step 6: the peer
message contains `(globalParent, globalFace, subSlot)` only. The peer installs
the owner's generated face ID but retains a transform computed against its
rank-local first parent's frame. Native interpolation includes the owner's
transform as a fourth message word. The implementation comment and per-element
face-slot multiset test document/check this distinction, including remote face
owners and ghost-cell faces. The test has passed on np=1/2/4/8.

The plan repo's `experiments/mesh-construction-correction` contains the common
C++ probe, portable CMake-build helper and independent comparator. Seven
comparator UTs exercise valid input and deliberately bad ownership, incomplete
or ambiguous mapping, changed second slots, wrong transforms and matching
missing boundary ghosts in both sources. Compact reports/manifests are in
`evidence/mesh-construction-2026-10-28`; raw JSON is outside the plan repository.

A standalone probe linked to the frozen legacy libraries now verifies the
actual periodic/elevation ID policy on IV10_10:

| MPI ranks | Elevation | Nodes | Unique IDs | ID range |
| --- | --- | --- | --- | --- |
| 1 | O1 | 100 | 100 | 0–99 |
| 1 | O2 | 400 | 400 | 0–399 |
| 2 | O1 | 100 | 100 | 0–99 |
| 2 | O2 | 400 | 400 | 0–399 |

The previous “historical duplicate IDs” test manufactured sparse retained IDs
through the then-buggy native ingress fixture; it did not reproduce frozen
legacy behavior. That attribution is withdrawn. The test now explicitly replaces
one distinct valid ID with another before import, retaining the corruption
rejection purpose. Probe logs: `/tmp/dnds-owned-20261028-frozen-ids-np{1,2}.log`;
SHA-256 values `ef5a696a82f582090e8ce6019c7959c8df03be26f5278271eca16d740b173b52`
and `042b2888db5d53735d094a4d43034ad2cf87481de31b7f2436e4005b00b7a2c6`.

The frozen legacy three-suite matrix passed 12/12 np=1/2/4/8 CTests. The fourth
native focused build passed 24/24 tests. The fifth build's broader matrix passed
21/24 and correctly exposed a lost rank-option agreement check: retained
coverage caused differing fallback depths to be ignored. The original failing
assertion was retained, the frontend agreement was restored, and an additional
explicit connected-recipe mismatch case was added. The sixth/eighth full Geom
matrices each had seven failures; these exposed depth overrides, a false
historical ID attribution and then stale periodic test/API assumptions. The
thirteenth focused build passes preparation and periodic np=1/2/4/8 (8/8),
including the owned pipeline and owner-frame invariant. The fourteenth full
Geom build and refresh passed. No corrected setup-performance claim follows
from these intermediate results.

The fourteenth full matrix subsequently passed **191/191** in 340.21 s; log
SHA-256 `86dc1419240cf02090383ba384d9f113031e6188bd9c2a8a6487739d7d333526`.
Further audit replaced the availability test's hard-coded key exclusion with
actual registry lookup, added positive typed variable/fixed/image access, and
made weight-branch disagreement fail collectively before coordinate exchange.
Python preparation now uses the combined owned setup path; partition precedes
refinement and preserves original labels. Tests retain source-unchanged checks
while correcting the old assumption that post-bisection input already had faces.

The same audit removed intermediate assembly from completed Partition and
legacy-primary ReadOwned, added native/compatible owned archive decoding, and
kept archive repartition owned until finalization. Existing archive UTs now
assert zero ghosts, writable owned input and no retained mesh scope before
performing their unchanged full restoration checks. The fifteenth broad consumer build was
terminated with status 143 without a compiler diagnostic; completed objects are
reused, but it is not counted as a successful build.

## Final validation and limits

- The sixteenth full rebuilt Geom matrix passed **191/191** in 325.95 s after
  the production corrections above. Later fixture-only changes passed the
  direct connectivity and index-conversion matrix **16/16** at np=1/2/4/8;
  the pipeline's restored strict growth/ownership conditions passed **4/4**.
  The final pipeline rerun also includes the direct boundary-membership check
  and the original NACA benchmark partition conditions.
  The separately refreshed interpolation suite passed **4/4**, explicitly
  rerunning the DSL over owned/appended parents in both real-grid partitions.
- Frozen legacy direct connectivity/ghost/interpolation tests passed **12/12**.
  Source/fixture equivalence and the full purpose/condition ledger accompany
  these results; matching case counts alone were insufficient.
- CFV reconstruction passed **4/4** and its remaining selected reconstruction3D/
  limiter tests **5/5**. Actual Euler evaluator pipeline tests passed **4/4**.
  The earlier target-name regex selected no Euler CTests and is not Euler evidence.
- All four Python modules were rebuilt/installed and all eight build/install
  ELF IDs checked. CPU-profile Geom Python tests passed 68/69/69 per rank at
  np=1/2/4, with 4/3/3 skips respectively: CUDA-only coverage in the CPU profile,
  two optional numba-dependent overset tests, and an extra single-rank skip.
- Refreshed CUDA mesh/device-field/two-GPU overlap/CFV tests passed **4/4**.
  Native-only memcheck passed two cases, 138 assertions and **zero errors**.
  The unfiltered executable passed 20 cases/1213 assertions but the sanitizer
  reported two API errors in legacy CUB and OpenMPI cleanup; it is not a clean
  full-profile memcheck. Native-only isolation used `OMPI_MCA_accelerator=null`;
  ordinary two-GPU tests retained the default MPI profile.
- Final IV and 32768-cell box comparisons retain the explicit legacy-frame
  differences above. They do not establish byte-identical numerical solver
  histories or a controlled setup-performance improvement.

Compact source/log/binary hashes and final reports live in the independent plan
repository's `evidence/mesh-construction-2026-10-28`. Raw probe JSON and complete
logs remain outside that repository. No commit/push was performed in this phase.
