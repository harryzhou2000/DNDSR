# Legacy connectivity test-purpose correspondence, 2026-10-28

Baseline `03891f8b5869017de355468d93294e3eb25f0feb`. All legacy
case purposes are listed below; equal names/counts are not proof. **Fixture and
condition coverage must at least include the legacy coverage.** Source anchors
refer to the frozen baseline and the active refactor checkout. Tests retain
np=1/2/4/8 execution, including empty ranks. This ledger must accompany the
actual rebuilt test result; it is not a substitute for execution.

The native setup DSL retires ConeAdj/SupportAdj/AdjVariant containers. Four
container cases are contract replacements, explicitly marked below. Numerical
incidence expectations are not waived. The real-mesh regression cases use
independent coordinate/lattice oracles rather than comparing a wrapper to the
same implementation. Every basic interpolation assertion (entity totals,
internal/shared support counts, element types, cone widths, expected vertex
sets and parent-slot membership) is retained with native access spellings.

## Fixture and condition floor

The frozen `test/cpp/Geom` files are content-checked against the baseline,
alongside Geom/DNDS/CFV sources. The following are input conditions, not merely
test-name correspondences. Additional native cases do not replace them.

| Legacy fixture / conditions | Native preservation and extra checks |
| --- | --- |
| Four explicit Quad4 cones over nine nodes | Same ordered cones and ownership: np=1 owns all; otherwise rank 0 owns cells 0/1 and nodes 0–4, rank 1 owns cells 2/3 and nodes 5–8; higher ranks are empty. Explicit owned-count canaries accompany the nine exact inverse star sets. |
| Manually supplied ghost-cell cones | Ranks 0/1 append cells 2/3 and 0/1 respectively; higher ranks append none. Explicit ghost-row counts/total shapes are checked before traversal. One/two-hop and union results retain exact cell/node IDs and totals 6/7. |
| UniformSquare_10 real regressions | Actual 100-cell/121-node/40-boundary CGNS input, reader rank 0, AutoAppendName2ID, periodic-identification tolerance 1e-8, METIS KWAY ufactor 30 / seed 42 / ncuts 1, primary redistribution and one cell ring. Inverse, composition and interpolation each run this variant **and** an additional direct-even variant. Interpolation separately runs the DSL over owned and appended parents, fetches resulting remote faces, and checks every parent against independent polygon edges; the prepared mesh also passes a coordinate-based oracle. |
| Fixed-width two-parent face input | Same five faces, two cells, exact absent second-parent slots, and expected inverse sets. An additional padded-stride variant must also pass. Standalone legacy width narrowing retains its six values. |
| Eight elementary interpolation fixtures | Same ordered Quad4, Tri3, Tet4, Hex8 and mixed Tri3/Quad4 payloads and node totals. Rank 0 owns the handcrafted payload; other ranks are empty collective participants. Original face/edge totals, support counts, types, widths, explicit vertex sets and parent slots remain assertions. |
| Small periodic fixtures | Reader-owned 2×2 doubly periodic quads and 2×2×2 triply periodic hexes, repeated canonical node IDs and ordered image bits. Both image-unaware false merging and image-aware correct physical subentities are exercised; the deliberately wrong seven-neighbor composition remains a negative control. |
| Distributed hex fixtures | N=4 cells in each local direction, x-slab ownership, last-rank end-node slice for nonperiodic input, wrapped ownership for x-periodic input; one neighboring cell slice on each available side. Nonperiodic/x-periodic faces and edges and triply periodic faces are retained; triply periodic edges and exact owner-frame checks are additional. |
| Tiled ghost traversal | Same Cartesian rank tiling, eight-neighbor cell adjacency and exact wrapped/clipped expected sets. N=32/100 and N=500 when np≤4; periodic N=32 with depths 1–4, plus N=2 wrap-around degeneracy. No reduced benchmark fixture substitutes for the original sizes. |
| Invalid/omitted definitions | All four malformed chain constructions, absent boundary-hop registration, wrong family/state, absent-slot handling and invalid construction checks remain explicit failures. Availability uses actual registry lookup, not a hard-coded list of missing keys. |
| Execution conditions | All three suites run np=1/2/4/8, including empty owners and serial no-ghost behavior. A pass with skipped MPI tests or stale installed bindings does not count. |

The final fixture audit found and corrected three earlier coverage regressions:
direct-even ownership had replaced reader/METIS setup; index tests had relaxed
owned-node/owned-boundary-parent assertions; and the pipeline's multilayer test
had retained monotonicity but dropped strict cell/node growth for np≥2. The
original conditions/assertions are restored, with additional native coverage
kept separately. These corrections change tests, not the production algorithm.

## Index-conversion and pipeline conditions

`test_MeshIndexConversion.cpp` retains all 24 legacy case purposes across
UniformSquare_10, IV10_10, NACA0012_H2 and IV10U_10 at np=1/2/4/8, plus Ball2
(958994 mixed 3D cells) at np=8 exactly as in the baseline. Reader-rank input,
name mapping, tolerance and KWAY 30/42/1 settings now match the original. The
fixture uses owned partition followed by one explicit final assembly. The
original `NumNode() >= 1` and owned boundary's first parent `< NumCell()` checks
are restored; native APIs still permit independently owned or empty node
families in other fixtures.

The conversion purposes include all owned/ghost/owned-only mappings, unknown
global indices, UnInitIndex through all twelve methods, `-1-global` decoding,
valid local ranges, stable primary round-trips repeated three times and the
boundary-only conversion subset. Published states are immutable: mutable group
round-trips are replaced by detached native conversions with exact ordered
value checks, explicit state checks and proof that unrelated live relations
remain unchanged. This is an API-contract replacement, not a retained mutable
group-state implementation.

`test_MeshPipeline.cpp` retains the same five files, periodic transforms,
rank selection and reader/METIS settings. Multilayer fixtures retain depths
1/2/3, nondecreasing cell/node/boundary counts, **strict** cell/node increases
between successive depths for np≥2, full two-hop local reachability and fixed
global cell count. Full relation/periodic slot checks use element incidence;
reassembly equality is an additional consistency check. Boundary membership
is also checked directly by walking owned boundary nodes and complete inverse
rows, so identical missing storage in two assemblies cannot pass that check.
No global-sized per-rank oracle table is introduced.

## test_MeshConnectivity.cpp (13 legacy cases)

| Legacy case / baseline line | Native line | Preserved purpose / explicit replacement |
| --- | --- | --- |
| Inverse: 4-quad serial correctness (L33) | [L19](../../test/cpp/Geom/test_MeshConnectivity.cpp#L19) | Nine explicit node-star sets for the four-quad fixture. |
| Inverse: round-trip covers original cell2node (L84) | [L31](../../test/cpp/Geom/test_MeshConnectivity.cpp#L31) | Every original incidence survives inverse-of-inverse exactly once. |
| ComposeFiltered: 4-quad cell2cell via node-neighbor (L150) | [L45](../../test/cpp/Geom/test_MeshConnectivity.cpp#L45) | Every quad has the other three as node neighbors; self excluded. |
| ComposeFiltered: face-share filter (minShared=dim) (L247) | [L61](../../test/cpp/Geom/test_MeshConnectivity.cpp#L61) | Exact four expected edge-sharing neighbor sets; diagonal contact excluded. |
| Regression: Inverse matches RecoverNode2CellAndNode2Bnd on UniformSquare_10 (L356) | [L108](../../test/cpp/Geom/test_MeshConnectivity.cpp#L108) | Actual UniformSquare_10 input; each node star independently derived from lattice coordinates, replacing legacy recovery as the oracle. |
| Regression: ComposeFiltered matches RecoverCell2CellAndBnd2Cell on UniformSquare_10 (L408) | [L134](../../test/cpp/Geom/test_MeshConnectivity.cpp#L134) | Actual UniformSquare_10 input; exact eight-neighborhood clipped at domain edges. Original case name mentions boundaries but its assertions only compare CellCells. |
| MeshConnectivity: cone management (L506) | [L164](../../test/cpp/Geom/test_MeshConnectivity.cpp#L164) | Absent/present lookup, from/to kind, stable found object, second cone and count. New duplicate/missing-definition rejection strengthens the retired container purpose. |
| MeshConnectivity: support management (L527) | [L190](../../test/cpp/Geom/test_MeshConnectivity.cpp#L190) | Inverse definition from/to, retained lookup identity and count; semantic inverse intentionally differs from the legacy facade key. |
| ConeAdj: AdjVariant typed access (L544) | [L211](../../test/cpp/Geom/test_MeshConnectivity.cpp#L211) | Retired variant/default-null internals are replaced by construction-time family/state validation and typed companion association. Compile-time checks forbid uninitialized relation construction; valid variable/fixed-width access, absent slots and actual ordered image values are exercised. |
| SupportAdj: no pbi member (L568) | [L274](../../test/cpp/Geom/test_MeshConnectivity.cpp#L274) | Support carries sorted unique entity membership, separately from optional cone occurrence images. Retired SupportAdj null/fatherSize internals have no native counterpart. |
| Inverse<2>: fixed-width face2cell input (L583) | [L288](../../test/cpp/Geom/test_MeshConnectivity.cpp#L288) | Fixed-width two-parent faces, exact absent second slot, inverse sets {0,1,2}/{1,3,4}; padded physical stride additionally exercised. |
| narrowAdjToFixed: variable-to-fixed-2 conversion (L651) | [L331](../../test/cpp/Geom/test_MeshConnectivity.cpp#L331) | Unchanged standalone generic-pair width conversion; all six values including absent second slot. |
| Periodic 2x2x2: ComposeFiltered cell2cellFace is WRONG without pbi filter (L684) | [L311](../../test/cpp/Geom/test_MeshConnectivity.cpp#L311) | Periodic torus node-count composition deliberately produces seven wrong neighbors; separate physical interpolation tests establish three distinct correct neighbors. |

## test_MeshConnectivity_Ghost.cpp (17 legacy cases)

| Legacy case / baseline line | Native line | Preserved purpose / explicit replacement |
| --- | --- | --- |
| GhostChain: defaultPrimary compiles (L112) | [L611](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L611) | Both anchor roots, connected boundary path, levels/collect flags and exact four required descriptors. |
| GhostChain: prefix merging (L190) | [L689](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L689) | One Cell root, shared first hop, nested ring levels, flags and leaf structure. Other-anchor continuations must not merge. |
| GhostChain: invalid chain detection (L218) | [L717](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L717) | All four original malformed chain cases throw: empty, wrong anchor, wrong target and disconnected hops. |
| GhostChain: checkAvailable (L245) | [L822](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L822) | Actual registered definitions must establish the two missing boundary hops; Evaluate must reject the incomplete recipe. Missing keys are computed through registry.Find; the earlier hard-coded exclusion was removed. The refreshed np=1/2/4/8 matrix passes. |
| GhostChain: dump (L265) | [L744](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L744) | Original metadata assertions retained: names, hash/equality, dimension depths, dump content or face-chain shape as named. |
| GhostChain: face ghost chain (L277) | [L756](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L756) | Original metadata assertions retained: names, hash/equality, dimension depths, dump content or face-chain shape as named. |
| GhostChain: AdjKind equality and hash (L300) | [L779](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L779) | Original metadata assertions retained: names, hash/equality, dimension depths, dump content or face-chain shape as named. |
| GhostChain: entityDepth (L319) | [L798](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L798) | Original metadata assertions retained: names, hash/equality, dimension depths, dump content or face-chain shape as named. |
| GhostChain: adjKindName (L334) | [L813](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L813) | Original metadata assertions retained: names, hash/equality, dimension depths, dump content or face-chain shape as named. |
| evaluateGhostTree: performance benchmark (L573) | [L432](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L432) | Execute equivalent sparse traversal with N=32/100 and N=500 for np<=4. Benchmark timing is not a correctness acceptance threshold. |
| evaluateGhostTree: single hop cell2cell (L642) | [L28](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L28) | Exact ordered remote cell IDs 2/3 versus 0/1, collective active-kind flags and empty ranks. |
| evaluateGhostTree: two-hop cell2cell2node with manual ghost (L692) | [L75](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L75) | Exact cell and node sets, totals 6/7 on the two active ranks; empty ranks remain empty. |
| evaluateGhostTree: 2-ring cell chain (L781) | [L130](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L130) | Tiny fixture one-ring and two-ring remote sets coincide exactly. |
| evaluateGhostTree: union of multiple chains (L824) | [L157](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L157) | Union traversal has exact expected node sets on active ranks and none on empty ranks. |
| evaluateGhostTree: np=1 produces no ghosts (L909) | [L213](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L213) | Single-rank traversal has no ghost cells and total zero. |
| evaluateGhostTree: doubly-periodic tiled grid (L937) | [L507](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L507) | Independent wrapped-lattice expected cell/node counts across ghost depths, including tiny periodic degeneracy. |
| evaluateGhostTree: small periodic tiled grid (L999) | [L564](../../test/cpp/Geom/test_MeshConnectivity_Ghost.cpp#L564) | Independent wrapped-lattice expected cell/node counts across ghost depths, including tiny periodic degeneracy. |

## test_MeshConnectivity_Interpolate.cpp (17 legacy cases)

| Legacy case / baseline line | Native line | Preserved purpose / explicit replacement |
| --- | --- | --- |
| Interpolate: 4-quad faces (2D) (L27) | [L96](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L96) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Interpolate: 2-tri faces (2D) (L123) | [L190](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L190) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Interpolate: single tet faces (3D) (L174) | [L239](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L239) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Interpolate: two tets sharing a face (3D) (L216) | [L279](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L279) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Interpolate: single hex faces (3D) (L257) | [L318](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L318) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Interpolate: single tet edges (3D) (L294) | [L353](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L353) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Interpolate: two-tet shared edges (3D) (L335) | [L392](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L392) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Interpolate: mixed tri+quad faces (2D) (L383) | [L438](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L438) | All original explicit totals, shared/internal counts, element types, widths and expected vertex/support sets retained. |
| Regression: Interpolate matches InterpolateFace on UniformSquare_10 (L432) | [L955](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L955) | Actual UniformSquare_10 input. Each cell has exactly the independently constructed polygon-edge set; no same-routine reference comparison. |
| Interpolate: 2x2 periodic quad mesh — collaborating check required (L535) | [L1029](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L1029) | Original periodic count/support/multiplicity expectations, including deliberate false merging without image checks. Physical image disambiguation retained. |
| Periodic 2x2x2: face interpolation (3D, collaborating check) (L677) | [L1084](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L1084) | Original periodic count/support/multiplicity expectations, including deliberate false merging without image checks. Physical image disambiguation retained. |
| Periodic 2x2x2: edge interpolation (3D, collaborating check) (L763) | [L1165](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L1165) | Original periodic count/support/multiplicity expectations, including deliberate false merging without image checks. Physical image disambiguation retained. |
| InterpolateGlobal: 4x4x4 distributed hex faces (non-periodic) (L832) | [L485](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L485) | Original analytical global total, boundary/internal support counts where applicable, type/width checks, support degree and all resolved parent-slot IDs in the correct global range. |
| InterpolateGlobal: 4x4x4 distributed hex faces (X-periodic) (L933) | [L579](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L579) | Original analytical global total, boundary/internal support counts where applicable, type/width checks, support degree and all resolved parent-slot IDs in the correct global range. |
| InterpolateGlobal: 4x4x4 distributed hex edges (non-periodic) (L1022) | [L659](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L659) | Original analytical global total, boundary/internal support counts where applicable, type/width checks, support degree and all resolved parent-slot IDs in the correct global range. |
| InterpolateGlobal: 4x4x4 distributed hex edges (X-periodic) (L1110) | [L741](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L741) | Original analytical global total, boundary/internal support counts where applicable, type/width checks, support degree and all resolved parent-slot IDs in the correct global range. |
| InterpolateGlobal: 4x4x4 distributed hex faces (triply-periodic) (L1196) | [L934](../../test/cpp/Geom/test_MeshConnectivity_Interpolate.cpp#L934) | Both faces and edges: exact global totals, no boundary support, full parent widths, first-parent stored nodes/images and every owned occurrence frame, including remote entity owners. Multi-axis corner images checked through exact physical coordinates. |

## Additional safeguards

Native registry lookup retains an immutable row lease after registry destruction.
Owned-construction tests redistribute nodes away from boundary owners before
face construction and check the connected boundary ring independently. Native
periodic tests compare image-labeled node multisets at each element face slot.
The plan repository contains an independent dual-source probe comparator with
negative tests for duplicate owners, ambiguous/incomplete ID mapping, a missing
boundary ring in *both* inputs, changed second boundary slots and wrong frame
flags. A matching corrupted reference is therefore insufficient for acceptance.
