# Focused Geom reorder audit, 2026-10-24

This closes a bounded GE0/GE4 mesh-reorder gap. It does not provide staged,
atomic publication of a mesh plus solver field bundle, nor universal preserved
edge/face transition coverage.

## Reproduced causes

1. The registry registered `cell2cellFace` but its separate hardcoded target
   pull-set inventory omitted that relation. A synthetic next-rank cell
   reference reproduced `LookupResult::resolve: old global 1 not found` at
   two ranks. Pull-set collection now follows the actual registered relation
   inventory, including only owned source rows. `ReorderPlan::apply` remaps
   those rows **before** relocation: arriving rows already contain new target
   IDs. The earlier comment attributing failure to unresolved arriving old
   face IDs did not describe the current execution order.
2. Reorder resets relation sons. Converting facial/C2F indices to local form
   without rebuilding their ghosts left owned cells without usable face
   references. `BuildGhostFace` now restores cell/boundary relation ghosts and
   their PBI companions using the current primary correspondence. Rebuild
   tests call it explicitly after primary ghost reconstruction.
3. Independent boundary ownership left adjacent cells outside the primary
   ghost set. Owned retained faces similarly needed cell/node/boundary
   closure. The mesh primary builder now requests these relations and makes
   boundary labels available to adjacent cell/face owners. Fresh face
   interpolation also requests faces incident to owned boundaries' cells.
4. Boundary matching used only owned boundary rows, leaving owned faces
   incorrectly classified when their boundary metadata belonged elsewhere.
   It now considers available ghost boundary metadata while retaining
   owned-only boundary side vectors. Periodic side selection uses adjacent
   global cell IDs rather than rank-dependent local/ghost ordering.
5. An unmatched periodic boundary's local side-vector sentinel `-1` was
   copied into tracked `bnd2face`. Local-to-global conversion interpreted
   this as encoded global face zero, producing a false face/cell dependency.
   Tracked absent entries now use `UnInitIndex`; the local side vector keeps
   its negative-sentinel convention.

## Evidence

All paths below are relative to `docs/dev/geom_validation_2026-10-24/`.

| Evidence | Log |
|---|---|
| Missing independent face-neighbor target lookup, before fix | `reorder-coverage-before.txt` |
| Former face identity pipeline missing usable face ghosts | `reorder-face-before.txt` |
| Node/Bnd follow missing boundary-adjacent primary cells | `reorder-follow-before.txt` |
| First closure iteration: periodic encoded face zero and boundary classification failures | `reorder-face-after-np2.txt`, `reorder-follow-after-np2.txt` |
| Diagnostic periodic identity showing encoded cell zero (`-1`) | `reorder-periodic-initial-check.txt` |
| Full-suite detection of the older rebuild test omitting face ghost reconstruction | `reorder-full-iteration5-np2.txt`, `reorder-full-iteration5-np4.txt` |
| Final target build | `reorder-build-final.txt` |
| Final focused two-rank face/lookup and Node/Bnd follow runs | `reorder-face-final-np2.txt`, `reorder-follow-final-np2.txt` |
| Final complete reorder executable, one/two/four ranks | `reorder-full-final-np1.txt`, `reorder-full-final-np2.txt`, `reorder-full-final-np4.txt` |

The final complete reorder executable exited zero at `np=1,2,4`, with
`OMP_NUM_THREADS=2` and a 120-second bound per invocation. Each rank reported
37 passing test cases. Assertion totals were 2,639,781 at one rank; 23,866 and
25,088 at two ranks; 12,152, 12,424, 12,560 and 12,583 at four ranks.
Existing rank guards inside older tests still apply; doctest's zero skipped
test-case count does not mean every old test body executed on every rank.

## Validated scope and limits

The new preserved-face fixture runs without rank skips at one, two and four
ranks on `UniformSquare_10.cgns` and periodic `IV10_10.cgns`. For each mesh it
checks identity, round-robin cell movement with unchanged face ownership,
and round-robin cell movement with an independent cyclic face-owner shift.
Node and boundary ownership remain explicit identity in this fixture.
Every owned `cell2node`, `cell2face`, `face2node`, `face2cell`, `face2bnd`,
`bnd2face` and `bnd2node` row is compared by exact transfer correspondence and
relation-slot order before and after the complete primary/facial/inverse
ghost and local/global conversion roundtrip. Periodic cell-node, cell-face
and face-node PBI bytes are compared in the same slots. Absent periodic
boundary-face references must remain absent through global conversion.

The Node-explicit/Bnd-follows-Node fixture now performs inverse recovery,
primary and inverse ghost construction, fresh face interpolation and face
validation at one, two and four ranks. The independent `cell2cellFace`
fixture verifies target coverage without relying on another relation
incidentally referencing the same remote IDs.

`AssertOnFaces` retains owned-cell face count and reciprocity checks, checks
face nodes/cells against the reconstructed primary closure, and additionally
checks each owned face occurs in its adjacent cells' relation rows. An owned
external face may reference a ghost cell after independent redistribution.
This validates mesh incidence; it does not certify numerical solver loops
that impose stronger ownership assumptions. Full preserved-edge rebuilds,
arbitrary unregistered solver state, exception rollback and atomic staged
mesh/field publication remain outside these results.
