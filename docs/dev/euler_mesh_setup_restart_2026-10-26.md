# Euler mesh setup and legacy restart correction

This follow-up to `071542f3` restores the source-setup ordering from the
`03891f8b` Euler implementation while retaining native mesh authority. It does
not change the Runtime Array archive format or reintroduce legacy mesh storage.

## Setup order and geometry

Source readers now partition/distribute before optional order elevation and
each distributed elevate/bisect pass. Refinement retains parent ownership;
there is no unconditional repartition after the derived mesh is built.
The setup helper and legacy `BuildO2FromO1Elevation` entry point request
preservation of numeric Cell IDs during elevation, matching legacy
`cell2cellOrig`. General native elevation retains its existing
fresh-ID default; bisection still creates rank-prefix child IDs. All derived
meshes retain the native new-namespace/provenance contract.

The common Euler sequence remains preparation/reordering, optional elevated
node smoothing, optional wall-distance construction, and the partition-only
save/return. Normal solver setup then extracts the boundary mesh, rotates,
scales, snaps near-plane/symmetry coordinates, refreshes invalidated wall
vectors, rebuilds output descriptors, and initializes FV/evaluator state.
Partition-only output therefore keeps geometry and wall vectors before the
solver-only transforms. A loaded wall field is refreshed after an actual snap
even when `meshBuildWallDist=false`; a coordinate no-op keeps its cache.

`RectifyNearPlaneCoordinates(mask,threshold)` has explicit canonical-owned-node
semantics: clean the selected components once on the owner, scatter ghosts,
and retain periodic image labels and frames. This supports periodic meshes
without interpreting an arbitrary callback in multiple image frames. General
nonlinear `TransformCoords` callbacks remain rejected on periodic meshes:
an arbitrary function need not respect their transformations. Actual coordinate
edits invalidate wall geometry; no-op rectification does not publish a new epoch.

## Legacy Euler solution reads

H5 reads prefer existing `u/origIndex`. If it is absent, the Euler-specific
fallback reads temporary legacy `u/father` and `cell2cellOrig/father` arrays
using the same EvenSplit row partition. It checks paired offsets/counts,
nonnegative unique keys, equal global cell counts and a bijection with current
mesh keys, then redistributes values using the existing sparse-ID directory.
This supports a different current MPI size without global-sized local buffers.
Missing identity metadata, duplicate keys and missing cells are errors before
the destination solution is replaced. Ordinary and selected-variable reads
share this fallback.

These keys must belong to the same cell identity domain. Importing a legacy
mesh adopts its provided `cell2cellOrig`; array positions cannot establish the
association for an independently constructed/refined mesh. Historical
rank-prefix child IDs are particularly dependent on their original refinement
partition. Loading the corresponding mesh archive preserves that association.

JSON remains a matching-rank partition format with local reordering. Duplicate,
negative or mismatched provided keys fail before assignment. The existing
metadata-free JSON positional fallback remains available with its warning;
the caller must establish identical row ordering for that legacy case.
Selected-variable JSON reads accept father-only checkpoints, reorder initialized
temporary data and copy only the requested components. Unselected components
remain attached to their current cells. Invalid variable selections are errors.

## Validation

Focused MPI geometry tests cover translated/rotated periodic frames, ghost
coordinate agreement, no-op publication, conflicting readers, rank-disagreeing
parameters, wall-vector refresh, partition-before-refinement ancestry and sparse
Cell IDs retained through setup elevation. Euler tests additionally exercise
old H5 wire data without `origIndex`, sparse high IDs, empty writer ranks,
one-writer checkpoints, different rank counts, fixed/dynamic source variables,
JSON local reorder, selected components, malformed keys and successful retry.

Final results: 24/24 CPU MPI registrations and 4/4 ordinary CUDA registrations
pass. Exact commands, source/log hashes, scope and retained initial fixture
failures are in the [validation manifest](euler_mesh_setup_restart_validation_2026-10-26/manifest.json).
Python bindings were not retested in this C++ follow-up; rebuild/install before
testing them against current sources. This is a compatibility regression suite,
not a claim of bitwise equivalence across every historical mesh/configuration.
