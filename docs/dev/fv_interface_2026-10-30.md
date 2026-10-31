# CPU FV/interface implementation validation

See the [API and caller contract](../architecture/FVFacility.md). Implementation
starts at refactor commit `c171c318`; existing legacy VR and solver consumers
remain available. The new code uses Runtime storage directly.

## Correctness

OMP=1 local focused C++ tests passed at np1/2/4/8:

* `cfv_test_fv_facility`: six cases covering degree 1–3 and factorial/HQM
  sweep agreement, independent polynomial cell means and point values, six-field
  characteristic WBAP, periodic triangles/quads, relaxed Jacobi/SOR, 3D tet/hex
  cubic recovery, rank-local/collective failures, stale plans, alias/shape checks
  and retained bulk metric reads.
* `geom_test_boundary_interface`: seven cases covering exact coordinates,
  seams and persistent-ID ties, rigid updates, relative sliding with changed
  donors, quarter/full rotations, 3D triangle/quad surfaces, overlap ambiguity,
  gap/coverage/ID failures, empty patches and caller-specified DSL stencils.
  Search is single-rank; unsupported MPI search is rejected. Single-rank-only
  cases return after mesh construction on other rank counts.
* `test/CFV/test_fv_facility.py`: two cases passed at np1/2/4 with `--with-mpi`,
  using rebuilt and installed Python modules. Covers shared allocation lifetime,
  native DNDS algebra field reuse, explicit halos, leases, matrix publication,
  two independent domain identities and interpolation.

Existing `cfv_test_reconstruction` (31 cases) and `cfv_test_limiters` (36 cases)
passed at np1/2 without weakening their assertions. New tests use both the
independent legacy implementation and analytic geometry/polynomial oracles.
No CUDA VR execution, distributed matching or complete flow-solver accuracy is
claimed by these tests.

## Performance

The benchmark target `cfv_benchmark_fv_facility` runs old/new mathematics on the
same mesh and checks their coefficient discrepancy. BSSCA **12405713** completed
14 cases, five trials each, one exclusive AMD EPYC 7452 node, OMP=1, np1/2/4/32.
Source, meshes, executable and MPI installation were checked and recorded.

For 262,144 periodic hexahedra at np32, polynomial degree two, five components:

| Measurement | Legacy | New |
| --- | ---: | ---: |
| Total metric + reconstruction setup | 3.374 s | 1.110 s |
| Per reconstruction sweep including halo | 36.501 ms | 18.476 ms |
| Full order-wise WBAP including halos | 105.495 ms | 51.448 ms |

These are medians, with setup/sweep ordering alternated each trial. Across the
14 cases the new implementation takes 24–42% of legacy setup time, 25–51% of
sweep time and 40–57% of WBAP time. Maximum coefficient difference is
`1.7181e-14`. Comparisons cover the supported isotropic functional; the boundary
derivative weights are deliberately matched, not compared at unlike defaults.

Raw local logs: `data/cfv-interface-20261030/`. Compact hashes, trial records,
test summaries and scheduler evidence are in the private plan repository under
`evidence/fv-interface-2026-10-30/`, summarized by
`docs/30-fv-interface-implementation.md`. Initial MPI-installation and launcher
slot failures are retained and excluded from final timings.
