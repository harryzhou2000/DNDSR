# Native geometry, smoothing, output and original DSL tests

This package removes duplicate legacy publication branches for rigid geometry,
boundary rectification, wall vectors and mesh output. Elevation, bisection and
the three production smoothing frontends use native implementations. The unused
compact-support predecessor explicitly rejects rather than silently selecting
another numerical method. Native wall/output mathematics and maps-only external
solver-field schedules remain. No complete legacy mesh is created by these paths.

The original index, connectivity, ghost and interpolation suites now use native
entities/relations. All 24/13/17/17 cases are retained, with exact independent
incidence, composition, closure, periodic-frame and physical-point oracles.
One narrow generic legacy Array conversion case remains an explicit field-level
compatibility test. Historical duplicate-ID archive validation constructs the
known old ID pattern independently in a wire payload, without retaining an old
mesh/elevation algorithm. Retained smoothing diagnostics replace legacy scratch
arrays. The audit exercises actual zero-displacement workspaces and unchanged
coordinates; it does not copy retired field algebra into a test-only product.

All **187 Geom registrations** are covered by 76 initial passes, 103 resumed
passes, and the four corrected index plus four mutation registrations. Final
audit cases pass separately after the diagnostics update. The initial broad run
was interrupted only for two inefficient index test processes: constructing a
fully validating reader inside row loops made them quadratic. Final tests retain
one reader per loop, with unchanged meshes/assertions. The original failure and
stack evidence remains. Other corrected test assumptions were an actually valid
2-D elevation destination and a rank allowed to own no nodes while still having
complete local node closure. They did not require production algorithm changes.

The full eight-rank Ball2 index test passes all 24 cases on 958,994 mixed cells
in 138.26 seconds. `/usr/bin/time` reports maximum RSS 1,580,784 KiB and no swaps;
the sum of eight separately observed rank high-water values is 8,545,632 KiB,
which is **not** a concurrent peak measurement or a scaling proof. MPI 1/2/4
passes separately. No mesh was capped or removed to obtain these results.

The changed CFV reconstruction suite passes at MPI 1/2/4/8, all 29 cases per
rank. Tests now use native graph preparation and a valid temporary no-face
publication for stale-basis/coefficient/metrics failure and explicit rebuild.
Its initial retired-call and invalid test-identity failures remain recorded;
numerical tolerances are unchanged.

Build and test commands, original and final logs, source hashes and binary IDs
are retained alongside this record. Source hashes precede the formatting hook.
No fresh Python, CUDA or whole Euler run is claimed for this checkpoint; their
checkpoint8 results are historical. The next storage-free facade/ownership/
Python packages are still drafts and are excluded from these execution claims.
GN7 remains active until the remaining legacy mesh payloads and producers are
removed and the final CPU/Python/two-GPU matrix passes.
