# Native defaults, transitions and affine geometry

Python mesh reading and EulerP now default to native construction. Preparation
keeps the caller's mesh object, serial readers are native output adapters, and
archive matching reads preserve saved ownership unless repartition is requested.
Reader mismatch and rank-local archive failures reject collectively and permit
retry. Native edit/transition candidates no longer clone a legacy mesh. Face or
edge reconstruction preserves retained families and assigns new IDs above the
previous high-water mark, including after archive restart.

The affine geometry operation publishes coordinates and periodic frames together,
rejects singular/reflected/nonfinite or inconsistent transforms and nonrigid
conjugated periodic rotations, and invalidates wall/device geometry. Independent
quad/hex physical-corner, volume and Jacobian checks preserve the original CFV
shear fixture. That wave-analysis fixture is serial: its one-free-cell algebra
is explicitly skipped on multiple ranks; distributed affine checks run in the
Geom Python and C++ suites at every registered rank count.

All **175 Geom CTest registrations passed**, including MPI 1/2/4/8. The earlier
focused run had seven failures from two test assumptions: republishing an
incomplete legacy cache oracle, and treating global output offsets as local.
Their corrected eight registrations pass separately and the broad final run
passes. The original logs remain here. Independent legacy numeric baselines are
confined to a test-only codec; this does not yet remove production legacy storage.

CPU Python passes **41 tests plus two skips per rank at MPI 1/2/4/8**.
One skip is CUDA-only; the other is a multi-rank mixed-authority test at MPI 1
and the serial wave fixture at MPI 2/4/8. Four bindings were built/installed and eight actual library
build IDs verified. The initial MPI wave-fixture failure is preserved. Commands
and statuses are in the matrix JSON. Source hashes describe the pre-hook source.
The subsequent archive depth fix forwards the requested ghost depth through
repartition. After rebuilding/reinstalling Geom and re-verifying all eight IDs,
the focused helper and wave fixture matrix passes 11 tests plus one expected
skip per rank at MPI 1/2/4/8. Python routes archive repartition separately; the
C++ combined read-and-repartition depth regression belongs to the next package.

No fresh DNDSR CUDA run is claimed for this checkpoint. The earlier nonzero
sanitizer registration finding now has a tiny independent reproducer, including
a plain CUDA two-translation-unit template with a real cross-file device call.
RDC reproduces the registration error; whole-program/single-file controls do not.
This establishes an external toolchain interaction, not harmlessness or a proven
fix. The complete small-source/log/provenance record is in
`cuda-registration-reproducer/`; no suppression or project RDC change was made.

GN7 remains open: replace remaining setup helpers/applications, retire legacy
mesh storage/construction, migrate the remaining independent test purposes and
run final CPU/Python/two-GPU gates. Legacy DNDS Array solver fields remain supported.
