# Legacy VR small-mode SVD correction

The nonzero `svdTolerance` path in `ConstructRecCoeff` previously selected the
inverse filter based on the smallest singular value. It could discard the best
constrained modes and retain weak modes. VR now selects the existing standard
relative filter: retain `sigma > svdTolerance * sigma_max`, with no absolute
cutoff. Equality is discarded. The zero setting still uses Eigen's automatic
SVD solve threshold. The general helper's explicitly requested inverse-filter
mode is unchanged for other callers.

Cases with nonzero tolerances can therefore change numerically. In particular,
the checked-in CRM, Orion and DLRF6 configurations exercise the corrected path;
historical results from those configurations are not bitwise reference results
for this correction. This change is independent of the Runtime/Geom migration.

The inverse helper optionally reports effective rank, raw spectral condition,
retained-subspace condition and relative threshold from the same factorization.
Raw condition is infinity for exactly singular matrices. The old return value
remains the regularized estimate for API compatibility. Zero rank has infinite
retained condition. These values describe the assembled local matrix; for a Gram
matrix its spectrum is squared relative to its least-squares factor.

VR reports MPI-global owned-cell counts for factored cells, filtered cells,
discarded modes and rank-zero cells via `GetInverseDiagnostics()`, plus maximum
raw condition. Counts exclude ghosts and count actual discarded modes even with
Eigen's automatic threshold. OpenMP reductions now protect these accumulators
and the previously racing maximum-condition update. Reporting occurs only at
coefficient setup, not inside reconstruction iteration.

Focused tests cover known spectra, a rotated spectrum, strict cutoff equality,
rank deficiency, zero matrices and the actual VR call path with all modes
discarded. Distributed VR tests check owned-cell counts independent of MPI
partitioning; existing reconstruction fixtures retain their original purposes.

Validation: `dnds_hard_eigen` and `cfv_reconstruction_np1/np2/np4` pass locally
with the configured two OpenMP threads. The focused VR setup regression also
passes at np=4 with `OMP_NUM_THREADS=4`. These are small correctness checks,
not new production-case or performance acceptance evidence.
