# Native preparation, periodic import and output checkpoint

This advances native construction and consumer compatibility. GN7 remains open:
production C++ setup, remaining OpenFOAM/smoothing/partition policies, Python
defaults and retirement of the old mesh authority still require integration.
Draft OpenFOAM and smoothing files are excluded from this checkpoint.

## Implemented behavior

Native local cell reordering uses the shared partition/RCM algorithm and returns
explicit transfers for all entity families. It preserves IDs, ordered incidences,
periodic companions and external field association. The small-graph tests exposed
two shared helper defects: disconnected graphs must not request contiguous Metis
parts, and insertion counters must allow more requested parts than cells.
Native preparation supports distributed unweighted KWAY partitioning, empty
owners and a single active owner, then completes the required sparse topology.

CGNS periodic import uses explicit transforms and main/donor BC labels. It
validates high-order node bijections, reconstructs physical images and reserves
removed donor IDs. Elevation, bisection and boundary extraction preserve images.
CGNS GridConnectivity Periodic properties and transform fixed-point node
identification remain explicit exclusions.

Serial PLT/VTU output gathers Runtime payloads only to the writer; parallel
writers consume scoped current geometry. Maps-only adapters connect solver-owned
output arrays. Native primary checkpoint output preserves the old wire schema
without exporting a legacy mesh; full native archives retain their richer
identity and provenance. The old primary archive decoder builds Runtime arrays
directly. It rejects historical duplicate original IDs rather than repairing
them silently. Reconstruction-matrix diagnostics preserve exact face slots and
pull owners' complete ghost rows before serialization.

Allocation accounting includes value and host/device shape buffers, counts each
allocation once, and reports imported backing capacity. Python has explicit
native preparation, partition, local reorder and periodic-reader arguments.
EulerP has an opt-in native setup/evaluator path.

## Validation

`build-refactor` passes all 214 Geom/DNDS CTests and all 26 CFV/Euler CTests with
Open MPI 5.0.9 and `OMP_NUM_THREADS=2`. The registered MPI cases cover 1/2/4/8
ranks. These broad runs precede the narrowly scoped EulerP entropy-input fix
described below. The final Riemann regression and rebuilt Python parity results
are recorded separately. The broad runs are not claimed as a rebuilt execution
of that later correction.

The four CPU Python modules were built, installed, and checked against all eight
extension/shared-library ELF IDs. The helper/periodic matrix passes six tests
plus one expected skip at np1, and seven per rank at np2/4/8. A test incorrectly
attempted to pickle a bound BC-map object; it now compares the map's dictionary.

CUDA Runtime allocation accounting passes on physical GPU 1. The full CUDA mesh
facade suite passes there (11 cases, 807 assertions). Fresh direct-MPI overlap
and field-migration executables pass at np2 on two distinct A100 GPUs, with one
rank per GPU. UUIDs, commands, source and binary hashes are in the two-GPU
provenance file. This establishes correctness and concurrent in-flight work,
not a network speedup measurement. Native topology setup remains on the CPU.

## EulerP orientation defect found by the new scenario

The first nonperiodic evaluator fixture reached an existing unimplemented EulerP
physical boundary condition and aborted. The replacement fixture exercises a
fully periodic mesh with a nonconstant state. Native and legacy geometry,
primitive fields, reconstructed face states and signed normals agree; their RHS
initially differed by about 2.4e-5 because valid face orientation reversals changed
EulerP's entropy dissipation.

EulerP passed signed normal velocity to its Harten-Yee threshold. The established
Euler scheme uses total velocity magnitude. The correction passes the Roe total
speed and documents that contract; Euler's formula is unchanged. This is an
intentional EulerP numerical bug fix. The independent regression checks a
tangential-speed-dominated threshold, equality with Euler scheme 8, reversal of
left/right state and normal, rigid rotation covariance, and failure of the old
signed-input convention. The numerical parity tolerance is unchanged.

The corrected single-rank/two-rank runs exposed a second defect at np4/8:
`CalculateOneRHS(zero_grad=True)` started a gradient halo transfer before zeroing
its send/receive arrays. Owner rows became zero while late receives restored
nonzero ghost gradients. An independent owner/ghost catalog measured errors up
to 1.5 even though the intended gradient was zero everywhere. The fix zeros
gradients before starting the transfer; the regression checks every owned and
ghost gradient as well as the final RHS. This changes only the invalid ordering
of existing operations and preserves the communication overlap afterward.

After both corrections, the complete eight-test Python matrix passes seven
tests plus one expected skip at np1, and eight per rank at np2/4/8. It includes
all 1024 periodic cells, RHS tolerance 1e-10, and exact zero gradients on every
owned/ghost row. The rebuilt Riemann executable passes 18 cases/251 assertions.
The final installed CPU artifacts are verified again; exact commands, exit
statuses and source hashes are in `geom-native-batch4-entropy-matrix.json`.

## Evidence and reproduction

```sh
cmake --build build-refactor --target geom_unit_tests dnds_unit_tests cfv_unit_tests euler_unit_tests -j12
OMP_NUM_THREADS=2 ctest --test-dir build-refactor -R '^(geom_|dnds_)' --output-on-failure -j2
OMP_NUM_THREADS=2 ctest --test-dir build-refactor -R '^(cfv_|euler_)' --output-on-failure -j2
cmake --build build-refactor --target euler_test_riemann_solvers dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j10
cmake --install build-refactor --component py
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 ctest --test-dir build-refactor-cuda -V -R '^geom_mesh_(facade_overlap|entity_field)_cuda_np2$' --output-on-failure
```

The checked-in logs retain earlier failures separately. These include a CUDA
test accessing a private query method, a serial-output test missing field
attachment, one run against its pre-relink binary, and the initial EulerP
boundary/parity failures. They are diagnostic evidence, not passing tests.
`precommit-artifacts.json` records pre-hook source hashes and available binary
identities. A listed binary is not proof it was exercised. Text normalization
removes terminal escapes, NUL MPI-version terminators and trailing whitespace;
the normalization manifest records original and stored hashes. No production
solver trajectory or physical accuracy claim is made.
