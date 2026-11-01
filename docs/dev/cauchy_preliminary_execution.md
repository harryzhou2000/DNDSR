# Cauchy preliminary implementation evidence

Implementation started 2026-11-01 from `1a16a781` on
`codex/refactor-dnds-array-geom`. The independent work order is document39 in
`workspace/dndsr-next-plan`; document40 there tracks package completion.

## P0/P1 source organization checkpoint

Canonical paths and historical include overlays are recorded in
[source organization](../architecture/source_organization.md) and its JSON
manifest. All246 moved file bodies matched the previous revision after
excluding include directives, before the normal commit formatting hook.
No public namespace or numerical algorithm changes are intended here.

CPU Release builds completed for `dnds_unit_tests`, `geom_unit_tests`,
`cfv_unit_tests`, and `euler_unit_tests`. The compatibility smoke also rebuilt
and passed with both historical and canonical Array headers. The broader
DNDS/Geom/CFV regression result is recorded at the checkpoint in the plan repo.

CUDA and rebuilt/installed Python validation remain pending integration gates.
This source move checkpoint does not claim a new physical layout or device
operator has been integrated or validated.

## P2 physical layout foundation

`ColMajorTable` and `TiledColMajorTable<W>` add physical uniform-table formats,
with padding options, stride-aware generic/static views, logical iterators,
provider-enforced base alignment and a centralized borrowed Eigen row-matrix
adapter. Array ownership is still typed only by element type. Metadata/layout
and plain view code now live in separate headers from the storage owner.

CPU:26 Array cases/26,868 assertions pass both default unchecked and separately
compiled checked views;71 DNDS CTest registrations pass. CUDA:3 selected Runtime/
algebra tests pass; new column/tiled access and cross-stream aligned async
allocation cases pass memcheck with0 errors (2,324 and1,388 assertions).
The frozen P1 Euler test binaries also pass17/17 tests.

This is an incremental checkpoint. Column/tiled MPI payloads, native I/O and
Python access deliberately reject until their direct layout adapters land in
P3. No silent row-contiguous interpretation is permitted meanwhile. Existing
AoS/padded/CSR interfaces retain their behavior. Incremental RowBuilder and
legacy row bridges accept contiguous physical rows only; callers explicitly
convert a column/tiled table before using those construction/legacy interfaces.

## P3/P4 CPU integration checkpoint

Direct MPI datatype payloads now address all three uniform physical formats;
transient/persistent and explicit HostPacked cases have independent logical
value/padding checks. Custom POD and builtin receive collision reductions pass.
Twelve MPI1/2/4/8 transfer/algebra/archive registrations pass. Four additional
native-I/O registrations pass with direct-pointer spies and independent
opposite-endian custom-POD encodings, now including column/tiled formats.

Native CFV U/URec and matching temporaries/halos accept mixed physical layouts.
Prepared metrics/basis/operator/BC/characteristic arrays keep their previous
format. Sixteen focused MPI1/2/4/8 CFV tests pass, including Jacobi/SOR, WBAP,
linear actions and existing numerical/overlap fixtures. CPU iteration performance
still needs the matched benchmark gate; no parity claim is made from correctness.

After all Python targets rebuilt and installed, the focused CPU Python pass has
45 passed/33 skipped. The MPI2 pass has56 passed/22 skipped per rank. Skips
include CUDA paths under CPU bindings; they are not CUDA acceptance. Twenty-eight
Array cases/26,974 assertions pass after adding explicit conversion and actual
checked-bounds assertion tests. GPU residency/VR/WBAP integration remains P5/P6.
