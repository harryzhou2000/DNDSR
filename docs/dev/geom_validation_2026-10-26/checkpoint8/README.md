# Native setup, conversion and first legacy retirement

Native setup replacement now stages and validates source state before one
publication, rejects active readers and incompatible destination/source metadata,
and invalidates old entity and field-route descriptors. C++ read/preparation/
archive/boundary helpers and five setup applications use native paths. Archive
repartition keeps the requested ghost depth. The second-ring regression compares
the same ownership against an independently checked Cartesian neighborhood.

`NativeRelation::Convert` and `ConvertedIndex` provide global/local conversion
against exact source/target orders. Compute publications remain local. Tests
check actual current-mesh owned/ghost rows, CSR/table/padded logical shapes,
sentinels, unavailable encodings, strict closure rejection, wrong states,
reorder/redistribution/reclosure, periodic ordered repeated nodes and persistent
IDs deliberately distinct from current global numbering. Python retained reads
report relation state. The shared composition mathematics is now in a neutral
header, outside the legacy ArrayPair DSL.

Legacy serial-output gathering and boundary reconstruction branches are removed.
Native assertions replace unsafe old array dereferences while preserving face
side/multiplicity and node-support consistency. They allow documented incomplete
outer ghost rows and unused nodes. The production Python EulerP frontend requires
native construction; its independent legacy **volume evaluator** baseline remains
test-only and still compares the same 1024 physical-cell RHS values.

Validation of the pre-format source:

- The broad Geom run passed 184/187 registrations. Three distributed-read tests
  dereferenced a legacy array after the now-native helper; replacing that test
  access with a retained native view preserves its local-resolution oracle.
  All four distributed-read MPI registrations then pass. Thus every one of the
  187 registrations is covered; this is not a single all-green 187-test log.
- Conversion and assertion cases pass at MPI 1/2/4/8. The original conversion
  fixture failures (unpublished named key and a 2-D edge request) remain recorded.
- All nine CFV and all 17 Euler CTest registrations pass.
- Rebuilt/installed CPU Python passes 41 tests plus two documented skips per
  rank at MPI 1/2/4/8. All eight loaded extension/library build IDs are checked.
- Fresh CUDA device-view, direct-MPI overlap and entity-field tests pass. The two
  communication tests use exactly one rank per physical local GPU. Their full
  output/UUID mapping is retained in `cuda-last-test.txt`. These are ordinary
  execution results; checkpoint7's nonzero sanitizer finding remains open.

Exact commands, original failed runs, corrected runs, source hashes and binary
identities are retained alongside this file. Logs normalize terminal formatting
and retain original hashes. Five migrated applications pass syntax checks;
their full runtime cases are not claimed here. CPU and CUDA builds share no
installed Python profile during testing.

GN7 is still active. Remaining legacy storage, readers, mutation/DSL producers
and test fixtures must be retired before final acceptance. Draft retirement
packages are not included in this checkpoint's execution claims. Legacy DNDS
Array solver fields and their maps-only native mesh bridge remain supported.
