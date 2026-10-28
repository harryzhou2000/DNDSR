# Runtime, algebra, legacy storage and unstructured mesh sources

The infrastructure finish phase separates implementation responsibilities without
changing numerical behavior or promoting Runtime symbols into the root DNDS
namespace. The exact relocation manifest is
[source_path_map.tsv](source_path_map.tsv); paths in that file are repository
relative. The [symbol map](source_symbol_map.tsv) records renamed types and
namespace spellings. Old public include paths remain compatibility forwarding
headers. The state/snapshot primary-type rename requires rebuilding C++ clients;
the aliases preserve source compatibility, not the old compiled ABI.

## DNDS boundaries

| Directory | Responsibility | C++ API |
| --- | --- | --- |
| `src/DNDS/Runtime` | Shared arrays, row layouts, placement, access leases and communication | `DNDS::Runtime::*` |
| `src/DNDS/Algebra` | Composed native DOF ownership and numerical operators | Explicit native algebra API |
| `src/DNDS/Legacy` | Old Array/ParArray, ArrayPair, ArrayTransformer, redistribution, derived arrays and DOF operations | `DNDS::Legacy::*` and identical historical `DNDS::*` names |
| `src/DNDS/Legacy/ArrayDerived` | Legacy adjacency and Eigen row adapters | Existing `DNDS::*` names |
| `src/DNDS/Runtime/LegacyArrayBridge.hpp` | Explicit shared-allocation bridge between the two storage contracts | `DNDS::Runtime` bridge functions |

`DNDS::Array` continues to mean the legacy class. New code names
`DNDS::Runtime::Array` explicitly. This phase does not introduce a competing
`DNDS::Array` alias or change the shared-handle semantics of Runtime arrays.
The old `DNDS/Array.hpp`, `ArrayTransformer.hpp`, `ArrayPair.hpp`, derived-array
headers and DOF headers forward to the implementation under `DNDS/Legacy`.
`ParArray` is defined by `ArrayTransformer.hpp`; it has no separate historical
header to move.

Legacy bindings, their width-specific instantiations and their generator travel
with the legacy implementation. CMake enumerates the actual translation units,
including CUDA instantiations and Python binding specializations. The legacy
host/device-vector primitive also lives under `Legacy`; old serializer and
device-helper clients retain its original include path through a forwarder.
The serializer still depends on that Legacy vector plumbing; this relocation
does not claim an independent serializer storage implementation.
Shared support such as index maps, permutation transfer, device providers,
serializers and PCH files stays at its existing path.

## Unstructured mesh responsibilities

| Canonical directory | Responsibility |
| --- | --- |
| `Geom/Unstructured` | Public mesh owner, immutable state and snapshot, builder and preparation entry points |
| `Geom/Access` | Query/read leases, entity spaces/fields and relation access |
| `Geom/Connectivity` | Incidence operations, topology construction and sparse closure |
| `Geom/Transition` | Detached transitions/assembly, reorder plans and coordinated publication |
| `Geom/Geometry` | Geometry editing, wall distance and smoothing |
| `Geom/Refinement` | Elevation, refinement and boundary extraction |
| `Geom/IO` | Native archive, CGNS/OpenFOAM import and output |
| `Geom/Compat` | Legacy read/device views, field-index adapters and legacy archive/serial operations |

`UnstructuredMeshState` is the immutable query/storage authority and
`UnstructuredMeshSnapshot` is a detached retained placement. Their old names
`NativeMeshState` and `NativeMeshSnapshot` remain source aliases in
`Geom/Unstructured/UnstructuredMeshFwd.hpp`; they introduce no second storage
authority. `UnstructuredMesh` and `MeshReadAccess` retain their names. The
remaining Native-prefixed service classes keep their existing names in this
phase to avoid unrelated API churn.

The Python snapshot class is named `UnstructuredMeshSnapshot` and the historical
`NativeMeshSnapshot` module attribute aliases that same class. Existing Python
mesh/read APIs retain their behavior.

Canonical `Geom/Access/MeshAccess.hpp` exposes the native read/query API without
defining the legacy mesh read owner. That owner lives in
`Geom/Compat/LegacyMeshRead.hpp`; `LegacyFieldBridge` lives in its own Compat
header. Old `Geom/Mesh/MeshAccess.hpp` and `EntitySpace.hpp` forwarders preserve
their historical transitive compatibility definitions. The public mesh facade
still includes explicit compatibility types used by existing solver/output APIs.
The native-only compile smoke checks that direct canonical Access/State includes
do not define legacy `DNDS::Array`; it does not claim that the entire Geom facade
or serializer is independent of all legacy plumbing.

## Compatibility and validation

Forwarding headers introduce no second class definition or storage authority.
Include both old and canonical headers in the same translation unit to check
source compatibility. Native-only access headers should not acquire legacy mesh
arrays through a compatibility adapter. The implementation move itself changes
only resolved include paths; symbol spelling changes are recorded separately.

The path manifest and explicit source lists support checks that every relocated
translation unit is compiled once, every forwarder resolves, generators retain
their relative output directory, and no draft file was moved. A successful
header/source audit does not replace rebuilding CPU/CUDA consumers and installing
the rebuilt Python extensions before Python tests.
