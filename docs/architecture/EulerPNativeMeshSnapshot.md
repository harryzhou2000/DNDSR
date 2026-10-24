# EulerP native mesh query integration

EulerP evaluator argument owners now retain a managed source-mesh read lease, a
cached immutable `NativeMeshSnapshot`, and its `MeshReadAccess`. Kernel payloads
contain only the plain query view. Existing operator kernels read connectivity
through semantic boundary/cell/face queries; the core numerical calculations and
field array interfaces remain unchanged.

The evaluator caches placed snapshots by current physical CUDA ordinal (queried
from the active device), source mesh identity, and a full source version stamp.
A new mesh epoch rebuilds the detached snapshot at the next launch boundary.
Repeated operators reuse native values and CSR metadata without topology copies,
metadata conversion or full adjacency validation. Finite-volume metrics are
validated against the same source epoch before dispatch. Snapshot acquisition
requires the complete face-cell closure used by EulerP's face-process loops.

`FiniteVolumeDeviceView` accepts a second mesh-view template argument, defaulting
to its established legacy view. EulerP selects `UnstructuredMeshView`; existing
legacy FV device callers retain their default path. Periodicity is copied once
into the small FV payload so mutable legacy transform conveniences can be called
without changing the readonly mesh facade or adding per-element ownership work.

The authoritative source mesh stays on host. EulerP's `to_device` places its
snapshot while FV metrics, BC and physics data retain their existing transfers.
First-party solver/test setup no longer mirrors the source mesh separately.
Previously mirrored source arrays must be explicitly returned to host and their
device mirror cleared before native snapshot capture; no implicit coherence or
legacy fallback is introduced.

Current EulerP CUDA kernels launch on the default stream. The argument owner's
mesh scope records/waits for that prefix during destruction, retaining topology
through completion even when the optional serializeCUDAExecution switch is off.
This does not claim arbitrary-stream asynchronous evaluator support. A future
stream API must pass the actual context and join every using stream explicitly.

Granular FV construction must finish with `finalize_metrics()` before evaluator
dispatch. It verifies the completed current-epoch metric stages and publishes
their dependency stamp. `ConstructMetrics` performs the corresponding publication
for bulk construction.
