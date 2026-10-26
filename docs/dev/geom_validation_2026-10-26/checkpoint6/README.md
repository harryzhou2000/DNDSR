# Native edits and retained Python mesh aliases

This checkpoint removes the legacy candidate workspace from `MeshEdit`.
Explicit legacy inputs must publish/import native state before editing. The
native-only edit tests preserve rollback, invalid metadata, neighbor-order,
source snapshot, namespace and collective conflict checks. All four CTests at
MPI 1/2/4/8 pass. Dense ghost metadata normalization now participates in the
collective failure boundary before subsequent communication.

Native Python pair properties share native storage and retain its value/shape
and semantic read leases. They support owned/ghost portions, readonly numeric
rows, whole contiguous buffers, historical IDs and readonly POD metadata.
Public properties no longer expose a detached legacy payload under native
authority. The two compatibility poison tests explicitly retain their old
fixture handles before publication, preserving their independent snapshot and
field-routing purposes. They remain transitional baseline tests, not production
native construction.

The final CPU Python matrix passes **26 tests plus one expected skip per rank**
at MPI 1/2/4/8. It covers 2-D quads and distributed 3-D tetrahedra with empty
ranks, every incidence alias, direct memory sharing, periodic bits, wall vectors,
POD snapshot edits, ragged-buffer rejection, escaped whole/row/slice lifetimes,
rank-local retained-ID conflict and successful retry. Existing facade and
overset field-routing tests remain included. All four bindings were rebuilt and
installed; `geom-native-batch6-sharing-profile.json` verifies all eight installed
extension/shared-library IDs and CPU mode. Commands and statuses are in the
final matrix JSON.

The initial Python run incorrectly requested 3-D edge completion on a 2-D
fixture. Its failure log is retained; the corrected suite uses independent
tetrahedral construction for edge coverage. This was a fixture correction, not
evidence of a fixed 2-D edge implementation. A read-only review also prompted
direct `shares_memory` assertions, explicit collective normalization, current
geometry docstrings and native pointer/layout checks before exporting one bulk
Python buffer.

No CUDA tests were rerun for this checkpoint. The checkpoint5 CUB sanitizer
finding remains open. GN7 still requires native Python default producers,
native-only transition candidates, retirement of old construction/storage, and
the final CPU/Python/CUDA matrix. The source manifest is captured before the
formatting hook; log normalization retains original hashes.
