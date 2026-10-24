# Legacy mesh CUDA read-view comparison, 2026-10-24

The legacy mesh/array device-view files were inspected read-only. No legacy
implementation was changed, no legacy CUDA mirror was imported into a second
Runtime Buffer controller, and no build or GPU run was performed for this
source comparison.

## Existing CUDA path

`src/DNDS/ArrayPair.hpp:29` implements father/son combined indexing once for
both mutable and const device views. `ArrayPairDeviceViewConst` at line 121
selects const underlying array views; the const `deviceView()` factory at
line 695 obtains them from both retained legacy arrays. The view itself is
non-owning, and its lifetime remains tied to those arrays and their mirrors.

`src/Geom/Mesh/Mesh_DeviceView.hpp:166` builds the mesh CUDA view out of those
views. `src/CFV/FiniteVolume_DeviceView.hpp:12` embeds it directly.
`src/EulerP/EulerP_Evaluator_impl.cu:16` generates real CUDA kernels over
`EvaluatorDeviceView<CUDA>`, and its launchers ship a retained evaluator view
copy to those kernels. The corresponding implementation reads cell/face
relations, counts, zones and face orientation; it is an established CUDA path,
not an unimplemented design.

The existing CUDA regression
`test/cpp/CFV/test_DeviceTransferable.cu:110` reads cell volumes, cell
barycenters and face areas through `FiniteVolumeDeviceView<CUDA>`. Its setup
at line 233 transfers mesh/FV data, constructs the device view and launches
that kernel. This verifies the operations actually instantiated there; it
does not instantiate every device-annotated mesh helper.

## Source findings

1. The mesh-view constructor at `Mesh_DeviceView.hpp:333` copies dimensions,
   periodic flag and adjacency states, but does not assign
   `mesh.periodicInfo` to its `periodicInfo` member. That member therefore
   starts from `Periodicity`'s own default constructor
   (`src/Geom/PeriodicInfo.hpp:197`: identity rotations and unit-axis
   translations). No compensating assignment was found in CFV/EulerP.
   This is a specific source-level omission for configured periodic
   transforms; it is not a claim that all legacy CUDA operation is broken.
   It was left unchanged as requested.
2. The same constructor does not call `create_view_N2CB`, although it copies
   the N2CB state and `UnstructuredMesh::op_on_device_arrays` enumerates
   inverse arrays for device transfer (`src/Geom/Mesh/Mesh.hpp:1100`).
   Transferring inverse arrays is therefore distinct from initializing
   those fields in this whole-mesh view. Raw separately obtained const
   pair views remain usable. This omission was also left unchanged.
3. Legacy coordinate-list helpers take `tSmallCoords`, an Eigen matrix with
   an unbounded dynamic column count (`src/Geom/Geometric.hpp:50`), and call
   `resize` inside device-annotated methods (`Mesh_DeviceView.hpp:426`).
   The examined CFV/EulerP CUDA kernels primarily consume already computed
   metrics and raw mesh relations; no CUDA call site for these dynamic
   coordinate-list helpers was found. Device annotation alone is not
   evidence that this exact Eigen output type was exercised on CUDA.

## New-view parity and ownership

The new `UnstructuredMeshView` preserves father/son combined local indexing,
ordered relation slots, sentinel values, ElemInfo data, PBI bytes and the
face-side rule for a periodic self-face. Its scalar coordinate and wall
vector transformations follow the legacy `Periodicity` transform order
(third, second, first). It explicitly captures the configured periodic
metadata and requested inverse arrays.

`MeshReadAccess` owns independent retained CUDA snapshots plus their read
leases; the legacy view still refers to its original device mirrors. These
are separate ownership strategies. Reusing the legacy indexing and query
semantics does not require wrapping its mirrors in independent access
controllers, which would falsely claim shared exclusion/version tracking.

The new probe instantiates additional read helpers, including coordinate
lists. Its CUDA output now uses bounded `MeshQueryCoords`; host overloads
retain `tSmallCoords`. A compile/runtime issue in that newly instantiated
probe must be described as a new-view/probe issue, not as evidence that the
working legacy raw-view and FV/EulerP CUDA paths cannot run.

Recommended direct parity coverage is raw const-pair row values and widths,
owned/ghost counts, raw coordinates, zones/element types, PBI bytes and face
orientation, comparing both views over independently retained owners. Check
new transformed geometry against explicit configured host truth. Whole-view
legacy periodic transforms and inverse fields should be reported separately
while the source omissions above remain intentionally untouched. Existing
CFV device-transfer regression results should retain their narrower metric
scope; a generic device annotation is not an exhaustive CUDA API test.
