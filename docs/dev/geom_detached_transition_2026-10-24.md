# Prepared host mesh transition, 2026-10-24

`MeshTransitionHandle` provides one serialized shared mesh publication point.
`PreparedMeshTransition::Prepare` retains the source mesh and a managed read
scope, clones owned rows into independent arrays and a fresh mutation controller,
converts copied local relation entries to globals using their original target
mapping, runs the existing reorder mathematics on the candidate, and rebuilds
primary ghosts, facial ghosts/boundary matching, inverse ghosts and the optional
cell-face-neighbor relation. Preserved-face and destroy/reconstruct-face policies
are explicit. Periodic geometry and every participating PBI row are copied.

Native fields are staged by deferring their existing preparation callback's
publication while the candidate mesh uses the same `ReorderEntitiesWithFields`
plan. No independent second plan is inferred for the fields. Source fields remain
unchanged through clone, reorder and rebuild. `Publish` checks the source handle
and full version stamp and field attachments collectively, then moves all field
handles and swaps the shared mesh pointer without allocation. Canonical source
entity map identities are also revalidated at publication. Existing shared
source owners and retained read scopes continue to refer to the unchanged source.
Field objects and the handle must outlive preparation/publication. Access to the
external field handles must share caller serialization with the mesh handle;
independent raw field readers are not made atomic by the handle mutex.

The supported inventory includes current legacy or published native primary/facial
storage, original IDs, periodic companions and wall vectors. The clone reads the
retained current query view and private canonical entity maps, never a stale raw
workspace underneath native publication. Original IDs are owned at native
publication and exported through `OriginalEntityId`. Wall vectors transfer as
Node companions and rebuild ghosts. Inverse and optional cell-face-neighbor
arrays are rebuilt. Existing 3D edges and their periodic companions preserve
exact row/slot order through the existing registry. Native Edge fields, explicit
Edge placement and supported Edge follows use that same correspondence. Primary
ghost setup retains owned-edge parent cells and their nodes; edge ghost setup
retains parent incidence and refreshes the edge cone transformer after ownership
changes. `destroyKinds{Edge}` explicitly selects reconstruction and prohibits
attached Edge fields/maps/follows. Parent/boundary-extraction links,
elevated original-node-prefix semantics and elevation workspaces are rejected
before cloning. Ghosts regenerate according to the explicit `ghostLayers` policy
(default one), rather than copying old halo rows. `TransitionCachePolicy` either
invalidates derived caches or rebuilds output caches; local partition and
factorization caches invalidate in both cases.

`CloneCurrentMeshForSetup` provides the same independent prepared export without
ownership changes. Its default still prohibits parent/extraction links. Output
may explicitly choose `BoundaryCachePolicy::DropBoundaryCache`: a temporary
read-only metadata alias shares the original array/controller/native owners but
clears only its copied `node2bndNode` source-extraction cache, then deep-exports.
The original cache, child mesh and source controller remain unchanged. Child
identity vectors `node2parentNode` and `cell2parentCell` remain prohibited; the
publishing transition never receives this export-only bypass. Legacy collective VTKHDF/CGNS and distributed serial-reader
output use this exporter when a native authority is present, then rebuild output
caches on the detached clone. Point-field callbacks map clone-local nodes through
global IDs back into the source local numbering, including ghost and periodic
recreated-node output. A missing source point row rejects collectively before
writing. A retained source read spans callback evaluation. Native serial-output mode with a retained stale
serial-reader cache rejects explicitly.

`WriteCurrentPrimaryMeshSnapshot` is an explicit collective **primary-only**
checkpoint API for both H5 and per-rank JSON; `WriteCurrentMeshSnapshot` remains
a compatibility spelling with the same policy. It agrees authority, serializer
mode/communicator/path and auxiliary policy, exports the current mesh, converts
the clone to global primary indices and serializes it. Python exposes both names
and `CurrentMeshCheckpointPolicy`. Default `RequireNoAuxiliaryValues` rejects wall
vectors, which the legacy primary format cannot preserve. Explicit
`DropAuxiliary` authorizes dropping them. Optional compatible metadata records
`checkpoint_kind=DNDSR.CurrentPrimaryMeshSnapshot`, `checkpoint_schema=1`,
`rebuild_inventory` and `auxiliary_policy`; legacy readers ignore these extras.
Inventory bits are Face=1, Edge=2, inverse=4, face-neighbor=8, output-cache=16,
partition/factorization-cache=32 and discarded wall values=64. Face/edge numbering,
registered fields, wall values and auxiliary caches are not a whole restart
checkpoint. Rebuild or restore those explicitly after loading primary topology.
Source authority and retained views remain unchanged. Legacy `WriteSerialize`
keeps its local per-rank JSON behavior and rejects native state with this helper's
name rather than introducing a conditional collective into a local API.

Local checkpoints agree failures before subsequent communication. Initial
inventory, reorder schedule, rank assignments and follows are validated before
the existing collective algorithms. Native payload adapters retain Runtime's
route/schema checks. Arbitrary callback divergence, failure inside unguarded
legacy collective internals, MPI process failure, and universal out-of-memory
recovery are outside this contract. Candidate failures never publish the source
mesh, even when the discarded candidate's own mutation state becomes failed.

Delayed native field attachments carry allocation identity, value/shape versions
and local row/distribution extents in addition to schema/domain/epoch/association
and destination handle identity. Publication rejects replacement or writes after
preparation, including equal-shape/equal-metadata rebinding. These stamps remain
process-local and do not enter cross-rank schema signatures. Caller serialization
still forbids concurrent external field writes during the final check/publication.
All registered native fields, including inactive entity families, now validate
exact owned row extents and zero ghosts before cloning. Valid inactive owned
handles retain their allocation and epoch; ghost routes are rebuilt explicitly
against the destination entity space. Kind-only field registration does not
prove a foreign field's semantic mesh/order identity: row ordering remains the
registering caller's contract, while domain/epoch/schema consistency is checked
among attachments and across ranks. `EntityField` is the opt-in typed boundary
that retains and validates the actual EntitySpace association; see
[EntityField ownership](../architecture/EntityField.md).

Detached preparation preserves native source authority by publishing the rebuilt
candidate's requested native groups before binding staged typed fields to its
final EntitySpaces. Export-only cloning explicitly selects a legacy workspace
destination. Active and inactive typed field states are allocated and bound
during preparation; publication only swaps their retained states. Retained old
typed reads remain usable against their old mesh snapshot, while old routes
reject the newly published field. Typed registration descriptors defer semantic
validation to collective preparation checkpoints, so a rank-local stale
registration cannot throw before peers enter preparation.

The coordinating integration 23 run passes the original typed field executable
at MPI 1/2/4/8 and transition/edit/checkpoint/overlap at 16/16 registrations after
native destination authority preservation. The final typed regression extension
adds rank-local stale registration/route rejection, collectively agreed local
shape/type failures, explicit padded POD MPI-layout halo transfer and cross-rank
schema disagreement; these additions require the subsequent fresh build/run.

Fresh runtime iteration 9 failed at MPI 1/2/4/8 before completing the first
preserved-face transition: cloned unchanged Face2Cell/Face2Bnd rows had no source
row global mapping, so primary ghost traversal asserted. The source fix assigns
each cloned relation/info/PBI/original-ID/wall father its canonical source-family
mapping before recovery/reorder; canonical target-map conversion is separate.
Iteration 11 then found an inappropriate second initial boundary match after
preserved relations were already local. The detached preserved-face path now
rebuilds only its derived local side caches from retained exact relations/labels.
Iteration 12/13 exposed fixture omissions: manually created wall rows lacked
canonical maps, and `UP3D_32.cgns` has three periodic BC pairs on the unit cube,
requiring periodic frames/deduplication and periodic-node recreation before VTK
setup. Those fixture stages are explicit; facade sentinel checks were preserved.

Fresh CPU iteration 14 passes the entire transition executable at MPI 1, 2, 4
and 8 (4/4 CTest registrations, 9.91 s). Evidence is
`geom_validation_2026-10-24/transition-iteration14-build.txt` and
`geom_validation_2026-10-24/transition-iteration14-tests.txt`. Earlier failed logs
retain their distinct iteration labels. The seven test cases include subcases
cover ordinary and periodic preserved/reconstructed faces, two fields with IDs
matched to moved mesh rows, exact stable-ID connectivity/slot PBI rows, independent
CSR metadata, retained old scopes/controller/coordinate isolation, rank-local
failure after clone/reorder/rebuild or second-field preparation, delayed native
field/Array edits/rebinding, authoritative export/checkpoint readback despite raw
workspace corruption, wall vectors, 3D edge reconstruction and unsupported parent
inventory rejection. The separate `test_MeshCurrentOutput.cpp` adds actual
periodic VTKHDF scalar/vector point-callback readback against authoritative
geometry/IDs with a three-layer source halo and corrupt raw workspace coordinates;
passed at MPI 1/2/4/8 in the coordinating integration 15 Geom suite (75/75
registrations overall, including current-output 4/4). This output evidence
predates the preserved-edge extension.

Fresh iteration 17 passes the full eight-case transition executable at MPI
1/2/4/8 (4/4 registrations, 23.84 s). Logs are
`geom_validation_2026-10-24/transition-edge-iteration17-build.txt` and
`geom_validation_2026-10-24/transition-edge-iteration17-tests.txt`. The new periodic
3D regression independently places cells on rank+1 and edges on rank+2, tests both
legacy and native current sources, and checks native edge IDs, exact endpoint and
incident-cell order, endpoint PBI, and cell edge incidence/PBI against staged
source-row oracles. Source controllers/read scopes remain valid. Reconstruction
remains a separate passing subcase. CPU success does not imply CUDA-resident
field transfer, all output modes or universal detached-transition completion.

Native direct output setup is synchronous/serialized. Export and rebuild use the
source mesh communicator, even when VTKHDF later uses a duplicated communicator
for I/O. MPI_THREAD_MULTIPLE alone does not permit concurrent mismatched
collective sequences on that source communicator. Prepare detached export before
launching a writer thread, or supply a future independently scoped setup route;
Euler native distributed volume output executes synchronously; its boundary and
serial-reader adapters remain explicitly guarded. The output
delegates do not establish cross-thread ordering of unrelated field writes.

Legacy `WriteSerialize` now guards possible CSR compression with an exclusive
managed setup mutation, validating host storage and idle buffers before taking
the writer or initiating I/O. Collective serializers agree preflight failures;
per-rank JSON remains local. Its allocation/interface generations change on
success; failure after destructive compression fails closed. The detached current
checkpoint path performs this only on independent scratch storage. Transition
cloning itself does not use serializers or compress source arrays.
