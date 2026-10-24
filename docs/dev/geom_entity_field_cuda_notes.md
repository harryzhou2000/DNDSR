# EntityField direct CUDA/MPI regression

`test/cpp/Geom/test_MeshEntityField_CUDA.cu` is a dedicated multi-GPU test source.
It requires at least two MPI ranks, CUDA-aware MPI, and one distinct CUDA UUID per
rank. Without device visibility isolation it selects ordinal `rank` and requires
at least `size` visible devices. Under the existing one-rank-per-GPU wrapper each
rank selects visible ordinal zero; an MPI UUID comparison also rejects accidental
oversubscription. Register/build/run through the CUDA profile, with a bounded
timeout. This source was added for root-managed integration; creation alone is
not compilation or runtime evidence.

The real CGNS mesh fixture publishes native authority before creating
`EntityField<double, FixedRows<5>>` from CUDA values. Its five-component state uses
an exactly representable formula of original cell ID and component. A separate
MPI gather of owned original IDs constructs the current-global-ID oracle; ghost
expectations do not read received values or reuse the payload route.

Device kernels consume `field.Read(...).View()` and `field.Write(...).View()`.
Pinned callback gates prove completions remain pending and mesh mutation leases
remain held until the using prefix completes. The gate owner opens/synchronizes
before lease/completion destruction, including on exceptions. Kernel functions
are preloaded before gating to avoid lazy module-loading synchronization deadlock.
Read/write conflicts are rejected before an unsafe pointer is issued, and sealed
scope getters reject further access. Kernel read residuals and halo payloads are
checked exactly on CPU after explicit device completion.

The halo uses `HaloRoute::Bind` and `BeginScheduled(context)` with direct transport;
no packed host fallback is substituted. While the operation is active, immediate
ghost reads and owned writes reject. Its explicit wait completes MPI before CPU
debug copying. The test also sends the CUDA field through the detached distributed
mesh transition, checks rejection of old field association/halo binding/route,
then repeats the kernel and halo oracle on the new native mesh.

Both nondefault CUDA streams outlive all default-context operations, retained
scopes, halo operations, registration callbacks, and prepared transitions. The
test proves ordering and association correctness, not communication speedup.
