# Mesh preparation communication policy

Native owned construction accepts an optional, collectively agreed Runtime
communication context. Existing callers keep collective safety preflight enabled.
This policy controls safety-only agreement, independently of mesh identity,
topology, audit selection, indexing and physical storage layout.

```cpp
auto communication = std::make_shared<DNDS::Runtime::CommunicationContext>(mpi.comm, false);
DNDS::Geom::NativeMeshBuildOptions options;
options.communication = communication;
auto owned = DNDS::Geom::NativeMeshBuilder::Build(std::move(input), options);
owned = DNDS::Geom::NativeMeshTopology::Build(std::move(owned), mpi);
auto specification = DNDS::Geom::DefaultNativeMeshGhostSpec(owned, 1, true);
auto state = DNDS::Geom::NativeMeshAssembler::Build(mpi, std::move(owned), specification);
```

Construction binds all input entity descriptors to that context. A null option
inherits the input cell descriptor's context; an absent descriptor context means
enabled compatibility behavior. `EntitySpace::WithCommunication(context)` returns
a descriptor sharing the exact numbering maps, identity and generation stamps.
It changes neither existing descriptors nor `SameOrder`. Siblings, ghost-map
replacements, interpolated entity families and final publications inherit the
selected context. A retained old descriptor remains usable under its old policy.
The complete-state topology frontends inherit that policy too, including their
mutation-availability preflight. `CheckCollectiveMutationAvailable(bool)` now
names and respects its safety-mode argument; its default remains enabled. It
does not acquire mutation ownership: the caller still serializes submission and
publication. Local version/access checks remain active in either mode.
Do not change context choices rank-locally or during active work; all ranks must
select the matching context at a drained, coordinated setup boundary.

```mermaid
flowchart LR
    C[Agreed Runtime context] --> O[Owned builder and entity descriptors]
    O --> T[Owned topology and temporary closure]
    T --> A[Assembler: DSL tree and canonical ghost mapping]
    A --> P[Immutable mesh publication]
    P --> R[EntitySpace row transfer plan]
    C --> S[Shared local-stage validation helper]
    O --> S
    T --> S
    A --> S
```

Owned records carry the context for checks before their descriptor inventory is
validated. Keep that field and every descriptor on the same policy; topology,
assembly, publication and registry registration reject mixed policies. Public
setup boundaries agree the actual selected mode before optional local stages.
Registries freeze that agreement for their collective lifetime. This protects
against selecting different already-valid on/off contexts on different ranks.

`LocalStage` still executes local checks. Enabled mode agrees failures and
preserves the originating local exception; disabled mode performs no successful
safety reduction and aborts on local failure before peers can strand in payload
work. Off mode does not provide recoverable distributed rollback. Required
ownership decisions, schedule/schema checks, sparse discovery, alias/reuse
agreement, row counts and numerical reductions remain. Full identity/closure
audits still execute when requested; this switch does not disable their content.

`EntitySpace::BuildRowTransferPlan()` inherits its descriptor's context. The
overload accepting an explicit congruent context remains available. Binding and
request ownership, direct MPI datatypes and explicit payload namespaces keep
their existing contracts; attaching a context does not create a transfer.

This is the owned-builder/topology/assembler core of plan 53 E1. Importer,
partition/reorder companion stages and production-loader propagation are still
being completed; the existing Cauchy switch is not yet a claim that every cold
mesh stage uses this context. Python exposes descriptor rebinding and inspection;
its runtime gate is queued with the remaining cross-module rebuild.

`geom_test_mesh_preflight` compares on/off topology and direct halo payloads,
including empty owners and a self-periodic mesh. PMPI counts prove four repeated
local stages and three repeated halo starts use zero safety reductions in off
mode. Required setup reductions are retained, so complete setup is not expected
to have zero MPI reductions. The test also checks immutable descriptor rebinding
and mutation preflight, and rejects inconsistent context selection before
optional preparation begins.
