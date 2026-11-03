# Native-field FAS cycles

`Solver/FAS.hpp` provides a synchronous CPU V-cycle over explicit `FASLevel`
operators and `FASTransfer` edges. Cauchy polynomial multigrid supplies spatial
levels and physics; the cycle does not construct meshes, own time history,
choose convergence criteria or perform hidden MPI. A future agglomerated
hierarchy supplies the same three transfer operations.

```text
FASCycle
  levels[i] -> retained state + apply F(U) + bounded smoother + update policy
  edges[i]  -> state restriction / residual restriction / prolong correction
  private scratch per level -> source, F(U), defect, entrance state, correction
```

For `F_f(U_f)=b_f`, the coarse equation is
`F_c(U_c)=F_c(I U_f)+R(b_f-F_f(U_f))`. After the coarse cycle, prolong
`U_c-I U_f`, apply the level's optional physical update policy, then post-smooth.
The incoming coarse reference remains unchanged throughout its recursive child
cycle. State and residual restriction need not be the same operator.

Callbacks write provided allocations without rebinding them, and return only
after their work and required communication completes. An operator may refresh
input ghosts but may not modify owned input values. Distinct levels have
nonoverlapping mutable state. Fixed bindings pin exact ArrayStorage objects,
not merely allocation IDs; this rejects rebinding to a disjoint range of the
same allocation. Temporary numerical views remain local to callbacks.

All ranks call the same hierarchy/cycle order. Callbacks coordinate local errors
before their communication; an optional LocalStage coordinates local vector
algebra errors. Caller serializes operations and releases conflicting leases.
The cycle reports work, not convergence: the physical controller must check its
original finest equation. A failed cycle can leave partial level updates, marks
the instance failed, and requires state restoration and hierarchy reconstruction
before retry. It is an in-place primitive, not an accepted-state transaction.

`FASCycleSettings::updateComponents` is an optional set of component indices;
empty means all. A complete explicit set is normalized to the same full-state
path. A proper subset requires equal component widths on all levels and each
level's `smoothSelected` callback. That callback receives the normalized set and
must solve its principal equations while holding the remaining owned components
fixed. Solving a full coupled system then discarding some entries is a different
method and violates the contract. Distributed callers agree the selection and
hierarchy before entering; the generic cycle performs no hidden MPI agreement.

Restriction still transfers the complete background state and residual for tau
construction. Each coarse level freezes its own restricted background, rather
than zeroing components excluded from updates. Prolongated corrections are
projected onto the selected components before the physical update. Optional
physical update callbacks must preserve the other means, including through PP.
Subset mode allocates one retained snapshot per level and checks frozen owned
values bit for bit after smooth/update callbacks, outside numerical kernels.
A violation fails the cycle; it does not silently restore an invalid update.
Full-state mode has no snapshot allocation or additional validation collective.
The outer controller still measures its original complete finest residual;
update selection and convergence selection are independent decisions.

Tests use independent two- and three-level numeric forcing/correction oracles,
a nonlinear fixed point, differing state/residual restrictions and mixed native
layouts. Negative tests exercise aliases, callback mutation, output rebinding
and same-allocation/different-view replacement. These are building-block tests;
they do not qualify Cauchy's pMG convergence or performance.
Subset tests additionally use an independent coupled 3-component/2-component
principal inverse, incompatible frozen equations, strided and empty fields,
prolongation projection, signed-zero preservation and rejected callback writes.
