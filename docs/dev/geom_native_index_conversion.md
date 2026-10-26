# Native adjacency index state and conversion

Native setup relations carry `MeshAdjState` and exact source/target `EntitySpace`
descriptors. An unknown state is invalid. Global entries use the target's current
distributed rank-offset numbering; they are **not** durable entity IDs. Reorder
and redistribution create new numbering, while persistent IDs travel as fields.

Published mesh compute views contain local indices. Keeping that invariant lets
CPU/CUDA kernels use plain adjacency views without interpreting a state enum in
their hot loops. Setup can convert detached relations in either direction:

```cpp
auto local = RelationReadAccess::Current(mesh, Adj::Cell2Node,
                                        MeshReadRequirements::Primary());
auto globalCopy = NativeRelation::Convert(local, Adj_PointToGlobal);
auto localCopy = NativeRelation::Convert(globalCopy.Read(), Adj_PointToLocal,
                                        false); // require complete local closure
```

The conversion retains source/target numbering, creates independent host values,
and preserves owned/ghost rows, logical row sizes, ordered/repeated incidences
and absent slots. Ordered periodic bits are companions: they are never converted
as indices, and their slot association remains unchanged. Table stride and
nonempty padded/CSR layouts are retained; empty layouts promise logical shape,
not a distinguishable storage kind through a zero-sized borrowed view.

`UnInitIndex` remains absent. A valid global target missing from local closure
becomes `-1-global` in local state when `allowUnavailable=true`; strict conversion
rejects it. Converting back recovers the global target. An encoded missing target
that is actually present locally, a negative global entry, an out-of-range index,
or a sentinel collision is an error. Neither conversion silently fetches ghosts.
`ConvertedIndex` supplies the same rule to primary construction and assembly.
When constructing a relation over arbitrary external values, the caller must
supply their true state and entity order. Numerically plausible values alone
cannot prove that a caller has labeled the correct numbering domain.

Conversion itself is rank-local, issues no MPI, and never mutates or publishes
the live mesh. If followed by a collective setup operation, callers must agree
conversion failures through the existing collective validation boundary. Source
leases retain values and maps through conversion. Use new entity descriptors
after a mesh transition; identical row counts do not establish the same order.

The old whole-mesh `AdjGlobal2Local*`/`AdjLocal2Global*` mutators remain legacy
workspace APIs during retirement. They must not silently change the advertised
state of a native compute view. Native callers use explicit detached conversions
and native assembly/publication instead. Legacy storage removal remains GN7 work.

The focused regression is `geom_test_native_relation_conversion`: exact slot
round trips for CSR/table/padded rows, actual current-mesh ghost views, unavailable
targets, invalid states, empty owners, reorder/redistribution/reclosure, and a
quadratic periodic torus with persistent IDs deliberately unlike global numbers.
Test registration is not an execution claim; checkpoint evidence records results.
