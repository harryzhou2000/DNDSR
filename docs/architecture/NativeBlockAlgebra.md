# Native block storage and local factors

`DNDS/Algebra/BlockMatrixStorage.hpp` owns uniform dense-block values in a native
Runtime Array and shares immutable sparse block-row offsets. It contains no mesh,
equation or symbolic-factorization policy. Copies share values. Read/write access
scopes retain allocation and row metadata; borrowed Eigen block views must not
outlive them. Lease ownership and layout setup occur outside block arithmetic.
These Eigen views are CPU-only.

`Solver/NativeBlockLU.hpp` is a rank-local native numeric matrix/factor adapter.
It uses generic `Direct::LocalLUBase` logic and a published `SerialSymLUStructure`;
it does not include EulerJacobian.hpp or legacy ArrayDOF. Sparse rows store
diagonal, lower, then upper blocks. Matrix and factors use separate allocations.
Fill matrix values before factoring. Diagonal inversion follows the active Euler
policy: row-diagonal scaling, partial-pivot LU with a1e-7 reciprocal-condition
guard, then filtered-SVD fallback. The storage-independent implementation is
`Solver/BlockInverse.hpp`; native LU and Cauchy diagonal preparation share it.
Fallback counts are safe across local parts.

The caller serializes mutation and execution. A new symbolic generation needs a
new native storage owner. Matrix writes invalidate factors. Solve pins matrix
and factor reads before checking versions, rejecting an active unpublished
writer. RHS, forward scratch and output must have separate allocations; each
can be an AoS, SoA or AoSoA uniform DOF table. Output acquisition failure cancels
an unused scratch write. No communication/error agreement is hidden here;
distributed callers coordinate local failures before subsequent MPI operations.

`SetCoreFirstOrdering` stably partitions the requested ordering within each local
part before symbolic fill. Factor positions remain distinct from cell indices.
`ReadySolveRows` propagates available RHS rows through actual lower/upper edges.
Short factor rows do not imply short solve-dependency chains. The ready core can
be empty; complete LU has no bounded-stencil core guarantee.

Generic numeric ranges are consecutive pivots. Lower solves advance in factor
order and upper solves reverse it. Lower-core completion produces forward scratch,
not final core solution values. Partial-range callers provide completed
predecessors and own phase/readiness validation. Native full Solve runs both
scans without MPI. An exposed middle transfer is a separate SGS orchestration
option. Legacy full-factor/full-solve entry points preserve their arithmetic.

Focused Solver tests cover independent dense solves, incomplete/full factors,
full/ranged agreement, core permutations/local parts, readiness counterexamples,
all three DOF layouts, empty ownership, inverse fallback, retained lifetimes,
aliases and stale/active-writer rejection. Cauchy's Jacobian/nonlinear/physical
integration remains a separate consumer and needs its own distributed gates.
