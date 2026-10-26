# Native smoothing migration

The staged native implementation retains the established Hermite boundary
displacement, local RBF, global RBF (V1), and bisect-FEM elasticity (V2)
algorithms. It replaces mesh storage and communication access; it does not
claim a new numerical method or accuracy result.

`NativeSmoothingWorkspace` retains its immutable source state and entity stamps,
read-protected Runtime displacement arrays, and explicit node constraint roles.
Original boundary corners come from actual element corner slots rather than an
owned-array prefix. This remains valid after redistribution or refinement;
older multi-parent ancestry alone does not identify corners of the current
quadratic element. Boundary preparation leaves source coordinates unchanged.
Each solve returns a detached geometry replacement with the same identity,
persistent IDs, ordered topology, and ancestry. Wall vectors and cached device
placements are invalidated when coordinates change.

Complete node/face and node/cell support comes from native inverse incidence
and sparse owner-directed row requests. In particular, V2 does not assume the
source's existing cell ghost layer contains every cell incident to an owned
node. Topology and work storage scale with local requested/served incidences
plus rank metadata; a genuine large support star can still be large. V1
coefficient iterations fetch only locally referenced coefficient columns.

The spatial boundary dataset is explicitly replicated per rank, as in the
legacy RBF algorithm. This is a recorded scalability limit, separate from
topology remapping. It costs O(global selected boundary points) storage on each
rank. Replacing it with a distributed spatial service is outside this port.

## Numerical conventions retained for separate review

1. Local RBF compares the nearest distance against a squared radius expression.
   The native port retains this units convention.
2. V1 passes a physical support radius directly to a squared-distance KD search.
   Construction and evaluation also use different distance exponents when
   `RBFPower != 1`. Both expressions remain unchanged.
3. V1's legacy preconditioner computes an LU result then overwrites it with the
   input. The native path explicitly uses its effective identity operator.
4. V2's constitutive scale uses exponent `-0.0`, giving the effective factor two.
   Its nearest-boundary distance and `refDWall` do not affect that assembly.
5. V2 transforms the accumulated periodic block after each cell contribution,
   using `jFrame.transpose() * accumulated * iFrame`. This convention and the
   placement of Dirichlet row resets inside the cell loop are retained.
6. Boundary normal filtering transforms endpoint normals using the new node's
   periodic image bits. It retains the legacy choice of basis.
7. Unselected boundary nodes are constrained by V2. Local RBF and V1 may move
   them, matching the existing distinction between those solver variants.

The solvers retain the existing restart sizes, iteration counts, and stopping
tolerances. Exhausting an iteration budget is not evidence of convergence.
Floating-point accumulation or nearest-neighbor tie order can change with
partition and native row order; exact bitwise solver parity is not asserted.

## Validation snapshot

The registered six-case target passes at MPI 1/2/4/8 after integration with the
public facade. It covers independent family ownership, empty ranks, immutable
source/work arrays, collective errors and retry, exact octagon and redistributed
octahedron Hermite displacements, all three solver variants, a flat periodic
physical-image oracle, mixed selected/unselected boundary roles, retained-read
rejection, stale workspaces and atomic publication. The existing mesh pipeline
also passes at MPI 1/2/4/8. Logs are retained in checkpoint 5.
These checks are geometry/contract regressions, not a physical accuracy study.
