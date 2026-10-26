# Native facial partitioning

`NativeMeshPreparation::Partition` builds an owned CSR graph with current global
cell targets from exact physical faces. It never gathers a distributed graph.
KWAY uses ParMETIS when several ranks own cells and METIS when one rank owns all
cells. RB requires that single active rank; requesting RB on distributed cell
ownership fails collectively. Serial CGNS/OpenFOAM ingress selects
`NativeSerialDistribution::ReaderOwned` for RB before partitioning. The public
reader default remains `Even`.

Edge weight method 0 is unweighted. Method 1 takes the largest pairwise physical
face-node distance, including periodic image transforms. Several faces joining
one cell pair contribute their maximum diameter. Nonfinite coordinates or
nonpositive interior face diameters fail collectively. This is a nodal diameter,
not a face area or integrated curved-surface measure.

The integer scale accounts for the actual METIS `idx_t` type. Let E be the global
directed-edge count, I the largest `idx_t`, M the global maximum diameter, and
C = min(INT_MAX, floor((I - 1) / E)). An edge of diameter d receives
floor((C - 1) d / M) + 1, clamped to [1, C]. Thus the sum of all directed edge
weights is at most I - 1. Edge counts are summed with checked 64-bit arithmetic;
a graph with no positive weight capacity is rejected before conversion. An
edgeless graph needs no weights.

This corrects the old legacy weight scale of 1..INT_MAX: on a 32-bit METIS
backend, even two incident edges could overflow degree/cut accumulators.
Quantization can merge nearby diameters; it preserves nondecreasing weights.

Nodes, boundaries and faces generated internally for partitioning follow cell
owners. A serial reader therefore releases its global facial ownership when
cells distribute. Caller-provided face/edge ownership remains unchanged. All
references, companions, namespaces, persistent IDs and ancestry use the existing
native transition machinery. No legacy mesh workspace is constructed.

The focused graph test checks independent nonuniform edge lengths, integer
weights, repeated faces of a cell pair, real weighted backend execution, empty
ranks, recursive reader-owned partitioning, face identity/incidences after
redistribution, and collective failures followed by retry. The graph and
preparation targets passed all eight CTests at MPI sizes 1, 2, 4 and 8 in the CPU
profile. The final log is `/tmp/geom-native-batch5-partition-tests.log`.
