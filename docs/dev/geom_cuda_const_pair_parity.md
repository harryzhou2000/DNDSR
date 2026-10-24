# CUDA comparison with legacy const pair views

`Native mesh CUDA queries match existing const legacy pair views` constructs
independent ordinary and self-periodic synthetic meshes. One mesh uses the new
scoped CUDA facade. The reference mesh transfers its legacy arrays through the
existing `to_device` API and exports only const `ArrayPair` CUDA views.

A device kernel compares owned/ghost counts, coordinates, cell/face incidences,
element types/zones, periodic bit fields and transformed coordinate components,
and ordered face-side/other-cell results. Periodic reference coordinates apply
the existing annotated `Periodicity::TransCoord` operation to the byte bits read
from the const legacy view, preserving the established third/second/first order.
Both separate meshes and their views remain retained through stream completion.

The test does not alter legacy view definitions. It deliberately avoids the
legacy whole-mesh dynamic coordinate-batch API, which requires dynamic Eigen
storage, and does not use legacy host-only bit getters inside device code. The
new facade's bounded batch helpers are separately compared against host results
by the existing full CUDA probe.

The existing const adjacency view hardcodes a mutable base class while declaring
a const base alias, so the specialization itself fails instantiation. The
reference therefore obtains established const generic `ParArray` device views
from the adjacency owners' public base and assembles the existing const pair
view. It uses `RowSize` and `operator()(row, column)`. That legacy defect is
recorded without modifying its implementation or claiming coverage of the
broken adjacency specialization.

The CPU real-mesh overlap test stores gate flags and counters in shared worker
state captured by value. An unwinding release guard opens the gate, while the
submitted command retains its atomics and borrowed mesh view until engine drain;
the outer mesh access owner outlives that engine. This protects exceptional paths
as well as the normal inline/worker/MPI schedules.
