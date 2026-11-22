# Experimental matrix storage for FV and VR

These choices are experimental. The default FV block representation and
precomputed VR Jacobi representation are unchanged. All layout selections
change physical storage; they do not change the equation unknown ordering.

## FV linear blocks

`implicit.linear.matrixStorage` separately selects the diagonal, inverse and
off-diagonal matrix payloads and the native sparse-index width. Each payload
supports `aos`, `soa` and `aosoa32`. The AoS `rowStridePaddingMultiple` is a
multiple of scalar elements, not bytes; `allocationAlignmentBytes` controls
the allocation base independently. These settings may increase capacity without
changing the logical block dimensions. Matrix actions must use the matching
layout-aware view. Compact AoS retains the static CPU map path.

The single-device AmgX adapter requires native compact row-major AoS diagonal
and off-diagonal blocks and native int32 CSR indices. The inverse layout is
independent. Incompatible storage is rejected; DNDS does not repack the matrix.
See [the AmgX guide](../guides/amgx.md) for provider-specific restrictions.

## Variational reconstruction

`numerics.cfv.storage` selects the sweep representation. VR matrix, inverse,
mean and boundary maps use contiguous AoS rows; losing experimental VR-map
SoA/AoSoA choices were retired in the stage-once follow-up, with their evidence
retained in plan history. Runtime/DOF and FV matrix layout options are unchanged.
In `precomputed` mode, a sweep
uses the previously assembled A-inverse-times-B maps. In `raw` mode, it first
accumulates the B products and mean/boundary terms, then applies the local
inverse. Both use the same functional and SVD cutoff, but reassociation can
change floating-point rounding.

For raw sweeps, `inverseTriangle` packs the validated symmetric local inverse.
`faceCoupling` stores one full B block per local face and transposes the
opposite incidence. A B block is not assumed symmetric. Periodic vector
transformations still apply in the corresponding incidence frame, including
self-periodic faces. `aosRowPadding` controls AoS pitch in scalar elements;
`allocationAlignmentBytes` independently requests base alignment.

The general FP64 sweep accepts each supported representation. The A100
`fp64V2` preset selects compact fixed-width fields and matrix rows for
C4–7 with 2, 3, 5, 9 or 19 modes. Raw storage applies the full inverse
after the neighbor sum. The triangular inverse, padded rows, arbitrary
C and M2/M3 LVR use general FP64. The optimized families read neighbor
blocks directly into registers. The raw inverse consumes the accumulated
cell right-hand side from shared memory. See the [A100 routing
table](../dev/cauchy_a100_kernel_defaults.md).

LVR supplies one frozen owned target of width C × M and one coefficient
`alpha` per cell. The target uses the same modal ordering as the input.
The sweep blends the unconstrained Jacobi result with this target before
relaxation. Target preparation can exchange halo values independently.

Historical pipelines and precision experiments require
`DNDS_CFV_CUDA_ENABLE_V1_KERNELS=ON`. Their explicit selectors retain
their original staging and arithmetic.

`floatFloatPipelined` requires `DNDS_USE_CCCL_COMPUTE` and the pinned CCCL
`cuda::fp32mp2` provider (`DNDS::Math::real_f32f32`). ALL numerical sweep
arithmetic uses the pair: matrix products, reductions, periodic rotation,
mean/BC terms, raw inverse application and relaxation. FP64 is used only at
input/output conversion. Matrix/state storage stays FP64; inputs are converted
directly into both limbs, without first rounding to a single float. Shared
staging transports FP64 bytes then converts each matrix/coefficient once.
The narrower exponent range can overflow. A device-only result check sets the
sticky workspace status before guarded commit; no host drain or FP64 fallback
is inserted. This changes arithmetic precision, not the reconstruction weights.

Quantized FP32-map placement also supports `fp32Pipelined` (full FP32 SIMT
arithmetic) and `tf32Pipelined` (TF32-rounded tensor products and FP32 update
arithmetic), through the same neighbor-buffered implementation.

Raw CPU and CUDA relaxation support `numerics.cfv.storage.rawBPrecision`
and `numerics.cfv.storage.inversePrecision`, each accepting `fp64` (default)
or `fp32`. These controls quantize only the selected resident matrix after FP64
setup and inversion. The raw mean and boundary maps, evolving coefficients,
and CPU raw update arithmetic remain FP64. The host Cauchy relaxation path
publishes each selected matrix directly in FP32, without retaining an FP64
copy. CUDA general FP64 and eligible V2 presets load packed values directly.
The inverse precision applies to both full and triangular inverse storage.

TF32 has a separately looser accuracy budget;
it is not an ordinary-FP32 alias. The general FP32/TF32 policies remain
available as numerical/comparison anchors. Matrix, state and output component
meaning is identical across those choices. All use the common queued CFV
operator with explicit placement/capability checks.

The shared staged numerical implementation is in
`Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh`. The paired `_FP64`,
`_FloatFloat` and `_FP32` translation units instantiate mode/component variants
independently; the small `.cu` wrapper checks arguments and selects a provider.
The general sweep uses the same mode/component dispatch discipline via
`ReconstructionDevice_General.cuh` and its precision-specific translation units.
Kernels receive only numeric views and POD parameters. Host dispatch descriptors
never become device kernel arguments or asynchronous owners.

Raw sweeps gather B products then apply the local inverse, loaded once per
cell and reused across variables. Periodic component transformations and face
transpose conventions are preserved. Distinct inputs/outputs, scheduled leases
and immutable plan retention remain responsibilities of the common queued
operator. Numerical kernels contain no allocation, lease acquisition, MPI or
host synchronization. Unsupported device/build/shape combinations reject.

## Interpreting measurements

Compare matched mesh, polynomial and quadrature choices, solver iterations,
precision, and placement. Report model/effective bandwidth based on useful
nonduplicated bytes separately from measured physical DRAM traffic. Include
capacity, alignment padding and host/device placement in memory comparisons.
Runtime allocation snapshots count registered allocations; provider-private,
allocator, driver and other-process memory are not implied by that total.
