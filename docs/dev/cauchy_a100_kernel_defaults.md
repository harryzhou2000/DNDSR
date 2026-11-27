# Automatic A100 kernel selection

CUDA operator placement resolves `auto` selectors once, using the actual
device and equation/reconstruction shape. CPU arithmetic and scheduling are
unchanged. Kernel launches perform no hardware-name queries or tuning.

For the measured combined AoS, 3D P3 flow5 profile with quadrature order at most
five, A100 defaults are:

| Operator | Automatic selection |
| --- | --- |
| RHS | `ctaSegments`, deterministic segmented reduction, 128 threads by default |
| VR Jacobi | `fp64V2`, one warp per cell for 3D P3, FP64 MMA on compact full-inverse or precomputed tables |
| Polynomial PP | Warp per cell, 512 threads per block |
| Diagonal inverse | `guardedLU`, retaining the condition guard and filtered SVD fallback |
| Linear Jacobi | Existing thread-per-cell, compact matrix storage |

For coupled AoS **3D SA P2/Q3**, the A100 preset selects 512-thread
polynomial PP and a 256-thread FP64 RHS block. It leaves the mapping and
grid selection to the existing RHS resolver, including its periodic presets.
The block default does not override explicit block sizes, face-thread mapping,
double buffering, row hoisting, custom reductions or reduced arithmetic.
Direct-P1 launch geometry is unchanged. The hardware families sharing the
A100 preset inherit these shape choices; this is not a performance guarantee
on every device or mesh.

Cauchy selects `fp64V2` for A100 and A800 combined AoS polynomial relaxation.
A800 uses the A100 VR preset at compute capability 8.0. The other automatic
policies in the table retain their own device and shape criteria. CFV
selects the mapping from the **placed** tables. For C4–7, M2/M3 use eight
SIMT lanes per cell. M5/M9/M19 use one warp per cell. M5 with FP32 raw B
uses SIMT for C4–6 and MMA for C7. Limited M5/C6 also selects SIMT for
precomputed and FP64 raw matrices. The remaining M5 cases and all M9/M19
cases use FP64 MMA when the device supports it. M5 precomputed MMA uses
64 threads per block, the other presets use 128. The existing
`numerics.cfv.cudaSweep` settings override launch dimensions.

General FP64 covers arbitrary component widths, padded storage, a triangular
inverse, and limited sweeps at M2/M3. `fp64V2SIMT` and `fp64V2MMA` force
the compact optimized routes and report unsupported shapes. B and inverse
storage precisions remain independent. The legacy pipeline and comparison
selectors require a build with `DNDS_CFV_CUDA_ENABLE_V1_KERNELS=ON`.

`execution.kernels.rhsMapping`, `rhsReduction`,
and `polynomialPPThreads` override their respective automatic choices. Existing
`DNDS_CAUCHY_CUDA_MODAL_PP_*` environment overrides still take precedence.
`implicit.linear.cudaDiagonalInverse` accepts explicit `batchedSVD` or
`guardedLU`; `auto` selects guarded LU on A100 and batched SVD elsewhere.

Outside the measured A100 shape, FP64 RHS uses the same CTA mapping because it
is the sole consolidated mapping, not because every shape was benchmarked.
Reduced-precision flow5 comparisons use thread mapping. ACM/SST P2/Q3 in
3D on A100 uses CTA segments for modal FP64 face flux. P3 with face rule
order at least two retains CTA segments. Direct-gradient and mean face
states use the face-thread route.

On the 3D DoubleEllipse ACM/SST P3 benchmark (six components, 128
threads), the face-thread and CTA-segment kernels took 30.95 and
20.78 ms per update at Q3, and 70.37 and 38.87 ms at Q5. These are
face-flux kernel times. For assembled ACM/SST matrices,
`execution.kernels.acmSSTScalarJacobian` enables the interior scalar-column
optimization. Set it to false for full-row reference assembly.

`FiniteVolume::Settings()` on the placed plan exposes its concrete spatial
choices. The canonical input/resolved configuration retains the requested
`auto` intent; archived explicit benchmark configurations remain reproducible.
Already-running processes keep their original executable and configuration.

Fresh builds with `DNDS_USE_CUDA=ON` also default `DNDS_DIST_MT_USE_OMP=ON`.
An explicit cache value, including OFF, is preserved. Existing build caches do
not change implicitly. This controls CPU setup/host-kernel compilation; CUDA
kernels do not execute OpenMP threads.

## Plan 61-6 RHS source contract

The FP64 CTA-segment catalog uses one shared quadrature-point arithmetic body.
It supports only combined calorically perfect-gas flow5,
flow5+SA and flow5+2EQ, with no species or chemical source. The immutable
`RHSKernelParameters` must be validated before device publication; mixtures
cannot enter this profile by supplying another component width.

The host boundary projects combined AoS mean/incoming/output rows to static
5/6/7 extents and polynomial rows to C*M extents for M=2/5/9 in 2D and
M=3/9/19 in 3D. M=0 denotes non-VR constant/direct-gradient evaluation. Actual
field rows must be compact; padded input rows are rejected by the wrapper.
Direct-gradient and primitive-gradient views retain their appropriate fixed
widths. The typed views reach numerical leaves and shared staging without
decaying to a generic Array view. Dimension also specializes the derivative
arithmetic and indexing. Optional viscous inputs select a typed role on the
host, without another instantiation axis or an input copy.

One manifest generates 24 one-kernel CUDA TUs in the build directory. Block
width remains runtime; it does not multiply the catalog. The private common
body and CMake generator dependency contract are documented in
[kernel contracts](../architecture/CauchyKernelContracts.md#cuda-rhs-shape-catalog).
The local register cap keeps 512-thread launches within the register file and
does not constrain reduced-precision comparison TUs.
Cooperative reductions stay deterministic and retain complete-face selection,
tail-lane/barrier and scheduled-lifetime contracts. Polynomial PP retains its
runtime shape dispatch and numerical policy.
