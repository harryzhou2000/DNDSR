# OpenMPI/CUDA finalize isolation, 2026-10-26

The CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED (713) at accelerator_cuda_finalize
reproduces independently of DNDSR. This executable includes only MPI and CUDA,
allocates one 4-byte device integer, executes one ordinary non-template kernel,
copies the exact result37 to host, performs a host-buffer MPI_Allreduce, frees
its device allocation, and calls MPI_Finalize. No DNDS, Runtime array, Thrust,
CUB, registered application host buffer, device reset or separable compilation
is present.

All execution used exactly one MPI process and only physical GPU1 UUID
GPU-0848655f-4530-ea83-ca9b-c091c5c6f36a. The visible GPU is logical device0.
No package, installed file, service or persistent environment configuration
was changed. Sources, binary and logs are under this /tmp directory only.

| Order | Compute Sanitizer | Ordinary reference |
| --- | --- | --- |
| MPI_Init, then CUDA context/kernel | exit91, 1 unsuppressed error713 | exit0 |
| CUDA context, then MPI_Init, same CUDA kernel | exit0, 0 errors | exit0 |

All four runs print value37/total37/expected37 and MPI_Finalize returned.
The changed order is a diagnostic control, not a tested DNDSR initialization
policy or a recommendation to alter rank-to-GPU assignment.

The failing minimal stack matches the full test at
/tmp/geom-native-batch10-cuda-memcheck.log exactly through
accelerator_cuda_finalize[0x25c5], opal_finalize_cleanup_domain, opal_finalize,
ompi_rte_finalize, ompi_mpi_instance_finalize_common,
ompi_mpi_instance_finalize, ompi_mpi_finalize.

Installed OpenMPI5.0.9 source:
/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/opal/mca/accelerator/cuda/accelerator_cuda_component.c
lines145-155: delayed initialization returns before registering checkmem if
there is no current CUDA context.
lines165-177: successful delayed initialization registers checkmem.
lines199-215: accelerator_cuda_finalize unconditionally attempts
cuMemHostUnregister(&checkmem), including when registration never occurred.
This is a source-grounded explanation consistent with the controlled change.
It establishes this particular finalize error does not require DNDSR; it does
not prove every full-application sanitizer finding external or harmless.

Recorded toolchain: nvcc13.2.51, Compute Sanitizer2026.1.0.0, OpenMPI5.0.9,
driver595.71.05, NVIDIA A10080GB PCIe. See provenance.log and libraries.log.
No API-error filtering or suppressions were used. In particular no accelerator
MCA override was applied. The checkpoint7 duplicate-CUB registration issue was
not exercised by this single-TU/CUB-free isolation; the full DNDSR sanitizer
run remains nonzero with both original findings recorded.

Evidence: commands.txt contains exact commands and exits; run.py reproduces
all four cases; results.json records elapsed times/exits; *memcheck.log and
*plain.log are complete stdout/stderr; sha256.json identifies sources, binary,
tools, installed accelerator component and matching source. artifacts-sha256.json
identifies every evidence file (except itself). No broad DNDSR test build/run
was performed by this isolation.
