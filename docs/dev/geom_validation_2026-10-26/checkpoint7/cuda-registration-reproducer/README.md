# CUDA duplicate kernel registration — bounded isolation

2026-10-26. No DNDSR source, build configuration, installed package or service was
changed. Minimal sources/binaries/logs are all in this directory. Each Thrust
translation unit allocates only 16 bytes plus eight doubles. The plain CUDA case
allocates one int. CUDA context/tool overhead is additional.

## Proven trigger and limit

CUDA 13.2.51 / CUB 3.2.0 / driver 595.71.05 emits duplicate-kernel-registration
extended API errors when the same kernel template is instantiated in multiple
CUDA translation units compiled with separable compilation (`-rdc=true`).
DNDSR mesh, storage, CFV and MPI APIs are not required. A plain CUDA template with
a real cross-TU device function also reproduces it, without Thrust/CUB.

This isolates an external compiler/runtime/driver registration interaction. It
does NOT establish which individual external component owns the defect, and
correct tiny numerical results do NOT prove all duplicate registrations safe.
No toolchain update was installed/tested, so no particular upgrade is a proven
fix. The DNDSR checkpoint's sanitizer summary remains NONZERO (two errors).

## Controlled results

All sanitizer runs used physical GPU1 only:
`CUDA_VISIBLE_DEVICES=1 OMPI_MCA_accelerator=null`, OpenMPI5 mpirun `-np 1`,
`compute-sanitizer --tool memcheck --error-exitcode 91`.
The repro does not call MPI; the launcher matches the requested environment.
GPU1 UUID: GPU-0848655f-4530-ea83-ca9b-c091c5c6f36a, NVIDIA A10080GB.
Read-only preflight free memory: 81140MiB of81920MiB.

| Case | Compilation | Result |
| --- | --- | --- |
| a.cu+b.cu Thrust vectors | RDC, shared CUDART, sm80 | 3 duplicate API errors; exit91; a9/b8 correct |
| Same sources | Whole-program mode, shared CUDART, sm80 | 0 errors; exit0; a9/b8 correct |
| Same RDC sources + static-global-template-stub=true | RDC | Still3 errors; exit91 |
| Concatenated one.cu | RDC | 0 errors; exit0 |
| plain_a.cu+plain_b.cu | RDC, no Thrust; real cross-TU device function | 1 duplicate API error; exit91; a7/b7 correct |
| Two-TU Thrust RDC without sanitizer, CUDA_LOG_FILE set | Uninstrumented | Same3 duplicates in CUDA's own log; exit0 |

`cuobjdump --list-elf rdc` reports a single final sm80 cubin; its ELF symbol table
contains one weak device entry per offending template. `nm` shows merged weak
host stubs. Thus this is not explained by DNDSR manually linking the same final
fatbinary twice. The observed registration issue still happens in nvcc's own
single-command compile+link of two independent tiny .cu files.

## Relationship to DNDSR

The affected CFV test and DeviceStorage_CUDA.cu both compile `-rdc=true`.
DeviceStorage uses thrust::device_vector<uint8_t>; the test instantiates double
vectors. DNDSR static archives defer device linking to the executable and contain
no separate cmake_device_link member. The executable has one final device-link
step consuming cfv/geom/dnds archives and shared libcudart.so.13. The checkpoint
errors are EmptyKernel<void> and CUB uninitialized_fill<double>, matching the
minimal reproducer's templates and extended logging mechanism.

## Next action

Keep RDC for actual cross-TU device dependencies. Do not globally disable it,
change kernel linkage speculatively, or suppress extended API errors to obtain a
clean result. The narrow next external fix investigation is to rerun this exact
reproducer against a vetted newer compiler/runtime/sanitizer combination (with
the same driver recorded), or obtain an NVIDIA fix using the attached plain-CUDA
reproducer. Compute Sanitizer2026.1.1 release notes mention silencing some
non-actionable extended errors, but do not identify this specific registration
case; that is context, not proof this issue is fixed. No upstream report was sent.

## Evidence

- `a.cu`, `b.cu`, `one.cu`, `plain.cuh`, `plain_a.cu`, `plain_b.cu`: minimal sources.
- `run.py`, `probe.py`, `commands.log`, `probes.log`, `plain-commands.log`: exact
  compiler/launcher commands and exit codes. No API-error filtering options.
- `*-memcheck.log`: full unsuppressed sanitizer output.
- `rdc-driver.log`, `rdc-uninstrumented.log`: no-sanitizer extended CUDA logging.
- `rdc-device-symbols.log`: actual final device symbols.
- `provenance.log`: compiler/tool versions, driver/GPU/free memory, imported
  CUDART path via ldd, and binary/runtime ELF build IDs.
- `toolchain-sha256.json`, `artifact-sha256.json`: exact source/tool/runtime hashes.

Primary documentation consulted:
https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html
  static-global-template-stub applies only in whole-program mode.
https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
  Extended mode captures internal CUDA logs not necessarily returned as API errors.
https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html
  2026.1 introduced default extended reporting;2026.1.1 adjusted some reports.
https://cmake.org/cmake/help/latest/prop_tgt/CUDA_RESOLVE_DEVICE_SYMBOLS.html
  Static-library device linking normally defers to the final executable/DSO.
