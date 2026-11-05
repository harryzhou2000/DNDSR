# Routine CPU acceptance, November6

All **428 routine registrations have passing evidence across two batches**.
The first serial sweep passed420 and failed8 because this isolated checkout
lacked `UP3D_32.cgns`. After copying the unchanged canonical fixture, exactly
those8 reorder/transition entries passed at np1/2/4/8. Original failures remain
in the archive. No test, threshold, executable or source changed for the retry;
this is not described as one all-green sweep.

Source `8964181b`, production source through `4decc781`; later changes are
documentation and the opt-in benchmark diagnostic. Release GCC13, Open
MPI5.0.9, Cantera on, CUDA off. Eight compiler workers; CTest runs serially
with its recorded per-test environments, normally OMP2. Other host work was
present, so these are correctness/test-cost observations, not matched solver
performance or a cold-build comparison.

|Gate|Observed wall time|Outcome|
|---|---:|---|
|Initial aggregate build|929.57s|exit0|
|Companion-policy focused build|75.98s|exit0|
|Focused policy np1/2/4|0.92s|3/3 pass|
|Incremental aggregate after header fix|1024.60s|exit0|
|Routine CTest sweep|333.52s|420 pass,8 missing-fixture failures|
|Unchanged-test fixture retry|50.40s|8/8 pass|
|Benchmark diagnostic build/smoke|6.27s/0.21s|exit0; np2 packed worker oracles pass|

The fixture is3,237,859bytes with SHA256
`720f484e11db3ff129c0086da9784f825bda18efe4080290773e5fd6da5f736e`.
It is ignored run data, not another committed mesh blob. Its missing-file
exceptions account for all8 initial failures. The first ad-hoc summary parser
required whitespace before CTest's failure marker and rejected the incomplete
inventory; the corrected parser checks all428 names and the exact8-entry retry.
That parsing failure was not a numerical rerun or an accepted pass.

The configured inventory retains469 entries:466 labeled C++ entries, comprising
428 routine/22 extended/16 integration, plus3 other entries. Extended and
integration tiers were not run in this final batch; earlier focused evidence
keeps its own pins. Installed Python extensions, CUDA/NCCL and distributed
sanitizers are not qualified by this CPU gate. The22 tooling tests passed in
3.108s at their recorded preceding source, unchanged in relevant script files.

Both aggregates completed before the routine run. The mechanical agent hit
model capacity after its incremental build; root verified no lingering build
or test process before starting CTest once. No duplicate suite was launched.
The corrected compute-worker tail probe uses actual completion timestamps;
the earlier E4/E5 campaign retains its explicitly documented zero placeholder.
Its smoke is a functional check, not a new timing comparison.

`summary.json` records every registration's outcome, commands, source/cache
pins and the separate retry. `records.tar.gz` preserves the full inventory,
logs and build statuses; `SHA256SUMS` checks both compact artifacts. The other
implementation checkout and MG/IV experiment work were not modified.
