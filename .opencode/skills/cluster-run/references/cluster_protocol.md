# Scheduled Cluster Protocol

## Profile Discovery

Capture the following facts in as few remote calls as practical. Treat all
scheduler capacity and queue observations as time-sensitive.

| Area | Facts to record |
|---|---|
| Access | SSH alias, resolved host, user, repository path |
| Scheduler | implementation/version, partitions, limits, account/QoS rules |
| Allocation | nodes, tasks per node, CPUs per task, GPUs, memory, wall time |
| Build | source path, build path, generator, compiler/MPI, login-node policy |
| Launch | example script, MPI launcher, environment script, threading rules |
| Evidence | log path, job ID, output path, accounting command, transfer path |

For Slurm, prefer a single probe containing `sinfo`, the selected partition's
`scontrol show partition`, `squeue -u "$USER"`, the launcher contents, one Git
status, and targeted build metadata. Avoid recursive `find` or repeated Git
commands on cold metadata unless a concrete question requires them.

## Local Gate Before Cluster Work

Before a cluster compile or run:

1. Check out the desired source locally without discarding unrelated work.
2. Build the exact required target in the normal local build directory.
3. Run the cheapest meaningful local smoke with the intended config overlay.
4. Record the local commit, command, status, and relevant executable/config
   hashes.

Platform-specific cluster compilation can still fail, but the cluster should
not be used to discover ordinary syntax, linkage, config, or startup errors.

## Remote Source Decision Tree

Resolve a full desired commit ID from the intended remote before changing the
checkout. Then perform one remote `git status --short --branch`.

### Clean Checkout

Fetch only the needed remote/ref, verify the commit, and move the established
checkout to it. Avoid an unqualified `git pull`, which mixes source selection
with merge/rebase behavior.

### Dirty Checkout

List tracked modifications and untracked collisions. Compare their content to
the desired commit:

- A tracked path is already represented only when its working-tree bytes match
  `<desired-commit>:<path>`.
- An untracked path is already represented only when the desired commit tracks
  the same path with identical bytes.
- If every affected path passes, record the evidence and align to the desired
  commit; no unique content is being discarded.
- Otherwise ask the user to choose a named stash or an isolated worktree/clone.
  Do not choose silently.

After a stash workflow, report the stash name and leave it intact unless the
user explicitly requests restoration. After any checkout change, verify `HEAD`,
status, and the source directory recorded in the build cache.

## Isolated Checkout Dependencies

Use an isolated checkout only when requested or when the established checkout
cannot be safely changed. Source and build must remain paired; do not point a
new worktree at an old build tree configured for another source directory.

Large populated dependencies may be linked from the established checkout:

1. Verify the dependency target exists and record its resolved path/version.
2. Create the expected parent directory in the isolated checkout.
3. Link `external/cfd_externals` and required per-library header-only
   directories rather than copying them.
4. Never replace a real directory or existing link without inspecting it.
5. Reconfigure only the isolated checkout's matching build directory.

## Slurm Compilation

When login-node compilation is prohibited, use one scheduled build task with an
explicit CPU allocation. Keep build parallelism at or below CPUs per task:

```bash
srun --partition=<compile_partition> --nodes=1 --ntasks=1 \
  --cpus-per-task=<build_cpus> --time=<build_limit> \
  bash -lc 'cd <repo> && source <cluster_env> && \
            cmake --build <build> --target <target> -j<build_cpus>'
```

Use the existing CMake configuration when it points to the current checkout.
Do not clean or rebuild unrelated targets. If the scheduled build cannot fit
the partition limit, ask before changing partition or build strategy.

## Slurm Submission Record

Create a run-specific script rather than modifying a shared launcher in place.
The script must make these controls explicit:

- partition/account/QoS and wall time;
- nodes, tasks per node, CPUs per task, GPUs, and memory;
- `OMP_NUM_THREADS` consistent with CPUs per task;
- cluster MPI/environment initialization;
- working directory and exact executable/config arguments;
- deterministic stdout/stderr paths and a printed job ID/context header.

Submit with `sbatch --parsable`, capture the returned ID, and print the absolute
log directory before returning control. Preserve the final rendered script and
config hash with the experiment record.

## Monitoring and Completion

Prefer one low-frequency monitor keyed by job ID:

```bash
squeue -j <job_id> -o '%.18i|%.12P|%.24j|%.2t|%.10M|%.10l|%.4D|%R'
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocCPUS,MaxRSS
scontrol show job <job_id>
```

Queued time is not runtime. When the job leaves `squeue`, use accounting plus
the declared log path to determine the terminal state. Do not declare success
from disappearance alone. Cancel or requeue only within explicit authorization
and the declared stopping policy.

Fetch compact evidence with `rsync` or `scp`; do not recursively copy large raw
output merely for convenience. Record remote paths so another researcher can
recover the full data later.
