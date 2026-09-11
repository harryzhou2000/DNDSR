---
name: cluster-run
description: Prepare, submit, monitor, and record jobs on Slurm or another batch-scheduled cluster. Use when compilation or execution must obtain compute resources through a scheduler; use numerical-experiment instead for local or direct-SSH runs without a scheduler.
---

# Cluster Run

Run reproducible jobs through a scheduler without treating a login node like a
compute node or losing source provenance on a slow shared filesystem.

## Routing Boundary

Use this skill when the machine requires `sbatch`, `srun`, `salloc`, or an
equivalent scheduler. Use `numerical-experiment` for local or direct-SSH
execution. For scheduled numerical experiments, use both skills: this skill
owns cluster preparation and job lifecycle; `numerical-experiment` owns the
scientific plan, runtime bounds, data placement, and result validation.

## Required Cluster Contract

Before compiling or submitting, establish:

- cluster SSH alias and scheduler;
- remote repository and matching build directory;
- desired source commit or branch;
- whether compilation is allowed on login nodes or must be scheduled;
- partition, account/QoS if needed, nodes, tasks, CPUs per task, GPUs, memory,
  wall time, and threading model;
- launcher or submission template, executable command, working directory, log
  directory, and output location.

If fields are missing, inspect and prepare without launching. Ask at the first
operation whose correctness depends on the missing value.

## Subagent Policy

When subagents are available, dispatch regular or mechanical cluster work to
Luna, or to the user's specified available low-cost model. This includes file
transfer, remote checkout or repository updates, compilation, job launch, and
polling. Give each dispatch the cluster profile, exact paths, desired source
state, scheduler request, bounds, stopping conditions, and evidence to return.
The main agent retains source-state and authorization decisions, technical
reasoning, result interpretation, and final verification; delegation does not
authorize checkout mutation, job submission, cancellation, or other external
side effects by itself.

## Workflow

1. Read [the cluster protocol](references/cluster_protocol.md). If the chosen
   cluster has a profile, read it and revalidate its time-sensitive facts.
2. Bundle read-only discovery into one SSH call. Scheduler state and remote
   filesystems can be slow; avoid repeated Git status, recursive searches, and
   many short logins.
3. Build and smoke-test the required target locally at the desired source and
   config state before spending cluster allocation time. A cluster build or
   production run is not the first functional test.
4. Inspect the remote checkout once. By default, update the established remote
   checkout rather than creating another clone. Never silently destroy dirty
   work; follow the source-state decision tree in the protocol.
5. Use the build directory paired with that source checkout. Reuse its existing
   configuration and build only required targets unless reconfiguration is
   demonstrably necessary.
6. Compile only where the cluster contract allows it. If compute allocation is
   required, compile inside `srun` or `salloc`, never on the login node.
7. Materialize a run-specific submission script and config record. Print the
   absolute log directory, exact command, source commit, and scheduler request
   before submission. Submit once and capture the scheduler job ID.
8. Monitor by job ID with bounded, low-frequency polls. Distinguish queued,
   running, completed, timed out, cancelled, and failed states; a zero exit
   status does not establish numerical correctness.
9. Fetch compact logs, manifests, profiles, tables, and plots. Leave ordinary
   raw solver output in the configured remote data tree.

## Source-State Rule

Resolve the desired commit before changing the remote checkout. If it is clean,
fetch once and move it to that commit. If it is dirty:

- when every dirty or colliding file is byte-identical to the desired commit,
  record that comparison and align the checkout to the commit;
- otherwise, stop and ask whether to stash the changes or create an isolated
  checkout/worktree.

Do not infer that similarly named edits are already committed. For an isolated
checkout, keep source and build paired and link large populated dependencies
from the established checkout only after verifying each target.

## Cluster Profiles

- For `ssh thtj1`, read [references/thtj1.md](references/thtj1.md). Its DNDSR
  compilation must use `srun` on `debug6`; do not compile on the login node.

## Safety

- Do not submit, cancel, requeue, or mutate a remote checkout without the
  authorization required for that action.
- Do not put credentials, tokens, or private environment values in job scripts
  or logs.
- Do not reuse a build directory configured for a different source tree.
- Do not copy a cluster example script blindly: reconcile its partition,
  task/thread layout, time limit, environment, output path, and launcher.
- If repeated scheduler or remote-filesystem failures prevent progress, report
  the exact job/command and pause instead of retrying indefinitely.
