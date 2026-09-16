---
name: numerical-experiment
description: Plan, execute, monitor, and record bounded DNDSR numerical experiments on a local or direct-SSH runner. Use for solver trials, parameter studies, convergence experiments, numerical-method comparisons, and CFD evidence generation; combine with cluster-run when execution requires Slurm or another batch scheduler.
---

# Numerical Experiment

Produce reproducible numerical evidence without allowing exploratory runs to become unbounded or raw output to pollute the tracked research record.

This skill directly manages local and non-scheduled SSH runners. When a remote
machine requires `sbatch`, `srun`, `salloc`, or an equivalent scheduler, also
use `cluster-run` for checkout preparation, allocation, submission, and job
lifecycle handling.

## Required Inputs

Before launching a numerical case, establish all four inputs:

- `runner`: `local` or an identified remote machine;
- allocation: MPI rank count `np` or an explicit CPU-usage limit;
- direction: the numerical question, hypothesis, or comparison;
- `workspace_name`: the experiment repository at `<repo>/workspace/<workspace_name>`.

If runner or allocation is absent, proceed with planning, source inspection, existing-data analysis, and dry-run preparation, but do not launch a solver. Ask for the missing execution input at the next meaningful run boundary rather than blocking earlier useful work.

Read [references/experiment_protocol.md](references/experiment_protocol.md) before creating the workspace or launching, monitoring, modifying, or reporting an experiment.

## Subagent Policy

When subagents are available, dispatch regular or mechanical execution work to
Luna, or to the user's specified available low-cost model. This includes file
transfer, remote repository updates, compilation, solver launches, and polling
running cases. Give each dispatch explicit paths, commands or intended state,
runtime bounds, stopping conditions, and the evidence it must return.

For one continuous experiment task, reuse the same subagent thread for its
successive mechanical actions (for example, staging, launch, polling, fetch,
and reduction). Start a new thread only when the prior one is unavailable or a
separate independent task needs parallel work; state why continuity was not
possible.

Use a maximum wait of 20 minutes for one ordinary subagent polling round. For
an explicitly authorized long-running task expected to finish within one hour,
one polling round may wait up to one hour. If the process is expected to run
for more than one hour, the main agent defaults to stopping active checks after
verifying that it is safely detached. Report the detached state, process or job
handle, log and record paths, last verified progress, stop conditions, and ETA;
resume polling only in a later turn or when the user explicitly requests it.
A polling timeout is only an observation timeout and never authorizes restarting
the process. Subagent completion is not a persistent hook that can wake a main
agent after its turn has ended, so do not rely on it for detached-job reporting.
These polling limits do not extend the solver's declared wall-time bound.

For a scientific, numerical, or provenance audit, use the default subagent
model rather than deliberately selecting a cheaper model, unless the user
specifies otherwise. The main agent retains experimental design, safety and
authorization decisions, code or numerical reasoning, interpretation, and
final verification; delegation does not expand permission to mutate Git state
or launch work.

## Core Workflow

1. Verify the repository root, applicable `AGENTS.md`, current Git status, available build, case path conventions, and whether `.codegraph/` requires CodeGraph-first navigation.
2. Create or reuse `<repo>/workspace/<workspace_name>` as an independent Git repository, never a DNDSR submodule. Preserve any existing workspace contents and history.
3. Record the experiment plan before execution: hypothesis, controls, observables, runner/allocation, time budget, stopping conditions, raw-output paths, and tracked derived artifacts.
4. Establish the cheapest discriminating trial first. Estimate wall time from a prior comparable run or a short timed pilot.
   For MPI solver runs, use the maximum rank count allowed by the user for the first timed run unless the runner cannot provide it or the case is known to require fewer ranks.
5. Make configuration changes in place with the editing tool. Preserve JSON notes and comments; never rewrite maintained configs with `json.dump`.
6. Build only required targets. For Python tests after C++ changes, rebuild and install every required pybind11 target before testing, as required by the repository instructions.
7. Launch only after the input contract is complete. Record the exact command, runner, allocation, source state, config snapshot/hash, start time, and expected stop time.
8. Monitor long expected runs through the low-cost subagent required above. Monitoring observes progress and enforces stop conditions; it does not silently alter the experiment. Apply the polling or detached-state policy above. Use the ordinary 10-minute solver bound for normal trials.
9. For numerous consecutive rows, launch through a checked orchestration script rather than a sequence of ad hoc shell commands. The script must enumerate the intended cases, default to serial execution unless parallelism is explicitly justified, enforce each row's bound, record or skip only verified terminal rows, stop or continue on failure by an explicit option, and print the campaign log/record roots before launch. Keep the orchestrator and its declared matrix in the workspace so the sequence is reproducible.
10. Fetch only compact post-processed lines, profiles, critical-point data, tables, and plots into the workspace. Keep ordinary raw numerical output under `<repo>/data/...` according to config conventions.
11. Validate numerical plausibility and compare against the stated reference or invariant. Report verified results separately from partial, failed, or unrun work.

## Non-Negotiable Bounds

- A normal end-to-end trial that produces numerical results must be designed to finish within 10 minutes.
- Bound total work with physical `nTimeStep` or final time and with pseudo/internal step caps. Also set convergence thresholds and output cadence deliberately.
- Never extrapolate a short pilot into a claimed result. Mark pilots as pilots.
- Do not start, stop, restart, or repair unrelated services.
- Do not commit, push, create PRs, or mutate remote Git state without the authorization required by repository policy.
- Do not discard user changes. Check status before branch, stash, restore, or checkout operations.

## When a Trial Is Too Slow

Use the protocol's adaptation order and record each change. Prefer more cores when scaling is credible. A smaller mesh is acceptable without asking only for simple reproducible geometry; ask the user before simplifying complex geometry. Coarser physical steps, mesh coarsening, relaxed internal thresholds, internal-step caps that stop before threshold, and CFL tuning can change accuracy or numerical behavior, so treat each as an experimental factor and re-establish result quality.

## Completion Standard

The experiment is complete only when another researcher can identify the exact source/config state, reproduce the command on the stated runner/allocation, locate raw and tracked artifacts, understand runtime bounds, and distinguish the observation from its interpretation and limitations.
