# Experiment and TPU workflow

This interface supplies editable **starter presets**, not optimal hyperparameters.
It manages experiments on existing TPU VMs. It does not install training packages,
provision VMs, or provide an automatic resume command. The trainer mathematics,
training/data implementations, and existing sweep launcher are unchanged.

## Controller and worker prerequisites

A **controller shell** is your laptop/workstation or another machine with a clean
checkout, Python 3.9+, Bash, Git, and (for live commands) authenticated `gcloud`
with TPU VM SSH access. Run `tpu.sh` there. A **TPU SSH shell** is a shell on just
one worker; `ssh` does not synchronize code or launch the other workers.

Each worker needs:

- A clean Git checkout at `/full-gauss-newton`, on the **same named branch and
  exact commit** as the controller for `run`. Include the tooling commit first.
- Bash, Linux `/proc`, Git, tmux, Python 3.9+, and the repository's working training
  environment, including JAX/libtpu, EasyLM dependencies, Hugging Face access,
  checkpoint bucket access, and W&B authentication for worker 0.
- `/etc/profile.d/gfn_env.sh`, readable by `ga2740`, which activates that environment
  and supplies required credentials. It is sourced over SSH and **again inside
  the tmux pane** (a preexisting tmux server may have an older environment).
- Optional `tpu-info` for status output; otherwise status queries libtpu monitoring
  for per-chip duty cycle, tensorcore utilization, and HBM usage/capacity. A missing
  or unavailable monitor is reported explicitly.

No new Python dependencies are needed by the tooling or its tests. Dry-run needs
only Bash/Python and the checkout: no gcloud, credentials, JAX, trainer imports,
network, or TPU access.

All site defaults are in the configuration section of `scripts/tpu_control.py`:

| Setting | Default | Optional environment override |
| --- | --- | --- |
| Target `1` | `gn-v4-32` | `GFN_TPU_NAME` overrides the selected name |
| Target `2` | `gn-v4-32-2` | `GFN_TPU_NAME` |
| Project | `fast-second-order` | `GFN_PROJECT` |
| Zone | `us-central2-b` | `GFN_ZONE` |
| SSH user | `ga2740` | `GFN_SSH_USER` |
| Worker repository | `/full-gauss-newton` | `GFN_REMOTE_REPO` |
| Worker environment | `/etc/profile.d/gfn_env.sh` | `GFN_REMOTE_ENV` |
| Worker records | `/tmp/gfn-tpu-ga2740` | `GFN_STATE_DIR` |

Use the same overrides for later status/logs/stop operations. The records directory
must be outside the repository (including symlinks). Normal usage requires no
configuration file. Local paths are resolved from the scripts, independent of CWD.

## First use and code synchronization

Preview from the controller's repository root:

```bash
git switch codex/tpu-workflow
bash scripts/tpu.sh help
bash scripts/tpu.sh 1 run gn_adam --dry-run --inner_loop_iter=100
bash scripts/tpu.sh 2 run pcg_jacobi --dry-run --cg_maxiter=1000
python3 -m unittest discover -s scripts/tests -v
```

Before a live run, an operator must bootstrap the tooling onto **every** worker.
The new helper cannot bootstrap itself onto a checkout that lacks it. For example,
after the tooling branch is pushed, use an existing administrative SSH workflow to
fetch and fast-forward each clean, idle worker to the controller's exact tooling
branch and SHA. Inspect each worker for running jobs and dirty files first. Do not
use a reset or stash to force agreement. This implementation work and its tests do
not perform that synchronization or contact the TPUs.

Once those prerequisites hold, these are live commands to run from the controller:

```bash
bash scripts/tpu.sh 1 status
bash scripts/tpu.sh 1 run gn_adam --inner_loop_iter=100
bash scripts/tpu.sh 1 logs
```

For the epoch version use `git switch codex/tpu-workflow-epoch` on the controller
and synchronize that same branch/SHA on every worker first. After the PRs are
merged separately, normal `main`/`epoch` checkouts can be used instead.

`pull`/`switch` require identical executable helpers (`tpu.sh`, `tpu_control.py`,
`tpu_worker.py`, and `run_experiment.py`) on the controller, current workers, and
**target commit**. Preset contents and inventory may differ: updates can deliver
committed preset-only changes while workers still have the previous presets.
All workers must still start idle, clean, and at the same branch/SHA, and converge
to one clean target commit. `run` continues to require the controller and every
worker to match in branch/SHA, helpers, and the full preset inventory and contents.

Helper-code changes require manual bootstrap of the same revision on the
controller and every worker first, including upgrading from the original workflow
to this corrected updater. Updates refuse missing or changed target helpers so
the helper remains available to finish verification and release reservations.
Following a remote `switch epoch` or preset update, also switch/update the
controller to the same resulting branch/SHA before `run`.

## Command reference

From the root use `bash scripts/tpu.sh ...`; from `scripts/` use `bash tpu.sh ...`.
Worker numbering is the zero-based index from the TPU description's
`networkEndpoints`, and is range-checked for `logs` and `ssh`.

```bash
bash scripts/tpu.sh 1 run gn_adam --inner_loop_iter=100
cd scripts
bash tpu.sh 2 run pcg_jacobi --cg_maxiter=1000
bash tpu.sh 2 run gn_adam --dry-run --inner_loop_iter=100
bash tpu.sh 2 status
bash tpu.sh 2 logs
bash tpu.sh 2 logs 2
bash tpu.sh 2 stop
bash tpu.sh 2 pull
bash tpu.sh 2 switch epoch
bash tpu.sh 2 ssh
bash tpu.sh 2 ssh 2
bash tpu.sh help
```

- `run PRESET [--flag=value ...]`: a preset is required; omission lists choices.
  Dry-run prints target, preset, fresh run ID, merged command, output location,
  environment identity, and launch plan, and launches nothing.
- `status`: shows branch/SHA, dirty paths, managed tmux sessions, managed records,
  training processes (GN, ordinary Adam, layerwise, and legacy sweep processes),
  unmanaged processes, and TPU usage for every worker. Worker failures are reported.
- `logs [WORKER]`: follows the latest managed run's worker-local log, default 0.
  Ctrl+C ends the log connection and leaves tmux/training running.
- `stop`: signals only recorded, owned process groups/supervisors on all workers.
  SIGTERM gets up to 10 seconds, then SIGKILL gets up to 5 seconds. PID birth times
  and an inherited ownership token guard against PID reuse/unrelated processes.
  It verifies that the managed group/session ended. Unmanaged/legacy activity is
  reported for explicit inspection and is never killed by this command. Logs and
  checkpoints are retained. Unverifiable ownership causes refusal, not a broad kill.
- `pull`: fetches and fast-forwards the workers' current named branch.
- `switch BRANCH`: fetches that origin branch, switches/creates its local tracking
  branch, and fast-forwards to one resolved commit. Both update commands reserve
  and preflight **all workers first**, refuse dirty/active/differing workers or
  divergent local target branches, and verify clean branch/SHA convergence.
  Fetch failures cause no working-tree updates. Apply/connection failures can
  leave different workers at different commits and are reported as incomplete;
  inspect status and reconcile explicitly. No reset, stash, or rollback is used.
- `ssh [WORKER]`: opens an interactive SSH shell, default worker 0.

## Presets and overrides

The flat JSON files in `scripts/presets/` are easy to edit. `_common.json`, `_gn.json`,
`_adamw.json`, and `_inner_adam.json` factor shared defaults; there is no recursive
inheritance. Resolution order is common, GN defaults when applicable, nested AdamW,
inner Adam defaults when applicable, preset settings, then explicit overrides.

| Preset | Trainer/solver | Starter settings |
| --- | --- | --- |
| `gn_adam` | GN, AdamW inner solver | 50 inner steps, LR .001, global cosine |
| `gn_muon` | GN, Muon inner solver | 50 inner steps, LR .003, global cosine |
| `gn_sgd` | GN, **SGD inner solver** | 200 inner steps, LR .1, constant, zero momentum/decay/clipping |
| `gn_rmsprop` | GN, AdamW inner family with `inner_b1=0` | Adam-GN settings with no first-moment averaging; no invented `rmsprop` type |
| `pcg_jacobi` | GN PCG with GN-Jacobi | lambda 1, damping .00001, 4 probes, maxiter 10000, nested AdamW b1=0 |
| `pcg_interpolated` | GN PCG with Adam diagonal | lambda .1, damping 0, maxiter 256, nested ordinary AdamW settings |
| `adam` | Ordinary first-order AdamW | direct `llama_train` execution, FP32 compute and parameters |

GN/PCG uses `llama_train_gn` through the existing sweep launcher's main routine.
Both PCG presets disable batch lambda scheduling, lambda ramps, condition logging,
and outer momentum. Overrides may enable supported scheduling explicitly.

Shared starters: LLaMA 150M, `google-t5/t5-base`, streaming `allenai/c4` English
train/validation text, no added BOS, sequence length 1024, global train batch 720,
eval batch 128, mesh `16,1,1`, BF16 compute/FP32 parameters (Adam uses FP32), 3000
steps, log/eval frequency 1, 50 evaluation batches. Training shuffle uses seed 42
and buffer 100000; validation is unshuffled. Checkpoint saving is disabled by
frequency; if enabled, optimizer-state saving and FP32 checkpoint dtype are set.
Nested AdamW starts at LR .01, b1 .9, b2 .95, cosine schedule, init LR .0001,
end LR .001, warmup 10, decay 3000, accumulation 1, weight decay 0, clip 1,
FP32 momentum, and no parameter-scale multiplication. See the JSON files for
all inner-solver settings and the supplied warmup checkpoint.

Use one shell argument per override. Dotted names, booleans, empty values, spaces,
quotes, `=`, and literal shell syntax are preserved. **Last explicit value wins**
and each final trainer flag appears once. There is no eval of argument strings.

```bash
bash scripts/tpu.sh 2 run gn_adam --dry-run \
  --inner_loop_iter=50 --inner_loop_iter=100 \
  --linesearch=False --load_dataset_state= \
  '--notes=trial A: x=y & literal $HOME' \
  --optimizer.adamw_optimizer.b1=0
bash scripts/tpu.sh 2 run adam --dry-run \
  --train_dataset.huggingface_dataset.batch_size=1440
```

Setting either `train_dataset_batch_size` or
`train_dataset.huggingface_dataset.batch_size` updates both. Conflicting explicit
values are rejected after resolving repeated overrides. Trainer-specific flags
are passed through for the selected branch; trainers remain responsible for their
semantic validation. Ordinary Adam presets contain no GN-only flags. The interface
does not translate GN overrides into ordinary optimizer settings.

`program`, `experiment_id`, `wandb_run_id`, and sweep-only `resume_from` are reserved
by the interface and rejected as overrides. Use checkpoint flags to load existing
state while creating a new experiment identity.

## Identity, launch safety, and records

The controller generates one timestamp/UUID run ID for all workers. GN sets
`SLURM_ARRAY_TASK_ID=1` and a shared `SLURM_ARRAY_JOB_ID`; the sweep launcher adds
`_0`. Adam directly uses the controller's ID. Output goes to
`output_dir/experiment_id`, by default under `gs://gfn-soo/checkpoints/script_runs`.
A normal invocation always creates a fresh identity; it does not reuse inherited
W&B IDs, names, resume settings, SLURM IDs, or previous experiment directories.
Worker 0 has inherited `WANDB_MODE` removed so a controller/worker environment
with `WANDB_MODE=disabled` cannot silently disable its reporting; all other
workers get `WANDB_MODE=disabled`.
`PYTHONUNBUFFERED=1` and the worker repository's `PYTHONPATH` are set for both routes.

A controller file lock and token-owned reservations on every worker serialize
managed run/stop/update operations. Run checks clean matching branch/SHA,
matching helper/preset hashes, the environment and required executables, no
managed tmux sessions, and no training processes **on every worker before any
launch**. A mismatch produces a refusal and sync guidance; run never auto-pulls.
Each worker rechecks preflight while holding its reservation just before launch.

Each worker starts a unique tmux session, a supervisor, and a separate training
process group. The supervisor records process identities and exit status. A partial
launch failure attempts cleanup on **all workers for that run ID only**, including
workers with an ambiguous SSH acknowledgement, and reports cleanup failures.
“Process started” confirms process creation, **not successful training
initialization**; inspect logs/status for JAX setup, model/data loading, the first
training update, and subsequent worker failures. Workers are not continually
monitored by the controller after it returns.

Records are in `/tmp/gfn-tpu-ga2740/RUN_ID/` on each worker:

- `worker.log`: stdout/stderr, retained after exit/stop.
- `manifest.json`: run ID, preset, explicit overrides, resolved arguments, branch
  and Git SHA, target and worker ID, process/session identities, lifecycle state,
  and exit status (null while running; negative signal status for signaled children,
  or 143 for cancellation before an exit acknowledgement).
- `plan.json`: the resolved, shared experiment plan.

No environment variables or credentials are dumped into records. Do not put
credentials in trainer overrides or notes. These are **worker-local** files: they
do not survive VM loss or `/tmp` expiry. Checkpoints use the configured output
bucket; copy wanted logs/manifests to durable storage through your existing
workflow. There is no bulk `/tmp` cleanup command.

A lost controller can leave an operation reservation. The next operation refuses
it with its path. Inspect the previous controller and managed processes before
an operator repairs that reservation; there is no automatic lock expiry that
could overlap an active launch.

## Checkpoint and recovery semantics

The starter is a **parameter-only** warm start:

```text
trainstate_params::gs://gfn-soo/checkpoints/warmup_150m/1_0/streaming_train_state
```

`load_dataset_state`, `load_llama_config`, and `update_llama_config` start empty.
Checkpoint flags are passed unchanged. A fresh run identity does not prevent
explicitly loading a checkpoint, and it does not mean optimizer/data continuation
is interchangeable between solvers.

- `load_checkpoint=trainstate_params::PATH` extracts weights from a streaming
  train state and creates new optimizer state. `params::PATH` loads a streaming
  parameter file; `flax_params::PATH` loads a Flax parameter file. These do not
  constitute exact optimizer/data recovery. `load_dataset_state=PATH` independently
  restores a compatible dataset cursor, subject to trainer checks.
- `log_step_offset`, `log_token_offset`, and `log_time_offset_s` are reporting
  controls, not optimizer/schedule continuation. Parameter-only cumulative branches
  need compatible parent metadata/dataset/completion companions and explicit
  offsets where required by the trainer. Zero-origin experiments use fresh counters.
- For ordinary raw-HF Adam, `load_checkpoint=trainstate::PATH` restores full state
  only with its matching `metadata[...].pkl`, `dataset[...].pkl`, and
  `complete[...].pkl` companion bundle. Exact recovery requires FP32 state,
  optimizer saving, accumulation 1, no weight averaging, matching trajectory
  settings and compatible raw-HF evaluation. The trainer validates snapshot/RNG,
  packing, and progress consistency. `start_step` is not an equivalent recovery.
- CG uses `cg_resume_state=PATH_TO_cg_state_0_OR_1`, including its checksum-validated
  metadata/dataset/completion companions. Matching trajectory settings and current
  data-consumption metadata are required. It preserves solver/data state; a plain
  parameter checkpoint is a new solve. The GN trainer rejects
  `load_checkpoint=trainstate::...` for all solvers; GN Adam/Muon/SGD must use
  parameter-only starts rather than full-state continuation.
- `load_ema_checkpoint` is a separate EMA parameter load when supported, not full
  recovery. See `docs/checkpointing.md` and `docs/cg_recovery.md` and the trainer's
  validation for the selected branch.

The existing sweep launcher checks output directories and can auto-load a previous
identity on its legacy invocation path. This interface supplies a new UUID-based
identity every time and does not expose `resume_from`. For its single experiment,
`run_experiment.py` resolves the launcher's static configuration before calling
its existing `main`; this avoids interpreting `&` in a quoted value as a sweep
and lets the launcher append the nested batch flag exactly once. Existing sweep
commands and parsing behavior remain untouched. Explicit checkpoint recovery
creates a new logging/output experiment here; reattaching an old W&B history is
outside this interface.

## Epoch replay restrictions

The same tooling commit applies to both bases; replay flags are absent from common
presets. `epoch` accepts explicit
`train_dataset.huggingface_dataset.repeat_first_n_batches=N`. `main` does not define
this flag. The tooling passes it through without promising solver support.

The existing epoch trainer authorizes GN replay only for non-adaptive **Adam-GN**
(`optimizer_type=adamw`, `gauss_newton=True`), positive inner iterations,
`single_batch_inner=True`, raw HF streaming, matching fixed positive batch sizes,
no batch growth, zero-origin solve tokens, explicit positive `ls_eval_batches`,
fresh solver/data initialization, parameter-only checkpoint loading, empty
`load_dataset_state`/`cg_resume_state`, **explicit zero** reporting offsets, and
zero model/milestone save frequencies. Validation replay stays disabled. The
line-search stream is independent and single-pass, using the next shuffle seed;
only the solve stream repeats. Muon-GN, SGD-GN, and PCG replay are unsupported.

An epoch-only preview for the authorized GN path:

```bash
bash scripts/tpu.sh 2 run gn_adam --dry-run \
  --train_dataset.huggingface_dataset.repeat_first_n_batches=100 \
  --log_step_offset=0 --log_token_offset=0
```

Epoch's first-order replay validator separately requires raw-HF ordinary AdamW,
accumulation 1, no pretokenization/weight averaging/outer-loop method, zero-origin
start tokens and start step, zero/default reporting offsets, and validation replay
disabled. A full-state transition from non-replay to replay has additional FP32,
complete packed checkpoint, parent budget, and progress restrictions. See
`validate_replay_config`/`validate_resume_metadata` in `EasyLM/training_resume.py`
on epoch. None of these restrictions are relaxed by the tooling.

## Branch delivery and local validation

The implementation is committed on `codex/tpu-workflow` from `origin/main`, then
cherry-picked onto `codex/tpu-workflow-epoch` from `origin/epoch`. Open separate PRs
against `main` and `epoch`; do not merge either base into the other. All changed
paths are under `scripts/` and `docs/`, preserving branch-specific trainer/data
behavior.

Run on each tooling branch:

```bash
bash -n scripts/tpu.sh
python3 -m py_compile scripts/run_experiment.py scripts/tpu_control.py scripts/tpu_worker.py
python3 -m unittest discover -s scripts/tests -v
git diff --check
```

Tests use a fake gcloud executable that decodes/logs RPC requests without executing
them, mocked tmux, local Git fixture repositories, and harmless local sleep
processes for ownership/termination checks. They do not contact real TPUs or import
trainers/JAX. AST checks verify every preset flag against the checked-out branch's
trainer and nested configuration definitions.

## TPU-resident synchronized GN-Adam queues (experimental, operator approval required)

Unlike `sweep_launcher.py` (one SLURM-selected job) and `gn_sweep_queue.py`
(which polls worker manifests from box248), the resident queue uses a **managed
queue process on every TPU worker**. A one-time authenticated launch from box248
reserves, preflights and starts the same immutable queue on every discovered
worker; the controller returns. Worker-local processes exchange small conditional
GCS markers, so **box248 authentication and SSH are not required between trials**.

The ordered default screening configuration is
`scripts/sweeps/gn_large_batch_remaining.json`. The **new six-trial
follow-up**, including K25/K100 and per-trial 75M/150M/200M solve-token
budgets, is `scripts/sweeps/gn_batch_k_followup.json`.

Before the expensive six-trial follow-up, run the minimal **two-trial
integration smoke test** in `scripts/sweeps/gn_resident_smoke.json`:
each B256 trial performs one outer update, first K1 then K2. This verifies
real TPU worker initialization, GCS ready/result synchronization, W&B,
and automatic progression after a completed trial. It **does not**
benchmark the scientific value of these K settings.

```bash
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_resident_smoke.json --dry-run
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_resident_smoke.json
bash scripts/tpu.sh 1 logs 0
bash scripts/tpu.sh 1 status
```

Do not start the full queue until all workers have returned successful
exit statuses and the worker0 log says `ALL TRIALS COMPLETE`.
On failure, inspect `logs` and the managed worker manifests;
**do not assume CPU-only CI proves GCS permissions or live JAX initialization**.

For the follow-up run, use:

```bash
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_batch_k_followup.json --dry-run
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_batch_k_followup.json
```

Its ordered trials are B8192/LR0.002/K50 (150M), B4096/LR0.002/K25
(150M), B4096/LR0.002/K100 (75M), B4096/LR0.004/K50 (200M),
B2048/LR0.002/K50 (200M), B3072/LR0.002/K50 (200M). The B3072
trial uses `n_micro=6`; each B3072, B2048, B4096, and B8192 configuration
uses 32 samples per data-shard microbatch with mesh `16,1,1`.
All runs preserve the **1B solve-token cosine horizon** and start from the
same warmup checkpoint, not from a preceding trial. K25/K100 entries
set `inner_loop_iter` individually, leaving legacy defaults at K50. Do not launch if the original
box248 queue controller is active or TPU workers show active processes.

```bash
cd ~/Desktop/full-gauss-newton
git status --short
bash scripts/tpu.sh 1 status

# After the branch has been safely deployed to the four idle TPU1 workers:
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_large_batch_remaining.json --dry-run
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_large_batch_remaining.json

# Inspect remotely (this requires auth only when you choose to inspect):
bash scripts/tpu.sh 1 status
bash scripts/tpu.sh 1 logs 0

# Immediate cancel of the managed queue AND current training:
bash scripts/tpu.sh 1 stop
```

Each trial receives a unique ID, its own W&B run, and the complete existing
GN-Adam scientific flags from `gn_sweep_queue.build_plan_for_config`. All
workers wait for each other's successful result before the next trial. Queue
coordination objects live under
`gs://gfn-soo/checkpoints/queue_runs/<queue-id>/` and are conditionally created
(immutable); mismatched/stale markers and failed workers prevent advancement.
Trial outputs remain under normal `script_runs/` paths. `logs` follows each
worker's combined queue and trainer output in the managed worker log.

The JSON supports per-trial `solve_tokens`; for a longer run edit a **new
committed file** with e.g. `"solve_tokens": 300000000` while retaining
`"schedule_tokens": 1000000000`. Do not reuse an active queue identity or
change its committed spec in place. Each new launch receives independent
experiment IDs. 300M runs start from the same warmup checkpoint, **not**
from an earlier 150M run's final state.

### Mandatory manual bootstrap for helper-changing branches

**Do not** attempt `bash scripts/tpu.sh 1 switch codex/tpu-resident-queue-implementation`
from the old `98b2928` checkout: `tpu_worker.verify_target` deliberately
refuses target commits with changed helpers.

1. Operator: from box248 first verify no old `tmux` sweep queue is active,
   and `bash scripts/tpu.sh 1 status` reports **all TPU1 workers idle and clean**.
   Leave TPU2 (`gn-v4-32-2`) entirely untouched.
2. For each TPU1 worker 0,1,2,3, **read-only first**, using the previously
   working version of `scripts/tpu.sh 1 ssh <WORKER>` or:
   ```bash
   gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32 \
     --project=fast-second-order --zone=us-central2-b --worker=0 \
     --command='cd /full-gauss-newton && git status --short && git branch --show-current && git rev-parse HEAD && tmux ls || true'
   ```
   Repeat for worker 1,2,3 and confirm no training processes with the
   original controller status before changing anything.
3. After the branch has been reviewed and the operator confirms all workers are
   idle and clean, **operator only**: on box248 fetch and switch the feature
   branch; on each TPU1 worker `cd /full-gauss-newton && git fetch origin &&
   git switch -c codex/tpu-resident-queue-implementation
   --track origin/codex/tpu-resident-queue-implementation`.
   If the branch already exists on a host, use `git switch` and
   `git merge --ff-only origin/codex/tpu-resident-queue-implementation`.
   Never reset, stash, or force checkout. Repeat for all workers, verify exact
   common SHA and clean work trees, then perform the dry-run and launch above.
4. Stop if ANY worker fails to synchronize. No training or branch deployment
   is part of the Codex/ChatGPT implementation task.

### Limitations

This removes the **box248 controller's repeated gcloud auth/polling** from the
critical path *after launch*. It does not guarantee recovery from spot
preemption, missing TPU-local GCS credentials, lost HF/W&B connectivity, or a
GN training crash. Workers time out rather than hanging forever. No intermediate
GN optimizer/data checkpoints are introduced; interrupted individual trials
are not automatically resumable.

## TPU2 AdamW target-loss queue (separate feature branch)

Branch `codex/tpu-resident-adamw-sweep` adds the `adam` route to the same
TPU-local synchronized queue runner, without changing the GN route. The ordered
`scripts/sweeps/adamw_followup.json` contains ten B64/B128/B192/B256 runs
testing peak LR, second-moment beta2, and weight decay. All runs use the
150M warmup checkpoint, FP32, seed 42, a **1.2B-token cap**, a **2B-token
cosine schedule horizon**, 1% schedule warmup, and target validation loss
`3.25`. Each run starts from the common checkpoint with fresh Adam moments,
not the preceding run. Short W&B notes look like
`Adam_B128_lr0.01_b20.95_wd0`; token budgets and seeds remain in structured
run configuration, not notes.

The trainer already supports `--target_loss` and evaluates it *after* validation
batches, not continuously. A run reaching the target exits normally; a run
that does not reach it completes its token cap. Plan for up to 10 full
runs; **the intended ~20-hour window is not a runtime guarantee** and
does not automatically stop the overall queue at 20 hours. The per-trial
resident timeout remains in place.

### Keep TPU1 unchanged

Do not switch the existing `~/Desktop/full-gauss-newton` checkout if it is
being used to monitor TPU1 GN. On box248, create a **separate** Git worktree for
the TPU2 controller, using the feature branch at one exact commit:

```bash
cd ~/Desktop/full-gauss-newton
git fetch origin
git worktree add --track -b codex/tpu-resident-adamw-sweep \
  ~/Desktop/full-gauss-newton-adamw origin/codex/tpu-resident-adamw-sweep
cd ~/Desktop/full-gauss-newton-adamw
git rev-parse HEAD
git status --short
```

If that branch/worktree already exists, inspect it and fast-forward it rather
than deleting or resetting. **Check `bash scripts/tpu.sh 2 status` first**.
Do not touch an active TPU2 MINRES-QLP run. Only after TPU2 is confirmed fully
idle/clean, operator manually fetches and switches **all** TPU2 workers
`gn-v4-32-2` (0,1,2,3), with project `fast-second-order`, zone
`us-central2-b`, and worker repo `/full-gauss-newton`. The old helper
fingerprint switch restriction applies, so use explicit read-only checks,
then `git fetch` and `git switch --track`, `git merge --ff-only`.
Verify every worker is on the *same named branch and commit* as the box248
TPU2 controller and all are idle. **Never** reset/stash/force a worker or
change TPU1.

```bash
cd ~/Desktop/full-gauss-newton-adamw
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_followup.json --dry-run
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_followup.json
bash scripts/tpu.sh 2 status
bash scripts/tpu.sh 2 logs 0
```

The queue is TPU-resident after initial launch; existing stop is immediate
and cancels the current trial:

```bash
bash scripts/tpu.sh 2 stop
```

Before running the full Adam queue, use
`scripts/sweeps/adamw_smoke.json` to test both the **direct AdamW trainer
route** and TPU2 GCS synchronization. Its two B128 configurations each do
one update and one 50-batch validation, and should automatically advance
between trials. This is an infrastructure test, **not** a scientific optimizer
comparison. Run:

```bash
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_smoke.json --dry-run
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_smoke.json
bash scripts/tpu.sh 2 logs 0
```

Verify both trials finished, all workers report successful exit status 0
and no managed training processes before starting the expensive Adam queue.

Do not merge this branch or start TPU experiments as part of implementation
or CI. The operator controls TPU2 deployment and run start.


### Historical-comparability rerun (new, opt-in)

The **original** `scripts/sweeps/adamw_followup.json` is retained unchanged
for provenance (50 validation batches, 2B-token LR schedule). The separate
`scripts/sweeps/adamw_followup_comparable.json` is the proposed next queue:
seven ordered runs, keeping B128/LR0.015 while omitting the redundant
B128/LR0.01, B64/LR0.015, and B128/LR0.02 comparisons. It uses `protocol:
"adam_followup_600m"` to match the original follow-up evaluation and
optimization schedule: 8 validation batches, no initial evaluation, roughly
20M-token validation intervals (611 updates for B32, 153 for B128),
a 600M-token cosine horizon, 10 warmup updates beginning at initial LR 0,
end LR equal to 10% of the peak, and optimizer-state saving enabled as
in the original CLI (while save frequencies remain 0). Unlike the original historical runs, these experiments may continue
up to 1.2B tokens unless their validation loss reaches 3.25. This **extends
the old schedule beyond its decay horizon**, rather than stretching decay
to 1.2B tokens, and is a deliberate experimental choice. The 1.2B token
ceiling and loss target are retained in resolved flags, not verbose W&B
notes.

Do not rerun already completed 50-batch experiments expecting their
historical charts to become directly comparable. Always compare the
new 8-batch series with the historical 8-batch runs, under the same
evaluation stream/configuration. If the original run configuration
differs in other flags, investigate that discrepancy before claiming
strict equivalence.

After confirming all TPU2 workers idle and clean, and manually
fast-forwarding the TPU2 feature branch across the controller worktree
and **all** TPU2 workers to the same commit, preview and launch:

```bash
cd ~/Desktop/full-gauss-newton-adamw
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_followup_comparable.json --dry-run
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_followup_comparable.json
```

Never change or restart active TPU2 training in order to deploy this.


### Long-horizon AdamW final-loss comparison

`scripts/sweeps/adamw_long_horizon_primary.json` (five experiments)
and `scripts/sweeps/adamw_long_horizon_optional.json` (three independent
follow-ups) are opt-in, full-budget queues. **Neither reruns the historical
B720 / LR .01 / WD0 / 3000-update result**. The primary set compares
B128/.01, B256/.01, B384/.01, B720/.01/WD.01, B128/.015
(β₂=.95, WD0 otherwise). Optional runs are B128/.01/WD.01,
B512/.01/WD0, and B192/.0125/WD0.

Every run uses **2,211,840,000** solve tokens (rounded down to full updates),
matching the historical B720/3000-step token budget and cosine horizon.
The common schedule has 1% peak initial LR (B720 historical run: .0001
at peak .01), 10% peak final LR, and 7,372,800 warmup tokens
(10 B720 updates, scaled to the other batch sizes). Each run uses
8 validation batches per ~74M solve-token interval, the same 150M warmup
checkpoint, and full FP32. `target_loss=0` disables target-based stopping
to compare **terminal validation loss** at a common budget. The previous
B720 reference evaluated 50 batches every update; its validation results
are not directly interchangeable without a common evaluation protocol.

These are **new independent training runs**, not resumes, and are not limited
to 20 hours total. Running all five full 2.2B-token primary experiments
may exceed available TPU time. Do not automatically launch optional runs.

Before launching, check TPU2 idle and clean and synchronize all four
TPU2 worker checkouts to the exact controller branch and commit.
Keep TPU1 untouched. Preview, then operator-only launch:

```bash
cd ~/Desktop/full-gauss-newton-adamw
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_long_horizon_primary.json --dry-run
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_long_horizon_primary.json
```

Only after results are reviewed, if worthwhile:

```bash
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_long_horizon_optional.json --dry-run
```

The two new fields in long-horizon specifications
(`init_lr_fraction`, `warmup_tokens`) affect only the explicit
`adam_followup_600m` planning mode, not the historical JSON,
previous runs, GN, or trainer math. Only make branch changes while TPU2
is confirmed fully idle; never switch running workers.


### Eight-trial overnight AdamW queue

`scripts/sweeps/adamw_long_horizon_eight.json` combines the five
primary and three optional long-horizon trials into a **single** ordered
eight-run queue. This is a separate new queue spec; the original primary
and optional files remain intact. Its order is B128/.01/WD0,
B720/.01/WD.01, B256/.01/WD0, B128/.015/WD0, B384/.01/WD0,
B128/.01/WD.01, B512/.01/WD0, B192/.0125/WD0; all use β₂=.95.
It does **not** repeat historical B720/.01/WD0.

This queue preserves 2,211,840,000-token training and cosine horizon,
7,372,800 warmup tokens, initial LR 1% peak, final LR 10% peak,
8 validation batches at **approximately 20M training-token intervals**
(controlled independently by `eval_interval_tokens: 20000000`),
a common checkpoint, and `target_loss=0` (full-budget final-loss comparison).
The earlier version of this queue evaluated approximately every 74M
tokens; the revised source changes only future launches. It cannot
retroactively add evaluations to a queue already running, whose trial
plans were resolved at launch. Do not replace running workers or restart
the existing queue solely for plotting cadence.
**Eight full runs may last far beyond one night.** There is no total
wall-clock deadline: a healthy queue keeps executing after the operator
disconnects. It does not recover automatically from preemption or
Hugging Face outages; the failed trial is not checkpoint-resumable
with `save_model_freq=0` and `save_milestone_freq=0`. No additional
training or checkpointing changes were introduced here.

After verifying TPU2 **all four workers idle and clean**, operator manually
syncs box248's AdamW worktree and all TPU2 worker checkouts to the exact
same new feature-branch SHA. Only then:

```bash
cd ~/Desktop/full-gauss-newton-adamw
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_long_horizon_eight.json --dry-run
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_long_horizon_eight.json
bash scripts/tpu.sh 2 logs 0
```

Do not launch this queue if the five-run primary queue has already begun;
the new queue would duplicate its first runs and preflight would reject
active workers. Preserve active training instead.


### Remaining AdamW runs with numbered W&B names

`scripts/sweeps/adamw_long_horizon_remaining.json` is a separate,
future-only, seven-run configuration that excludes the first B128/.01/WD0
experiment, which the operator has already mostly completed. It retains
the other distinct B128 experiments and all B720/B256/B384/B512/B192
candidates. Names and W&B `notes` explicitly preserve original ordinal
labels `adam_trial2_...` through `adam_trial8_...`.
Inside this new queue their synchronization/index sequence is 1 through 7;
the preserved numbering is only a readable W&B name, **not** a resume
index, and every run starts independently from the warmup checkpoint.

The `notes_from_trial_name: true` flag is opt-in and affects only the
new queue; older queues retain existing W&B notes. The 20M-token
evaluation cadence, eight validation batches, 2.21B-token common
schedule, token-matched warmup, and `target_loss=0` are unchanged.
No checkpoint/resume behavior was added.

**Do not deploy while the old eight-run queue is still active.** A
currently running resident queue has fixed plans, so changing the
file or updating its repository cannot remove the queued trial safely.
If it is already running, preserve it and use the revised queue only
after the original queue exits and the operator confirms which
experiments remain necessary. Do not launch duplicate trials.

```bash
# After all four TPU2 workers are idle, clean, and on the same new commit:
cd ~/Desktop/full-gauss-newton-adamw
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_long_horizon_remaining.json --dry-run
bash scripts/tpu.sh 2 queue scripts/sweeps/adamw_long_horizon_remaining.json
```
