# Codex task: TPU-resident synchronized GN-Adam sweep queue

## Assignment and boundaries

Implement against `gmbabreu/full-gauss-newton`, based on **`gn-adam-sweep` commit `98b2928047c2aaf89418075cc6a3d8af01159580`**. Create a feature branch for implementation and prepare a **PR targeting `gn-adam-sweep`**, not `main`. Do not merge. Only local/CPU tests; **do not provision, switch, stop, launch, or access live TPUs**. Do not touch TPU 2 / MINRES-QLP or TPU 1 live sweep. Preserve model/data/trainer, optimizer math, learning-rate schedule, and standalone run behavior. This file is a **task handoff, not an implemented feature**.

## Inspected source, current behavior

- `sweep_launcher.py` chooses a **single** Cartesian-product configuration using `SLURM_ARRAY_TASK_ID`, then `os.execvp`s one trainer. It does not queue trials.
- `scripts/run_experiment.py` resolves presets in `make_plan`, assigns per-run UUIDs, and executes one configuration through the existing `sweep_launcher.py` under `SLURM_ARRAY_TASK_ID=1`; GN experiment IDs end in `_0`.
- `scripts/gn_sweep_queue.py` builds the established GN-Adam 150M solve-token / 1B schedule-token experiment settings but polls all TPU manifests over `gcloud ssh` from box248, which failed during noninteractive Google reauthentication.
- `scripts/tpu_control.py` discovers, reserves, preflights, and launches workers via `Controller.operate('run',plan)`; it requires clean matching branch/SHA and fingerprints.
- `scripts/tpu_worker.py` supervises one managed training process in tmux, using process group and token/PID-birth-time safety, and retains worker-local manifests and logs. The existing `stop` kills *owned* processes only.
- `EasyLM/gcs_utils.py` uses `google.cloud.storage`; import lazily in new coordination module so CPU tests do not require Google credentials.

## Required CLI

```bash
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_large_batch_remaining.json --dry-run
bash scripts/tpu.sh 1 queue scripts/sweeps/gn_large_batch_remaining.json
bash scripts/tpu.sh 1 status
bash scripts/tpu.sh 1 logs 0
bash scripts/tpu.sh 1 stop
```

Dry-run validates exact ordered plans and prints budgets/configs with **no gcloud / JAX / trainer import**. Real command starts **one persistent managed queue process per discovered TPU worker**; `box248` controller returns after all acknowledgments and is not required to poll subsequently. No Slurm, external VM, long-lived keys, or controller process on box248.

## JSON ordered queue

Add committed `scripts/sweeps/gn_large_batch_remaining.json` with:

```json
{
  "preset": "gn_adam",
  "default_solve_tokens": 150000000,
  "schedule_tokens": 1000000000,
  "trials": [
    {"name": "B8192_lr0.001", "batch": 8192, "inner_lr": 0.001, "n_micro": 16},
    {"name": "B8192_lr0.002", "batch": 8192, "inner_lr": 0.002, "n_micro": 16},
    {"name": "B4096_lr0.004", "batch": 4096, "inner_lr": 0.004, "n_micro": 8},
    {"name": "B8192_lr0.004", "batch": 8192, "inner_lr": 0.004, "n_micro": 16}
  ]
}
```

Ordered entries, **not Cartesian-product**. Support optional per-trial solve-token budget for 300M followups, preserving a separate 1B schedule-token horizon. Validate finite positive batches, LRs, microbatches, token limits, distinct safe names, no duplicate trials, no arbitrary shell commands or path traversal. Require a committed repo config. Controller must resolve *all* per-trial plans and embed an **immutable shared queue payload**, not rely on the controller's file path at execution time.

Reuse or extract scientific flag builder from `gn_sweep_queue.build_plan`; keep its external API and behavior unchanged. Exact settings: `fp32` dtype/param_dtype, `mesh_dim=16,1,1`, 150M warmup checkpoint, seed42, `inner_loop_iter=50`, `lr_sched=global_cosine`, `reset_start=False`, `single_batch_inner=True`, `adaptive_inner_loop=False`, `inner_loop_wd=0.01`, `optimizer_wd=0.001`, Armijo halving with beta0.5/init1, one LS eval batch, existing shuffle and eval cadence, no model/milestone checkpoints, `checkpointer.save_optimizer_state=False`. `total_steps=floor(solve_tokens/(batch*1024))`; `schedule_steps=ceil(schedule_tokens/(batch*1024))`. Print actual consumed solve tokens. Keep all preset flags, including nested `train_dataset.huggingface_dataset.batch_size`. Every trial gets unique globally fresh run/W&B identity, **identical across workers**; retry in a new queue must not overwrite or auto-resume old outputs. Enable W&B on worker0 only.

## Architecture

1. Extend `tpu_control.py` CLI minimally with `queue FILE [--dry-run]`. Use established `Controller.operate('run', queue_plan)`; indicate route `queue` so queue is not mistaken for a trainer preset. Discover actual worker IDs/count; no hardcoded four. All workers must pass clean/idle matching SHA/fingerprint preflight *before launching any*. Validate queue payload consistency across workers. The controller should **not poll** after launching the resident processes.
2. In `tpu_worker.py`, extend managed supervision minimally: `route='queue'` launches `scripts/tpu_resident_queue.py` within the existing supervisor's **owned process group**, preserving tmux, PID/token birth-time checks, cancellation semantics, status/logs, and standalone routes. Include new helper in fingerprint and verify relevant tests. The persistent queue process must still be visible between trials.
3. The resident runner invokes each existing `scripts/run_experiment.py --execute per_trial_plan.json` as a **subprocess** (no trainer/JAX import into resident queue), with same owned process group/token, same run ID on all workers. Append `START/COMPLETE/FAILED` and child stdout/stderr to managed `worker.log`. Each trial's plan/output/W&B identity is independent.
4. **Distributed barriers using durable GCS**: allocate fresh per-queue prefix under `gs://gfn-soo`. Before first trial, all workers publish `ready`; after each trial publish immutable `result` with queue UUID, SHA256 of plan, trial index, run ID, worker ID, and exit status. Use conditional-create (`if_generation_match=0`) and verify every expected worker and identity. **No worker may start next trial until all workers report success on prior trial**. On any nonzero exit or malformed/mismatched marker, stop queue everywhere. Stale queue markers never satisfy a new queue. Best-effort fail marker on errors, but don't rely on it after SIGKILL/preemption.
5. Bounded, realistic deadlines for ready barrier, each training trial, and completion barrier; on disappeared worker others eventually exit nonzero rather than hanging indefinitely. Signal handlers must terminate and wait for child training process. GCS permission/credential errors fail clearly. Test read/write access in initial ready stage; don't report healthy startup if coordination cannot function.
6. Keep GCS limited to small queue control/completion objects, with ordinary per-trial model output unchanged. `status` must reflect queue between trials; `logs 0` follows queue's aggregated worker log. Immediate `tpu.sh 1 stop` kills **only** TPU1 owned queue and active training, not TPU2/unmanaged jobs. Optional graceful finish-trial GCS `STOP` marker distinct from immediate stop, with exact documented invocation if implemented.

## Required CPU-only validation

- Fake in-memory GCS store and subprocess stubs. Never contact real GCP/GCS, call live gcloud, invoke TPU/trainer/JAX, or run real tmux.
- Four-worker full-success; one-worker training failure; missing worker/startup timeout; GCS permission or read/write failures; termination mid-trial; mismatched queue ID/plan hash/trial index; stale marker rejection; duplicate conditional-create; new IDs on rerun; dynamic worker count; no next-trial starts after failure.
- Exact new/old GN plan parity for B4096/LR0.002/n_micro8 and B8192/LR0.001/n_micro16: `total_steps`, `schedule_steps`, inner and nested HF batch, notes, W&B identities and all science flags.
- Existing `scripts/tests/test_tpu_workflow.py` must pass. Verify dry-run no external commands, managed stop process ownership, helper fingerprints, and unchanged standalone `run`.
- `bash -n scripts/tpu.sh`; `python3 -m py_compile` of changed scripts; focused unit tests; `git diff --check`. Record **actual** results in PR. Do not change `EasyLM/models/llama/llama_train_gn.py`, dataset code, GN/CG math or `sweep_launcher.py` absent a separately explained critical issue.

## Deployment and docs

At base, `tpu_worker.verify_target` rejects a normal switch when executable helper bytes change. Document a **read-only-first, manual operator-only bootstrap** of the new feature branch onto box248 and all *idle* TPU1 workers, including clean state/branch/SHA checks and safe fetch/fast-forward. Project `fast-second-order`, zone `us-central2-b`, TPU1 `gn-v4-32`, worker repo `/full-gauss-newton`. No reset, stash, forced updates, or live operations by Codex. TPU2 `gn-v4-32-2` remains untouched.

Deliver unmerged PR against `gn-adam-sweep`, committed JSON, code/tests, `docs/tpu_workflow.md` operator guide, exact launch/preview/status/stop/300M followup commands. Mention limits explicitly: removes dependence on personal box248 gcloud auth **after launch** but does not solve TPU spot preemption, per-worker GCS credential failure, W&B/dataset failure, or lack of intermediate GN checkpoints.
