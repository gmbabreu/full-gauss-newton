# GN-Adam solve-token sweep

Run from a clean main checkout matching the TPU workers. User launches training.
TPU1 is the default; do not stop or update TPU2's independent Adam queue.

    python3 scripts/gn_sweep_queue.py preview --target 1
    python3 scripts/gn_sweep_queue.py preview --target 1 --json
    python3 -u scripts/gn_sweep_queue.py run --target 1 --tokens 299000000 --schedule-tokens 1000000000

Ten trials: B720/256/1024/2048/64, each with peak inner LR .001/.003.
FP32 computation/parameters, K50, b1 .9, b2 .999, optimizer_wd .001,
inner_loop_wd .01, global cosine to zero, no warmup, n_micro1/1/2/4/1.
Uses the shared warmup parameter checkpoint and preset C4 data settings.
Evaluation8 batches of128 sequences; logging/evaluation about30 times per trial.
No checkpoint saves. Names include trial, batch, LR, K, WD and token budgets.

Budgets count solve positions only (sequence length1024). Reusing a solve batch
across inner iterations or microbatches does not increase the count. Additional
line-search positions remain separately reported and excluded from the cap and
schedule. Compare historical runs only with matching evaluation settings; old
Adam runs with shorter decay horizons are not matched1B-horizon controls.

The queue sets total_steps=floor(tokens/(B*1024)) and
lr_schedule_steps=ceil(schedule_tokens/(B*1024)). The GN trainer uses
lr_schedule_steps (or total_steps when zero) as the outer-step schedule horizon;
global_cosine internally multiplies it by K. The queue is specifically for fixed
batches, fixed K, single_batch_inner=True, no batch growth/adaptive iterations.

Controller runs sequentially. Each next trial requires all workers exited0 plus
normal managed preflight. Failed/cancelled runs halt the queue; W&B cancellation
is not a supported skip mechanism. Three consecutive manifest-read failures or
a72-hour per-trial timeout halt the controller without stopping live training.
No automatic restart of failed trials or stale-reservation deletion.

Plans/completion records live in ~/adam-queues/<queue-id>. Create STOP in the
printed queue directory to pause after the current trial. Restart with explicit
--trials IDs to run only remaining trials. A restart creates fresh runs from the
warmup checkpoint; it is not checkpoint continuation. Interrupting the queue
leaves training running. The shared target queue lock prevents duplicate queued
controllers on the same controller host. Do not interrupt launch/stop operations.

Keep the controller host awake and connected. A tmux session survives terminal
disconnection, not controller-host shutdown. The full queue can exceed one night.
