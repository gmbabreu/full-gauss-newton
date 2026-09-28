# Adam inner-solver state ablations

Use the existing best **Adam-GN inner-solver** command, with
`optimizer_type=adamw`, `gauss_newton=True`, `adaptive_inner_loop=False`, and
`reset_start=False`. This is not the ordinary AdamW baseline in `llama_train`.
Keep the checkpoint, data seed/order, solve batches, inner iteration count,
learning-rate schedule and horizon, decay, line search, precision, and
evaluation settings identical across runs.

`adam_reset_components` resets the selected state **before every outer solve**.
It leaves all unselected state unchanged. Default: `none`.

| Run | `--adam_reset_components=` | Change at each solve boundary |
| --- | --- | --- |
| Baseline | `none` | Carry everything |
| Parameters | `params` | Start inner parameters at the current outer weights |
| First moment | `first_moment` | Zero Adam's `mu`; retain both counters and `nu` |
| Second moment | `second_moment` | Zero Adam's `nu`; retain both counters and `mu` |
| Bias correction | `bias_count` | Zero the shared Adam count; retain both moments |
| LR schedule | `schedule_count` | Restart the actual LR clock; retain Adam's count/moments |
| Full-reset control | `all` | Same reset path as legacy `reset_start=True` |

For example, add this to the existing best command for the first-moment run:

```bash
--reset_start=False --adam_reset_components=first_moment
```

Replace an existing `reset_start` argument instead of duplicating it. Give
every arm a fresh experiment ID/output location and new W&B run; do not resume
an existing run or reuse its optimizer state. The launcher can automatically
try to resume when an output checkpoint already exists. A parameters-only
start initializes optimizer history, so compare fresh runs from the same
starting checkpoint, not a new arm against a stateful mid-run continuation.

## Seven configurations through the existing sweep launcher

In the existing `python sweep_launcher.py ...` command, set these two arguments
(the quotes around `&` are required):

```bash
--reset_start=False \
--adam_reset_components='none&params&first_moment&second_moment&bias_count&schedule_count&all'
```

Remove other swept (`&`) arguments so that only this flag varies. Run that same
command separately with `SLURM_ARRAY_TASK_ID` set to 1 through 7, corresponding
to the table order. The launcher selects **one** arm per invocation; it does
not launch seven jobs or allocate TPUs. Choose a new `SLURM_ARRAY_JOB_ID` for
this experiment group. The launcher stores arms with zero-based suffixes
`<job_id>_0` through `<job_id>_6`.

Combinations are allowed as comma-separated names, e.g.
`--adam_reset_components=first_moment,second_moment`. These should be separate
follow-up experiments, not part of the first one-state-at-a-time comparison.
`none` and `all` must be used alone. Active selective resets reject
`reset_start=True`, other solvers, non-GN training, and adaptive inner loops.

## Interpretation and logging

- Momentum is Adam's first moment; there is no extra momentum or residual
  buffer to ablate. `inner_state.step` is preserved by both legacy and selective
  resets, so it cannot explain the `reset_start=True` performance gap.
- A moment-only reset retains the bias-correction count by design. It is a
  literal state deletion, not fresh Adam. Conversely, resetting only the count
  applies fresh bias-correction factors to retained moments; it is a diagnostic
  intervention, not a standard optimizer restart. Do not substitute `b1=0` or
  `b2=0`: those change every inner update rather than only boundary memory.
- With line search, the carried inner parameters are the raw endpoint, whereas
  the outer model accepts a scaled direction. Without line search, ordinary
  Adam accepts the inner endpoint directly, so the parameter-only arm should
  coincide with the baseline. A constant LR similarly makes the schedule-only
  arm numerically identical to the baseline.
- Adam-GN `learning_rate` now reports the LR actually applied to the logged
  update. `adam/bias_count` and `adam/schedule_count` are the **pre-update**,
  zero-based counters. Ordinary per-outer logging describes the last inner
  update; `log_inner_steps=True` includes the counters with the existing inner
  diagnostic rows. The old LR metric used the post-update `TrainState.step`,
  which did not track actual LR after a reset. Optimization is unchanged for
  `none`; the metric correction applies to Adam-GN controls as well.
- Compare evaluation loss on a common solve-token/outer-update budget first,
  then wall time. These arms have identical configured solve work, but state
  changes can change line-search outcomes and cost. A large individual drop
  identifies a sensitive component; the combined full-reset effect need not
  equal the sum of individual drops.

## CPU verification

From the repository root in an environment with JAX, Optax, Flax, and pytest:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python -m pytest -q tests/test_adam_reset.py
```

Tests exercise the actual trainer boundary, clipped/unclipped Adam construction,
multi-solve trajectories against independent NumPy Adam arithmetic and explicit
quadratic derivatives, and the actual GN train step against an explicit softmax
Jacobian. They also check full-reset equivalence, unselected state preservation,
applied LR logging, invalid configurations, and old CG checkpoint flag defaults.
They do not establish LLaMA performance or TPU memory/runtime behavior.
