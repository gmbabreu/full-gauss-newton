# CG batch lambda and recovery

## Lambda

The default `cg_lambda_batch_denominator=0` preserves the existing constant or
update-based ramp. Constant lambda 0.1 uses:

```text
--cg_lambda_batch_denominator=0
--cg_interpolation_lambda=0.1
--cg_lambda_final=-1
--cg_lambda_ramp_steps=0
```

For lambda equal to actual global solve-batch sequences divided by 10240:

```text
--cg_lambda_batch_denominator=10240
--cg_lambda_final=-1
--cg_lambda_ramp_steps=0
--condition_log=False
```

This gives 0.025 at batch 256, 0.1 at 1024, and 0.2 at 2048, before device or
microbatch splitting. Both scheduled and effective lambda report this value.
Negative/nonfinite denominators, non-CG solvers, simultaneous ramps, matrix-norm
rescaling, and batches above the denominator are rejected. Startup checks include
the growth cap; actual batch size is checked again at each solve. No clamping is
performed. The LR schedule remains update-based; this rule does not eliminate
all coupling between batch schedules and token-budget comparisons, or guarantee
training stability. Set the denominator to zero AND disable the old ramp when
requesting constant lambda.

## Condition diagnostics

`--condition_log=True` measures the frozen operator every
`condition_every` outer updates. CG reports `G`, the actual damped `A`, and the
symmetric `D^-1/2 A D^-1/2`; Muon-GN reports the first already-fetched inner
batch's `G`. Diagnostics neither fetch data nor consume training RNG, and their
matrix products are excluded from solve-token accounting. Dropout and FCM must
be disabled for Muon diagnostics.

Endpoint values use the `_est` suffix because agreement and residual checks do
not certify global extremality. Failed checks retain residuals, counters, and
failure reasons but withhold `condition_est`. The Rayleigh conditioning lower
bound is an exact-arithmetic PSD implication from two evaluated quotients; its
floating-point value is not a certified bound. Four unnormalised Rademacher
probes are used by default for trace and trace-square plug-in estimates and rough
sample standard errors. The same `Gz` product is reused for `A`; no preconditioned
trace estimate is attempted.

The diagnostic controls, including `condition_trace_probes`, may change on exact
resume. A legacy checkpoint with `cg_log_matrix_norms=True` is rejected because
that old logging path changed effective lambda and therefore the trajectory.

## Data order and compatibility

The unused pre-solve fetch has been removed for all solver paths. The later solve
fetch remains reusable with `single_batch_inner=True`. Historical skipped-token
fields are retained; new updates do not add skipped tokens. Solve-token counting
and line-search fetching are unchanged, including pre-existing search computation
in fixed-step CG runs.

New CG snapshots record `data_consumption_version=2`. Exact resume of older
snapshots is rejected: use their original revision to continue the original data
order. Params-only warmup initialization still works. New runs use different
examples than the old discarded-fetch version, even with the same seed.

## Recovery and milestones

The existing StreamingCheckpointer state, packed cursor, metadata, checksum
manifest, and marker-last protocol remain in use on local storage and GCS.
The trainer still restores moments, CG warm start, RNG, optional outer momentum
and EMA, progress, timing, and rolling generation, and validates trajectory flags.

The launcher honors an explicit `--cg_resume_state` before auto-discovery.
Otherwise, an existing experiment must have a valid rolling bundle; failed
recovery stops rather than restarting from warmup. A corrupted latest slot may
fall back to the older valid slot. Missing explicitly requested experiments fail.
Permission and non-404 storage errors propagate. Intentional fresh runs should
use a new experiment ID. Automatic resume retains the experiment's W&B ID;
explicit checkpoint selection leaves the requested logging identity unchanged.
Replayed updates after fallback may already exist in W&B history; this change
adds no W&B history rewriting or deduplication policy.

Regular saves alternate `cg_state_0` and `cg_state_1`. At a milestone, one rolling
save is followed by a writer-only copy to:

```text
<output_dir>/<experiment_id>/milestones/step_<local_completed_updates>/cg_state_0
```

Milestone numbering uses the trainer's completed update count (`train_state.step`),
not an optional reporting offset. The copy uses the same companion names and
writes its completion marker last. Valid milestones are immutable; incomplete
copies may be retried. Milestones are not pruned. GCS copies stay within GCS.
All hosts participate in the original state gathers and synchronize after saving;
copying the committed bundle requires no additional gather. Coincident save
frequencies write one rolling bundle and one retained copy. Milestone-only
configurations also refresh rolling recovery at each milestone.

Select a retained snapshot explicitly with `--cg_resume_state=<path above>`.
Automatic recovery examines only the rolling slots. The milestone retains its
source generation without advancing the live rolling generation a second time.

## CPU verification

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m py_compile EasyLM/cg_resume.py EasyLM/models/llama/llama_train_gn.py sweep_launcher.py
```

Tests cover recovery failures, explicit selection, local serialization, mocked
GCS writes/copies/reads, interrupted milestone publication, immutability, lambda
validation, and a small quadratic PCG continuation using the trainer's extracted
save function, actual StreamingCheckpointer, and packed HF loader over synthetic
examples. The next batch and numerical state are compared with uninterrupted
execution across a batch-growth boundary. This is not a full LLaMA trainer test.
Real TPU multi-host execution, GCS credentials/network failures, and W&B recovery
remain untested here.
