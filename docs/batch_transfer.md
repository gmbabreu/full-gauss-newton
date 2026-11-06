# One-step first-order Adam batch transfer

`EasyLM/models/llama/llama_train.py` accepts `--batch_transfer_probe=True`
(default `False`). It applies **one native AdamW update** on training batch A,
measures data loss on A and a separate training batch B at the same updated
parameters, writes a completion result, finishes W&B, and returns. B contributes
no gradients, learning-rate selection, or update selection. There is no line
search, GN product, prediction, or direction export.

## Checkpoint and configuration

Start with **checkpoint500**, global batch **720**, sequence length **1024**.
Later, use the original checkpoint as a separately labeled **fresh optimizer
state** control. Ignore checkpoint1000 for this experiment.
For that later control, restore parameters only (for example, the loader's
`trainstate_params::` mode for a full-state file) to initialize fresh moments;
a full-state load is labeled `restored` even when its step is zero.

Use the user's original first-order Adam configuration and checkpoint companions.
Do not substitute GN inner-Adam settings. No complete launch command is supplied
because that original configuration and its companions must first be identified.

Keep `total_steps`, optimizer/schedule settings, precision, and the original
resume configuration unchanged; the flag itself selects one-step mode. Do **not**
set `total_steps=1` or disable `save_optimizer_state`. Existing strict full-state
loading and matching metadata, dataset, and completion companions remain required.
The new flag is outside the validated trajectory settings. Native clipping,
weight decay, moments, bias correction, and schedule all come from the existing
Optax state. The diagnostic branch bypasses initial, periodic, and final training
checkpoint writes, initial validation, and the ordinary training loop.

Supported configurations:

- Ordinary AdamW (`multiply_by_parameter_scale=False`), accumulation 1,
  `weight_average=False`, and `outer_loop_method=None`.
- Raw Hugging Face streaming, without a pretokenized directory.
- Zero-start dataset configuration: `tokens_count_at_start=0` and `start_step=0`.
  The actual optimizer step comes from restored state.
- Replay disabled: leave `wandb_run_id` empty and use a new diagnostic run and
  output experiment ID. This trainer has no separate replay flag.
- Finalized model embedding, feedforward, attention, and residue dropout and
  both FCM ratios must be zero. Checks use resolved model settings.

Full raw-HF Adam resume currently requires FP32 compute/state and FP32 params.
Historical GN probes used BF16 compute with FP32 params. Preserve authentic Adam
resume settings and label this difference on the initial plot: this is a native
method comparison, not an isolated optimizer test at matched precision. A later
matched-precision GN control can address the difference.

## Data and measurements

A separate `DatasetFactory` loader receives a deep copy of the training dataset
configuration and the same tokenizer, without saved dataset state. Its first
batch is A and its next batch is B. Preserve the historical packing, fields,
masks, shuffle seed 42, shuffle buffer 100000, batch 720, and sequence 1024 in
the original configuration. The restored training dataset's cursor and packing
remainder remain untouched. The evaluation dataset is not used.

This is a **controlled checkpoint probe with stored Adam moments on a restarted
training pair**, not continuation of the saved stream. The checkpoint may have
seen A or B in training history; B must not be labeled globally unseen.
This first/next routing matches historical GN runs with `single_batch_inner=True`,
`adaptive_inner_loop=False`, `ls_eval_batches=1`.

Before donation, all hosts capture fingerprints of the base parameters and A/B
using sorted paths, native shapes/dtypes, and the existing
`update_geometry.fingerprint_leaves` convention. Global leaves are replicated
through gathers before host access, one leaf at a time. No extra full parameter
tree or direction is retained or exported.

The probe evaluates B before the update, invokes `sharded_train_step` exactly once
on A with the original training RNG, and evaluates A then B afterward. The native
train step's pre-update `loss` supplies A's before value. The zero-stochasticity
guards make this reuse valid. Diagnostic evaluations use independent RNG buffers
and do not advance the training RNG. This adds exactly three forward evaluations.
All computations finish before publication.

All four losses use `cross_entropy_loss_and_accuracy`: the mean of per-sequence
masked losses. No weight-decay penalty is added to reported data loss; native
AdamW weight decay remains in the applied update.

## Output and inspection

Only process 0 writes the result, through `mlxu.open_file` for local paths or GCS:

```text
<output_dir>/<experiment_id>/batch_transfer/metadata.json
```

The JSON contains:

- `format_version=1`, `method=first_order_adamw`, original `checkpoint_uri`,
  and `repository_commit` (null if unavailable).
- `optimizer_state_source` (`restored` for full train state, otherwise `fresh`),
  `optimizer_step_before`, `optimizer_step_after` (exactly one higher).
- `data_mode=fresh_first_two_training_batches` and the data interpretation.
- `base_parameters`, `update_batch`, `transfer_batch`: each has `sha256` and
  `leaves` containing sorted paths, shapes, and native dtypes.
- Finalized `model`, `dtype`, `param_dtype`, `tokenizer`, `dataset`, `optimizer`
  (including schedules), `total_steps`, `seed`, `train_dataset_batch_size`, and
  `precision_comparison`.
- `metrics`, with the six scalars below.
- `completed=true`, only after finite measurements and a verified single step.
  Nonfinite losses or decreases raise an error instead.

| Metric under `metrics` | Meaning |
| --- | --- |
| `batch_transfer/loss_update_before` | L_A(w), from the native train step |
| `batch_transfer/loss_update_after` | L_A(w_plus) |
| `batch_transfer/loss_transfer_before` | L_B(w) |
| `batch_transfer/loss_transfer_after` | L_B(w_plus) |
| `batch_transfer/improvement_update` | L_A(w) minus L_A(w_plus) |
| `batch_transfer/improvement_transfer` | L_B(w) minus L_B(w_plus) |

The same scalars are logged to W&B regardless of `log_freq`, at the optimizer
step after the update. An all-host barrier follows the JSON write. Process 0 then
prints `BATCH_TRANSFER_COMPLETE path=...`, and all hosts finish W&B.
Use a distinct experiment ID for every probe so an older result cannot be
mistaken for a new completion after a failed rerun.

Inspect a downloaded/local result without loading JAX or model parameters:

```shell
python -m json.tool /path/to/experiment/batch_transfer/metadata.json
```

For checkpoint500, compare the recorded hashes with:

```text
base_parameters.sha256:
d969a27135d8a00faa5a77fee7f73db4d7d12c7c65be6408da77f106444c1075
update_batch.sha256:
778a755ee646bcadf89623c38cd988a0843adf0083a9aede7e27ccd0380aeaad
```

Historical GN metadata lacks B's fingerprint. Matching verified first/next
routing supports matching B; it does not independently hash-verify historical B.
The code records hashes rather than asserting checkpoint-specific values, so
CPU fixtures and the later original-checkpoint control remain usable.

## Panel C handoff: Adam/Jacobi Main Findings

Map these metrics to **actual loss decrease**, retaining its sign:

| Legend | Update batch | Separate training batch |
| --- | --- | --- |
| Adam-GN | `gn_quad/accepted_actual_improvement_solve` | `gn_quad/accepted_actual_improvement_linesearch` |
| Jacobi-PCG | `gn_quad/accepted_actual_improvement_solve` | `gn_quad/accepted_actual_improvement_linesearch` |
| Adam (first order) | `batch_transfer/improvement_update` | `batch_transfer/improvement_transfer` |

Rename the categories to **Update batch** and **Separate training batch**. Use a
zero line and allow negative bars. Keep original and checkpoint500 in separate
panels/groups, with fresh/restored optimizer-state labels and the FP32 Adam versus
BF16 GN compute annotation. The existing Main Findings plotting implementation
is not present in this checkout; this table is the handoff for that figure.
Do not populate quadratic or fidelity panels with first-order values, because
those quantities were not computed.

Relative transfer (Delta_B / Delta_A) may supplement absolute bars only when
Delta_A is meaningfully positive; otherwise it is undefined. Different step
lengths and GN's selection using B mean one pair cannot establish a general
first-order advantage. A later multi-pair/common untouched-batch control is optional.

## Validation

Run with the trainer dependencies installed:

```shell
JAX_PLATFORMS=cpu python -m pytest -q tests/test_batch_transfer.py
```

CPU coverage includes independent analytic softmax/NumPy AdamW with nonzero
moments, schedule, clipping, and decay; exact diagnostic on/off state and RNG
equivalence; actual compiled donation; raw-HF first/next packing and untouched
saved cursor; zero gains and adverse transfer; strict resume validation; invalid
configurations; disabled-flag isolation; writer gating and completion ordering.
Tiny real LLaMA checkpoints exercise production `main()` with restored and fresh
optimizer state, with ordinary saving and initial evaluation configured to verify
their bypass.

Validated on 2026-11-06 with Python 3.12, JAX/JAXlib 0.4.35, Flax 0.10.2, and
Optax 0.2.4 in an isolated CPU environment: `pytest -q -rs tests` passed **162
tests and 60 subtests**, with one existing opt-in real multiprocess CPU transport
test skipped. The new focused file contains 31 passing cases. Python compilation
and `git diff --check` also passed. Existing JAX clip deprecation warnings remain.

Multi-host TPU execution, checkpoint500 fingerprint checks, and real losses remain
pending smoke testing with the original configuration and companions. This task
does not launch TPUs or production training.
