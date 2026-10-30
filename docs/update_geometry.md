# Opt-in update geometry

`EasyLM.models.llama.llama_train_gn` can export the first completed outer
solve of an invocation. AdamW-GN, Muon-GN, SGD-GN, and CG/PCG use the same
capture helper. It records `d = candidate_params - base_params` before outer
scaling, and calls `sharded_condition_apply_g(base_params, solve_batch, d, 0)`
once. The exported operator is deterministic raw data-loss GN, excluding CG
damping, interpolation, and preconditioning. Ordinary GN uses the final
`inner_state.params` as the candidate. Training updates and the existing
`quadratic_fidelity` formula are unchanged.

Add these two flags to the existing training invocation:

```bash
--export_update_geometry=True \
--stop_after_first_outer_step=True
```

The export flag alone allows training to continue, with no further geometry
exports. The stop flag also works independently and stops after the completed
step's diagnostics and outer logging. It never modifies `total_steps`; retain
the original schedule horizon, warmups, and solver budgets.

These existing flags must have the following values (append overrides if needed):

```bash
--single_batch_inner=True \
--adaptive_inner_loop=False \
--normalize_step=False \
--outer_momentum_beta=0 \
--outer_weight_decay=0 \
--llama.embedding_dropout=0 \
--llama.feedforward_dropout=0 \
--llama.attention_dropout=0 \
--llama.residue_dropout=0 \
--llama.fcm_min_ratio=0 \
--llama.fcm_max_ratio=0
```

Select `--gauss_newton=True` with `--optimizer_type=adamw`, `muon`, or `sgd`;
use `--optimizer_type=cg` for CG/PCG. Keep `--condition_log=False` to avoid
independent spectrum diagnostics. Keep `--log_gn_quadratic=True` (the existing
default) to include existing `g_dot_d`, `d_G_d`, predicted/actual improvement,
and fidelity metrics in the sidecar. If quadratic logging is disabled, its
unavailable scalars are represented by JSON `null`; the vector export still
works. The accepted scalar step is always recorded.

Run each solver separately with the same checkpoint, model/precision,
`cg_n_micro`, mesh, dataset configuration/cursor, batch size, and seed. Use
unique output/experiment IDs and fresh W&B run IDs. Load parameters only;
do not select a recovered CG bundle or reuse solver state. Both the original
and late experiments require fresh solver state. For the late experiment use:

```bash
--load_checkpoint='trainstate_params::gs://gfn-soo/checkpoints/late_training/adam/streaming_train_state_500'
```

For the original experiment retain its original checkpoint with the appropriate
params-only prefix (`trainstate_params::`, `params::`, or `flax_params::`).
The exporter rejects full-state/recovered-CG loading. The original checkpoint
URI is retained before GCS download rewrites the load path.

Each run produces these files under
`<output_dir>/<experiment_id>/update_geometry/`:

| File | Contents |
| --- | --- |
| `direction.msgpack` | FP32 parameter tree `d`, existing streaming checkpoint format |
| `g_direction.msgpack` | FP32 parameter tree `Gd`, same format |
| `metadata.json` | SHA256 fingerprints, parameter paths/shapes, operator and solver settings, scalar diagnostics, commit ID |

The sidecar is written only after the outer update completes. Partial vector
files without a sidecar are not a completed bundle. Use a new experiment
directory for each invocation so old files cannot be confused with a new run.
Each FP32 tree takes approximately `4 * parameter_count` bytes; together they
take `8 * parameter_count` bytes plus a small sidecar and serialization overhead.
For 45M parameters that is about 360 MB (343 MiB) per solver/checkpoint; for
150M parameters it is about 1.2 GB (1.12 GiB). No base weights or batch tensors
are duplicated on disk. Fingerprinting additionally gathers base weights and
the complete three-field solve batch, one leaf at a time.

All hosts participate, in this order: one raw Gv, sorted base-parameter gathers,
sorted batch gathers, direction gathers in serializer order, product gathers in
serializer order, then a completion barrier after the writer saves metadata.
Each gather uses the existing replication-before-host-transfer helper. Peers
execute the writer's serialization traversal but never open output files.
With export disabled, none of these additional products, gathers, or writes run.

Download the three files from each GCS output directory before comparing. The
comparison command needs only Python, NumPy, and msgpack, with no Flax, JAX,
TPU, or model initialization:

```bash
python scripts/compare_update_geometry.py \
  exports/original/adam/update_geometry \
  exports/original/pcg/update_geometry \
  --output-prefix reports/original_adam_vs_pcg

python scripts/compare_update_geometry.py \
  exports/late/adam/update_geometry \
  exports/late/pcg/update_geometry \
  --output-prefix reports/late_adam_vs_pcg
```

Compare Muon/SGD and other pairs in the same way, always **within one checkpoint**.
The original and late checkpoints define different GN operators and require
separate reports. SHA256 comparisons verify actual base weights and all
`input_tokens`, `target_tokens`, and `loss_masks`, including shapes and dtypes.
The command aborts for identity, parameter-structure, or operator mismatches;
solver settings are allowed to differ. Operator checks conservatively include
model configuration, precision, requested and actual microbatch grouping, and
data-shard count.

The command prints a compact summary and writes `<prefix>.json` plus
`<prefix>_per_matrix.csv`. All dots accumulate on CPU in float64, leaf by leaf.
The JSON contains global Euclidean/GN cosines, both GN cross terms, their
absolute and relative symmetry discrepancies, each direction's Rayleigh
quotient, and the norm/curvature/Rayleigh quotient of the Euclidean residual
`e = c - (a.T @ c / (a.T @ a)) * a`. It obtains `Ge` by linearity from saved
products. The CSV contains Euclidean cosines for two-dimensional parameters.
It does not interpret layer contributions to coupled GN products as layer GN
cosines. Undefined metrics print `undefined` and serialize as JSON `null`.
Relative symmetry discrepancy uses `abs(a_Gc-c_Ga)/max(abs(a_Gc),abs(c_Ga))`
and is undefined when both cross terms are zero. Saved products retain the
training computation's numerical precision, even though storage is FP32 and
comparison accumulation is float64.

Focused CPU validation:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_update_geometry.py \
  tests/test_accepted_gn_comparison.py tests/test_gn_fixed_step.py
```

The tests cover explicit SPD ground truth with cross-layer coupling,
degeneracies, incompatible bundles, serialization, BF16 fingerprints, raw
candidate/first-export gates, validation restrictions, and logging before stop.
A separate four-device CPU subprocess exercises real sharding and replication
gathers. Writer/peer traversal ordering is tested on CPU. This does not establish
real multi-host TPU behavior: a short TPU smoke test must still verify global
base/batch/vector contents, synchronized first-export behavior, GCS writes,
peak memory/compile overhead, one extra Gv, and normal outer logging before
shutdown. No TPU job is launched by the implementation or CPU tests.
