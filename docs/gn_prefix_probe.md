# Controlled GN prefix experiment

This diagnostic measures additional inner optimization along a committed **K=10**
training trajectory. It does not compare independently trained K50/K100 models,
select an adaptive K, or make spectral claims. The PR base is
`main` (`origin/main` at branch creation: `ede9de0`). The experimental branch is
`codex/gn-prefix-transfer-diagnostic-main`; it contains no queue infrastructure or
GN sweep commits.

## Interface and constraints

Only these trainer flags are new:

```text
--gn_prefix_probe_steps=0,10,100,1000,2500
--gn_prefix_ks=10,20,50,100
--gn_prefix_holdout_batches=2
```

An empty `gn_prefix_probe_steps` disables all probe dataset construction,
compilation, counter inspection, extra evaluations and logging. Steps are
zero-based outer loop indices (the trainer's `step`, before progress offsets).
Ks must increase strictly from 10. The committed iteration count must stay 10.

The validator requires GN, AdamW or Muon, `global_cosine`, a fixed solve batch,
non-adaptive warm-started inner optimization, line search with one fresh batch,
and zero solve penalty (`inner_loop_wd=0`). It rejects weight averaging,
normalization, outer momentum/decay, fixed step overrides and reused LS batches.
All dropout and FCM settings must be zero. Inner optimizer clipping and optimizer
weight decay retain their configured semantics. Other LR families have not been
verified and are rejected. The cosine horizon always uses K10; extra iterations
continue the original optimizer counters even when that reaches the schedule end.

## Launch example (no experiment launched)

The independent preset `scripts/presets/gn_prefix_probe.json` uses main's existing
single-run launcher. Preview it with:

```bash
bash scripts/tpu.sh 1 run gn_prefix_probe --dry-run
```

It inherits the ordinary GN-Adam preset settings and overrides only K10, the
undamped solve objective, required outer/LS constraints and probe settings.
The default 3000 outer updates retain a K10 global cosine horizon; probes at
0, 10, 100, 1000 and 2500 sample early through later training. For a controlled
comparison, use the same settings in the disabled run with
`--gn_prefix_probe_steps=`. Adjust the run horizon and probe steps together.

For Muon, `run gn_muon --dry-run` accepts the same three flags plus the required
constraints, particularly `--inner_loop_iter=10`, `--inner_loop_wd=0`,
`--lr_sched=global_cosine`, `--ls_reuse_lag=0`, and zero outer decay/momentum.
No resident-queue code or configuration changes are part of this experiment.

## Isolation, measurements and memory

The normal ten updates run first. The state and RNG are captured immediately
following iteration ten, before the ordinary LS helper splits the training RNG.
The ordinary search and parameter/optimizer/warm-start commit finish normally.
The diagnostic then runs after the ordinary training timer stops. Its callbacks
are the same compiled GN inner step, nonlinear loss, and host line search used
by training. K10 reuses the selected scale and recorded losses without another
search. Later prefixes use the same LS data/RNG and original search configuration.
Matched-norm candidates are evaluated directly, without any search.

A separately constructed evaluation loader with a deep-copied configuration owns
the probe cursor and packing buffers. Exact held-out batches and local derived
RNG keys are reused across baselines and candidates. This stream advances only
on probe steps. It may overlap the ordinary evaluation stream in content; its
cursor is independent. Operators must configure an actual held-out evaluation
split. Finite probe data exhaustion fails explicitly rather than recycling data.
Probe cursor state is not checkpointed, so restarting a run restarts that stream.

`gn_prefix/kK/` contains held-out improvement, marginal improvement over K10,
matched-norm improvement, raw/accepted quadratic predictions, **post-update**
raw-system residual and relative residual, norms, selected alpha, LS improvement,
and candidate count. Each candidate's scale/loss is also recorded. The undamped
quadratic uses the existing linearize/JVP/transpose GN helpers; residual norm is
computed on the final parameters, with denominator `||g|| + 1e-12`.
`gn_prefix/updateN/` records every update's actual Optax schedule counter and
pre-update LR, including updates 1–10. Muon's masked schedule counters must agree.
SHA256 fingerprints identify solve, LS and held-out batches and evaluation RNGs.
Non-addressable arrays are fingerprinted through local shards and small collective
hashes; complete optimizer states are never transferred to the host.

The normal GN inner step does **not** donate its arguments. Thus the K10 state
can share immutable device buffers with the committed state. Only one rolling
diagnostic state is advanced; no K20/K50/K100 optimizer snapshots are retained.
Transient directions, candidates, gradients and operator intermediates still cost
memory. Adding donation later requires revisiting this sharing contract.

`gn_prefix/runtime_s` synchronizes at the probe boundaries and includes held-out
pulls, evaluations, fingerprinting, continuation, compilation and pre-update
counter inspection. Normal training time excludes these intervals. Probe rows are
published on the corresponding outer step even off the regular logging cadence;
ordinary evaluation remains gated by the unchanged regular cadence.

## CPU validation and limits

Run with CPU JAX, Optax (with `contrib.muon`), Flax, and pytest:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_gn_prefix.py \
  tests/test_accepted_gn_comparison.py tests/test_iterative_microbatch.py \
  tests/test_ls_reuse.py tests/test_inner_residual_logging.py \
  tests/test_gn_fixed_step.py -q
python -m unittest discover -s scripts/tests -q
```

Validation on the main baseline passed: 29 tests and 60 subtests in the CPU
suite above, plus all 42 existing-launcher tests. The CPU environment uses Python
3.12, JAX 0.4.38, Optax 0.2.5 and Flax 0.10.4. The source-branch
comparison executes the untouched main host branch against the enabled
implementation for two outer updates with both optimizers. It verifies
byte-identical serialized states and identical RNG, ordinary logs and data pulls.
Fetch `origin/main` or set `GN_PREFIX_BASELINE_REF` to its commit when running
that check in a shallow clone.

The harness extracts and executes actual production functions and the ordinary
host update/commit branch, replacing only model/service/mesh dependencies. It
uses a tiny nonlinear model, real Optax AdamW/Muon, and dense JAX Jacobian and
logits-loss Hessian ground truth. Enabled/disabled runs compare serialized
parameters, moments, counters, warm starts, RNG, LS consumption, ordinary logs,
and the next outer update. Additional checks cover nonlinear measurements,
matched norms/zeros, packed-loader factory isolation, microbatch continuation,
synchronized timing exclusion, and launch-preset constraints. The earlier accepted-GN
harness is updated to handle the existing LS closure and the new disabled gate.

These are CPU mathematical and host-control checks, not a complete LLaMA/tokenizer
startup, a TPU HBM benchmark or a multi-host runtime test. No TPU training is run.
Actual scientific findings await the controlled experiment; none are inferred
from these unit tests.
