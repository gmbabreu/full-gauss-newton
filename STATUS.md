# Muon--GN outer weight-decay experiment

Implementation is ready for CPU review; no TPU training was launched. The source
checkout was `bd2782b72bf3b75ce8ee5e2567ce805650774f4c`, rather than the audited
`a927145c...`, and includes later CG/interpolation work. The controlled feature is
restricted to ordinary, stateful, non-adaptive Muon--GN.

The supplied container has Python 3.14.4 but no installed JAX or Optax. Network
installation is blocked (HTTP 403), and no TPU runtime is attached, so runtime
versions could not be recorded here and the CPU JAX suite could not be executed.
Syntax and shell checks pass. Before launch, run the printed CPU suite and record
`jax.__version__` / `optax.__version__` on the actual TPU image; the earlier audit's
JAX 0.11.1 / Optax 0.2.8 result is historical evidence, not this verification.

The A/B template is `scripts/run_muon_gn_outer_decay_ab.sh`. Arm A uses `rho=0`;
arm B uses the dimensionless per-outer-update shrink `rho=1e-4`. Over 1,000 outer
updates the decay-only factor is `(1-0.0001)^1000 = 0.9048328936`. This is not
equivalent to `optimizer_wd=0.001`.

Both arms must use the same checkpoint kind and restart policy. Full checkpoints
now restore Muon optimizer state/counters and the raw warm-start iterate. A
parameters-only checkpoint initializes fresh identical states. Existing production
checkpoints do **not** save dataset state (`dataset=...` is commented out), and the
training RNG is recreated, so such a resume is not an exact continuation of the
old data/RNG trajectory. Supply the same explicit dataset state where available,
seed, and restart policy to both arms.

Both commands explicitly set `weight_average=False`. This avoids the no-search
EMA confound where the historical baseline averages every inner iterate while a
nonzero-decay run averages only the accepted outer update. Start with a short smoke
run, then run long enough to determine whether the logged norms actually separate;
the 1,000-step factor above does not imply a visible effect over a much shorter run.

## Analysis plan

1. Plot per-leaf weight norms and solver-relative updates versus outer step,
   including early/middle/late attention and MLP matrices plus embedding/head.
2. Plot held-out loss against consistently counted inner-work and line-search
   tokens; record wall-clock time separately.

Norm growth causing slowdown remains an untested hypothesis. A short run can show
whether decay changes norms, but cannot exclude benefits that occur only later.
