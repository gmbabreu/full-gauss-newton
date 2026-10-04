# CG batch lambda and recovery

## Explicit scalar damping and preconditioning

Setting `cg_damping_mu` to a positive finite value replaces the legacy
interpolated CG system with

```text
(G + mu I) x = -g.
```

Here `g` and `G` are computed on the same full frozen solve batch. Scalar
damping changes the system and therefore its solution. In contrast,
`cg_preconditioner=adam_diag` applies the inverse bias-corrected Adam
second-moment diagonal only as a PCG preconditioner, while
`cg_preconditioner=gn_jacobi` applies the inverse of
`maximum(d_hat, 0) + mu`, where `d_hat` is a sequential Hutchinson estimate of
`diag(G)`. These preconditioners can change convergence speed but not the
converged solution. `cg_x0` remains the warm start; `reset_start=True` retains
its existing zero-start behavior.

The damped system requires fixed pure-GN interpolation flags and Adam `b1=0`,
so its right-hand side is the current full-batch gradient rather than a
momentum average. `gn_jacobi` additionally requires positive scalar damping.
Condition diagnostics are temporarily rejected with positive scalar damping:
the existing `spectrum/A` diagnostics describe the legacy interpolated
operator, not `G + mu I`.

The damped solve reports two distinct, unpreconditioned residuals:

```text
relative_residual = ||(G + mu I)x + g|| / ||g||
cg_raw_gn_relative_residual = ||Gx + g|| / ||g||
```

The first measures convergence of the damped system actually passed to CG. The
second measures violation of the original undamped GN equation and need not
vanish: an exact damped solution satisfies `Gx + g = -mu*x`. Neither metric is
a preconditioned residual, and choosing `adam_diag` versus `gn_jacobi` does not
change either definition.

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
Negative/nonfinite denominators, non-CG solvers, simultaneous ramps, and batches
above the denominator are rejected. Startup checks include
the growth cap; actual batch size is checked again at each solve. No clamping is
performed. The LR schedule remains update-based; this rule does not eliminate
all coupling between batch schedules and token-budget comparisons, or guarantee
training stability. Set the denominator to zero AND disable the old ramp when
requesting constant lambda.

## Condition diagnostics

`--condition_log=True` runs spectral diagnostics after the first successfully
completed outer update of each invocation and whenever the completed-update
number is divisible by `condition_every` (default 100). Thus a fresh run with
the default cadence diagnoses updates 1, 100, 200, and so on; initial or loaded
weights are not diagnosed before an update. `condition_log=False` disables all
of these diagnostics.

Every supported solver (CG, Adam-GN, Muon-GN) estimates the leading
`spectrum_top_k` eigenvalues of raw `G` at the committed post-update raw model
weights, including its largest eigenvalue. CG additionally estimates both
endpoints of `A=lambda*G+(1-lambda)/eta*D` at those post-update weights, using
the effective lambda, safe Adam learning rate, and bias-corrected diagonal from
the just-completed solve. Consequently `spectrum/A` is not the exact operator
used during that preceding solve. The existing maximum of
symmetric `P=D^-1/2*A*D^-1/2` and its damping proxy are retained. The new minimum
estimate is for `A`, not `P`.

Diagnostics reuse the frozen solve batch, never fetch data or consume training
RNG, and their products are excluded from solve-token accounting. For Adam-GN
and Muon-GN with multiple inner batches, the first already-fetched inner batch
is used. Dropout and FCM must be disabled. `cg_n_micro` microbatches diagnostic
`Gv` for every supported solver, including Muon; it does not microbatch Muon's
inner training solve.

When diagnostic and checkpoint cadences coincide, both refer to the same raw
model parameters; for example, the row after zero-based loop step 99 aligns with
checkpoint 100. The completed-update and checkpoint recovery axes are unchanged.
Curvature comparisons between Muon and PCG also require matching diagnostic
data: equal weights alone do not imply equal spectra when the retained solve
batches differ.

`spectrum/G/lambda_1_est` is the one canonical maximum series. An accepted
top-k result supplies it directly. If full top-k acceptance fails, a bounded
power estimate supplies the same series only when that endpoint is resolved and
finite. If the fallback is unresolved, the last completely direct-checked
Lanczos maximum may be reported as unresolved; there is no duplicate
fallback-value series. `lambda_1_from_fallback`,
`lambda_1_resolved`, `lambda_1_residual`, and `lambda_1_residual_tol` make the
source and validation standard explicit. The top-k direct tolerance defaults to
0.01, while the maximum-only fallback defaults to 0.05. A maximum-only fallback
never fabricates higher ranks or condition ratios. Raw-G metrics live only under
`spectrum/G/*`; failure and fallback status remain available when unresolved.

`A`'s maximum uses power iteration; its minimum uses inverse iteration with
compiled, diagonally preconditioned inner CG solves. Defaults are
`spectrum_endpoint_maxiter=24` outer iterations for each endpoint,
`spectrum_endpoint_num_starts=2`, `spectrum_inverse_cg_maxiter=100`, and
`spectrum_inverse_cg_tol=0.001`. Endpoint stability and eigenpair acceptance use
`spectrum_endpoint_agreement_tol` and `spectrum_endpoint_residual_tol`.
Actual inner-solve residuals, eigenpair residuals, and agreement between starts
must pass; unresolved minima and condition ratios are withheld.
`spectrum/A/lambda_min_est` and `spectrum/A/condition_est` are estimates, not
certified spectral bounds. Singular undamped systems may remain unresolved.
Inverse iteration adds GN products beyond the spectrum product budget; its cap
is separate. All damped and preconditioned results live under `spectrum/A/*`;
preconditioned fields use a `preconditioned_` prefix. For
`P=cI+B`, the maximum is found on
`B=lambda*D^-1/2*G*D^-1/2` before adding the known identity shift, avoiding
premature convergence on an identity-dominated `P`. The descriptive
`preconditioned_damping_condition_proxy` is retained. Trace and trace-square
probes are no longer run by the trainer because they added full GN products but
were not needed for the eigenvalue objective.

### CPU-resident top-100 spectrum

`--condition_log=True` includes a thick-restarted Lanczos estimate of raw
`G`; `spectrum_top_k` defaults to 100. One FP32 basis stays on CPU. The small
symmetric recurrence matrix and its eigendecomposition use FP64. A preflight
includes the basis, active blocks, bounded transformation/transfer workspace,
cgroup-aware available memory, and an 8 GiB reserve. The default 600-vector
basis is intended for the high-memory TPU host used by this project; at 150M
parameters the basis alone is about 335 GiB. On multiple hosts, only process 0
allocates that basis and runs the unchanged CPU estimator. It broadcasts each
probe and directs all hosts through the same sharded GN product; global products
are gathered before CPU calculations. The final report is broadcast before the
fallback/A diagnostics so every host takes the same path. Other hosts hold
transient full vectors, not a second basis. Process 0 still needs sufficient
host memory; adding a worker does not pool RAM for the Lanczos basis.

CPU preflight/estimator exceptions release waiting peers. A killed process or
failed device collective still relies on JAX distributed failure handling.
Single-host execution, flags, training updates and estimator mathematics are
unchanged. Multi-host transfers add diagnostic wall time.

After a multi-host diagnostic, every probe and product buffer is explicitly
deleted and the diagnostic/collective compilation caches are released on each
host. This avoids carrying transfer executables into an HBM-constrained inner
solve. Because JAX does not expose the collective helpers' individual caches,
`jax.clear_caches()` clears caches globally within that process; the first training dispatch
after a later periodic diagnostic may therefore recompile. Single-host cache
reuse is unchanged. The terminal line
`[spectrum] cleanup: released multi-host diagnostic caches` confirms that the
boundary completed.

The two-process CPU integration check is opt-in (run once on a machine with a
working JAX CPU collective transport; no TPU training is launched):

```bash
RUN_MULTIHOST_TESTS=1 JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q -s tests/test_multihost_spectrum.py
```

It checks top-100 values against dense NumPy eigenvalues of an explicit J^T J,
parameter-sharded products, replicated leaves, identical metrics/counts,
unchanged parameters, failed-spectrum fallback, A endpoints, and CPU error
propagation. `SPECTRUM_TEST_MPI=1` optionally uses an installed MPI-enabled JAX
CPU runtime and `mpiexec`; ordinary runs use Gloo.

Validation before the cleanup change: 63 CPU tests passed
(one opt-in transport test skipped), including numerical/routing/Adam
regressions, four local sharded devices, and a simulated-transport controller.
The four-device top-100 test has max relative error 5.90e-8 and resolves
fallback G and damped/preconditioned A endpoints. The latter checks a
real JAX J^T J and top-100 values (max relative error 7.21e-8), but is **not** a
real multi-host test. The opt-in integration test could not reach diagnostics:
Gloo failed local transport setup (EPERM), and MPI lacked its trampoline wrapper.
The cleanup-specific tests cover successful, failed, and single-host boundaries,
but the current scratch runtime cannot import its stale CPU JAX build. A real
two-host TPU smoke run remains required before a long ablation run.

The short recurrence uses an FP32 overlap scan and adaptive two-pass corrective
reorthogonalization. Thick restart retains `spectrum_restart_keep` Ritz vectors
and their residual coupling `beta * Y[-1, :keep]`, producing an arrowhead
projection before ordinary Lanczos expansion resumes. Numerical breakdown
starts a random direction orthogonal to the current basis. The old residual
expansion solver and stored `GQ` buffer have been removed.

`spectrum_check_every` controls the projected-eigensolve cadence and the bounded
validation reconstruction batch. The FP64 Gram matrix is cached: validation
retains old-old entries and computes only old-new and new-new blocks in bounded
coordinate chunks. A thick restart transforms and rounds stored rows to FP32,
so it invalidates this cache and the next validation rebuilds it from actual
stored rows. The cache is included in host-memory preflight accounting.
Recurrence residuals screen convergence, together with the existing eigenvalue
stability tolerance. Before acceptance, a full Gram check and fresh direct
residual checks of every scalar rank that will be published (1, 10, 20, ...,
top-k) guard against recurrence drift and lost orthogonality. Consequently
`spectrum_max_gn_products=600` reserves 11 products for final validation when
`spectrum_top_k=100`. This budget includes all spectrum operator calls; PCG
endpoint work remains separate.

FP32 recurrence residuals can become optimistic at slightly different basis
sizes when device reductions are regrouped across hosts. A recurrence-qualified
candidate that fails fresh direct residual validation therefore no longer ends
the estimate immediately. Retries are spaced by 8, 16, ...
`spectrum_check_every` expansion steps. `spectrum_max_validation_attempts=3`
caps total complete validations, and `spectrum_max_seconds=3600` adds a soft,
cooperative host-0 deadline; zero disables only the elapsed-time limit. The
deadline is checked around products and within chunked CPU work, never by peers
using independent clocks and never by asynchronously interrupting a collective.
An in-flight device/BLAS call can overrun it, and fallback plus cleanup take
additional time. Controlled stops report `validation_attempt_budget_exhausted`
or `time_budget_exhausted`, withhold incomplete candidates, and then use the
existing bounded maximum fallback. All validation products count against
`spectrum_max_gn_products`.

All products use the same frozen parameters, batch and microbatch weighting.
The existing `spectrum/G/lambda_*_est` series report the last complete direct
validation attempt, even when its residual threshold fails; interpret them with
`accepted`, `max_direct_residual`, and the residual tolerance. Values remain
withheld when validation did not complete or found an invalid/nonfinite basis,
candidate, or operator product. Every tenth rank through top-k and the top-k
endpoint are reported without extra operator products. Condition ratios use
the matching Lanczos values from that completed validation snapshot, even
when it is unaccepted; they never mix a fallback maximum with a Lanczos
endpoint and should also be interpreted with `accepted`. Validation
attempts, phase and transfer timers, capacity/memory status, maximum direct
residual, and the rank with that worst residual are forwarded to metrics.
Validation starts/ends, restarts, and long chunked CPU work also produce terminal
progress so quiet TPU-product periods are distinguishable from a hang.

Lanczos still uses Rayleigh--Ritz on a small recurrence matrix. Its advantage
here is avoiding repeated full-basis projection/residual reconstruction and
storing only one basis, not making the small eigensolve faster. Direct residuals
are not a proof that no larger eigenvalue was missed; independent-seed and
larger-budget repeats remain useful controls. As a single-vector method, this
estimator also does not guarantee recovery of the full multiplicity of an
exactly repeated leading eigenvalue; the stochastic GN spectrum is expected to
be generic, but multiplicity-sensitive studies require a block method. TPU-host
speedup must be measured.

Suggested validation commands (run on a host with JAX and sufficient RAM):

```bash
PYTHONPATH=. python -m unittest tests.test_matrix_condition tests.test_matrix_spectrum -v
# One frozen measurement with the trial budget:
python -m EasyLM.models.llama.llama_train_gn ... --condition_log=True --condition_every=1
# Independent seed and larger-budget repeats:
python -m EasyLM.models.llama.llama_train_gn ... --condition_log=True --spectrum_seed=1
python -m EasyLM.models.llama.llama_train_gn ... --condition_log=True --spectrum_max_gn_products=1200
```

The diagnostic controls, including the two new `spectrum_*` limits, may change
on exact resume because they are observational and do not alter optimizer state
or data consumption.

This implementation was transplanted from diagnostics-only commits `b891ac7`,
`bd5401d`, `0ce84ca`, and `db86be7` onto main `1ac6532`, then extended here.
CPU tests cover exact/rotated spectra, thick restart, explicit-Jacobian GN,
incremental Gram reuse/rebuild, retry/product limits, cooperative deadlines,
canonical maximum routing, and simulated multi-host coordination/cleanup. They
do not diagnose the observed 12-hour event, establish a TPU speedup, or replace
a real multi-host run. Before another full run, perform a short training
continuation smoke test and repeat diagnostics on the saved difficult
checkpoint. If the residual plateau remains, next test the frozen operator's
repeatability, linearity, symmetry, and BF16/FP32 discrepancy; a
diagnostic-only precision change is separate work requiring its own validation.

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

## Reset-start memory validation status

An AdamW GN run has failed at outer update 1 while allocating a program after a
successful update 0 with `reset_start=True`.  Retention of the previous inner
optimizer state is a hypothesis, not a confirmed root cause.  The trainer now
releases the synchronization-only result list immediately after its barrier and
drops both train-state references to the completed inner optimizer slots before
initializing their replacement.  CPU tests establish reset-state semantic
equivalence only; they do not
establish a TPU memory reduction or rule out condition diagnostics.

For user-operated TPU validation, start from the original batch-600 command and
keep all learning-rate and schedule arguments, especially the original
`--total_steps`: it determines schedule decay (and, depending on `--lr_sched`,
the per-outer-step schedule construction).  Do not shorten it to three.  First
run with:

```bash
<original command> --train_dataset_batch_size=600 --reset_start=True \
  --condition_log=False
```

Observe completion of outer updates 0, 1, and 2, then stop the run manually (or
use the existing job controller) after the third update.  Repeat from a fresh
run/checkpoint with the original condition cadence restored, for example:

```bash
<original command> --train_dataset_batch_size=600 --reset_start=True \
  --condition_log=True
```

Again exercise at least updates 0 through 2.  Record peak HBM and whether the
allocation failure recurs in each run; until this is done, the memory-lifetime
explanation remains unverified.
