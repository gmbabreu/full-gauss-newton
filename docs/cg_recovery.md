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

`--condition_log=True` enables all spectral diagnostics at outer step 0 and every
`condition_every` updates (default 50). It is the sole enable switch; remove
`--spectrum_log` and replace `--spectrum_every=N` with `--condition_every=N` in
existing commands. `condition_log=False` disables all of these diagnostics.

Every supported solver (CG, Adam-GN, Muon-GN) estimates the leading
`spectrum_top_k` eigenvalues of raw `G`, including its largest eigenvalue. CG
additionally estimates both endpoints of the actual damped solve matrix
`A=lambda*G+(1-lambda)/eta*D`, using the effective lambda, safe Adam learning
rate, and bias-corrected diagonal from that update. The existing maximum of
symmetric `P=D^-1/2*A*D^-1/2` and its damping proxy are retained. The new minimum
estimate is for `A`, not `P`; structural lower bounds remain separately named.

Diagnostics reuse the frozen solve batch, never fetch data or consume training
RNG, and their products are excluded from solve-token accounting. For Adam-GN
and Muon-GN with multiple inner batches, the first already-fetched inner batch
is used. Dropout and FCM must be disabled. `cg_n_micro` microbatches diagnostic
`Gv` for every supported solver, including Muon; it does not microbatch Muon's
inner training solve.

An accepted top-k result supplies `spectrum/G/lambda_1_est`; the largest
eigenvalue is not recomputed. If top-k convergence fails, a power estimate is
still attempted and recorded as `spectrum/G/fallback_lambda_max_est`. Raw-G
metrics live only under `spectrum/G/*`. W&B retains the accepted eigenvalues,
top-10/top-100 ratios, residual and orthogonality checks, basis size, total raw-G
products, elapsed time, and failure details when unresolved. Detailed phase and
transfer timings remain in terminal output rather than W&B.

`A`'s maximum uses power iteration; its minimum uses inverse iteration with
compiled, diagonally preconditioned inner CG solves. Defaults are
`condition_top_maxiter=24` outer iterations for each endpoint,
`condition_num_starts=2`, `condition_inner_cg_maxiter=100`, and
`condition_inner_cg_tol=0.001`. Actual inner-solve residuals, eigenpair residuals,
and agreement between starts must pass; unresolved minima and condition ratios
are withheld. `spectrum/A/lambda_min_est` and `spectrum/A/condition_est` are
estimates, not certified spectral bounds. Singular undamped systems may remain
unresolved. Inverse iteration adds GN products beyond the spectrum product
budget; its cap is separate. All damped and preconditioned results live under
`spectrum/A/*`; preconditioned fields use a `preconditioned_` prefix. For
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
cgroup-aware available memory, and an 8 GiB reserve. At 150M parameters, one
160-vector buffer is about 89.4 GiB, before workspace and reserve. The diagnostic
refuses unsupported multi-host runs and insufficient host memory.

The short recurrence uses an FP32 overlap scan and adaptive two-pass corrective
reorthogonalization. Thick restart retains `spectrum_restart_keep` Ritz vectors
and their residual coupling `beta * Y[-1, :keep]`, producing an arrowhead
projection before ordinary Lanczos expansion resumes. Numerical breakdown
starts a random direction orthogonal to the current basis. The old residual
expansion solver and stored `GQ` buffer have been removed.

`spectrum_block_size` now controls the projected-eigensolve cadence and the
bounded validation reconstruction batch (not a block Lanczos recurrence).
Recurrence residuals screen convergence, together with the existing eigenvalue
stability tolerance. Before acceptance, a full Gram check and fresh direct
residual checks of every scalar rank that will be published (1, 10, 20, ...,
top-k) guard against recurrence drift and lost orthogonality. Consequently
`spectrum_max_gn_products=600` reserves 11 products for final validation when
`spectrum_top_k=100`. This budget includes all spectrum operator calls; PCG
endpoint work remains separate.

All products use the same frozen parameters, batch and microbatch weighting.
Accepted scalar metrics include `spectrum/G/lambda_1_est`, every tenth rank through top-k
(`lambda_10_est`, `lambda_20_est`, ...), and the top-k endpoint even if it is
not divisible by ten. Intermediate ranks need no additional eigensolve or
operator products for logging. Unresolved estimates remain withheld. Candidate
tables, memory estimates, and historical phase timing keys are omitted from
W&B to keep the dashboard compact; the terminal completion record remains
detailed enough for performance debugging.

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

The diagnostic controls may change on exact resume. The retired
`condition_trace_probes` field is ignored when reading legacy checkpoints. A
legacy checkpoint with `cg_log_matrix_norms=True` is rejected because that old
logging path changed effective lambda and therefore the trajectory.

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
