# Undamped MINRES-QLP project notes

Implemented and CPU-validated on 2026-11-10, starting from main `ee92b75`.
The earlier plain-GN integration and acceptance guard were found in local
`origin/MINRES` (`4d292ae`, `c5adf6a`); those patterns are reused narrowly.
The supplied ZIP is left untouched and untracked. No TPU work, push, or merge.

Enable with:

```text
--optimizer_type=cg --cg_solver=minres_qlp --cg_plain_gn=True
--cg_preconditioner=none --cg_damping_mu=0
--cg_interpolation_lambda=1 --cg_lambda_batch_denominator=0
--cg_lambda_final=-1 --cg_lambda_ramp_steps=0
--normalize_step=False --outer_momentum_beta=0
```

`cg_maxiter`, `cg_tol`, and `cg_atol` retain their roles. Set
`cg_qlp_rank_rtol` explicitly in (0,1), or leave it at 0 to select 32 times the
working dtype epsilon (FP32: 3.8147e-6, FP64: 7.1054e-15). This truncates small
projected QLP pivots. It is numerical-rank regularization, not an exact inverse
of every small positive eigenvalue, nor the same policy as a dense SVD cutoff.
Solver, plain-mode, and rank-tolerance changes remain resume-incompatible.
Old bundles missing these fields normalize to legacy CG defaults. The existing
checkpoint layout, moment/warm-start slots, RNG consumption and step counter
are retained; QLP leaves the three vector slots unchanged. Nested Adam settings
are inactive for this direction. The default legacy route is unchanged.

The solver uses a zero start, RHS normalization and the single probe
`||H(b/||b||)||` to scale its operator. This probe is not a spectral norm.
It uses QLP from the first step, one JAX while loop and a fixed number of
vectors, with no Krylov basis, reorthogonalization, restart or damping.
The corrected projected column norm includes the current Lanczos beta after
the first step. Both normalized and returned original-coordinate iterates are
checked using operator products; the recurrence residual never certifies success.

Status values are 0 equation convergence, 1 least-squares stationarity,
2 iteration limit (also failed final verification), 3 breakdown without verified
success, and 4 nonfinite arithmetic. `converged` denotes equation convergence
only. Stationarity additionally requires numerical-rank detection; it is not a
universal minimum-length certificate. A finite descent direction can still be
useful at an iteration limit. Invalid/non-descent directions select the original
parameters directly. Every actual outer candidate must improve the fixed
line-search baseline, including a fixed-step override; otherwise the accepted
scale is zero. This also works when all proposed steps fail.

`gn_quad` retains raw and accepted quadratic predictions, accepted actual
solve-batch and line-search-batch improvements, and residual diagnostics.
`cg_true_residual`, `cg_direction_norm`, `cg_accepted_scale`, and
`cg_step_rejected` report direction/acceptance behavior. `qlp/*` contains status,
iterations, rank events/tolerance, true residual, stationarity, norm estimate,
operator scale, and matvec counts. Solver matvec count is `3 + 3*iterations`
(one probe, three products per step, two final checks), or zero for zero RHS.
`qlp/total_matvec_count` also includes the optional raw `gn_quad` Gd product.
Separate condition/spectrum diagnostics keep their own budgets; each aggregated
Gv here may itself visit several microbatches. A runtime callback test checks
the solver count, including verification products.

## CPU evidence

Windows CPU: Python 3.11.5, JAX/jaxlib 0.10.2, NumPy 2.3.5, SciPy 1.17.1.
The existing `.venv` pointed to a missing interpreter. An ignored local
`.cache/qlp-venv` was used. Dense controls also ran initially with NumPy 2.4.6;
NumPy 2.3.5 is necessary to run the unchanged supplied prototype adapter.
Its pinned source blob SHA-1 was checked against
`d5f80ccbbc0f81b5ce22942e4d7dd2312930a5c6`.

Committed measurements: [dense controls](minres_qlp_cpu_results.json) and
[masked GN controls](minres_qlp_gn_cpu_results.json). Dense residuals are also
recomputed independently with NumPy FP64 from the working-precision matrix,
RHS and returned solution. The 42 acceptance controls cover SPD, singular
compatible/inconsistent, diagonal/rotated systems, jointly scaled matrices/RHS
and independently scaled RHS (1e-8, 1, 1e8), with JIT and pytree inputs.
They reproduce the prototype's previous inconsistent-system/scaling failures.

| Controls | FP32 maximum | FP64 maximum |
| --- | ---: | ---: |
| Relative solution error vs dense pseudoinverse | 1.30e-6 | 4.25e-15 |
| Equation-control relative residual | 3.53e-7 | 7.97e-16 |
| Prototype relative solution error, same 42 controls | 1.51e-6 | 4.23e-15 |
| Masked GN independent operator entry error | 1.78e-8 | 4.17e-17 |
| Masked GN solution error | 1.51e-4 | 7.96e-13 |
| Masked GN absolute residual | 2.18e-7 | 3.65e-13 |

The GN check executes the actual trainer `apply_G` closure and weighted
microbatch reducer at 1, 2, 3 and 5 microbatches over five sequences. It compares
with independently assembled `J.T @ W @ J`, using analytic softmax Hessian
blocks and masks with unequal valid-token counts and a fully masked sequence.
FP32 uses the actual production loss; FP64 uses the same mathematical loss
without production's intentional FP32 logits cast. Trainer tests separately
exercise the actual update kernel and persistent slots.

Severe conditioning remains an explicit limitation. At nominal condition
1e10, diagonal/rotated FP32 controls have relative residuals 6.72e-7/5.52e-7
but solution errors 0.108/0.0577 against the specified truncated pseudoinverse.
FP64 residuals 1.52e-13/9.49e-13 coexist with solution errors
3.46e-6/4.27e-6. These are diagnostic measurements, not accuracy acceptance
tests. No claim of general robustness, exact rank recovery, or minimum-norm
certification follows from these small controls. Near-threshold clusters,
finite-precision nonsymmetry and null-space pollution remain limitations.

Focused CPU command:

```text
python -m pytest tests/test_minres_qlp.py tests/test_plain_gn_qlp.py tests/test_gn_jacobi_pcg.py tests/test_accepted_gn_comparison.py tests/test_gn_fixed_step.py tests/test_condition_routing.py tests/test_microbatch_flag.py -q
```

The focused run passed 47 tests and 59 subtests (including duplicate collection
of seven legacy harness tests, subsequently removed). A final targeted run
checks observed runtime matvec counts and the production-loss GN controls:
3 tests passed, 9 deselected. Syntax compilation and `git diff --check` pass.
The final trainer-guard rerun passed all four new integration tests.

Additional local ignored `test_gn_quadratic_logging.py` was examined. Against
the original main source it already has two failing tests from stale AST
harness namespaces (`n_micro`/`FLAGS`) and six failing dropout subtests.
Those unrelated local tests and production behavior were left intact.

To regenerate JAX diagnostic reports: `python -m tests.test_minres_qlp`.
The prototype comparison used the supplied ZIP only in `.cache`; its runtime
string replacement and exec machinery is not part of the production solver.

## Source and licensing

The supplied prototype fetches Yang Liu's Python recurrence, whose header
requests translation permission. The earlier official CPS11 MATLAB package
has the same restriction. Neither source is redistributed or used as the
licensed code-port basis. Instead, the recurrence source is the later official
[SOL Fortran v27 distribution](https://web.stanford.edu/group/SOL/software/minresqlp/minresqlp-f90/),
whose `readme.txt` explicitly offers CPL or BSD. BSD is selected and attribution
and terms are retained in [LICENSES/minres_qlp.txt](../LICENSES/minres_qlp.txt).
The normalized CPU prototype supplies the tested scaling, corrected norm,
rank-threshold and explicit residual/stationarity policy. See also Choi, Paige
and Saunders (2011), and Choi and Saunders, Algorithm 937 (2014).
