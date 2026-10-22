Plain Gauss–Newton with MINRES
=============================

Use the existing CG training route with:

```sh
--optimizer_type=cg --cg_plain_gn=True --cg_solver=minres --cg_damping=0.1
```

This solves `(G + mu I)d = -g` using the current gradient, deterministic
model evaluation, and the same fixed solve batch for every matrix product.
Damping is added once after the existing weighted microbatch aggregation.
`cg_maxiter`, `cg_tol`, `cg_atol`, and `cg_n_micro` apply to both plain solvers.
`cg_solver=cg` selects JAX CG with `x0=None` and `M=None` in plain mode.
MINRES requires plain mode; damping must be finite and nonnegative and requires
plain mode. With all three new flags at their defaults (`cg`, `False`, `0.0`),
the existing Adam-interpolated, preconditioned, warm-started CG path is unchanged.

Plain mode passes `params+d` into the existing line search. Adam learning rate,
moments, interpolation, and AdamW decay do not enter the solve or candidate.
The serialized first moment, second moment, and warm-start slots remain unchanged;
the compatibility counter advances once per outer update and still equals the
training step. Missing new checkpoint flags mean legacy defaults. Explicit
solver, mode, or damping changes fail trajectory compatibility checks.

MINRES uses the ordinary Paige–Saunders Lanczos/QR recurrence, with one
`lax.while_loop`, constant vector storage, and one GN product per iteration.
A final product recomputes the true residual. Success requires
`||rhs-A d|| <= max(cg_atol, cg_tol*||rhs||)`. Loss of accuracy in FP32 may make
the recurrence residual optimistic; this is reported as failure, with no
restart or QLP fallback. Singular inconsistent systems need not produce a
minimum-norm least-squares solution. A numerical failure retains the last
finite iterate. The existing line search still evaluates the resulting candidate.

Plain-mode metrics include `cg_true_residual`, `cg_estimated_residual`,
`relative_residual`, `cg_converged`, `cg_stop_reason`, `cg_stop_reason_name`,
`minres_iterations`, `cg_damping`, and `cg_quadratic_change`. The quadratic
change is `g^T d + 0.5 d^T (G+mu I)d`, before line-search scaling. Stop codes
are 0=converged, 1=iteration_limit, 2=breakdown, 3=nonfinite, and
4=true_residual_above_tolerance. JAX CG does not expose an iteration count or
estimated residual: these are logged as -1 and NaN, respectively. For that
solver, a failed finite true-residual check is reported as
true_residual_above_tolerance because the internal termination cause is unavailable.
Direction norm, accepted scale, timing, and evaluation loss keep their existing
logging. Optional spectrum diagnostics in plain mode describe G without the
legacy Adam-interpolated operator.

CPU validation (SciPy is only a test dependency):

```sh
JAX_PLATFORMS=cpu python -m unittest tests.test_minres tests.test_plain_gn -v
```

The tests cover SciPy and dense references, indefinite and singular systems,
zero RHS, pytrees, nonfinite values, iteration limits, FP32 residual drift,
and the actual training kernel against an independently assembled masked
cross-entropy GN matrix, including unequal microbatches and legacy PCG.
