"""Ordinary, unpreconditioned Paige--Saunders MINRES for real pytrees.

The recurrence follows Paige & Saunders (1975), as in SciPy MINRES. SciPy
is only a test reference. Storage is constant in the iteration budget; each
iteration uses one A product, followed by one final true-residual product.
"""
import math

import jax
import jax.numpy as jnp


CONVERGED, ITERATION_LIMIT, BREAKDOWN, NONFINITE, RESIDUAL_MISMATCH = range(5)
STOP_REASONS = ('converged', 'iteration_limit', 'breakdown', 'nonfinite',
                'true_residual_above_tolerance')


def _dot(x, y):
    return sum(jnp.vdot(a, b).real for a, b in
               zip(jax.tree.leaves(x), jax.tree.leaves(y)))


def _norm(x):
    return jnp.sqrt(_dot(x, x))


def _finite(x):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(a))
                              for a in jax.tree.leaves(x)]))


def minres(A, b, *, tol=1e-5, atol=0.0, maxiter=100):
    """Solve A x = b from zero, without preconditioning, for symmetric A.

    Returns (x, info). Convergence means ||b-Ax|| <= max(atol, tol*||b||),
    verified after the loop. An optimistic recurrence estimate can stop the
    loop but cannot certify success. Singular inconsistent systems report
    breakdown or iteration limit, rather than least-squares convergence.
    ``stop_reason`` is an integer indexing STOP_REASONS (safe under jit).
    No QLP, basis storage, warm start, or automatic differentiation is provided.
    """
    if maxiter < 0 or not math.isfinite(tol) or not math.isfinite(atol) or tol < 0 or atol < 0:
        raise ValueError('MINRES tolerances must be finite and nonnegative; maxiter must be nonnegative')
    b = jax.tree.map(jnp.asarray, b)
    if not jax.tree.leaves(b) or any(
            not jnp.issubdtype(a.dtype, jnp.floating) for a in jax.tree.leaves(b)):
        raise ValueError('MINRES requires a nonempty real floating point pytree')
    zero = jax.tree.map(jnp.zeros_like, b)
    bnorm = _norm(b)
    scalar = lambda x: jnp.asarray(x, dtype=bnorm.dtype)
    threshold = jnp.maximum(scalar(atol), scalar(tol) * bnorm)
    # k, x, r_{k-1}, r_k, w_{k-2}, w_{k-1}, beta_{k-1}, beta_k,
    # dbar, epsln, cos, sin, phibar, numerical stop code (-1 = running).
    initial_reason = jnp.where(~jnp.isfinite(bnorm), NONFINITE,
                               jnp.where(bnorm <= threshold, CONVERGED, -1))
    initial = (jnp.int32(0), zero, zero, b, zero, zero, scalar(1), bnorm,
               scalar(0), scalar(0), scalar(-1), scalar(0), bnorm, initial_reason)

    def condition(s):
        return (s[0] < maxiter) & (s[-1] == -1)

    def body(s):
        k, x, r1, r2, w1, w2, oldb, beta, dbar, epsln, cs, sn, phibar, _ = s
        v = jax.tree.map(lambda r: r / beta, r2)
        av = A(v)
        y = jax.tree.map(lambda a, r: a - (beta / oldb) * r, av, r1)
        alpha = _dot(v, y)
        y = jax.tree.map(lambda a, r: a - (alpha / beta) * r, y, r2)
        next_beta = _norm(y)
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        gamma = jnp.hypot(gbar, next_beta)
        safe_gamma = jnp.where(gamma > 0, gamma, scalar(1))
        next_cs, next_sn = gbar / safe_gamma, next_beta / safe_gamma
        phi = next_cs * phibar
        next_phibar = next_sn * phibar
        w = jax.tree.map(lambda vi, a, c: (vi - epsln * a - delta * c)
                         / safe_gamma, v, w1, w2)
        candidate = jax.tree.map(lambda a, wi: a + phi * wi, x, w)
        finite = (_finite((av, y, w, candidate)) & jnp.isfinite(gamma)
                  & jnp.isfinite(next_phibar))
        # A vanishing Lanczos subdiagonal exhausts the available Krylov space.
        # On the first iteration beta is ||b||, not an operator coefficient.
        exhausted = next_beta <= jnp.finfo(bnorm.dtype).eps * (
            jnp.abs(alpha) + jnp.where(k > 0, beta, scalar(0)))
        reason = jnp.where(~finite, NONFINITE,
                   jnp.where(gamma == 0, BREAKDOWN,
                   jnp.where(jnp.abs(next_phibar) <= threshold, CONVERGED,
                   jnp.where(exhausted, BREAKDOWN, -1))))
        # Keep the last finite iterate on numerical failure.
        x = jax.tree.map(lambda a, c: jnp.where(finite & (gamma > 0), c, a), x, candidate)
        return (k + 1, x, r2, y, w2, w, beta, next_beta,
                -cs * next_beta, sn * next_beta, next_cs, next_sn,
                next_phibar, reason)

    result = jax.lax.while_loop(condition, body, initial)
    k, x = result[:2]
    # The zero RHS needs no operator evaluation (including A(0)).
    ax = jax.lax.cond(bnorm == 0, lambda _: zero, lambda _: A(x), operand=None)
    residual = jax.tree.map(lambda a, rhs: rhs - a, ax, b)
    true_norm = _norm(residual)
    finite = _finite((x, ax, residual)) & jnp.isfinite(true_norm) & jnp.isfinite(bnorm)
    converged = finite & (true_norm <= threshold) & (result[-1] != NONFINITE)
    reason = jnp.where(~finite | (result[-1] == NONFINITE), NONFINITE,
              jnp.where(converged, CONVERGED,
              jnp.where(result[-1] == CONVERGED, RESIDUAL_MISMATCH,
              jnp.where(result[-1] == -1, ITERATION_LIMIT, result[-1]))))
    return x, dict(iterations=k, converged=converged, stop_reason=reason,
                   true_residual_norm=true_norm,
                   estimated_residual_norm=jnp.abs(result[-2]),
                   relative_residual=true_norm / jnp.where(bnorm > 0, bnorm, scalar(1)),
                   quadratic_change=scalar(.5) * _dot(x, ax) - _dot(b, x))
