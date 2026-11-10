"""Undamped, unpreconditioned MINRES-QLP for real symmetric operators.

QLP from iteration one; zero start; constant vector storage. Recurrence adapted
from Choi and Saunders' BSD-licensed SOL Fortran v27 (see
docs/minres_qlp.md and LICENSES/minres_qlp.txt). Normalization and verified
stopping follow the supplied scaled CPU prototype, not its code-generation
adapter. Numerical rank truncation is an explicit policy, not exact inversion
of every small eigenvalue. No minimum-norm certificate is implied by LS exit.
"""
import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


STOP_REASONS = ('equation', 'least_squares_stationary', 'iteration_limit',
                'breakdown', 'nonfinite')


def _norm(v):
    """Scaled Euclidean norm, avoiding squared-norm overflow/underflow."""
    scale = jnp.max(jnp.abs(v), initial=0.)
    return scale * jnp.sqrt(jnp.sum(jnp.square(v / jnp.where(scale > 0, scale, 1))))


def _givens(a, b):
    radius = jnp.hypot(a, b)
    safe = jnp.where(radius > 0, radius, 1)
    return jnp.where(radius > 0, a / safe, 1), b / safe, radius


def minres_qlp(matvec, b, *, tol=1e-5, atol=0., maxiter=100,
               rank_rtol=None):
    """Return ``(x, info)`` for H x = b or stationary min ||H x - b||.

    ``b`` is a real FP32/FP64 array pytree and ``matvec`` preserves its structure
    and dtypes. JIT compatible; tolerances/maxiter are static configuration.
    Equation exit requires ||b-Hx|| <= max(atol, tol ||b||). LS exit additionally
    requires a truncated QLP pivot and ||Hr|| <= max(tol, rank_rtol) Anorm ||r||.
    Anorm is a projected norm estimate, not a spectral norm certificate.
    Every iteration uses three matvecs; setup uses one and final verification
    uses two (zero RHS uses none). ``converged`` means equation convergence;
    ``least_squares_stationary`` is separate. Nonfinite results are not success.
    """
    if (not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 0
            or not math.isfinite(tol) or tol < 0
            or not math.isfinite(atol) or atol < 0):
        raise ValueError('maxiter, tol and atol must be finite and nonnegative')
    rhs, unravel = ravel_pytree(b)
    if rhs.dtype not in (jnp.float32, jnp.float64):
        raise TypeError('MINRES-QLP requires real FP32 or FP64 inputs')
    eps = jnp.finfo(rhs.dtype).eps
    rank_rtol = 32 * eps if rank_rtol is None else rank_rtol
    if not math.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError('rank_rtol must be finite and in (0, 1)')
    ls_tol = max(tol, rank_rtol)
    tiny = jnp.finfo(rhs.dtype).tiny
    zero = jnp.zeros_like(rhs)
    scalar = lambda value: jnp.asarray(value, rhs.dtype)

    def original(v):
        return ravel_pytree(matvec(unravel(v)))[0]

    bn = _norm(rhs)
    unit = rhs / jnp.where(bn > 0, bn, 1)
    probe = jax.lax.cond((bn > 0) & jnp.isfinite(bn), original,
                         lambda v: zero, unit)
    scale0 = _norm(probe)
    scale = jnp.where(scale0 > 0, scale0, scalar(1))

    def av(v):
        return original(v) / scale

    # Scalar recurrence names follow the SOL notation: l denotes one lag.
    s = {name: scalar(0) for name in (
        'beta', 'tau', 'taul', 'dltan', 'eplnn', 'gama', 'gamal', 'gamal2',
        'eta', 'etal', 'vepln', 'veplnl', 'ul3', 'ul2', 'ul', 'anorm')}
    s.update(cs=scalar(-1), sn=scalar(0), cr1=scalar(-1), sr1=scalar(0),
             cr2=scalar(-1), sr2=scalar(0), phi=scalar(1), betan=scalar(1),
             r1=zero, r2=unit, w=zero, wl=zero, xl2=zero, x=zero,
             iterations=jnp.int32(0), rank_events=jnp.int32(0),
             status=jnp.where(~jnp.isfinite(bn) | ~jnp.isfinite(scale0), 4,
                              jnp.where(bn == 0, 0, 2)).astype(jnp.int32),
             rn=scalar(1))

    def body(old):
        s = dict(old)
        k = old['iterations'] + 1
        beta = s['betan']
        v = s['r2'] / jnp.where(beta > 0, beta, 1)
        y = av(v) - jnp.where(k > 1, beta / jnp.where(s['beta'] > 0, s['beta'], 1), 0) * s['r1']
        alpha = jnp.vdot(v, y)
        y = y - alpha * v
        betan = _norm(y)
        # Correct column norm: current beta, not the previous beta; no b norm.
        pnorm = jnp.hypot(jnp.where(k > 1, beta, 0), jnp.hypot(alpha, betan))
        delta = s['cs'] * s['dltan'] + s['sn'] * alpha
        gbar = s['sn'] * s['dltan'] - s['cs'] * alpha
        eplnn = s['sn'] * betan
        dltan = -s['cs'] * betan
        gamal2, gamal = s['gamal'], s['gama']
        cs, sn, gama = _givens(gbar, betan)
        taul2, taul = s['taul'], s['tau']
        tau, phi = cs * s['phi'], sn * s['phi']
        veplnl2, etal2 = s['veplnl'], s['etal']
        etal = jnp.where(k > 2, s['eta'], s['etal'])
        veplnl = jnp.where(k > 2, s['cr2'] * s['vepln'] + s['sr2'] * delta, s['veplnl'])
        delta = jnp.where(k > 2, s['sr2'] * s['vepln'] - s['cr2'] * delta, delta)
        eta = jnp.where(k > 2, s['sr2'] * gama, s['eta'])
        gama = jnp.where(k > 2, -s['cr2'] * gama, gama)
        c1, s1, gl = _givens(gamal, delta)
        cr1, sr1 = jnp.where(k > 1, c1, s['cr1']), jnp.where(k > 1, s1, s['sr1'])
        gamal = jnp.where(k > 1, gl, gamal)
        vepln = jnp.where(k > 1, sr1 * gama, s['vepln'])
        gama = jnp.where(k > 1, -cr1 * gama, gama)
        ul3 = s['ul2']
        # Safe denominators avoid NaNs in inactive first/second-step branches.
        ul2 = jnp.where(k > 2, (taul2 - etal2 * s['ul3'] - veplnl2 * ul3)
                        / jnp.where(gamal2 != 0, gamal2, 1), s['ul2'])
        ul = jnp.where(k > 1, (taul - etal * ul3 - veplnl * ul2)
                       / jnp.where(gamal != 0, gamal, 1), s['ul'])
        norm_est = jnp.maximum(s['anorm'], pnorm)
        truncated = jnp.abs(gama) <= rank_rtol * norm_est
        u = jnp.where(truncated, 0, (tau - eta * ul2 - vepln * ul)
                      / jnp.where(gama != 0, gama, 1))

        def later(_):
            wl2 = s['wl'] * s['cr2'] + v * s['sr2']
            w0 = s['wl'] * s['sr2'] - v * s['cr2']
            return wl2, s['w'] * cr1 + w0 * sr1, s['w'] * sr1 - w0 * cr1

        wl2, wl, w = jax.lax.cond(k > 2, later, lambda _: (
            s['wl'],
            jnp.where(k == 1, v * sr1, s['w'] * cr1 + v * sr1),
            jnp.where(k == 1, -v * cr1, s['w'] * sr1 - v * cr1)), None)
        xl2 = s['xl2'] + wl2 * ul2
        x = xl2 + wl * ul + w * u
        cr2, sr2, gamal = _givens(gamal, eplnn)
        anorm = jnp.maximum(norm_est, jnp.maximum(gamal, jnp.abs(gama)))
        r = unit - av(x)
        ar = av(r)
        rn, arn = _norm(r), _norm(ar)
        finite = jnp.all(jnp.isfinite(x)) & jnp.isfinite(rn) & jnp.isfinite(arn) & jnp.isfinite(anorm)
        equation = rn <= jnp.maximum(tol, atol / jnp.where(bn > 0, bn, 1))
        stationary = truncated & (arn <= ls_tol * jnp.maximum(anorm, tiny) * jnp.maximum(rn, tiny))
        broken = (betan == 0) | ((k > 2) & (gamal2 == 0)) | ((k > 1) & (gl == 0))
        status = jnp.where(~finite, 4, jnp.where(equation, 0,
                 jnp.where(stationary, 1, jnp.where(broken, 3, 2)))).astype(jnp.int32)
        s.update(beta=beta, betan=betan, r1=s['r2'], r2=y, cs=cs, sn=sn,
                 dltan=dltan, eplnn=eplnn, gama=gama, gamal=gamal, gamal2=gamal2,
                 tau=tau, taul=taul, phi=phi, eta=eta, etal=etal, vepln=vepln,
                 veplnl=veplnl, ul3=ul3, ul2=ul2, ul=ul, cr1=cr1, sr1=sr1,
                 cr2=cr2, sr2=sr2, wl=wl, w=w, xl2=xl2, x=x, anorm=anorm,
                 iterations=k, rank_events=s['rank_events'] + truncated.astype(jnp.int32),
                 status=status, rn=rn)
        return s

    s = jax.lax.while_loop(lambda s: (s['status'] == 2) & (s['iterations'] < maxiter), body, s)
    x = s['x'] * (bn / scale)
    # Verify the returned, rescaled iterate against the ORIGINAL equation.
    def verify(x):
        r = rhs - original(x)
        return _norm(r), _norm(original(r))

    rn, arn = jax.lax.cond((bn > 0) & jnp.isfinite(bn), verify,
                           lambda x: (bn, bn), x)
    finite = jnp.all(jnp.isfinite(x)) & jnp.isfinite(rn) & jnp.isfinite(arn) & jnp.isfinite(scale0)
    eq = finite & (rn <= jnp.maximum(atol, tol * bn))
    ls = finite & ~eq & (s['rank_events'] > 0) & (arn <= ls_tol * s['anorm'] * scale * jnp.maximum(rn, tiny))
    status = jnp.where(~finite, 4, jnp.where(eq, 0, jnp.where(ls, 1,
                 jnp.where(s['status'] == 3, 3, 2)))).astype(jnp.int32)
    info = dict(status=status, iterations=s['iterations'], rank_events=s['rank_events'],
                converged=eq, least_squares_stationary=ls,
                true_residual_norm=rn, true_stationarity_norm=arn,
                relative_residual=rn / jnp.where(bn > 0, bn, 1),
                estimated_residual_norm=jnp.abs(s['phi']) * bn,
                direction_norm=_norm(x), operator_scale=scale,
                operator_norm_estimate=s['anorm'] * scale, rank_rtol=scalar(rank_rtol),
                matvec_count=jnp.where((bn > 0) & jnp.isfinite(bn), 3 + 3 * s['iterations'], 0))
    return unravel(x), info
