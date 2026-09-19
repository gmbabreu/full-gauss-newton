"""Constant-memory, matrix-free condition diagnostics.

The reported endpoints are *estimates*: residual and agreement checks are not a
proof that an eigenpair is globally extremal.  A failed check is unresolved; it
is never reclassified as numerical singularity.
"""
from dataclasses import dataclass, fields
import math
import time

import jax
import jax.numpy as jnp
import numpy as np


def tree_dot(x, y):
    products = [jnp.vdot(a.astype(jnp.float32), b.astype(jnp.float32)).real
                for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y))]
    return jnp.sum(jnp.stack(products)) if products else jnp.float32(0)


def tree_norm(x):
    return jnp.sqrt(jnp.maximum(tree_dot(x, x), jnp.float32(0)))


def tree_scale(scale, tree):
    return jax.tree.map(lambda value: scale * value, tree)


def tree_add(left, right, *, alpha=1.0):
    return jax.tree.map(lambda x, y: x + alpha * y, left, right)


def random_unit_pytree(key, template):
    leaves, structure = jax.tree.flatten(template)
    keys = jax.random.split(key, len(leaves))
    result = jax.tree.unflatten(structure, [
        jax.random.normal(k, leaf.shape, dtype=jnp.float32).astype(leaf.dtype)
        for k, leaf in zip(keys, leaves)])
    norm = float(tree_norm(result))
    return None if not math.isfinite(norm) or norm <= 0 else tree_scale(1 / norm, result)


def random_rademacher_pytree(key, template):
    """An unnormalised independent +/-1 pytree for Hutchinson probes."""
    leaves, structure = jax.tree.flatten(template)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(structure, [
        jax.random.rademacher(k, leaf.shape, dtype=jnp.float32).astype(leaf.dtype)
        for k, leaf in zip(keys, leaves)])


@dataclass
class Endpoint:
    value_est: float | None
    residual: float | None
    resolved: bool
    iterations: int
    operator_matvecs: int
    inner_solves: int = 0
    inner_solves_converged: bool = True
    rayleigh_agreement: bool = False
    starts_agree: bool = False
    failure_reasons: tuple = ()
    vector: object = None

    def metrics(self):
        # asdict recursively copies a parameter-sized vector before discarding it.
        return {field.name: getattr(self, field.name) for field in fields(self)
                if field.name != 'vector'}


def eigenpair_residual(operator, vector, operator_args=()):
    """Return a direct Rayleigh quotient and scale-invariant residual."""
    if vector is None:
        return None, None
    bv = operator(vector, *operator_args)
    vv, bv_norm = float(tree_dot(vector, vector)), float(tree_norm(bv))
    if not math.isfinite(vv) or vv <= 0 or not math.isfinite(bv_norm):
        return None, None
    theta = float(tree_dot(vector, bv)) / vv
    if not math.isfinite(theta):
        return None, None
    residual_norm = float(tree_norm(tree_add(bv, vector, alpha=-theta)))
    vector_norm = math.sqrt(vv)
    denominator = max(abs(theta) * vector_norm,
                      float(jnp.finfo(jnp.float32).eps) * bv_norm)
    # The zero operator has a valid quotient but no meaningful relative residual.
    if not math.isfinite(residual_norm) or denominator <= 0:
        return theta, None
    return theta, residual_norm / denominator


def _last_three_agree(history, tolerance):
    if len(history) < 3 or not all(math.isfinite(x) for x in history[-3:]):
        return False
    scale = max(abs(history[-1]), max(abs(x) for x in history[-3:]), 1e-30)
    return (max(history[-3:]) - min(history[-3:])) / scale <= tolerance


def _start_agreement(candidates, tolerance):
    values = [candidate['value'] for candidate in candidates]
    if len(values) < 2 or not all(value is not None and math.isfinite(value)
                                  for value in values):
        return False
    scale = max(max(abs(value) for value in values), 1e-30)
    return max(values) - min(values) <= tolerance * scale


def _endpoint(candidates, matvecs, inner_solves, agreement_tol, residual_tol,
              require_inner):
    starts_agree = _start_agreement(candidates, agreement_tol)
    every_residual = all(item['residual'] is not None
                         and math.isfinite(item['residual'])
                         and item['residual'] <= residual_tol for item in candidates)
    all_inner = all(item['inner_ok'] for item in candidates)
    reasons = []
    if not all(item['converged'] for item in candidates): reasons.append('outer_not_converged')
    if not every_residual: reasons.append('eigenpair_residual')
    if not starts_agree: reasons.append('starts_disagree')
    if require_inner and not all_inner: reasons.append('inner_pcg_failed')
    resolved = not reasons
    valid = [item for item in candidates if item['value'] is not None]
    chosen = min(valid, key=lambda item: item['residual'] if item['residual'] is not None
                 else math.inf) if valid else candidates[0]
    return Endpoint(chosen['value'] if resolved else None, chosen['residual'], resolved,
                    max(item['iterations'] for item in candidates), matvecs,
                    inner_solves, all_inner, chosen['agreement'], starts_agree,
                    tuple(reasons), chosen['vector'])


def power_iteration(operator, template, *, key=jax.random.PRNGKey(0), maxiter=64,
                    miniter=0, agreement_tol=.05, residual_tol=.05, num_starts=2,
                    operator_args=()):
    candidates, matvecs = [], 0
    for start in range(num_starts):
        vector = random_unit_pytree(jax.random.fold_in(key, start), template)
        history, converged, value, residual = [], False, None, None
        for iteration in range(1, maxiter + 1):
            if vector is None: break
            product = operator(vector, *operator_args); matvecs += 1
            norm = float(tree_norm(product))
            if not math.isfinite(norm) or norm <= 0: break
            vector = tree_scale(1 / norm, product)
            value, residual = eigenpair_residual(
                operator, vector, operator_args); matvecs += 1
            if value is None: break
            history.append(value)
            converged = (iteration >= miniter and _last_three_agree(history, agreement_tol)
                         and residual is not None and math.isfinite(residual)
                         and residual <= residual_tol)
            if converged: break
        candidates.append(dict(value=value, residual=residual, vector=vector,
            converged=converged, iterations=iteration, agreement=_last_three_agree(
                history, agreement_tol), inner_ok=True))
    return _endpoint(candidates, matvecs, 0, agreement_tol, residual_tol, False)


def make_pcg_solver(operator, *, preconditioner=None, maxiter=512,
                    tolerance=1e-5):
    """Compile one reusable diagnostic PCG kernel.

    The returned callable is reused by every start and the calibration pass.
    Array-valued state belongs in ``rhs`` (and in explicit arguments of the
    supplied operator), rather than in this controller's Python loop.
    """
    preconditioner = preconditioner or (lambda value, *_: value)
    def solve(initial_rhs, *operator_args):
        rhs_norm = tree_norm(initial_rhs)
        initial_x = jax.tree.map(jnp.zeros_like, initial_rhs)
        initial_z = preconditioner(initial_rhs, *operator_args)
        initial_rz = tree_dot(initial_rhs, initial_z)
        zero_rhs = rhs_norm == 0
        initial_valid = (jnp.isfinite(rhs_norm) & (zero_rhs |
            (jnp.isfinite(initial_rz) & (initial_rz > 0))))
        state = (jnp.int32(0), initial_x, initial_rhs, initial_z, initial_rz,
                 initial_valid, jnp.where(zero_rhs, 0., jnp.inf))

        def cond(value):
            iteration, _, _, _, _, valid, relative = value
            return (iteration < maxiter) & valid & (relative > tolerance)

        def body(value):
            iteration, current_x, r, p, rz, valid, _ = value
            ap = operator(p, *operator_args)
            curvature = tree_dot(p, ap)
            curvature_ok = jnp.isfinite(curvature) & (curvature > 0)
            safe_curvature = jnp.where(curvature_ok, curvature, jnp.float32(1))
            alpha = rz / safe_curvature
            next_x = tree_add(current_x, p, alpha=alpha)
            next_r = tree_add(r, ap, alpha=-alpha)
            relative = tree_norm(next_r) / jnp.where(rhs_norm > 0, rhs_norm, 1.)
            z = preconditioner(next_r, *operator_args)
            next_rz = tree_dot(next_r, z)
            rz_ok = jnp.isfinite(next_rz) & (next_rz > 0)
            still_running = relative > tolerance
            next_valid = valid & curvature_ok & jnp.isfinite(relative) \
                & (~still_running | rz_ok)
            beta = jnp.where(rz_ok, next_rz / rz, jnp.float32(0))
            next_p = jax.tree.map(lambda zi, pi: zi + beta * pi, z, p)
            return (iteration + 1, next_x, next_r, next_p, next_rz,
                    next_valid, relative)

        return (*jax.lax.while_loop(cond, body, state), rhs_norm)

    return jax.jit(solve)


def pcg(operator, rhs, *, preconditioner=None, maxiter=512, tolerance=1e-5,
        compiled_solver=None, operator_args=()):
    """Run a reusable compiled PCG kernel and validate its direct residual."""
    solver = compiled_solver or make_pcg_solver(
        operator, preconditioner=preconditioner, maxiter=maxiter,
        tolerance=tolerance)
    iteration, x, _, _, _, valid, _, rhs_norm = solver(rhs, *operator_args)
    rhs_norm = float(rhs_norm)
    if not math.isfinite(rhs_norm):
        return x, False, math.inf, int(iteration), int(iteration), 'nonfinite_rhs'
    if rhs_norm == 0:
        return x, True, 0., 0, 0, None

    actual = tree_add(operator(x, *operator_args), rhs, alpha=-1)
    relative = float(tree_norm(actual)) / rhs_norm
    valid, iteration = bool(valid), int(iteration)
    converged = valid and math.isfinite(relative) and relative <= tolerance
    reason = None if converged else ('invalid_curvature' if not valid
                                     else 'tolerance_not_met')
    return x, converged, relative, iteration, iteration + 1, reason


def inverse_iteration(operator, template, *, key=jax.random.PRNGKey(1), maxiter=20,
                      miniter=0, inner_maxiter=512, inner_tol=1e-5,
                      agreement_tol=.05, residual_tol=.05, num_starts=2,
                      preconditioner=None, compiled_solver=None,
                      operator_args=()):
    candidates, matvecs, inner_solves = [], 0, 0
    compiled_solver = compiled_solver or make_pcg_solver(
        operator, preconditioner=preconditioner, maxiter=inner_maxiter,
        tolerance=inner_tol)
    for start in range(num_starts):
        vector = random_unit_pytree(jax.random.fold_in(key, start), template)
        history, converged, all_inner, value, residual = [], False, True, None, None
        for iteration in range(1, maxiter + 1):
            if vector is None: break
            vector, ok, _, _, count, _ = pcg(operator, vector,
                preconditioner=preconditioner, maxiter=inner_maxiter,
                tolerance=inner_tol, compiled_solver=compiled_solver,
                operator_args=operator_args)
            matvecs += count; inner_solves += 1; all_inner &= ok
            norm = float(tree_norm(vector))
            if not ok or not math.isfinite(norm) or norm <= 0: break
            vector = tree_scale(1 / norm, vector)
            value, residual = eigenpair_residual(
                operator, vector, operator_args); matvecs += 1
            if value is None: break
            history.append(value)
            converged = (iteration >= miniter and all_inner
                         and _last_three_agree(history, agreement_tol)
                         and residual is not None and math.isfinite(residual)
                         and residual <= residual_tol)
            if converged: break
        candidates.append(dict(value=value, residual=residual, vector=vector,
            converged=converged, iterations=iteration, agreement=_last_three_agree(
                history, agreement_tol), inner_ok=all_inner))
    return _endpoint(candidates, matvecs, inner_solves, agreement_tol,
                     residual_tol, True)


def condition_diagnostic(operator, template, **kwargs):
    started = time.monotonic()
    common = {name: kwargs[name] for name in
              ('agreement_tol', 'residual_tol', 'num_starts') if name in kwargs}
    seed = kwargs.get('key', jax.random.PRNGKey(0))
    operator_args = kwargs.get('operator_args', ())
    top = power_iteration(operator, template, key=seed,
        maxiter=kwargs.get('top_maxiter', 64), operator_args=operator_args,
        **common)
    inverse_args = dict(inner_maxiter=kwargs.get('inner_maxiter', 512),
        inner_tol=kwargs.get('inner_tol', 1e-5),
        preconditioner=kwargs.get('preconditioner'), **common)
    inverse_budget = kwargs.get('inverse_maxiter', 20)
    inverse_key = jax.random.fold_in(seed, 9176)
    compiled_solver = kwargs.get('compiled_solver') or make_pcg_solver(
        operator, preconditioner=kwargs.get('preconditioner'),
        maxiter=kwargs.get('inner_maxiter', 512),
        tolerance=kwargs.get('inner_tol', 1e-5))
    bottom = inverse_iteration(operator, template, key=inverse_key,
        maxiter=inverse_budget, compiled_solver=compiled_solver,
        operator_args=operator_args, **inverse_args)
    calibration, calibration_ok = None, True
    if kwargs.get('calibrate', False):
        # miniter forces genuinely additional work rather than identical early exit.
        calibration = inverse_iteration(operator, template, key=inverse_key,
            maxiter=2 * inverse_budget, miniter=inverse_budget + 1,
            compiled_solver=compiled_solver, operator_args=operator_args,
            **inverse_args)
        calibration_ok = (bottom.resolved and calibration.resolved
            and abs(bottom.value_est - calibration.value_est)
                <= kwargs.get('agreement_tol', .05)
                   * max(abs(calibration.value_est), 1e-30))
    order_ok = (top.value_est is not None and bottom.value_est is not None
        and 0 < bottom.value_est <= top.value_est)
    reasons = list(top.failure_reasons) + list(bottom.failure_reasons)
    if not order_ok: reasons.append('endpoint_order')
    if not calibration_ok: reasons.append('calibration_failed')
    resolved = top.resolved and bottom.resolved and order_ok and calibration_ok
    rayleigh_bound = None
    rayleigh_products = 0
    if top.vector is not None and bottom.vector is not None:
        q_hi, _ = eigenpair_residual(operator, top.vector, operator_args)
        q_lo, _ = eigenpair_residual(operator, bottom.vector, operator_args)
        rayleigh_products = 2
        if q_hi is not None and q_lo is not None and q_hi > 0 and q_lo > 0:
            rayleigh_bound = max(q_hi, q_lo) / min(q_hi, q_lo)
    return dict(lambda_max_est=top.value_est, lambda_max_residual=top.residual,
        lambda_min_est=bottom.value_est, lambda_min_residual=bottom.residual,
        condition_est=(top.value_est / bottom.value_est if resolved else None),
        condition_rayleigh_lower_bound_est=rayleigh_bound, resolved=resolved,
        failure_reasons=tuple(dict.fromkeys(reasons)), top=top.metrics(),
        bottom=bottom.metrics(), calibration=(calibration.metrics() if calibration else None),
        calibration_attempted=calibration is not None,
        calibration_status=('passed' if calibration is not None and calibration_ok
                            else 'unresolved' if calibration is not None
                            else 'previously_passed' if kwargs.get('calibrated', False)
                            else 'not_attempted'),
        calibration_agreement=calibration_ok,
        inner_solves=bottom.inner_solves + (calibration.inner_solves if calibration else 0),
        operator_matvecs=top.operator_matvecs + bottom.operator_matvecs
            + (calibration.operator_matvecs if calibration else 0)
            + rayleigh_products,
        seconds=time.monotonic() - started)


def shifted_operator(operator, shift):
    return lambda vector, *args: tree_add(
        operator(vector, *args), vector, alpha=shift)


def symmetric_diagonal_operator(operator, diagonal):
    def apply(vector, *args):
        actual_diagonal = diagonal(*args) if callable(diagonal) else diagonal
        scaled = jax.tree.map(lambda x, d: x / jnp.sqrt(d), vector, actual_diagonal)
        return jax.tree.map(lambda x, d: x / jnp.sqrt(d),
                            operator(scaled, *args), actual_diagonal)
    return apply


def structural_lower_bounds(effective_lambda, safe_adam_lr, diagonal):
    c = (1 - float(effective_lambda)) / float(safe_adam_lr)
    min_d = min(float(jnp.min(leaf)) for leaf in jax.tree.leaves(diagonal))
    return {'A_lambda_min_lower_bound': c * min_d,
            'P_lambda_min_lower_bound': c}


def summarize_probe_samples(trace_samples, square_samples, lambda_max_est):
    """Summarize paired Hutchinson samples; rank ratios are plug-in estimates."""
    t, s = np.asarray(trace_samples, np.float64), np.asarray(square_samples, np.float64)
    if len(t) < 2 or len(t) != len(s):
        raise ValueError('Need at least two paired probe samples')
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(s))):
        return {'trace_probes': len(t), 'trace_estimates_valid': False}
    trace, trace_square = float(t.mean()), float(s.mean())
    top_ok = lambda_max_est is not None and math.isfinite(lambda_max_est) and lambda_max_est > 0
    return dict(trace_probes=len(t), trace_estimates_valid=trace >= 0 and trace_square >= 0,
        trace_est=trace, trace_sample_se=float(t.std(ddof=1) / np.sqrt(len(t))),
        trace_square_est=trace_square,
        trace_square_sample_se=float(s.std(ddof=1) / np.sqrt(len(s))),
        participation_rank_est=(trace * trace / trace_square
                                if trace > 0 and trace_square > 0 else None),
        stable_rank_est=(trace_square / (lambda_max_est * lambda_max_est)
                         if top_ok and trace_square > 0 else None))


def probe_operators(apply_g, template, *, num_probes=4,
                    key=jax.random.PRNGKey(0), apply_a_from_g=None):
    """Sequential G probes, optionally reusing Gz to probe A without another G product."""
    if num_probes < 2: raise ValueError('Need at least two trace probes')
    samples = {'G': ([], [])}
    if apply_a_from_g is not None: samples['A'] = ([], [])
    for index in range(num_probes):
        z = random_rademacher_pytree(jax.random.fold_in(key, index), template)
        gz = apply_g(z)
        samples['G'][0].append(float(tree_dot(z, gz)))
        samples['G'][1].append(float(tree_dot(gz, gz)))
        if apply_a_from_g is not None:
            az = apply_a_from_g(gz, z)
            samples['A'][0].append(float(tree_dot(z, az)))
            samples['A'][1].append(float(tree_dot(az, az)))
    return samples


def gauss_newton_diagnostics(operator, template, *, shifts=(1e-2, 1e-4), **kwargs):
    """Raw G plus explicitly shifted estimates; no singularity inference."""
    calibration = kwargs.pop('calibrate_by_operator', None)
    solvers = kwargs.pop('compiled_solver_by_operator', None)
    raw_kwargs = dict(kwargs, calibrate=(not calibration.get('G', False)
        if calibration is not None else kwargs.get('calibrate', False)),
        calibrated=(calibration.get('G', False) if calibration is not None else False),
        compiled_solver=(solvers.get('G') if solvers is not None else None))
    raw = condition_diagnostic(operator, template, **raw_kwargs)
    output = {'G': raw, 'shifted': {}}
    if raw['lambda_max_est'] is None: return output
    for relative in shifts:
        mu = relative * raw['lambda_max_est']
        name = f'condition_shift_{relative:.0e}'.replace('e-0', 'e-')
        base_args = kwargs.get('operator_args', ())
        def dynamic_shifted(vector, *args):
            return tree_add(operator(vector, *args[:-1]), vector,
                            alpha=args[-1])
        shifted_kwargs = dict(kwargs, calibrate=(not calibration.get(name, False)
            if calibration is not None else kwargs.get('calibrate', False)),
            calibrated=(calibration.get(name, False) if calibration is not None else False),
            compiled_solver=(solvers.get(name) if solvers is not None else None),
            operator_args=base_args + (mu,))
        shifted = condition_diagnostic(
            dynamic_shifted, template, **shifted_kwargs)
        shifted.update(mu=mu, lambda_min_lower_bound=mu)
        output['shifted'][name] = shifted
    return output
