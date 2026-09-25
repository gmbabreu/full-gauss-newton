"""Spectral endpoint and trace diagnostics for symmetric PSD operators."""
from dataclasses import dataclass, fields
import math
import time

import jax
import jax.numpy as jnp
import numpy as np


def tree_dot(x, y):
    values = [jnp.vdot(a.astype(jnp.float32), b.astype(jnp.float32)).real
              for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y))]
    return jnp.sum(jnp.stack(values)) if values else jnp.float32(0)


def tree_norm(x):
    return jnp.sqrt(jnp.maximum(tree_dot(x, x), jnp.float32(0)))


def tree_add(x, y, *, alpha=1.0):
    return jax.tree.map(lambda a, b: a + alpha * b, x, y)


def random_unit_pytree(key, template):
    leaves, structure = jax.tree.flatten(template)
    keys = jax.random.split(key, len(leaves))
    value = jax.tree.unflatten(structure, [
        jax.random.normal(k, leaf.shape, jnp.float32).astype(leaf.dtype)
        for k, leaf in zip(keys, leaves)])
    norm = float(tree_norm(value))
    return None if not math.isfinite(norm) or norm <= 0 else jax.tree.map(
        lambda leaf: leaf / norm, value)


def random_rademacher_pytree(key, template):
    leaves, structure = jax.tree.flatten(template)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(structure, [
        jax.random.rademacher(k, leaf.shape, jnp.float32).astype(leaf.dtype)
        for k, leaf in zip(keys, leaves)])


@dataclass
class Endpoint:
    value_est: float | None
    rayleigh_quotient: float | None
    residual: float | None
    resolved: bool
    iterations: int
    operator_matvecs: int
    rayleigh_agreement: bool
    starts_agree: bool
    failure_reasons: tuple
    vector: object = None

    def metrics(self):
        return {field.name: getattr(self, field.name) for field in fields(self)
                if field.name != 'vector'}


def eigenpair_residual(operator, vector, operator_args=()):
    product = operator(vector, *operator_args)
    vv = float(tree_dot(vector, vector))
    product_norm = float(tree_norm(product))
    if not math.isfinite(vv) or vv <= 0 or not math.isfinite(product_norm):
        return None, None
    theta = float(tree_dot(vector, product)) / vv
    denominator = max(abs(theta) * math.sqrt(vv),
                      float(jnp.finfo(jnp.float32).eps) * product_norm)
    if not math.isfinite(theta) or denominator <= 0:
        return theta, None
    residual = float(tree_norm(tree_add(product, vector, alpha=-theta))) / denominator
    return theta, residual if math.isfinite(residual) else None


def make_power_solver(operator, *, maxiter=24, agreement_tol=.05,
                      residual_tol=.05):
    """Compile a one-product-per-iteration power loop."""
    def solve(initial_vector, *operator_args):
        history = jnp.full((3,), jnp.nan, jnp.float32)
        state = (jnp.int32(0), initial_vector, initial_vector, history,
                 jnp.float32(jnp.nan), jnp.float32(jnp.nan),
                 jnp.bool_(True), jnp.bool_(False))
        def cond(state):
            iteration, _, _, _, _, _, valid, converged = state
            return (iteration < maxiter) & valid & ~converged
        def body(state):
            iteration, vector, _, history, _, _, valid, _ = state
            product = operator(vector, *operator_args)
            vv, product_norm = tree_dot(vector, vector), tree_norm(product)
            theta = tree_dot(vector, product) / jnp.where(vv > 0, vv, 1.)
            residual_vector = tree_add(product, vector, alpha=-theta)
            denominator = jnp.maximum(jnp.abs(theta) * jnp.sqrt(vv),
                jnp.finfo(jnp.float32).eps * product_norm)
            residual = tree_norm(residual_vector) / jnp.where(
                denominator > 0, denominator, 1.)
            valid &= (jnp.isfinite(vv) & (vv > 0) & jnp.isfinite(product_norm)
                      & (product_norm > 0) & jnp.isfinite(theta)
                      & jnp.isfinite(residual) & (denominator > 0))
            history = jnp.roll(history, -1).at[-1].set(theta)
            scale = jnp.maximum(jnp.max(jnp.abs(history)), jnp.float32(1e-30))
            agreement = ((iteration + 1) >= 3) & jnp.all(jnp.isfinite(history)) \
                & ((jnp.max(history) - jnp.min(history)) / scale <= agreement_tol)
            converged = valid & agreement & (residual <= residual_tol)
            next_vector = jax.tree.map(
                lambda leaf: leaf / jnp.where(product_norm > 0, product_norm, 1.),
                product)
            return (iteration + 1, next_vector, vector, history, theta,
                    residual, valid, converged)
        return jax.lax.while_loop(cond, body, state)
    return jax.jit(solve)


def make_inverse_solver(operator, *, maxiter=24, inner_maxiter=100,
                        inner_tol=1e-3, agreement_tol=.05, residual_tol=.05,
                        preconditioner=None):
    """Compile inverse iteration with counted PCG and true solve residuals.

    This estimates the minimum of the original SPD operator. The diagonal
    preconditioner accelerates each solve without changing that eigenproblem.
    """
    def solve(initial_vector, *args):
        def precondition(value):
            return value if preconditioner is None else preconditioner(value, *args)

        def body(state):
            iteration, vector, _, history, _, _, _, _, products = state
            zero = jax.tree.map(jnp.zeros_like, vector)
            z = precondition(vector)
            rz = tree_dot(vector, z)
            target = inner_tol ** 2 * tree_dot(vector, vector)
            inner = (jnp.int32(0), zero, vector, z, rz, jnp.bool_(True))

            def inner_cond(state):
                i, _, r, _, _, valid = state
                return (i < inner_maxiter) & valid & (tree_dot(r, r) > target)

            def inner_body(state):
                i, x, r, p, rz, valid = state
                ap = operator(p, *args)
                pap = tree_dot(p, ap)
                valid &= jnp.isfinite(pap) & (pap > 0) & jnp.isfinite(rz) & (rz > 0)
                alpha = jnp.where(valid, rz / jnp.where(pap > 0, pap, 1.), 0.)
                x = tree_add(x, p, alpha=alpha)
                r = tree_add(r, ap, alpha=-alpha)
                z = precondition(r)
                next_rz = tree_dot(r, z)
                beta = next_rz / jnp.where(rz > 0, rz, 1.)
                p = tree_add(z, p, alpha=beta)
                valid &= jnp.isfinite(next_rz) & (next_rz >= 0)
                return i + 1, x, r, p, next_rz, valid

            inner_iterations, y, _, _, _, valid = jax.lax.while_loop(
                inner_cond, inner_body, inner)
            ay = operator(y, *args)
            solve_r = tree_add(ay, vector, alpha=-1.)
            # Check the true residual, not just the PCG recurrence residual.
            valid &= tree_dot(solve_r, solve_r) <= target
            length = tree_norm(y)
            q = jax.tree.map(lambda v: v / jnp.where(length > 0, length, 1.), y)
            theta = tree_dot(y, ay) / jnp.where(length > 0, length ** 2, 1.)
            denominator = jnp.abs(theta) * length
            residual = tree_norm(tree_add(ay, y, alpha=-theta)) / jnp.where(
                denominator > 0, denominator, 1.)
            valid &= (jnp.isfinite(length) & (length > 0) & jnp.isfinite(theta)
                      & (theta > 0) & jnp.isfinite(residual))
            history = jnp.roll(history, -1).at[-1].set(theta)
            scale = jnp.maximum(jnp.max(jnp.abs(history)), jnp.float32(1e-30))
            agreement = ((iteration + 1) >= 3) & jnp.all(jnp.isfinite(history)) \
                & ((jnp.max(history) - jnp.min(history)) / scale <= agreement_tol)
            converged = valid & agreement & (residual <= residual_tol)
            return (iteration + 1, q, q, history, theta, residual, valid,
                    converged, products + inner_iterations + 1)

        state = (jnp.int32(0), initial_vector, initial_vector,
                 jnp.full((3,), jnp.nan, jnp.float32), jnp.float32(jnp.nan),
                 jnp.float32(jnp.nan), jnp.bool_(True), jnp.bool_(False), jnp.int32(0))
        return jax.lax.while_loop(
            lambda s: (s[0] < maxiter) & s[6] & ~s[7], body, state)
    return jax.jit(solve)


def power_iteration(operator, template, *, key=jax.random.PRNGKey(0), maxiter=24,
                    agreement_tol=.05, residual_tol=.05, num_starts=2,
                    operator_args=(), compiled_solver=None):
    solver = compiled_solver or make_power_solver(operator, maxiter=maxiter,
        agreement_tol=agreement_tol, residual_tol=residual_tol)
    candidates = []
    for start in range(num_starts):
        vector = random_unit_pytree(jax.random.fold_in(key, start), template)
        if vector is None:
            candidates.append((None, None, None, 0, False, False, 0))
            continue
        result = solver(vector, *operator_args)
        iteration, _, evaluated, history, theta, residual, valid, converged = result[:8]
        products = result[8] if len(result) > 8 else iteration
        history = np.asarray(jax.device_get(history), dtype=np.float64)
        agreement = bool(np.all(np.isfinite(history)) and
            (history.max() - history.min()) /
            max(float(np.max(np.abs(history))), 1e-30) <= agreement_tol)
        candidates.append((float(theta), float(residual), evaluated,
                           int(iteration), bool(valid) and bool(converged), agreement,
                           int(products)))
    values = [item[0] for item in candidates]
    finite = all(value is not None and math.isfinite(value) for value in values)
    scale = max([abs(value) for value in values if value is not None] + [1e-30])
    starts_agree = finite and max(values) - min(values) <= agreement_tol * scale
    reasons = []
    if not all(item[4] for item in candidates): reasons.append('outer_not_converged')
    if not all(item[1] is not None and math.isfinite(item[1])
               and item[1] <= residual_tol for item in candidates):
        reasons.append('eigenpair_residual')
    if not starts_agree: reasons.append('starts_disagree')
    resolved = not reasons
    valid_candidates = [item for item in candidates
                        if item[0] is not None and math.isfinite(item[0])]
    chosen = min(valid_candidates, key=lambda item: item[1]
                 if item[1] is not None else math.inf) if valid_candidates else candidates[0]
    return Endpoint(chosen[0] if resolved else None, chosen[0], chosen[1], resolved,
        max(item[3] for item in candidates), sum(item[6] for item in candidates),
        chosen[5], starts_agree, tuple(reasons), chosen[2])


def condition_diagnostic(operator, template, **kwargs):
    """Maximum-only routine diagnostic."""
    started = time.monotonic()
    endpoint = power_iteration(operator, template,
        key=kwargs.get('key', jax.random.PRNGKey(0)),
        maxiter=kwargs.get('top_maxiter', 24),
        agreement_tol=kwargs.get('agreement_tol', .05),
        residual_tol=kwargs.get('residual_tol', .05),
        num_starts=kwargs.get('num_starts', 2),
        operator_args=kwargs.get('operator_args', ()),
        compiled_solver=kwargs.get('compiled_power_solver'))
    return dict(lambda_max_est=endpoint.value_est,
        lambda_max_residual=endpoint.residual, resolved=endpoint.resolved,
        failure_reasons=endpoint.failure_reasons, top=endpoint.metrics(),
        operator_matvecs=endpoint.operator_matvecs,
        seconds=time.monotonic() - started)


def damped_condition_diagnostic(operator, template, **kwargs):
    """Estimate both endpoints of SPD A; withhold an unresolved minimum/ratio."""
    started = time.monotonic()
    report = condition_diagnostic(operator, template, **kwargs)
    solver = kwargs.get('compiled_inverse_solver')
    if solver is None:
        solver = make_inverse_solver(operator,
            maxiter=kwargs.get('top_maxiter', 24),
            inner_maxiter=kwargs.get('inner_cg_maxiter', 100),
            inner_tol=kwargs.get('inner_cg_tol', 1e-3),
            agreement_tol=kwargs.get('agreement_tol', .05),
            residual_tol=kwargs.get('residual_tol', .05),
            preconditioner=kwargs.get('preconditioner'))
    bottom = power_iteration(operator, template,
        key=jax.random.fold_in(kwargs.get('key', jax.random.PRNGKey(0)), 0x4d494e),
        agreement_tol=kwargs.get('agreement_tol', .05),
        residual_tol=kwargs.get('residual_tol', .05),
        num_starts=kwargs.get('num_starts', 2),
        operator_args=kwargs.get('operator_args', ()), compiled_solver=solver)
    reasons = list(report['failure_reasons'])
    reasons.extend('minimum_' + reason for reason in bottom.failure_reasons)
    minimum, maximum = bottom.value_est, report['lambda_max_est']
    if minimum is not None and maximum is not None and minimum > maximum:
        reasons.append('endpoint_order')
        minimum = None
    report.update(lambda_min_est=minimum, lambda_min_residual=bottom.residual,
        condition_est=(maximum / minimum if maximum is not None
                       and minimum is not None and minimum > 0 else None),
        resolved=report['resolved'] and minimum is not None,
        failure_reasons=tuple(reasons), bottom=bottom.metrics(),
        operator_matvecs=report['operator_matvecs'] + bottom.operator_matvecs,
        seconds=time.monotonic() - started)
    return report


def preconditioned_condition_diagnostic(apply_b, apply_p, template, c, **kwargs):
    """Maximum-only P=cI+B diagnostic, with convergence checked on B's scale."""
    started = time.monotonic()
    if c > 0 and kwargs.get('effective_lambda') == 0:
        return dict(lambda_max_est=c, lambda_max_residual=0., resolved=True,
            damping_condition_proxy=1., failure_reasons=(), operator_matvecs=0,
            seconds=time.monotonic() - started, top={'special_case': 'P=cI'})
    endpoint = power_iteration(apply_b, template,
        key=kwargs.get('key', jax.random.PRNGKey(0)),
        maxiter=kwargs.get('top_maxiter', 24),
        agreement_tol=kwargs.get('agreement_tol', .05),
        residual_tol=kwargs.get('residual_tol', .05),
        num_starts=kwargs.get('num_starts', 2),
        operator_args=kwargs.get('operator_args', ()),
        compiled_solver=kwargs.get('compiled_power_solver'))
    p_max, p_residual, products = None, None, endpoint.operator_matvecs
    reasons = list(endpoint.failure_reasons)
    if endpoint.vector is not None:
        _, p_residual = eigenpair_residual(
            apply_p, endpoint.vector, kwargs.get('operator_args', ()))
        products += 1
        if endpoint.resolved and p_residual is not None \
                and p_residual <= kwargs.get('residual_tol', .05):
            p_max = c + endpoint.value_est
        elif endpoint.resolved:
            reasons.append('P_max_residual')
    resolved = p_max is not None
    proxy = p_max / c if resolved and c > 0 and math.isfinite(c) else None
    return dict(lambda_max_est=p_max, lambda_max_residual=p_residual,
        resolved=resolved, damping_condition_proxy=proxy,
        failure_reasons=tuple(dict.fromkeys(reasons)), top=endpoint.metrics(),
        operator_matvecs=products, seconds=time.monotonic() - started)


def structural_lower_bounds(effective_lambda, safe_adam_lr, diagonal):
    c = (1 - float(effective_lambda)) / float(safe_adam_lr)
    min_d = min(float(jnp.min(leaf)) for leaf in jax.tree.leaves(diagonal))
    return {'A_lambda_min_lower_bound': c * min_d,
            'P_lambda_min_lower_bound': c}


def symmetric_diagonal_operator(operator, diagonal):
    def apply(vector, *args):
        actual = diagonal(*args) if callable(diagonal) else diagonal
        scaled = jax.tree.map(lambda value, d: value / jnp.sqrt(d), vector, actual)
        product = operator(scaled, *args)
        return jax.tree.map(lambda value, d: value / jnp.sqrt(d), product, actual)
    return apply


def summarize_probe_samples(trace_samples, square_samples, lambda_max_est,
                            dimension=None):
    t, s = np.asarray(trace_samples, np.float64), np.asarray(square_samples, np.float64)
    if len(t) < 2 or len(t) != len(s):
        raise ValueError('Need at least two paired probe samples')
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(s))):
        return {'trace_probes': len(t), 'trace_estimates_valid': False}
    trace, trace_square = float(t.mean()), float(s.mean())
    top_ok = lambda_max_est is not None and math.isfinite(lambda_max_est) \
        and lambda_max_est > 0
    concentration = (dimension * lambda_max_est / trace
                     if top_ok and dimension is not None and dimension > 0
                     and trace > 0 else None)
    return dict(trace_probes=len(t), trace_estimates_valid=trace >= 0 and trace_square >= 0,
        trace_est=trace, trace_sample_se=float(t.std(ddof=1) / np.sqrt(len(t))),
        trace_square_est=trace_square,
        trace_square_sample_se=float(s.std(ddof=1) / np.sqrt(len(s))),
        spectral_concentration_est=concentration,
        participation_rank_est=(trace * trace / trace_square
                                if trace > 0 and trace_square > 0 else None),
        stable_rank_est=(trace_square / (lambda_max_est * lambda_max_est)
                         if top_ok and trace_square > 0 else None))


def probe_operators(apply_g, template, *, num_probes=4,
                    key=jax.random.PRNGKey(0), apply_a_from_g=None):
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
