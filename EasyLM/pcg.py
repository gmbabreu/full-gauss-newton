"""Small helpers for damped matrix-free preconditioned CG."""
import math
import numbers

import jax
import jax.numpy as jnp


PRECONDITIONERS = ('adam_diag', 'gn_jacobi')


def validate_damped_pcg(damping_mu, preconditioner, probes, *,
                        interpolation_lambda, lambda_batch_denominator,
                        lambda_final, lambda_ramp_steps, adam_b1,
                        condition_log):
    """Validate flags whose combinations define the damped PCG system."""
    if not math.isfinite(damping_mu) or damping_mu < 0:
        raise ValueError('cg_damping_mu must be finite and nonnegative')
    if preconditioner not in PRECONDITIONERS:
        raise ValueError(
            'cg_preconditioner must be one of: ' + ', '.join(PRECONDITIONERS))
    if (isinstance(probes, bool) or not isinstance(probes, numbers.Integral)
            or probes <= 0):
        raise ValueError('cg_gn_jacobi_probes must be a positive integer')
    if preconditioner == 'gn_jacobi' and damping_mu <= 0:
        raise ValueError('gn_jacobi requires cg_damping_mu > 0')
    if damping_mu > 0:
        if (interpolation_lambda != 1
                or lambda_batch_denominator != 0
                or lambda_final != -1
                or lambda_ramp_steps != 0):
            raise ValueError(
                'cg_damping_mu > 0 requires fixed pure-GN interpolation')
        if adam_b1 != 0:
            raise ValueError('cg_damping_mu > 0 requires Adam b1 == 0')
        if condition_log:
            raise ValueError(
                'condition_log is not supported with cg_damping_mu > 0')


def damped_operator(apply_g, damping_mu):
    """Return ``v -> G(v) + damping_mu * v`` for a full-batch ``apply_g``."""
    mu = jnp.asarray(damping_mu)

    def apply_a(vector):
        gv = apply_g(vector)
        return jax.tree.map(lambda g, v: g + mu.astype(g.dtype) * v, gv, vector)

    return apply_a


def estimate_gn_diagonal(apply_g, template, rng, probes):
    """Estimate diag(G) sequentially with deterministic Rademacher probes.

    Only the FP32 accumulator, one probe, and one G-product are live per loop
    iteration. Leaf keys are derived rather than split so no host RNG is used.
    """
    leaves, treedef = jax.tree.flatten(template)
    accumulator = jax.tree.map(
        lambda leaf: jnp.zeros(leaf.shape, dtype=jnp.float32), template)

    def body(index, diagonal_sum):
        probe_leaves = [
            jax.random.rademacher(
                jax.random.fold_in(jax.random.fold_in(rng, index), leaf_index),
                leaf.shape, dtype=leaf.dtype)
            for leaf_index, leaf in enumerate(leaves)
        ]
        probe = jax.tree.unflatten(treedef, probe_leaves)
        product = apply_g(probe)
        return jax.tree.map(
            lambda total, z, gz: total + z.astype(jnp.float32)
            * gz.astype(jnp.float32), diagonal_sum, probe, product)

    diagonal_sum = jax.lax.fori_loop(0, probes, body, accumulator)
    scale = jnp.asarray(probes, dtype=jnp.float32)
    return jax.tree.map(lambda value: value / scale, diagonal_sum)


def gn_jacobi_preconditioner(diagonal_estimate, damping_mu):
    """Return inverse application for max(diag estimate, 0) + damping."""
    denominator = jax.tree.map(
        lambda diagonal: jnp.maximum(diagonal, 0) + damping_mu,
        diagonal_estimate)

    def apply_inverse(tree):
        return jax.tree.map(
            lambda value, divisor: value / divisor.astype(value.dtype),
            tree, denominator)

    return apply_inverse
