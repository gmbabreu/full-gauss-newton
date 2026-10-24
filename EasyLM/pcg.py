"""Small diagonal-estimation and inverse-preconditioner helpers for GN CG."""
import jax
import jax.numpy as jnp


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
