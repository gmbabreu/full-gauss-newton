"""Pure pytree helpers for the controlled Muon--GN outer-decay experiment."""

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict


def muon_matrix_mask(params):
    """Select exactly rank-2 hidden leaves (the effective Muon matrix route)."""
    flat = flatten_dict(params, sep=".")
    selected = {
        name: (value.ndim == 2 and name not in (
            "params.transformer.wte.embedding", "params.lm_head.kernel"))
        for name, value in flat.items()
    }
    return unflatten_dict(selected, sep=".")


def construct_outer_candidate(base, direction, alpha, rho, mask):
    """Return theta + alpha*d - rho*M*theta; retain the old expression at rho=0."""
    if rho == 0.0:
        return jax.tree_util.tree_map(lambda p, d: p + alpha * d, base, direction)
    return jax.tree_util.tree_map(
        lambda p, d, selected: p + alpha * d - rho * p if selected else p + alpha * d,
        base, direction, mask,
    )


def commit_outer_update(outer_state, raw_inner_state, direction, alpha, rho, mask):
    """Production commit preserving raw iterate and optimizer state for warm starts."""
    params = construct_outer_candidate(outer_state.params, direction, alpha, rho, mask)
    return outer_state.replace(
        step=outer_state.step + 1, params=params,
        opt_state=raw_inner_state.opt_state,
        warmstart_params=raw_inner_state.params,
    )


def run_outer_linesearch(base, direction, batches, rngs, evaluate, candidate,
                         *, armijo, candidates, init_step, beta, patience):
    """Production line-search loop; ``candidate`` is also used by commit."""
    losses = []
    if armijo:
        step_size = init_step
        best_loss, best_step_size, bad = float("inf"), step_size, 0
        while step_size > 1e-6:
            trial = candidate(base, direction, step_size)
            average_loss = float(sum(evaluate(trial, batch, rng)
                                     for batch, rng in zip(batches, rngs)) / len(batches))
            losses.append((step_size, average_loss))
            if average_loss < best_loss:
                best_loss, best_step_size, bad = average_loss, step_size, 0
            else:
                bad += 1
            if bad >= patience:
                break
            step_size *= beta
        return best_step_size, losses

    for step_size in candidates:
        trial = candidate(base, direction, step_size)
        average_loss = sum(evaluate(trial, batch, rng)
                           for batch, rng in zip(batches, rngs)) / len(batches)
        losses.append((step_size, average_loss))
    return min(losses, key=lambda item: item[1])[0], losses


def restore_inner_solver_state(fresh_inner_state, loaded_outer_state,
                               full_checkpoint, reset_start):
    """Install checkpointed moments/counters only for a full stateful resume."""
    if full_checkpoint and not reset_start:
        return fresh_inner_state.replace(opt_state=loaded_outer_state.opt_state)
    return fresh_inner_state


def _norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                        for x in jax.tree_util.tree_leaves(tree)))


def outer_update_stats(before, direction, after, alpha, rho, mask, epsilon=1e-12):
    """Compute only scalar device reductions; callers transfer the returned scalars."""
    solver = jax.tree_util.tree_map(lambda d: alpha * d, direction)
    decay = jax.tree_util.tree_map(
        lambda p, selected: -rho * p if selected else jnp.zeros_like(p), before, mask)
    total = jax.tree_util.tree_map(lambda a, b: a - b, after, before)
    groups = {
        "muon_matrices": mask,
        "embedding_head": jax.tree_util.tree_map(
            lambda p, selected: p.ndim == 2 and not selected, before, mask),
        "auxiliary_vectors": jax.tree_util.tree_map(lambda p: p.ndim != 2, before),
    }
    out = {"outer/accepted_alpha": jnp.asarray(alpha), "outer/rho": jnp.asarray(rho)}
    for label, group_mask in {"": jax.tree_util.tree_map(lambda _: True, before), **groups}.items():
        select = lambda tree: jax.tree_util.tree_map(
            lambda x, chosen: x if chosen else jnp.zeros_like(x), tree, group_mask)
        w0, w1, ds, dd, dt = map(_norm, map(select, (before, after, solver, decay, total)))
        prefix = "outer" if not label else f"outer/{label}"
        denom = jnp.maximum(w0, epsilon)
        out.update({
            f"{prefix}/weight_norm_before": w0, f"{prefix}/weight_norm_after": w1,
            f"{prefix}/solver_update_norm": ds, f"{prefix}/decay_update_norm": dd,
            f"{prefix}/total_update_norm": dt, f"{prefix}/solver_relative_update": ds / denom,
            f"{prefix}/decay_relative_update": dd / denom,
            f"{prefix}/total_relative_update": dt / denom,
        })
    flat_before = flatten_dict(before, sep=".")
    flat_after = flatten_dict(after, sep=".")
    flat_solver = flatten_dict(solver, sep=".")
    flat_decay = flatten_dict(decay, sep=".")
    for name, value in flat_before.items():
        # W&B treats slashes as namespaces; dots retain the full stable leaf name.
        leaf_prefix = f"outer/leaf/{name}"
        w0 = jnp.linalg.norm(value.astype(jnp.float32))
        w1 = jnp.linalg.norm(flat_after[name].astype(jnp.float32))
        ds = jnp.linalg.norm(flat_solver[name].astype(jnp.float32))
        dd = jnp.linalg.norm(flat_decay[name].astype(jnp.float32))
        dt = jnp.linalg.norm((flat_after[name] - value).astype(jnp.float32))
        denom = jnp.maximum(w0, epsilon)
        out.update({
            f"{leaf_prefix}/weight_norm_before": w0,
            f"{leaf_prefix}/weight_norm_after": w1,
            f"{leaf_prefix}/solver_relative_update": ds / denom,
            f"{leaf_prefix}/decay_relative_update": dd / denom,
            f"{leaf_prefix}/total_relative_update": dt / denom,
        })
    return out


def selected_matrix_descriptions(params, mask):
    flat_params, flat_mask = flatten_dict(params, sep="."), flatten_dict(mask, sep=".")
    return [(name, tuple(flat_params[name].shape)) for name in flat_params if flat_mask[name]]
