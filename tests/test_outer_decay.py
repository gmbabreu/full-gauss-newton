"""CPU integration checks through production outer search/commit/restore helpers."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState

from EasyLM.outer_decay import (
    commit_outer_update, construct_outer_candidate, muon_matrix_mask, outer_update_stats,
    restore_inner_solver_state, run_outer_linesearch,
)


@struct.dataclass
class State(TrainState):
    warmstart_params: object = None


def params():
    return {"params": {"transformer": {
        "wte": {"embedding": jnp.arange(6.).reshape(3, 2)},
        "h": {"0": {"attn": {"kernel": jnp.arange(4.).reshape(2, 2)},
                      "norm": {"scale": jnp.array([2., 3.])}}},
    }, "lm_head": {"kernel": jnp.ones((2, 3))}}}


def tree_assert(actual, expected, exact=False):
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        if exact:
            np.testing.assert_array_equal(a, e)
        else:
            np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-7)


def no_op_apply(*_args, **_kwargs):
    return None


def state(p, schedule=lambda count: 0.05 / (count + 1)):
    # The scheduled Optax transform provides real moments and schedule counters.
    return State.create(params=p, tx=optax.adam(schedule), apply_fn=no_op_apply)


def candidate(mask, rho):
    return lambda outer, direction, alpha: construct_outer_candidate(
        outer, direction, alpha, rho, mask)


def actual_outer_step(outer, inner, rho, rng, batch, log):
    """Exercise actual Optax update, production search, commit, and diagnostics."""
    rng, used_rng = jax.random.split(rng)
    gradient = jax.tree.map(lambda x: batch * jnp.ones_like(x), inner.params)
    inner = inner.apply_gradients(grads=gradient)
    direction = jax.tree.map(lambda i, o: i-o, inner.params, outer.params)
    mask = muon_matrix_mask(outer.params)
    build = candidate(mask, rho)
    evaluate = lambda trial, _batch, _rng: sum(jnp.sum(x*x) for x in jax.tree.leaves(trial))
    alpha, losses = run_outer_linesearch(
        outer.params, direction, [batch], [used_rng], evaluate, build,
        armijo=False, candidates=[1., .5], init_step=1., beta=.5, patience=1)
    committed = commit_outer_update(outer, inner, direction, alpha, rho, mask)
    stats = outer_update_stats(outer.params, direction, committed.params,
                               alpha, rho, mask) if log else None
    return committed, inner, rng, alpha, losses, stats


def test_rho_zero_two_outer_steps_logging_is_observational():
    initial = params(); old_outer = state(initial); new_outer = state(initial)
    old_inner = state(initial); new_inner = state(initial)
    old_rng = new_rng = jax.random.PRNGKey(7)
    for batch in (jnp.float32(.2), jnp.float32(-.1)):
        old_outer, old_inner, old_rng, old_alpha, _, _ = actual_outer_step(
            old_outer, old_inner, 0., old_rng, batch, False)
        new_outer, new_inner, new_rng, new_alpha, _, stats = actual_outer_step(
            new_outer, new_inner, 0., new_rng, batch, True)
        assert stats is not None and old_alpha == new_alpha
    tree_assert(old_outer, new_outer, exact=True)
    tree_assert(old_inner, new_inner, exact=True)
    tree_assert(old_rng, new_rng, exact=True)
    print("PASS rho=0: params, moments/counters, carry, alpha, RNG/data over two updates")


def test_zero_direction_decay_is_once_and_independent_of_inner_count():
    p = params(); outer = state(p); mask = muon_matrix_mask(p)
    raw_one = state(p).replace(step=jnp.array(1))
    raw_many = state(p).replace(step=jnp.array(17))
    zero = jax.tree.map(jnp.zeros_like, p)
    one = commit_outer_update(outer, raw_one, zero, 1., .1, mask)
    many = commit_outer_update(outer, raw_many, zero, 1., .1, mask)
    tree_assert(one.params, many.params, exact=True)
    for before, after, selected in zip(jax.tree.leaves(p), jax.tree.leaves(one.params),
                                       jax.tree.leaves(mask)):
        np.testing.assert_allclose(after, before * (.9 if selected else 1.))
    print("PASS zero direction: selected matrices shrink exactly once independent of inner count")


def test_nonzero_direction_matches_differentiated_decay_and_stats():
    p = params(); outer = state(p); raw = state(jax.tree.map(lambda x: x+.3, p))
    mask = muon_matrix_mask(p); direction = jax.tree.map(lambda i, o: i-o, raw.params, p)
    got = commit_outer_update(outer, raw, direction, .5, .02, mask)
    penalty = lambda q: .01 * sum(jnp.sum(x*x) for x, m in
        zip(jax.tree.leaves(q), jax.tree.leaves(mask)) if m)
    decay_gradient = jax.grad(penalty)(p)
    reference = jax.tree.map(lambda x, d, g: x+.5*d-g, p, direction, decay_gradient)
    tree_assert(got.params, reference)
    stats = outer_update_stats(p, direction, got.params, .5, .02, mask)
    np.testing.assert_allclose(stats["outer/muon_matrices/decay_relative_update"], .02)
    print("PASS nonzero direction: differentiated sign/mask and logged decomposition")


def test_production_grid_backtracking_and_fixed_commit_loss():
    p = params(); outer = state(p); raw = state(jax.tree.map(lambda x: .8*x, p))
    mask = muon_matrix_mask(p); direction = jax.tree.map(lambda i, o: i-o, raw.params, p)
    build = candidate(mask, .01)
    loss = lambda q, _b, _r: sum(jnp.sum(x*x) for x in jax.tree.leaves(q))
    for armijo in (False, True):
        alpha, losses = run_outer_linesearch(
            p, direction, [0], [jax.random.PRNGKey(0)], loss, build,
            armijo=armijo, candidates=[1., .5], init_step=1., beta=.5, patience=1)
        committed = commit_outer_update(outer, raw, direction, alpha, .01, mask)
        np.testing.assert_allclose(loss(committed.params, 0, None), dict(losses)[alpha])
    fixed = .25
    committed = commit_outer_update(outer, raw, direction, fixed, .01, mask)
    np.testing.assert_allclose(loss(committed.params, 0, None), loss(build(p, direction, fixed), 0, None))
    print("PASS production grid/backtracking/fixed search loss equals committed loss")


def test_full_state_save_reload_continuation_nonconstant_schedule(tmp_path):
    p = params(); outer = state(p); inner = state(p); rng = jax.random.PRNGKey(11)
    uninterrupted, uninterrupted_inner, uninterrupted_rng, *_ = actual_outer_step(
        outer, inner, 0., rng, jnp.float32(.2), False)
    checkpoint = tmp_path / "full.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(uninterrupted))
    template = state(uninterrupted.params).replace(
        warmstart_params=uninterrupted.warmstart_params)
    loaded = serialization.from_bytes(template, checkpoint.read_bytes())
    fresh = state(loaded.params).replace(params=loaded.warmstart_params)
    restored = restore_inner_solver_state(fresh, loaded, True, False)
    resumed, resumed_inner, resumed_rng, *_ = actual_outer_step(
        loaded, restored, 0., uninterrupted_rng, jnp.float32(-.1), False)
    reference, reference_inner, reference_rng, *_ = actual_outer_step(
        uninterrupted, uninterrupted_inner, 0., uninterrupted_rng, jnp.float32(-.1), False)
    tree_assert(resumed, reference, exact=True)
    tree_assert(resumed_inner, reference_inner, exact=True)
    tree_assert(resumed_rng, reference_rng, exact=True)
    print("PASS full save/reload: params, moments, schedule counters, warmstart, RNG")
