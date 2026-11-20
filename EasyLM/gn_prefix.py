"""Observational prefixes on the committed K=10 trajectory.

Callbacks are the trainer's normal inner step, quadratic, loss and line search.
No optimizer snapshots are accumulated; only scalar measurements survive a prefix.
The inner-step callback MUST NOT donate buffers shared with committed state.
"""
import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils


def validate(flags):
    if not flags.gn_prefix_probe_steps.strip():
        return frozenset(), ()
    def integers(value):
        try:
            return tuple(int(x.strip()) for x in value.split(','))
        except ValueError as exc:
            raise ValueError('GN prefix steps and Ks must be comma-separated integers') from exc
    steps, ks = integers(flags.gn_prefix_probe_steps), integers(flags.gn_prefix_ks)
    if any(s < 0 for s in steps) or len(set(steps)) != len(steps):
        raise ValueError('GN prefix probe steps must be distinct zero-based steps')
    if not ks or ks[0] != 10 or any(b <= a for a, b in zip(ks, ks[1:])):
        raise ValueError('GN prefix Ks must be strictly increasing and begin with 10')
    required = dict(gauss_newton=True, inner_loop_iter=10, single_batch_inner=True,
                    adaptive_inner_loop=False, reset_start=False, linesearch=True,
                    ls_eval_batches=1, weight_average=False, normalize_step=False,
                    outer_momentum_beta=0, outer_weight_decay=0, fixed_step_size=0,
                    ls_reuse_lag=0, inner_loop_wd=0, lr_sched='global_cosine')
    for name, value in required.items():
        if getattr(flags, name) != value:
            raise ValueError(f'GN prefix requires {name}={value!r}')
    if flags.optimizer_type not in ('adamw', 'muon'):
        raise ValueError('GN prefix requires AdamW or Muon')
    if flags.gn_prefix_holdout_batches < 1:
        raise ValueError('GN prefix holdout batch count must be positive')
    return frozenset(steps), ks


def optimizer_usage(state, schedule):
    """Read actual Optax schedule counters, before update (including Muon masks)."""
    leaves = jax.tree.leaves(
        state.opt_state,
        is_leaf=lambda x: type(x).__name__ == 'ScaleByScheduleState')
    counts = [int(jax.device_get(x.count)) for x in leaves
              if type(x).__name__ == 'ScaleByScheduleState']
    if not counts or len(set(counts)) != 1:
        raise ValueError(f'GN prefix needs synchronized Optax schedule counters: {counts}')
    count = counts[0]
    return count, float(jax.device_get(schedule(count)))


def fingerprint(batch):
    digest = hashlib.sha256()
    distributed = False
    for name, value in sorted(batch.items()):
        digest.update(name.encode())
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            distributed = True
            # Never materialize a non-addressable global array on one host.
            for shard in value.addressable_shards:
                digest.update(str(shard.index).encode())
                local = np.asarray(jax.device_get(shard.data))
                digest.update(str((local.shape, local.dtype.str)).encode())
                digest.update(local.tobytes())
        else:
            value = np.asarray(jax.device_get(value))
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
    if distributed:
        parts = multihost_utils.process_allgather(
            np.frombuffer(digest.digest(), dtype=np.uint8), tiled=False)
        return hashlib.sha256(np.asarray(parts).tobytes()).hexdigest()
    return digest.hexdigest()


def norm(tree):
    return jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2)
                        for x in jax.tree.leaves(tree)))


def matched_direction(direction, radius):
    length = norm(direction)
    scale = jnp.where(length > 0, radius / jnp.where(length > 0, length, 1.), 0.)
    return jax.tree.map(lambda d: scale * d, direction)


def run_probe(*, step, ks, base, state, rng, solve_batch, solve_rng,
              ls_batches, ls_rngs, baseline_loss, selected_step, losses,
              holdout_iterator, holdout_batches, inner_step, quadratic,
              loss_fn, search, schedule, used_updates):
    """Run after normal commit and timer stop; never return training state or RNG."""
    jax.block_until_ready((base, state, rng))
    started = time.perf_counter()
    heldout = [next(holdout_iterator)[0] for _ in range(holdout_batches)]
    # A local derived stream; the normal stream is neither split nor replaced.
    keys = list(jax.random.split(jax.random.fold_in(rng, 0x4750), holdout_batches))
    def evaluate(params):
        return sum(float(jax.device_get(loss_fn(params, b, r)[0]))
                   for b, r in zip(heldout, keys)) / len(heldout)
    baseline = evaluate(base)
    result = {'gn_prefix/probe_step': step,
              'gn_prefix/solve_fingerprint': fingerprint(solve_batch),
              'gn_prefix/linesearch_fingerprint': fingerprint(ls_batches[0]),
              'gn_prefix/linesearch_rng_fingerprint': fingerprint({'rng': ls_rngs[0]}),
              'gn_prefix/heldout_baseline_loss': baseline}
    for i, (batch, key) in enumerate(zip(heldout, keys)):
        result[f'gn_prefix/heldout_{i}_fingerprint'] = fingerprint(batch)
        result[f'gn_prefix/heldout_{i}_rng_fingerprint'] = fingerprint({'rng': key})
    radius = None
    k10_improvement = None
    previous = 10
    for k in ks:
        for update in range(previous, k):
            used_updates.append(optimizer_usage(state, schedule))
            state, rng, _ = inner_step(state, base, rng, solve_batch, 0., jnp.bool_(False))
        direction = jax.tree.map(lambda p, p0: p - p0, state.params, base)
        alpha, candidates = ((selected_step, losses) if k == 10 else
                             search(base, direction, ls_batches, ls_rngs))
        alpha = float(jax.device_get(alpha))
        raw_norm = float(jax.device_get(norm(direction)))
        update_norm = abs(alpha) * raw_norm
        if radius is None:
            radius = update_norm
        accepted = jax.tree.map(lambda p, d: p + alpha * d, base, direction)
        improvement = baseline - evaluate(accepted)
        del accepted
        matched = matched_direction(direction, radius)
        matched_params = jax.tree.map(lambda p, d: p + d, base, matched)
        matched_improvement = baseline - evaluate(matched_params)
        del matched, matched_params
        q = jax.device_get(quadratic(base, state.params, solve_rng, solve_batch))
        a, b = float(q['gn_quad/g_dot_d']), float(q['gn_quad/d_G_d'])
        chosen_loss = next(float(jax.device_get(loss)) for scale, loss in candidates
                           if float(jax.device_get(scale)) == alpha)
        if k10_improvement is None:
            k10_improvement = improvement
        values = dict(heldout_improvement=improvement,
                      marginal_heldout_improvement=improvement - k10_improvement,
                      matched_norm_heldout_improvement=matched_improvement,
                      predicted_raw_improvement=-a - .5 * b,
                      predicted_accepted_improvement=-alpha * a - .5 * alpha**2 * b,
                      residual_norm=float(q['residual_norm']),
                      relative_residual=float(q['relative_residual']),
                      raw_direction_norm=raw_norm, accepted_update_norm=update_norm,
                      selected_alpha=alpha, candidate_count=len(candidates),
                      linesearch_improvement=baseline_loss - chosen_loss)
        result.update({f'gn_prefix/k{k}/{name}': value for name, value in values.items()})
        for i, (scale, loss) in enumerate(candidates):
            result[f'gn_prefix/k{k}/candidate{i}/alpha'] = float(jax.device_get(scale))
            result[f'gn_prefix/k{k}/candidate{i}/loss'] = float(jax.device_get(loss))
        previous = k
        del direction, candidates
    for i, (counter, lr) in enumerate(used_updates, 1):
        result[f'gn_prefix/update{i}/optimizer_counter'] = counter
        result[f'gn_prefix/update{i}/learning_rate_used'] = lr
    jax.block_until_ready((state, rng, result))
    result['gn_prefix/runtime_s'] = time.perf_counter() - started
    return result
