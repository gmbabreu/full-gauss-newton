"""Selective state resets between ordinary Adam-GN inner solves.

Moment-only resets deliberately preserve Adam's shared bias-correction count.
This removes exactly one carried state; it does not restart fresh Adam.
"""

import jax
import jax.numpy as jnp
import optax


ADAM_RESET_COMPONENTS = frozenset({
    'params', 'first_moment', 'second_moment', 'bias_count', 'schedule_count',
})
_SLOT_TYPES = (optax.ScaleByAdamState, optax.ScaleByScheduleState)


def parse_adam_reset_components(value, *, optimizer_type, gauss_newton,
                                adaptive_inner_loop, reset_start):
    """Validate the opt-in experiment before loading data or model weights."""
    value = value.strip()
    if value in ('', 'none'):
        return frozenset()
    components = (ADAM_RESET_COMPONENTS if value == 'all' else
                  frozenset(part.strip() for part in value.split(',')))
    unknown = components - ADAM_RESET_COMPONENTS
    if unknown:
        raise ValueError(f'Unknown adam_reset_components: {sorted(unknown)}; '
                         f'choose from {sorted(ADAM_RESET_COMPONENTS)}, none, all')
    if optimizer_type != 'adamw' or not gauss_newton or adaptive_inner_loop:
        raise ValueError('adam_reset_components requires optimizer_type=adamw, '
                         'gauss_newton=True, adaptive_inner_loop=False')
    if reset_start:
        raise ValueError('Use reset_start=False with adam_reset_components; '
                         'use adam_reset_components=all for the full-reset control')
    return components


def _is_slot(value):
    return isinstance(value, _SLOT_TYPES)


def _adam_slots(opt_state):
    # Match public Optax state types rather than tuple indices: optional
    # clipping adds another chain wrapper but no optimizer history.
    leaves = jax.tree.leaves(opt_state, is_leaf=_is_slot)
    adam = [x for x in leaves if isinstance(x, optax.ScaleByAdamState)]
    schedule = [x for x in leaves if isinstance(x, optax.ScaleByScheduleState)]
    if len(adam) != 1 or len(schedule) != 1:
        raise ValueError('Expected one Adam state and one callable LR schedule')
    return adam[0], schedule[0]


def reset_adam_inner_state(state, outer_params, components):
    """Reset only selected components; preserve step and unrelated fields.

The caller also updates the outer state's checkpoint copy of opt_state, so
obsolete moment arrays are not kept alive throughout the next inner solve.
"""
    if not components:
        return state
    _adam_slots(state.opt_state)

    def reset_slot(slot):
        if isinstance(slot, optax.ScaleByAdamState):
            fields = {}
            if 'first_moment' in components:
                fields['mu'] = jax.tree.map(jnp.zeros_like, slot.mu)
            if 'second_moment' in components:
                fields['nu'] = jax.tree.map(jnp.zeros_like, slot.nu)
            if 'bias_count' in components:
                fields['count'] = jnp.zeros_like(slot.count)
            return slot._replace(**fields) if fields else slot
        if (isinstance(slot, optax.ScaleByScheduleState)
                and 'schedule_count' in components):
            return slot._replace(count=jnp.zeros_like(slot.count))
        return slot

    return state.replace(
        params=outer_params if 'params' in components else state.params,
        opt_state=jax.tree.map(reset_slot, state.opt_state, is_leaf=_is_slot),
    )


def adam_inner_metrics(opt_state, lr_schedule):
    """Report the LR and zero-based counters used by the upcoming update."""
    adam, schedule = _adam_slots(opt_state)
    return {
        'learning_rate': lr_schedule(schedule.count),
        'adam/bias_count': adam.count,
        'adam/schedule_count': schedule.count,
    }
