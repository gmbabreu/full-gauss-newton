"""CPU checks of state isolation, actual trainer routing, and Adam arithmetic."""
import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from EasyLM.adam_reset import (
    ADAM_RESET_COMPONENTS, adam_inner_metrics, parse_adam_reset_components,
    reset_adam_inner_state,
)
from EasyLM import cg_resume


TRAINER = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'
UTILS = TRAINER.parents[2] / 'jax_utils.py'
MODES = ['none', *sorted(ADAM_RESET_COMPONENTS), 'all']


class State(TrainState):
    warmstart_params: object = None


def parse(value, **kwargs):
    flags = dict(optimizer_type='adamw', gauss_newton=True,
                 adaptive_inner_loop=False, reset_start=False)
    return parse_adam_reset_components(value, **(flags | kwargs))


def source_function(path, name, namespace):
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def boundary(inner, outer, mode, legacy=False):
    # Execute the trainer's actual boundary block without importing TPU/data
    # dependencies. This also catches accidental changes to the full-reset arm.
    tree = ast.parse(TRAINER.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and 'adam_reset_components == ADAM_RESET_COMPONENTS' in ast.unparse(n.test))
    ns = dict(inner_state=inner, train_state=outer,
              FLAGS=SimpleNamespace(reset_start=legacy, optimizer_type='adamw'),
              adam_reset_components=parse(mode), ADAM_RESET_COMPONENTS=ADAM_RESET_COMPONENTS,
              CustomTrainState=State, reset_adam_inner_state=reset_adam_inner_state)
    ns['create_reset_train_state'] = source_function(UTILS, 'create_reset_train_state', {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAINER), 'exec'), ns)
    return ns['inner_state'], ns['train_state']


def make_state(clip):
    schedule = lambda count: .02 / (1. + .1 * count)
    build = source_function(TRAINER, 'build_optimizer', dict(optax=optax, jnp=jnp))
    tx = build(schedule, .9, .95, grad_clip=clip, wd=.01, optimizer_type='adamw')
    state = State.create(apply_fn=None, params={'w': jnp.array([.8, -.3])}, tx=tx)
    for g in ([.2, -.6], [.5, .3], [-.1, .4]):
        state = state.apply_gradients(grads={'w': jnp.array(g)})
    return state.replace(warmstart_params={'w': jnp.array([7., 8.])}), schedule


def assert_same_values(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('clip', [0., .25])
@pytest.mark.parametrize('mode', MODES)
def test_only_selected_states_change_and_full_reset_matches_legacy(mode, clip):
    state, _ = make_state(clip)
    outer = state.replace(params={'w': jnp.array([.7, -.1])}, step=1)
    actual, outer_after = boundary(state, outer, mode)
    assert actual.step is state.step
    assert actual.warmstart_params is state.warmstart_params
    assert actual.tx is state.tx
    assert outer_after.params is outer.params
    assert outer_after.step is outer.step
    assert outer_after.opt_state is actual.opt_state
    if mode == 'none':
        assert actual is state
        assert outer_after is outer
    elif mode == 'all':
        expected, _ = boundary(state, outer, 'none', legacy=True)
        assert_same_values(actual, expected)
    else:
        # Compare all leaves by path; retained buffers must also keep identity.
        before = dict((jax.tree_util.keystr(p), v) for p, v in jax.tree.leaves_with_path(state))
        for path, value in jax.tree.leaves_with_path(actual):
            path = jax.tree_util.keystr(path)
            old = before[path]
            if mode == 'params' and path.startswith('.params'):
                np.testing.assert_array_equal(value, outer.params['w'])
            elif (mode == 'first_moment' and '.mu' in path
                  or mode == 'second_moment' and '.nu' in path):
                np.testing.assert_array_equal(value, np.zeros_like(old))
            elif mode in ('bias_count', 'schedule_count') and path.endswith('.count'):
                # These two scalar slots are checked separately below.
                pass
            else:
                assert value is old
        metrics = adam_inner_metrics(actual.opt_state, lambda n: n)
        assert int(metrics['adam/bias_count']) == (0 if mode == 'bias_count' else 3)
        assert int(metrics['adam/schedule_count']) == (0 if mode == 'schedule_count' else 3)


@pytest.mark.parametrize('clip', [0., .25])
@pytest.mark.parametrize('mode', MODES)
def test_repeated_outer_solves_match_independent_numpy_adam(mode, clip):
    """Explicit J^T J ground truth, with fractional outer steps and varying LR."""
    build = source_function(TRAINER, 'build_optimizer', dict(optax=optax, jnp=jnp))
    schedule = lambda n: .02 / (1. + .1 * n)
    tx = build(schedule, .9, .95, grad_clip=clip, wd=.01, optimizer_type='adamw')
    state = State.create(apply_fn=None, params={'w': jnp.array([.8, -.3])}, tx=tx)
    outer = state
    x = np.array([.8, -.3]); theta = x.copy()
    m = np.zeros(2); v = np.zeros(2); age = clock = 0
    components = parse(mode)
    design = np.array([[1., 2.], [-2., .5], [.3, -1.]])
    target = np.array([.2, -.4, .8])
    loss = lambda z: .5 * jnp.sum((jnp.asarray(design) @ z - target) ** 2)
    update = jax.jit(lambda s: s.apply_gradients(
        grads={'w': jax.grad(loss)(s.params['w'])}))
    np.testing.assert_allclose(jax.hessian(loss)(state.params['w']), design.T @ design,
                               rtol=1e-6, atol=1e-6)
    for outer_step in range(4):
        state, outer = boundary(state, outer, mode)
        if 'params' in components:
            x = theta.copy()
        if 'first_moment' in components:
            m = np.zeros(2)
        if 'second_moment' in components:
            v = np.zeros(2)
        if 'bias_count' in components:
            age = 0
        if 'schedule_count' in components:
            clock = 0
        for _ in range(3):
            metrics = adam_inner_metrics(state.opt_state, schedule)
            assert int(metrics['adam/bias_count']) == age
            assert int(metrics['adam/schedule_count']) == clock
            np.testing.assert_allclose(metrics['learning_rate'], schedule(clock), rtol=1e-6)
            # Autodiff gradient on the implementation side; explicit Jacobian
            # and manual Adam (including clipping/decay) on the reference side.
            state = update(state)
            g = design.T @ (design @ x - target)
            if clip:
                g = g * min(1., clip / np.linalg.norm(g))
            m = .9 * m + .1 * g
            v = .95 * v + .05 * g * g
            age += 1
            x = x - schedule(clock) * (
                (m / (1. - .9**age)) / (np.sqrt(v / (1. - .95**age)) + 1e-8) + .01 * x)
            clock += 1
            np.testing.assert_allclose(state.params['w'], x, rtol=2e-5, atol=2e-6)
        theta += .25 * (x - theta)
        outer = outer.replace(
            params=jax.tree.map(lambda a, b: a + .25 * (b - a), outer.params, state.params),
            opt_state=state.opt_state, step=outer_step + 1,
            warmstart_params=state.params)
        np.testing.assert_allclose(outer.params['w'], theta, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('mode', ['none', 'schedule_count', 'all'])
def test_actual_gn_step_and_applied_lr_against_explicit_softmax_jacobian(mode):
    state, schedule = make_state(.25)
    outer = state.replace(params={'w': jnp.array([.7, -.1])}, step=1)
    state, _ = boundary(state, outer, mode)
    design = np.array([[[1., 2.], [-2., .5], [.3, -1.]]], dtype=np.float32)
    target = np.array([[0., 1., 0.]], dtype=np.float32)

    def ce(logits, *args, **kwargs):
        return -jnp.sum(jnp.asarray(target) * jax.nn.log_softmax(logits)), 0.

    namespace = dict(
        jax=jax, jnp=jnp, linearize=jax.linearize, linear_transpose=jax.linear_transpose,
        JaxRNG=lambda rng: lambda *args: rng,
        with_sharding_constraint=lambda batch, _: batch,
        PS=lambda *args: None, FLAGS=SimpleNamespace(optimizer_type='adamw'),
        adam_inner_metrics=adam_inner_metrics, lr_sched=schedule,
        LLaMAConfigurator=SimpleNamespace(rng_keys=lambda: ()),
        model=SimpleNamespace(apply=lambda p, *args, **kwargs:
            SimpleNamespace(logits=jnp.asarray(design) @ p['w'])),
        cross_entropy_loss_and_accuracy_with_weight_decay=ce,
        global_norm=optax.global_norm, get_gpu_memory=lambda: [0],
    )
    step = source_function(TRAINER, 'train_step_gauss_newton', namespace)
    actual, _, metrics = jax.jit(step)(
        state, outer.params, jnp.int32(0),
        dict(input_tokens=jnp.zeros((1, 1)), target_tokens=jnp.ones((1, 1)),
             loss_masks=jnp.ones((1, 1))), 0., True)

    jacobian = design[0].astype(np.float64)
    logits = jacobian @ np.asarray(outer.params['w'])
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    hessian_logits = np.diag(probabilities) - np.outer(probabilities, probabilities)
    g0 = jacobian.T @ (probabilities - target[0])
    gn = jacobian.T @ hessian_logits @ jacobian
    gradient = g0 + gn @ (np.asarray(state.params['w']) - np.asarray(outer.params['w']))
    expected = state.apply_gradients(grads={'w': jnp.asarray(gradient, dtype=jnp.float32)})
    np.testing.assert_allclose(actual.params['w'], expected.params['w'], rtol=1e-6, atol=1e-7)
    expected_metrics = adam_inner_metrics(state.opt_state, schedule)
    for key in expected_metrics:
        np.testing.assert_allclose(metrics[key], expected_metrics[key], rtol=1e-6)
    np.testing.assert_allclose(metrics['gradient_norm'], np.linalg.norm(gradient), rtol=1e-6)


def test_configuration_and_old_cg_checkpoint_compatibility():
    assert parse(' params,first_moment ') == {'params', 'first_moment'}
    assert parse('none', optimizer_type='cg', gauss_newton=False, reset_start=True) == set()
    for value, overrides in [('residual', {}), ('params,', {}), ('none,params', {}),
                             ('all', {'reset_start': True}),
                             ('params', {'optimizer_type': 'muon'}),
                             ('params', {'gauss_newton': False}),
                             ('params', {'adaptive_inner_loop': True})]:
        with pytest.raises(ValueError):
            parse(value, **overrides)
    cg_resume.validate_flags({'optimizer_type': 'cg'},
                             {'optimizer_type': 'cg', 'adam_reset_components': 'none'})
    with pytest.raises(ValueError, match='adam_reset_components'):
        cg_resume.validate_flags({'optimizer_type': 'cg'},
                                 {'optimizer_type': 'cg', 'adam_reset_components': 'all'})
