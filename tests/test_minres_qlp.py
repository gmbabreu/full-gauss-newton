"""CPU acceptance controls for the constant-storage QLP recurrence."""
import json
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
try:
    from jax import enable_x64
except ImportError:  # Older project JAX releases.
    from jax.experimental import enable_x64

from EasyLM.minres_qlp import minres_qlp

GN_RESULTS = []


@pytest.fixture(autouse=True)
def precision_context():
    with enable_x64():
        yield


def measurements(A, b, x, info):
    ref = np.linalg.pinv(A.astype(float), rcond=1e-6 if A.dtype == np.float32 else 1e-13) @ b.astype(float)
    r = b.astype(float) - A.astype(float) @ np.asarray(x).astype(float)
    return dict(residual=float(np.linalg.norm(r)),
                measured_relative_residual=float(np.linalg.norm(r) / max(np.linalg.norm(b.astype(float)), 1e-300)),
                solution_error=float(np.linalg.norm(np.asarray(x) - ref) / max(np.linalg.norm(ref), 1e-300)),
                **{k: np.asarray(v).item() for k, v in info.items()})


def cases(dtype):
    rng = np.random.default_rng(51)
    for rotated in (False, True):
        Q = np.linalg.qr(rng.normal(size=(6, 6)))[0] if rotated else np.eye(6)
        for singular in (False, True):
            eigs = np.array([1, 2, 3, 4, 0, 0] if singular else [1, 2, 3, 4, 5, 6])
            for compatible in (False, True) if singular else (True,):
                for scale in (1e-8, 1., 1e8):
                    A = np.asarray((Q * eigs) @ Q.T * scale, dtype)
                    b = np.asarray(Q @ (eigs if compatible else np.ones(6)) * scale, dtype)
                    yield f'rotated={rotated},singular={singular},compatible={compatible},scale={scale}', A, b
    for scale in (1e-8, 1., 1e8):
        yield f'rhs_scale={scale}', np.diag(np.array([1, 2, 3], dtype)), np.ones(3, dtype) * scale


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dense_acceptance(dtype):
    tol = 1e-6 if dtype == np.float32 else 1e-12
    for name, A, b in cases(dtype):
        # Two leaves exercise pytree raveling under JIT, including an empty leaf.
        operator = lambda p: {'a': jnp.asarray(A) @ p['a'], 'empty': p['empty']}
        solve = jax.jit(lambda rhs: minres_qlp(operator, rhs, tol=tol, maxiter=100))
        x, info = solve({'a': jnp.asarray(b), 'empty': jnp.zeros(0, dtype)})
        row = measurements(A, b, x['a'], info)
        assert row['solution_error'] < (2e-4 if dtype == np.float32 else 1e-9), (name, row)
        assert int(info['status']) in (0, 1), (name, row)
        np.testing.assert_allclose(float(info['true_residual_norm']), row['residual'], rtol=1e-5, atol=tol * np.linalg.norm(b.astype(float)))
        assert int(info['matvec_count']) == 3 + 3 * int(info['iterations'])


def test_exits_and_rank_policy():
    x, info = minres_qlp(lambda v: v, jnp.zeros(3))
    assert int(info['matvec_count']) == 0 and bool(info['converged'])
    x, info = minres_qlp(lambda v: v, jnp.ones(3), maxiter=0)
    assert int(info['status']) == 2 and int(info['iterations']) == 0
    x, info = minres_qlp(lambda v: v, jnp.ones(3))
    np.testing.assert_allclose(x, 1.)
    assert bool(info['converged'])
    x, info = minres_qlp(lambda v: v * 0, jnp.ones(3))
    assert int(info['status']) == 1 and not bool(info['converged'])
    np.testing.assert_array_equal(x, 0)
    for operator, rhs in [(lambda v: v * jnp.nan, jnp.ones(3)),
                           (lambda v: v, jnp.array([jnp.inf]))]:
        _, info = minres_qlp(operator, rhs)
        assert int(info['status']) == 4
    for kwargs in [dict(tol=-1), dict(rank_rtol=0), dict(maxiter=-1)]:
        with pytest.raises(ValueError):
            minres_qlp(lambda v: v, jnp.ones(3), **kwargs)


def test_runtime_matvec_accounting_and_truncation():
    calls = []
    matrix = jnp.diag(jnp.array([1., 2., 0.], jnp.float32))
    def operator(v):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return matrix @ v
    x, info = jax.jit(lambda b: minres_qlp(operator, b, rank_rtol=1e-5))(jnp.ones(3, jnp.float32))
    jax.block_until_ready((x, info))
    jax.effects_barrier()
    assert len(calls) == int(info['matvec_count'])
    assert int(info['rank_events']) > 0 and int(info['status']) == 1
    assert not bool(info['converged']) and bool(info['least_squares_stationary'])
    np.testing.assert_allclose(x, [1., .5, 0.], atol=3e-6)
    calls.clear()
    _, info = jax.jit(lambda b: minres_qlp(operator, b))(jnp.zeros(3, jnp.float32))
    jax.block_until_ready(info)
    jax.effects_barrier()
    assert len(calls) == int(info['matvec_count']) == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_masked_softmax_unequal_microbatches(dtype):
    # Execute the actual trainer apply_G closure, without service/TPU imports.
    root = Path(__file__).resolve().parents[1]
    trainer = ast.parse((root / 'EasyLM/models/llama/llama_train_gn.py').read_text())
    ns = dict(jax=jax, jnp=jnp, linearize=jax.linearize,
              linear_transpose=jax.linear_transpose, wd=0.)
    def load(name):
        node = next(n for n in ast.walk(trainer) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<trainer>', 'exec'), ns)
    for name in ('microbatch_groups', 'weighted_microbatch_sum'):
        load(name)
    X = jnp.asarray(np.random.default_rng(3).normal(size=(5, 2, 2)), dtype)
    theta = {'w': jnp.asarray(np.arange(6).reshape(2, 3) / 10, dtype)}
    masks = jnp.asarray([[1, 0], [0, 0], [1, 1], [0, 1], [1, 0]], dtype)
    targets = jnp.array([[0, 1], [1, 2], [2, 0], [0, 1], [1, 2]])
    from types import SimpleNamespace as NS
    logits = lambda p, inputs: jnp.tanh(inputs @ p['w'])
    ns.update(model=NS(apply=lambda p, inputs, deterministic: NS(logits=logits(p, inputs))), params0=theta)
    # Match trainer per-sequence normalization, with unequal valid token counts.
    def loss(z, t, p, base, mask, weight_decay):
        value = -jnp.take_along_axis(jax.nn.log_softmax(z), t[..., None], -1)[..., 0]
        return jnp.mean(jnp.sum(jnp.where(mask > 0, value, 0), -1) / jnp.maximum(jnp.sum(mask, -1), 1e-10)), 0.
    ns['cross_entropy_loss_and_accuracy_with_weight_decay'] = loss
    if dtype == np.float32:
        # Production loss intentionally casts logits to FP32. Use it unchanged
        # for that control; the FP64 control retains FP64 softmax arithmetic.
        utils = ast.parse((root / 'EasyLM/jax_utils.py').read_text())
        node = next(n for n in utils.body if getattr(n, 'name', '') == 'cross_entropy_loss_and_accuracy_with_weight_decay')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<loss>', 'exec'), ns)
    batch = dict(input_tokens=X, target_tokens=targets, loss_masks=masks)
    ns['batch_'] = batch
    J = np.asarray(jax.jacfwd(lambda w: logits({'w': w}, X))(theta['w'])).reshape(30, 6).astype(float)
    p = np.asarray(jax.nn.softmax(logits(theta, X))).reshape(10, 3).astype(float)
    weights = np.asarray(masks / jnp.maximum(masks.sum(-1, keepdims=True), 1e-10) / 5).ravel()
    W = np.zeros((30, 30))
    for i in range(10):
        W[i*3:i*3+3, i*3:i*3+3] = weights[i] * (np.diag(p[i]) - np.outer(p[i], p[i]))
    H = J.T @ W @ J
    g = np.asarray(jax.grad(lambda w: loss(logits({'w': w}, X), targets, None, None, masks, 0)[0])(theta['w'])).ravel()
    for micro in (1, 2, 3, 5):
        ns['groups'] = ns['microbatch_groups'](5, micro, 1)[1]
        load('apply_G')
        operator = ns['apply_G']
        actual = np.column_stack([np.asarray(operator({'w': jnp.asarray(v.reshape(2, 3), dtype)})['w']).ravel() for v in np.eye(6)])
        np.testing.assert_allclose(actual, H, atol=2e-8 if dtype == np.float32 else 1e-15)
        x, info = jax.jit(lambda b: minres_qlp(operator, b, tol=1e-6 if dtype == np.float32 else 1e-12))({'w': jnp.asarray(-g.reshape(2, 3), dtype)})
        ref = np.linalg.pinv(H, rcond=1e-6 if dtype == np.float32 else 1e-13) @ -g
        err = np.linalg.norm(np.asarray(x['w']).ravel() - ref) / np.linalg.norm(ref)
        GN_RESULTS.append(dict(dtype=str(np.dtype(dtype)), microbatches=micro,
                               operator_max_error=float(np.max(np.abs(actual-H))),
                               residual=float(np.linalg.norm(H @ np.asarray(x['w']).ravel() + g)),
                               solution_error=float(err),
                               **{k: np.asarray(v).item() for k, v in info.items()}))
        assert err < (2e-3 if dtype == np.float32 else 1e-8), (micro, err, info)


def diagnostic_report():
    """Reproducible measurements; severe conditioning remains diagnostic only."""
    rows = []
    for dtype in (np.float32, np.float64):
        tol = 1e-6 if dtype == np.float32 else 1e-12
        controls = list(cases(dtype))
        for cond in (1e4, 1e10):
            for rotated in (False, True):
                Q = np.linalg.qr(np.random.default_rng(42).normal(size=(20, 20)))[0] if rotated else np.eye(20)
                A = np.asarray((Q * np.geomspace(1, 1 / cond, 20)) @ Q.T, dtype)
                controls.append((f'condition={cond},rotated={rotated}', A, A @ np.ones(20, dtype)))
        for name, A, b in controls:
            x, info = jax.jit(lambda rhs: minres_qlp(lambda v: jnp.asarray(A) @ v, rhs, tol=tol, maxiter=500))(jnp.asarray(b))
            rows.append(dict(case=name, dtype=str(np.dtype(dtype)), **measurements(A, b, x, info)))
    Path('docs/minres_qlp_cpu_results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    with enable_x64():
        diagnostic_report()
        for dtype in (np.float32, np.float64):
            test_masked_softmax_unequal_microbatches(dtype)
        Path('docs/minres_qlp_gn_cpu_results.json').write_text(json.dumps(GN_RESULTS, indent=2) + '\n')
