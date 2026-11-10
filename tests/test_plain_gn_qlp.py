"""Opt-in trainer route, legacy slots, resume policy and acceptance guards."""
import ast
import math
import numbers
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from EasyLM import cg_resume
from tests import test_gn_jacobi_pcg as pcg_tests
from tests.test_gn_jacobi_pcg import flags, load, TREE


def config(**changes):
    return flags(cg_solver='minres_qlp', cg_plain_gn=True,
                 cg_qlp_rank_rtol=0., cg_preconditioner='none',
                 cg_damping_mu=0., normalize_step=False, outer_momentum_beta=0.,
                 **changes)


def test_trainer_plain_direction_slots_metrics_and_legacy():
    harness = pcg_tests.TrainerSolveTest()
    harness.setUp()
    cfg = config()
    for micro in (1, 3, 5):
        cfg.n_micro = micro
        result, _ = harness.solve(cfg, compiled=True, moments=7., warm={'w': jnp.ones(2) * 11})
        candidate, m, v, warm, count, rng, metrics = result
        direction = np.linalg.solve(harness.G, -harness.g)
        np.testing.assert_allclose(candidate['w'] - harness.params['w'], direction, atol=3e-6)
        for slot, value in ((m, 7.), (v, 7.), (warm, 11.)):
            np.testing.assert_array_equal(slot['w'], value)
        assert int(count) == 3 and bool(metrics['cg_direction_valid'])
        assert bool(metrics['qlp/converged'])
        assert int(metrics['qlp/total_matvec_count']) == 4 + 3 * int(metrics['qlp/iterations'])
        np.testing.assert_allclose(metrics['gn_quad/g_dot_d'], harness.g @ direction, atol=1e-6)
        np.testing.assert_allclose(metrics['gn_quad/d_G_d'], direction @ harness.G @ direction, atol=1e-6)
    # Moments and LR cannot affect a plain-GN direction; NaN slots remain inert.
    other, _ = harness.solve(cfg, compiled=True, lr=9., moments=float('nan'))
    np.testing.assert_allclose(other[0]['w'], candidate['w'], atol=3e-6)


def test_flag_and_resume_policies():
    validate = load('validate_cg_solver', dict(math=math, numbers=numbers))
    validate(config())
    changes = dict(cg_plain_gn=False, cg_solver='bogus', optimizer_type='adamw',
                   cg_preconditioner='adam_diag', cg_damping_mu=.1,
                   cg_interpolation_lambda=.5, cg_lambda_batch_denominator=10.,
                   cg_lambda_final=.5, cg_lambda_ramp_steps=10,
                   normalize_step=True, outer_momentum_beta=.5,
                   cg_tol=float('nan'), cg_atol=-1., cg_maxiter=-1,
                   cg_qlp_rank_rtol=float('nan'))
    for key, value in changes.items():
        cfg = config()
        setattr(cfg, key, value)
        with pytest.raises(ValueError):
            validate(cfg)
    old = dict(optimizer_type='cg')
    defaults = dict(cg_solver='cg', cg_plain_gn=False, cg_qlp_rank_rtol=0.)
    cg_resume.validate_flags(old, dict(old, **defaults))
    cg_resume.validate_flags(dict(old, **defaults), old)
    for key, value in dict(cg_solver='minres_qlp', cg_plain_gn=True, cg_qlp_rank_rtol=1e-6).items():
        with pytest.raises(ValueError, match=key):
            cg_resume.validate_flags(old, dict(old, **{key: value}))


def test_actual_scale_guard_and_zero_fallback():
    choose = load('plain_gn_linesearch_scale', dict(jax=jax, jnp=jnp, np=np))
    p, d = {'w': jnp.ones(2)}, {'w': -jnp.ones(2)}
    for loss in (1., 2., float('nan'), float('inf')):
        assert choose(p, d, 1., [(1., loss)], 1., True, lambda p: 0.) == 0.
    assert choose(p, d, 1., [(1., .5)], 1., True, lambda p: 2.) == 1.
    assert choose(p, d, .25, [(1., .5)], 1., True, lambda p: 2.) == 0.
    assert choose(p, d, .25, [(1., 2.)], 1., True, lambda p: .5) == .25
    assert choose(p, d, 1., [(1., .5)], 1., False, lambda p: 0.) == 0.
    # Execute the actual outer guard and parameter commit with a nonfinite dir.
    guard = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
                 and any(isinstance(c, ast.FunctionDef) and c.name == 'evaluate_plain_candidate' for c in n.body))
    for parent in ast.walk(TREE):
        body = getattr(parent, 'body', [])
        if isinstance(body, list) and guard in body:
            tail = body[body.index(guard):]
            break
    update = next(n for n in tail if isinstance(n, ast.If) and ast.unparse(n.test) == 'FLAGS.outer_momentum_beta > 0.0')
    advance = next(n for n in tail if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'train_state')
    state = NS(params=p, step=4, replace=lambda **kw: NS(**kw))
    ns = dict(FLAGS=config(), plain_gn_linesearch_scale=choose, jax=jax, jnp=jnp,
              train_state=state, cg_metrics={'cg_direction_valid': True},
              dir={'w': jnp.full(2, jnp.inf)}, effective_step_size=1.,
              losses=[(1., .1)], baseline_loss=1.)
    exec(compile(ast.Module(body=[guard, update, advance], type_ignores=[]), '<outer>', 'exec'), ns)
    np.testing.assert_array_equal(ns['train_state'].params['w'], p['w'])
    assert ns['train_state'].step == 5 and ns['cg_metrics']['cg_step_rejected']


def test_invalid_solver_directions_are_discarded():
    harness = pcg_tests.TrainerSolveTest()
    harness.setUp()
    from unittest.mock import patch
    from EasyLM.minres_qlp import minres_qlp
    # Patch the harness's solver, retaining real diagnostic schema.
    for values in (harness.g, np.full(2, np.nan), np.full(2, np.inf)):
        def bad_solver(operator, rhs, **kwargs):
            _, info = minres_qlp(operator, rhs, **kwargs)
            return {'w': jnp.asarray(values, jnp.float32)}, info
        with patch('tests.test_gn_jacobi_pcg.minres_qlp', bad_solver):
            result, _ = harness.solve(config(), compiled=True)
        np.testing.assert_array_equal(result[0]['w'], harness.params['w'])
        assert not bool(result[-1]['cg_direction_valid'])
