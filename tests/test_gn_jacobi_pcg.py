"""Exercise the real trainer solve on CPU without TPU/service imports."""
import ast
import math
import numbers
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np

from EasyLM import cg_resume
from EasyLM.pcg import estimate_gn_diagonal, gn_jacobi_preconditioner

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'EasyLM/models/llama/llama_train_gn.py'
TREE = ast.parse(TRAINER.read_text())


def load(name, namespace, tree=TREE):
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAINER), 'exec'), namespace)
    return namespace[name]


def flags(**changes):
    result = NS(cg_preconditioner='gn_jacobi', cg_damping_mu=.35,
                cg_gn_jacobi_probes=4, optimizer_type='cg',
                cg_interpolation_lambda=1., cg_lambda_batch_denominator=0.,
                cg_lambda_final=-1., cg_lambda_ramp_steps=0, condition_log=False,
                cg_n_micro=3, cg_tol=1e-6, cg_atol=0., cg_maxiter=30,
                log_gn_quadratic=True, log_freq=1,
                optimizer=NS(adamw_optimizer=NS(b1=0., b2=.9, weight_decay=.2)))
    result.__dict__.update(changes)
    return result


def norm(tree):
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree.leaves(tree)))


class TrainerSolveTest(unittest.TestCase):
    def setUp(self):
        self.params = {'w': jnp.array([.2, -.3])}
        # Five rows force unequal microbatches. Each row is a two-class example.
        self.features = np.array([[1., 2.], [-1., 1.], [2., .5], [.3, -2.], [1., -1.]], np.float32)
        self.targets = np.array([0, 1, 1, 0, 1])
        self.batch = dict(input_tokens=jnp.array(self.features),
                          target_tokens=jnp.array(self.targets), loss_masks=jnp.ones(5))
        # Independent analytic softmax GN, without differentiating trainer code.
        logits = self.features @ np.array(self.params['w'])
        prob = 1 / (1 + np.exp(-logits))
        self.g = self.features.T @ (prob - (self.targets == 0)) / 5
        self.G = self.features.T @ ((prob * (1 - prob))[:, None] * self.features) / 5

    def solve(self, config=None, *, lr=.03, moments=1., warm=None, inspect=False, compiled=False):
        config = config or flags()
        ns = dict(jax=jax, jnp=jnp, FLAGS=config, data_shards=1,
                  with_sharding_constraint=lambda x, spec: x, PS=lambda *args: None,
                  linearize=jax.linearize, linear_transpose=jax.linear_transpose,
                  adamw_lr_schedule=lambda step: jnp.float32(lr), global_norm=norm,
                  get_gpu_memory=lambda: [0], estimate_gn_diagonal=estimate_gn_diagonal,
                  gn_jacobi_preconditioner=gn_jacobi_preconditioner)

        def model_apply(params, inputs, deterministic):
            z = inputs @ params['w']
            return NS(logits=jnp.stack((z, jnp.zeros_like(z)), axis=-1))

        def loss(logits, targets, params, base, masks, weight_decay):
            values = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], axis=-1)[:, 0]
            return jnp.mean(values * masks), jnp.float32(0)

        ns.update(model=NS(apply=model_apply),
                  cross_entropy_loss_and_accuracy_with_weight_decay=loss)
        load('JaxRNG', ns, ast.parse((ROOT / 'EasyLM/jax_utils.py').read_text()))
        for name in ('microbatch_groups', 'weighted_microbatch_sum', 'gn_quadratic_summary'):
            load(name, ns)
        captured = {}

        def checked_cg(operator, rhs, **kwargs):
            if inspect:
                # Inspect the actual full operator, RHS, warm start, and fixed M.
                eye = np.eye(2, dtype=np.float32)
                captured['A'] = np.column_stack([operator({'w': jnp.array(v)})['w'] for v in eye])
                captured['rhs'] = rhs['w']
                captured['x0'] = kwargs['x0']
                captured['M'] = kwargs['M']({'w': jnp.ones(2)})['w']
            return cg(operator, rhs, **kwargs)

        def counted_estimate(*args):
            captured['estimates'] = captured.get('estimates', 0) + 1
            return estimate_gn_diagonal(*args)

        ns['estimate_gn_diagonal'] = counted_estimate
        ns['cg'] = checked_cg
        step = load('train_step_cg', ns)
        if compiled:
            step = jax.jit(step)
        result = step(self.params, {'w': jnp.full(2, moments)},
                      {'w': jnp.full(2, moments)},
                      warm if warm is not None else {'w': jnp.zeros(2)},
                      jnp.int32(2), jnp.int32(0), jax.random.PRNGKey(17),
                      self.batch, 0., config.cg_interpolation_lambda)
        return result, captured

    def test_dense_system_microbatch_damping_and_raw_metrics(self):
        for micro in (1, 2, 3, 5, 9):
            with self.subTest(micro=micro):
                result, captured = self.solve(flags(cg_n_micro=micro), inspect=True)
                self.assertEqual(captured['estimates'], 1)
                expected_A = self.G + .35 * np.eye(2)
                np.testing.assert_allclose(captured['A'], expected_A, atol=2e-7)
                np.testing.assert_allclose(captured['rhs'], -self.g, atol=1e-7)
                expected = np.linalg.solve(expected_A, -self.g)
                x = result[3]['w']
                np.testing.assert_allclose(x, expected, atol=2e-6)
                metrics = result[-1]
                residual = np.linalg.norm(expected_A @ x + self.g) / np.linalg.norm(self.g)
                self.assertAlmostEqual(float(metrics['relative_residual']), residual, places=6)
                d = np.array(result[0]['w'] - self.params['w'])
                self.assertAlmostEqual(float(metrics['gn_quad/d_G_d']), d @ self.G @ d, places=6)
                self.assertAlmostEqual(float(metrics['gn_quad/g_dot_d']), self.g @ d, places=6)
                self.assertEqual(int(metrics['cg_gn_jacobi_gvp_count']), 4)
                self.assertAlmostEqual(float(metrics['cg_damping_mu']), .35, places=6)

    def test_adam_independence_rng_logging_and_warm_start(self):
        baseline, _ = self.solve()
        warm = {'w': jnp.array([.1, -.2])}
        for probes, lr, moment, logging in ((1, 0., 1e4, False), (7, 5., float('nan'), True)):
            result, captured = self.solve(flags(cg_gn_jacobi_probes=probes,
                                                log_gn_quadratic=logging),
                                           lr=lr, moments=moment, warm=warm, inspect=True)
            np.testing.assert_allclose(result[0]['w'], baseline[0]['w'], atol=2e-6)
            np.testing.assert_allclose(result[3]['w'], baseline[3]['w'], atol=2e-6)
            np.testing.assert_array_equal(captured['x0']['w'], warm['w'])
            np.testing.assert_array_equal(result[-2], baseline[-2])
            self.assertEqual(int(result[4]), 3)
        np.testing.assert_array_equal(baseline[-2], jax.random.split(jax.random.PRNGKey(17))[1])
        repeat, _ = self.solve()
        np.testing.assert_array_equal(repeat[0]['w'], baseline[0]['w'])

    def test_jitted_solve_and_fixed_preconditioner(self):
        eager, captured = self.solve(inspect=True)
        compiled, _ = self.solve(compiled=True)
        np.testing.assert_allclose(compiled[0]['w'], eager[0]['w'], atol=2e-7)
        key = jax.random.fold_in(jax.random.PRNGKey(17), 0x474E4A)
        diagonal = estimate_gn_diagonal(
            lambda v: {'w': jnp.array(self.G) @ v['w']}, self.params, key, 4)
        expected = gn_jacobi_preconditioner(diagonal, .35)({'w': jnp.ones(2)})
        np.testing.assert_allclose(captured['M'], expected['w'], atol=2e-7)

    def test_legacy_interpolated_adam_system(self):
        config = flags(cg_preconditioner='adam_diag', cg_damping_mu=0.,
                       cg_interpolation_lambda=.6)
        config.optimizer.adamw_optimizer.b1 = .4
        result, captured = self.solve(config, inspect=True)
        self.assertNotIn('estimates', captured)
        m = .4 * np.ones(2) + .6 * self.g
        v = .9 * np.ones(2) + .1 * self.g ** 2
        D = np.sqrt(v / (1 - .9 ** 3)) + 1e-8
        A = .6 * self.G + .4 / .03 * np.diag(D)
        rhs = -m / (1 - .4 ** 3)
        np.testing.assert_allclose(captured['A'], A, rtol=2e-6)
        np.testing.assert_allclose(captured['rhs'], rhs, atol=2e-7)
        np.testing.assert_allclose(captured['M'], 1 / D, rtol=2e-6)
        expected = np.linalg.solve(A, rhs)
        np.testing.assert_allclose(result[3]['w'], expected, atol=2e-7)
        np.testing.assert_allclose(result[0]['w'],
                                   np.array(self.params['w']) + expected - .4 * .03 * .2 * np.array(self.params['w']), atol=2e-7)
        self.assertEqual(int(result[-1]['cg_gn_jacobi_gvp_count']), 0)


class HelpersAndFlagsTest(unittest.TestCase):
    def test_sequential_rademacher_estimator_and_inverse(self):
        matrix = jnp.array([[2., .4, -.2], [.4, 1., .5], [-.2, .5, 3.]])
        template = {'a': jnp.zeros(2), 'b': jnp.zeros(1)}
        key = jax.random.PRNGKey(41)

        def apply(v):
            product = matrix @ jnp.concatenate([v['a'], v['b']])
            return {'a': product[:2], 'b': product[2:]}

        expected = np.zeros(3)
        for k in range(5):
            z = np.concatenate([np.array(jax.random.rademacher(
                jax.random.fold_in(jax.random.fold_in(key, k), leaf), (size,), dtype=jnp.float32))
                for leaf, size in enumerate((2, 1))])
            expected += z * (np.array(matrix) @ z) / 5
        actual = jax.jit(lambda: estimate_gn_diagonal(apply, template, key, 5))()
        np.testing.assert_allclose(np.concatenate([actual['a'], actual['b']]), expected, atol=2e-7)
        inverse = gn_jacobi_preconditioner({'a': jnp.array([-2., 0., 3.])}, .5)
        np.testing.assert_allclose(inverse({'a': jnp.ones(3)})['a'], [2., 2., 1/3.5])

    def test_validation(self):
        validate = load('validate_cg_preconditioner', dict(math=math, numbers=numbers))
        validate(flags())
        validate(flags(cg_preconditioner='adam_diag', cg_damping_mu=0.))
        for field, values in {
            'cg_damping_mu': [0., -1., float('nan'), float('inf')],
            'cg_preconditioner': ['unknown'], 'cg_gn_jacobi_probes': [0, -1, 1.5, True],
            'cg_interpolation_lambda': [.5], 'cg_lambda_batch_denominator': [10.],
            'cg_lambda_final': [.5], 'cg_lambda_ramp_steps': [10],
            'condition_log': [True], 'optimizer_type': ['adamw'],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate(flags(**{field: value}))
        config = flags()
        config.optimizer.adamw_optimizer.b1 = .1
        with self.assertRaisesRegex(ValueError, 'b1=0'):
            validate(config)

    def test_resume_legacy_defaults_and_jacobi_changes(self):
        old = {'optimizer_type': 'cg'}
        current = dict(old, cg_preconditioner='adam_diag', cg_damping_mu=0., cg_gn_jacobi_probes=4)
        cg_resume.validate_flags(old, current)
        cg_resume.validate_flags(current, old)
        self.assertEqual(old, {'optimizer_type': 'cg'})
        jacobi = dict(current, cg_preconditioner='gn_jacobi', cg_damping_mu=.2)
        cg_resume.validate_flags(jacobi, dict(jacobi))
        for key, value in (('cg_preconditioner', 'adam_diag'), ('cg_damping_mu', .3),
                           ('cg_gn_jacobi_probes', 8)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                cg_resume.validate_flags(jacobi, dict(jacobi, **{key: value}))
        with self.assertRaises(ValueError):
            cg_resume.validate_flags(old, jacobi)


if __name__ == '__main__':
    unittest.main()
