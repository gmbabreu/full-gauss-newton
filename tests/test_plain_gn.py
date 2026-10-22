"""Execute the actual trainer kernel on CPU without loading the TPU trainer."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg
import numpy as np

from EasyLM import cg_resume
from EasyLM.minres import minres
from tests.test_minres import reference

TRAINER = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'

# Load the unchanged utility definitions without optional model/TPU dependencies.
_utils = ast.parse((TRAINER.parents[2] / 'jax_utils.py').read_text())
_names = ('JaxRNG', 'global_norm', 'tree_dot',
          'cross_entropy_loss_and_accuracy_with_weight_decay')
exec(compile(ast.Module(body=[n for n in _utils.body if getattr(n, 'name', '') in _names],
                        type_ignores=[]), str(TRAINER), 'exec'))


def kernel(flags):
    tree = ast.parse(TRAINER.read_text())
    names = ('train_step_cg', 'microbatch_groups', 'weighted_microbatch_sum')
    nodes = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == name) for name in names]
    def apply(params, tokens, deterministic):
        features = jnp.stack([tokens / 7., jnp.ones_like(tokens)], axis=-1)
        return NS(logits=features @ params['w'] + params['bias'])
    namespace = dict(jax=jax, jnp=jnp, linearize=jax.linearize,
        linear_transpose=jax.linear_transpose, FLAGS=flags, data_shards=1,
        model=NS(apply=apply), with_sharding_constraint=lambda b, ps: b,
        PS=lambda *args: None, JaxRNG=JaxRNG, global_norm=global_norm,
        tree_dot=tree_dot, cg=cg, minres=minres, get_gpu_memory=lambda: [0],
        adamw_lr_schedule=lambda step: jnp.float32(.03),
        cross_entropy_loss_and_accuracy_with_weight_decay=cross_entropy_loss_and_accuracy_with_weight_decay)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(TRAINER), 'exec'), namespace)
    return namespace['train_step_cg']


def assembled(params, batch):
    """Assemble J^T (diag(p)-p p^T) J explicitly, independently of autodiff."""
    flat, unravel = ravel_pytree(params)
    columns = [unravel(np.eye(len(flat), dtype='float32')[i]) for i in range(len(flat))]
    g = np.zeros(len(flat)); G = np.zeros((len(flat), len(flat)))
    for tokens, targets, masks in zip(*[np.asarray(batch[k]) for k in
                                          ('input_tokens', 'target_tokens', 'loss_masks')]):
        features = np.stack([tokens / 7., np.ones_like(tokens)], axis=-1)
        logits = features @ np.asarray(params['w']) + np.asarray(params['bias'])
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        for t, (p, mask) in enumerate(zip(probs, masks)):
            J = np.stack([features[t] @ c['w'] + c['bias'] for c in columns], axis=1)
            weight = (mask > 0) / max(masks.sum(), 1e-10) / len(batch['input_tokens'])
            err = p - np.eye(3)[targets[t]]
            g += weight * J.T @ err
            G += weight * J.T @ (np.diag(p) - np.outer(p, p)) @ J
    return g, G


class PlainGNTest(unittest.TestCase):
    def setUp(self):
        self.params = {'w': jnp.array([[.1, -.2, .3], [.3, .1, -.1]]),
                       'bias': jnp.array([.2, -.1, .1])}
        self.batch = dict(input_tokens=jnp.arange(15).reshape(5, 3),
                          target_tokens=jnp.arange(15).reshape(5, 3) % 3,
                          loss_masks=jnp.array([[1, 1, 1], [1, 0, 0], [0, 0, 0],
                                                [1, 0, 1], [0, 1, 1]]))
        self.flags = NS(cg_plain_gn=True, cg_solver='minres', cg_damping=.2,
                        cg_n_micro=3, cg_maxiter=80, cg_tol=2e-6, cg_atol=0.,
                        optimizer=NS(adamw_optimizer=NS(b1=.9, b2=.99, weight_decay=.1)))
        self.first = jax.tree.map(lambda p: jnp.full_like(p, 7.), self.params)
        self.second = jax.tree.map(lambda p: jnp.full_like(p, 11.), self.params)
        self.warm = jax.tree.map(lambda p: jnp.full_like(p, 3.), self.params)

    def run_kernel(self, step=4):
        return jax.jit(kernel(self.flags))(
            self.params, self.first, self.second, self.warm, jnp.int32(step),
            jnp.int32(step), jax.random.PRNGKey(3), self.batch, .7, .37)

    def test_masked_ce_unequal_microbatches_dense_reference(self):
        g, G = assembled(self.params, self.batch)
        for solver in ('cg', 'minres'):
            for micros in (1, 2, 3, 5, 9):
                with self.subTest(solver=solver, micros=micros):
                    self.flags.cg_solver, self.flags.cg_n_micro = solver, micros
                    params, first, second, warm, count, rng, metrics = self.run_kernel()
                    d = np.asarray(ravel_pytree(jax.tree.map(lambda a, b: a-b, params, self.params))[0])
                    A = G + .2 * np.eye(len(g))
                    np.testing.assert_allclose(d, np.linalg.solve(A, -g), rtol=1e-4, atol=3e-6)
                    np.testing.assert_allclose(d, reference(A, -g), rtol=1e-4, atol=3e-6)
                    np.testing.assert_allclose(metrics['cg_quadratic_change'], g @ d + .5 * d @ A @ d, atol=2e-7)
                    for actual, expected in ((first, self.first), (second, self.second), (warm, self.warm)):
                        np.testing.assert_array_equal(ravel_pytree(actual)[0], ravel_pytree(expected)[0])
                    self.assertEqual(int(count), 5)
                    self.assertTrue(metrics['cg_converged'])
                    self.assertFalse(any('adam' in k or 'lambda' in k for k in metrics))

    def test_zero_damping_singular_masked_ce_gn(self):
        self.flags.cg_damping = 0.
        g, G = assembled(self.params, self.batch)
        params, *_, metrics = self.run_kernel()
        d = np.asarray(ravel_pytree(jax.tree.map(lambda a, b: a-b, params, self.params))[0])
        self.assertTrue(metrics['cg_converged'])
        np.testing.assert_allclose(G @ d, -g, atol=3e-7)
        np.testing.assert_allclose(d, np.linalg.pinv(G, rcond=1e-6) @ -g, atol=3e-6)
        np.testing.assert_allclose(d, reference(G, -g), atol=3e-6)

    def test_legacy_pcg_moments_preconditioner_and_warm_start(self):
        self.flags.cg_plain_gn = False
        g, G = assembled(self.params, self.batch)
        p, m0, s0 = [np.asarray(ravel_pytree(v)[0]) for v in (self.params, self.first, self.second)]
        for step in (0, 4):
            params, m, s, warm, count, rng, metrics = self.run_kernel(step)
            expected_m = .9 * m0 + .1 * g
            expected_s = .99 * s0 + .01 * g**2
            diag = np.sqrt(expected_s / (1 - .99**(step+1))) + 1e-8
            A = .37 * G + (1-.37) / .03 * np.diag(diag)
            d = np.linalg.solve(A, -expected_m / (1 - .9**(step+1)))
            np.testing.assert_allclose(ravel_pytree(m)[0], expected_m, rtol=2e-6)
            np.testing.assert_allclose(ravel_pytree(s)[0], expected_s, rtol=2e-6)
            np.testing.assert_allclose(ravel_pytree(warm)[0], d, rtol=2e-4, atol=3e-6)
            np.testing.assert_allclose(ravel_pytree(params)[0], p + d - .63 * .03 * .1 * p, atol=3e-6)
            self.assertEqual(int(count), step + 1)
            self.assertIn('adamw_learning_rate', metrics)

    def test_flags_reject_minres_without_plain_mode_before_initialization(self):
        main = next(n for n in ast.parse(TRAINER.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        validation = []
        for node in main.body:
            if isinstance(node, ast.Expr):
                break  # distributed initialization follows validation
            validation.append(node)
        for settings, message in (
                (dict(cg_solver='minres', cg_plain_gn=False, cg_damping=0.), 'requires'),
                (dict(cg_solver='other', cg_plain_gn=True, cg_damping=0.), 'cg_solver'),
                (dict(cg_solver='cg', cg_plain_gn=True, cg_damping=-1.), 'nonnegative'),
                (dict(cg_solver='cg', cg_plain_gn=True, cg_damping=float('nan')), 'finite'),
                (dict(cg_solver='cg', cg_plain_gn=False, cg_damping=.1), 'requires')):
            flags = NS(**settings, optimizer_type='cg', cg_maxiter=100,
                       cg_tol=1e-5, cg_atol=0.)
            with self.assertRaisesRegex(ValueError, message):
                exec(compile(ast.Module(body=validation, type_ignores=[]),
                             str(TRAINER), 'exec'), dict(FLAGS=flags, np=np))

    def test_legacy_checkpoint_defaults_and_explicit_changes(self):
        saved = dict(optimizer_type='cg')
        defaults = dict(cg_solver='cg', cg_plain_gn=False, cg_damping=0.)
        cg_resume.validate_flags(saved, dict(saved, **defaults))
        cg_resume.validate_flags(dict(saved, **defaults), saved)
        for key, value in dict(cg_solver='minres', cg_plain_gn=True, cg_damping=.1).items():
            with self.assertRaisesRegex(ValueError, key):
                cg_resume.validate_flags(saved, dict(saved, **dict(defaults, **{key: value})))


if __name__ == '__main__':
    unittest.main()
