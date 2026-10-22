"""CPU references for ordinary MINRES; SciPy is test-only."""
import inspect
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import minres as scipy_minres

from EasyLM.minres import (minres, CONVERGED, ITERATION_LIMIT, BREAKDOWN,
                           NONFINITE, RESIDUAL_MISMATCH)


def reference(a, b, tol=1e-10, maxiter=1000):
    key = 'rtol' if 'rtol' in inspect.signature(scipy_minres).parameters else 'tol'
    return scipy_minres(a, b, maxiter=maxiter, **{key: tol})[0]


class MinresTest(unittest.TestCase):
    def test_dense_symmetric_and_pytree_references(self):
        rng = np.random.default_rng(32)
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        for eigenvalues in (np.arange(1., 9.), np.array([-5., -3., -1., 1., 2., 4., 5., 8.])):
            a = (q * eigenvalues) @ q.T
            b = rng.normal(size=8)
            def unpack(v):
                return {'a': v[:3], 'b': (v[3:7].reshape(2, 2), v[7:])}
            def pack(v):
                return jnp.concatenate([v['a'], v['b'][0].ravel(), v['b'][1]])
            x, info = jax.jit(lambda b: minres(
                lambda v: unpack(jnp.asarray(a, dtype=jnp.float32) @ pack(v)),
                b, tol=2e-6, maxiter=40))(unpack(jnp.asarray(b, dtype=jnp.float32)))
            actual = np.asarray(pack(x))
            np.testing.assert_allclose(actual, reference(a, b), rtol=1e-4, atol=2e-6)
            np.testing.assert_allclose(actual, np.linalg.solve(a, b), rtol=1e-4, atol=2e-6)
            self.assertTrue(info['converged'])
            self.assertEqual(int(info['stop_reason']), CONVERGED)
            np.testing.assert_allclose(info['true_residual_norm'], np.linalg.norm(
                a.astype('float32') @ actual - b.astype('float32')), atol=2e-6)

    def test_zero_rhs_and_iteration_limit(self):
        calls = []
        def operator(v):
            jax.debug.callback(lambda: calls.append(1), ordered=True)
            return 2 * v
        x, info = jax.jit(lambda b: minres(operator, b))(jnp.zeros(3))
        jax.block_until_ready((x, info))
        self.assertEqual(calls, [])
        self.assertEqual(int(info['iterations']), 0)
        self.assertTrue(info['converged'])
        np.testing.assert_array_equal(x, 0)
        for budget in (0, 1):
            x, info = minres(lambda v: jnp.array([1., 2., 3.]) * v,
                             jnp.ones(3), tol=1e-8, maxiter=budget)
            self.assertEqual(int(info['iterations']), budget)
            self.assertEqual(int(info['stop_reason']), ITERATION_LIMIT)
            self.assertFalse(info['converged'])

    def test_singular_and_breakdown(self):
        for diagonal, b, success in (([1., 0., 2.], [1., 0., 1.], True),
                                     ([0., 0., 0.], [1., 2., 3.], False),
                                     ([1., 0., 2.], [1., 1., 1.], False),
                                     ([2., 2., 2.], [1., 2., 3.], True)):
            a, rhs = jnp.diag(jnp.array(diagonal)), jnp.array(b)
            x, info = minres(lambda v: a @ v, rhs, tol=2e-6, maxiter=20)
            self.assertEqual(bool(info['converged']), success)
            self.assertTrue(np.isfinite(x).all())
            if success:
                np.testing.assert_allclose(x, reference(np.asarray(a), np.asarray(rhs)), atol=1e-5)
            else:
                self.assertIn(int(info['stop_reason']), (BREAKDOWN, ITERATION_LIMIT, RESIDUAL_MISMATCH))
                self.assertGreater(float(info['true_residual_norm']), 2e-6 * np.linalg.norm(b))

    def test_nonfinite_rhs_and_operator(self):
        for bad in (jnp.nan, jnp.inf):
            for operator, rhs in ((lambda v: v, jnp.array([bad, 1.])),
                                  (lambda v: v * bad, jnp.ones(2))):
                x, info = minres(operator, rhs)
                self.assertFalse(info['converged'])
                self.assertEqual(int(info['stop_reason']), NONFINITE)
                self.assertTrue(np.isfinite(x).all())

    def test_ill_conditioned_fp32_true_residual_controls_success(self):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        a = ((q * np.geomspace(1e-5, 1., 12)) @ q.T).astype('float32')
        b = rng.normal(size=12).astype('float32')
        for tol in (1e-2, 1e-7):
            x, info = minres(lambda v: jnp.asarray(a) @ v, jnp.asarray(b), tol=tol, maxiter=300)
            true_norm = np.linalg.norm(a @ np.asarray(x) - b)
            np.testing.assert_allclose(info['true_residual_norm'], true_norm, atol=2e-3)
            if info['converged']:
                self.assertLessEqual(float(info['true_residual_norm']), tol * np.linalg.norm(b))
                # Forward error is amplified by conditioning; compare the equation too.
                expected = reference(a.astype('float64'), b.astype('float64'))
                self.assertLess(np.linalg.norm(a @ (np.asarray(x) - expected)), 2 * tol * np.linalg.norm(b))
            else:
                self.assertNotEqual(int(info['stop_reason']), CONVERGED)

    def test_ill_conditioned_fp32_matches_reference_at_attainable_tolerance(self):
        rng = np.random.default_rng(42)
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        a = ((q * np.geomspace(1e-3, 1., 12)) @ q.T).astype('float32')
        b = rng.normal(size=12).astype('float32')
        x, info = minres(lambda v: jnp.asarray(a) @ v, jnp.asarray(b),
                         tol=2e-3, maxiter=200)
        expected = reference(a.astype('float64'), b.astype('float64'))
        self.assertTrue(info['converged'])
        self.assertLess(np.linalg.norm(a @ np.asarray(x) - b), 2e-3 * np.linalg.norm(b))
        np.testing.assert_allclose(x, expected, rtol=.02, atol=.1)

    def test_false_recurrence_convergence_is_rejected(self):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        a = jnp.asarray(((q * np.geomspace(1e-5, 1., 12)) @ q.T).astype('float32'))
        b = jnp.asarray(rng.normal(size=12).astype('float32'))
        _, info = minres(lambda v: a @ v, b, tol=1e-7, maxiter=300)
        self.assertLessEqual(float(info['estimated_residual_norm']), 1e-7 * np.linalg.norm(b))
        self.assertGreater(float(info['true_residual_norm']), 1e-7 * np.linalg.norm(b))
        self.assertFalse(info['converged'])
        self.assertEqual(int(info['stop_reason']), RESIDUAL_MISMATCH)

    def test_one_matvec_per_iteration_plus_final_verification(self):
        calls = []
        def operator(v):
            jax.debug.callback(lambda: calls.append(1), ordered=True)
            return jnp.arange(1., 7.) * v
        x, info = jax.jit(lambda b: minres(operator, b, tol=1e-6))(jnp.ones(6))
        jax.block_until_ready((x, info))
        self.assertEqual(len(calls), int(info['iterations']) + 1)
        graph = jax.make_jaxpr(lambda b: minres(operator, b))(jnp.ones(6))
        self.assertEqual(sum(e.primitive.name == 'while' for e in graph.jaxpr.eqns), 1)

    def test_atol(self):
        x, info = minres(lambda v: 2 * v, jnp.array([1e-8]), tol=0., atol=1e-7)
        self.assertTrue(info['converged'])
        self.assertEqual(int(info['iterations']), 0)
        np.testing.assert_array_equal(x, 0)


if __name__ == '__main__':
    unittest.main()
