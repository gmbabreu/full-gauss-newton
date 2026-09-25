import unittest
from unittest import mock

import numpy as np

from EasyLM import matrix_spectrum as ms


class MatrixSpectrumTest(unittest.TestCase):
    def test_memory_preflight_counts_one_basis_buffer(self):
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            report = ms.memory_preflight(1000, 160, 4, reserve_bytes=0)
        self.assertEqual(report['basis_buffer_bytes'], 160 * 1000 * 4)

    def test_preflight_respects_effective_limit(self):
        with mock.patch.object(ms, 'available_host_memory', return_value=1024):
            with self.assertRaises(MemoryError):
                ms.memory_preflight(1000, 160, 4, reserve_bytes=0)

    def test_small_diagonal_candidates_are_ordered(self):
        matrix = np.diag(np.array([9., 7., 5., 3., 1.], np.float32))
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            scalars, table = ms.estimate_top_spectrum(
                lambda vector: matrix @ vector, 5, top_k=3, block_size=2,
                max_basis=5, restart_keep=4, max_products=40,
                residual_tol=1e-4, stability_tol=1e-4, seed=3)
        self.assertTrue(scalars['accepted'], table)
        np.testing.assert_allclose(table['values'][:3], [9., 7., 5.], rtol=1e-4)
        self.assertLess(table['orthogonality_error'], 1e-3)
        self.assertLessEqual(scalars['gn_products'], 40)

    def test_dimension_above_basis_restarts_and_preserves_spectrum(self):
        diagonal = np.zeros(192, np.float32)
        diagonal[:5] = [20., 19., 18., 17., 16.]
        calls = 0
        def apply(vector):
            nonlocal calls
            calls += 1
            return diagonal * vector
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            scalars, table = ms.estimate_top_spectrum(
                apply, 192, top_k=5, block_size=4, max_basis=12,
                restart_keep=8, max_products=100, residual_tol=1e-3,
                stability_tol=1e-3, seed=7)
        self.assertTrue(scalars['accepted'], table)
        np.testing.assert_allclose(table['values'][:5], diagonal[:5], rtol=1e-3)
        self.assertLess(table['orthogonality_error'], 1e-3)
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 100)

    def test_clean_budget_exhaustion_counts_calls(self):
        calls = 0
        def apply(vector):
            nonlocal calls
            calls += 1
            return np.arange(vector.size, 0, -1, dtype=np.float32) * vector
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            scalars, table = ms.estimate_top_spectrum(
                apply, 192, top_k=5, block_size=4, max_basis=12,
                restart_keep=8, max_products=8, residual_tol=1e-8,
                stability_tol=1e-8)
        self.assertFalse(scalars['accepted'])
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 8)
        self.assertIn('product_budget_exhausted', table['failure_reasons'])

    def test_top100_exceeds_basis_capacity_and_restarts(self):
        diagonal = np.concatenate((
            np.linspace(10., 9., 100),
            np.linspace(4., .1, 92),
        )).astype(np.float32)
        calls = 0
        def apply(vector):
            nonlocal calls
            calls += 1
            return diagonal * vector
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            scalars, table = ms.estimate_top_spectrum(
                apply, 192, top_k=100, block_size=4, max_basis=160,
                restart_keep=120, max_products=600,
                residual_tol=.01, stability_tol=.02, seed=11)
        self.assertTrue(scalars['accepted'], table)
        np.testing.assert_allclose(table['values'][:100], diagonal[:100], rtol=.01)
        self.assertGreater(table['restart_count'], 0)
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 600)
        self.assertAlmostEqual(
            scalars['top100_condition_est'], diagonal[0] / diagonal[99],
            delta=.01)
        self.assertEqual(set(table['direct_residuals']), {1, *range(10, 101, 10)})
        for rank in range(10, 101, 10):
            self.assertAlmostEqual(
                scalars[f'lambda_{rank}_est'], diagonal[rank - 1], delta=.01)

    def test_top10_condition_and_timings(self):
        diagonal = np.concatenate((
            np.arange(30., 20., -1),
            np.ones(38),
        )).astype(np.float32)
        calls = 0
        def apply(vector):
            nonlocal calls
            calls += 1
            return diagonal * vector
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            scalars, table = ms.estimate_top_spectrum(
                apply, diagonal.size, top_k=10, block_size=4,
                max_basis=32, restart_keep=16, max_products=120,
                residual_tol=.01, stability_tol=.02, seed=0)
        self.assertTrue(scalars['accepted'], table)
        np.testing.assert_allclose(
            table['values'][:10], diagonal[:10], rtol=.01, atol=.01)
        self.assertAlmostEqual(
            scalars['top10_condition_est'], 30. / 21., delta=.01)
        self.assertIsNone(scalars['top100_condition_est'])
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 120)
        timing_names = (
            'seconds_operator', 'seconds_orthogonalization', 'seconds_gram',
            'seconds_projection', 'seconds_eigensolve',
            'seconds_ritz_residuals', 'seconds_expansion', 'seconds_restart',
            'seconds_validation')
        for name in timing_names:
            self.assertTrue(np.isfinite(scalars[name]), name)
            self.assertGreaterEqual(scalars[name], 0., name)
        self.assertLessEqual(
            scalars['max_relative_ritz_residual'], .01)

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_clustered_rotated_spectrum_and_every_tenth_rank(self, _memory):
        rng = np.random.default_rng(42)
        basis = np.linalg.qr(rng.normal(size=(96, 96)))[0]
        diagonal = np.r_[np.linspace(10., 9.9, 25), np.linspace(4., .1, 71)]
        matrix = (basis * diagonal) @ basis.T
        scalars, table = ms.estimate_top_spectrum(
            lambda v: matrix @ v, 96, top_k=25, max_basis=45,
            restart_keep=30, max_products=240, residual_tol=1e-4)
        self.assertTrue(scalars['accepted'], table)
        self.assertGreater(scalars['restart_count'], 0)
        np.testing.assert_allclose(table['values'], diagonal[:25], rtol=1e-4)
        self.assertEqual(set(table['direct_residuals']), {1, 10, 20, 25})
        for rank in (1, 10, 20, 25):
            self.assertAlmostEqual(scalars[f'lambda_{rank}_est'], diagonal[rank-1], places=3)
        self.assertIsNone(scalars['lambda_100_est'])

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_fresh_validation_rejects_changed_operator(self, _memory):
        diagonal = np.arange(8., 0., -1, dtype=np.float32)
        calls = 0
        def apply(v):
            nonlocal calls
            calls += 1
            return diagonal * v + (v if calls > 8 else 0)
        scalars, table = ms.estimate_top_spectrum(
            apply, 8, top_k=4, max_basis=8, restart_keep=6,
            max_products=30, residual_tol=1e-4)
        self.assertFalse(scalars['accepted'])
        self.assertIn('direct_residual_failed', table['failure_reasons'])
        self.assertIsNone(scalars['lambda_1_est'])
        self.assertEqual(scalars['gn_products'], calls)

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_nonfinite_product_is_unresolved(self, _memory):
        scalars, table = ms.estimate_top_spectrum(
            lambda v: v * np.nan, 8, top_k=3, max_basis=8,
            restart_keep=5, max_products=20)
        self.assertFalse(scalars['accepted'])
        self.assertEqual(scalars['gn_products'], 1)
        self.assertIn('invalid_operator_product', table['failure_reasons'])

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_jax_gn_matches_explicit_jacobian(self, _memory):
        import jax
        import jax.numpy as jnp
        rng = np.random.default_rng(15)
        x = jnp.asarray(rng.normal(size=(9, 24)), dtype=jnp.float32)
        theta = jnp.asarray(rng.normal(size=24), dtype=jnp.float32)
        def logits(t): return jnp.sin(x @ t)
        jacobian = jax.jacfwd(logits)(theta)
        hessian = jax.hessian(jax.scipy.special.logsumexp)(logits(theta))
        dense = np.asarray(jacobian.T @ hessian @ jacobian) + .1 * np.eye(24)
        @jax.jit
        def product(v):
            _, tangent = jax.jvp(logits, (theta,), (v,))
            return jax.vjp(logits, theta)[1](hessian @ tangent)[0] + .1 * v
        scalars, table = ms.estimate_top_spectrum(
            lambda v: np.asarray(product(jnp.asarray(v))), 24, top_k=5,
            max_basis=12, restart_keep=7, max_products=120, residual_tol=1e-4)
        self.assertTrue(scalars['accepted'], table)
        expected = np.linalg.eigvalsh(dense)[::-1][:5]
        np.testing.assert_allclose(table['values'], expected, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
