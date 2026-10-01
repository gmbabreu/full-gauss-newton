import unittest
from unittest import mock
from contextlib import redirect_stdout
import io

import numpy as np

from EasyLM import matrix_spectrum as ms


class MatrixSpectrumTest(unittest.TestCase):
    def test_memory_preflight_counts_one_basis_buffer(self):
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            report = ms.memory_preflight(1000, 160, 4, reserve_bytes=0)
        self.assertEqual(report['basis_buffer_bytes'], 160 * 1000 * 4)
        self.assertEqual(report['gram_cache_bytes'], 160 * 160 * 8)

    def test_incremental_gram_matches_dense_and_rebuilds_after_restart(self):
        rng = np.random.default_rng(91)
        q = rng.normal(size=(40, 37)).astype(np.float32)
        cache = np.zeros((40, 40), np.float64)
        cached_count = 0
        for count in (12, 20, 28, 40):
            cache, cached_count = ms._incremental_gram(
                q, count, cache, cached_count, chunk=7)
            expected = q[:count].astype(np.float64) @ q[:count].astype(np.float64).T
            np.testing.assert_allclose(cache[:count, :count], expected,
                                       rtol=0, atol=5e-14)

        coefficients = np.linalg.qr(rng.normal(size=(40, 13)))[0]
        ms._transform_in_place(q, 40, coefficients, chunk=9)
        cache.fill(0.)
        cache, cached_count = ms._incremental_gram(
            q, 13, cache, 0, chunk=7)
        expected = q[:13].astype(np.float64) @ q[:13].astype(np.float64).T
        np.testing.assert_allclose(cache[:13, :13], expected, rtol=0, atol=5e-14)
        self.assertEqual(cached_count, 13)

    def test_preflight_respects_effective_limit(self):
        with mock.patch.object(ms, 'available_host_memory', return_value=1024):
            with self.assertRaises(MemoryError):
                ms.memory_preflight(1000, 160, 4, reserve_bytes=0)

    def test_basis_capacity_shrinks_to_current_memory(self):
        available = ms._memory_report(1000, 6, 2, 0, 0)['required_bytes']
        with mock.patch.object(ms, 'available_host_memory',
                               return_value=available):
            capacity, report = ms.fit_basis_to_memory(
                1000, 8, 5, 2, reserve_bytes=0)
        self.assertEqual(capacity, 6)
        self.assertEqual(report['requested_max_basis'], 8)
        self.assertEqual(report['effective_max_basis'], 6)
        self.assertTrue(report['memory_limited'])

    def test_basis_capacity_still_rejects_when_minimum_does_not_fit(self):
        available = ms._memory_report(1000, 4, 2, 0, 0)['required_bytes']
        with mock.patch.object(ms, 'available_host_memory',
                               return_value=available):
            with self.assertRaisesRegex(MemoryError,
                                        'spectrum preflight requires'):
                ms.fit_basis_to_memory(1000, 8, 5, 2, reserve_bytes=0)

    def test_memory_limited_estimator_restarts_and_converges(self):
        dimension = 192
        diagonal = np.concatenate((
            np.array([20., 19., 18., 17., 16.]),
            np.linspace(4., .1, dimension - 5),
        )).astype(np.float32)
        available = ms._memory_report(
            dimension, 12, 4, 8 << 30, 0)['required_bytes']
        with mock.patch.object(ms, 'available_host_memory',
                               return_value=available):
            scalars, table = ms.estimate_top_spectrum(
                lambda vector: diagonal * vector, dimension,
                top_k=5, check_every=4, max_basis=32, restart_keep=8,
                max_products=100, residual_tol=1e-3,
                stability_tol=1e-3, seed=7)
        self.assertTrue(scalars['accepted'], table)
        self.assertEqual(scalars['basis_capacity'], 12)
        self.assertEqual(scalars['configured_max_basis'], 32)
        self.assertTrue(scalars['memory_limited'])
        self.assertGreater(scalars['restart_count'], 0)
        np.testing.assert_allclose(table['values'][:5], diagonal[:5],
                                   rtol=1e-3)

    def test_small_diagonal_candidates_are_ordered(self):
        matrix = np.diag(np.array([9., 7., 5., 3., 1.], np.float32))
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            scalars, table = ms.estimate_top_spectrum(
                lambda vector: matrix @ vector, 5, top_k=3, check_every=2,
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
                apply, 192, top_k=5, check_every=4, max_basis=12,
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
                apply, 192, top_k=5, check_every=4, max_basis=12,
                restart_keep=8, max_products=8, residual_tol=1e-8,
                stability_tol=1e-8)
        self.assertFalse(scalars['accepted'])
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 8)
        self.assertIn('product_budget_exhausted', table['failure_reasons'])
        self.assertEqual(table['values'], [])
        self.assertIsNone(scalars['lambda_1_est'])

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
                apply, 192, top_k=100, check_every=4, max_basis=160,
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
                apply, diagonal.size, top_k=10, check_every=4,
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
        with redirect_stdout(io.StringIO()) as output:
            scalars, table = ms.estimate_top_spectrum(
                apply, 8, top_k=4, max_basis=8, restart_keep=6,
                max_products=30, residual_tol=1e-4)
        self.assertFalse(scalars['accepted'])
        self.assertIn('direct_residual_failed', table['failure_reasons'])
        self.assertAlmostEqual(scalars['lambda_1_est'], 8., places=4)
        self.assertGreater(scalars['max_direct_residual'], 1e-4)
        self.assertIn('direct_residuals={', output.getvalue())
        self.assertIsNone(scalars['top10_condition_est'])
        self.assertEqual(scalars['gn_products'], calls)

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_failed_direct_validation_continues_and_retries(self, _memory):
        diagonal = np.r_[np.linspace(10., 9., 5),
                         np.linspace(4., .1, 59)].astype(np.float32)
        calls = 0
        def apply(v):
            nonlocal calls
            calls += 1
            image = diagonal * v
            # The first candidate uses 18 expansion products, followed by the
            # two published-rank validations. Make only that validation fail.
            if calls in (19, 20):
                image = image + .1 * np.roll(v, 1)
            return image
        validation_sizes = []
        incremental = ms._incremental_gram
        def record_gram(q, count, *args, **kwargs):
            validation_sizes.append(count)
            return incremental(q, count, *args, **kwargs)
        with mock.patch.object(ms, '_incremental_gram', side_effect=record_gram):
            scalars, table = ms.estimate_top_spectrum(
                apply, 64, top_k=5, check_every=2, max_basis=48,
                restart_keep=24, max_products=120, residual_tol=1e-3,
                stability_tol=1e-3, seed=3)
        self.assertTrue(scalars['accepted'], table)
        self.assertEqual(scalars['validation_attempts'], 2)
        self.assertGreaterEqual(validation_sizes[1] - validation_sizes[0], 16)
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 120)
        np.testing.assert_allclose(
            table['values'], diagonal[:5], rtol=1e-3, atol=1e-3)
        self.assertLessEqual(scalars['max_direct_residual'], 1e-3)

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_persistent_direct_failure_stops_at_attempt_cap(self, _memory):
        diagonal = np.r_[np.linspace(10., 9., 5),
                         np.linspace(4., .1, 59)].astype(np.float32)
        calls = 0
        original = ms._transform_chunked

        def corrupt(source, coefficients, destination, **kwargs):
            original(source, coefficients, destination, **kwargs)
            destination[:] = np.roll(destination, 1, axis=1)

        def apply(vector):
            nonlocal calls
            calls += 1
            return diagonal * vector

        with mock.patch.object(ms, '_transform_chunked', side_effect=corrupt):
            scalars, table = ms.estimate_top_spectrum(
                apply, 64, top_k=5, check_every=2, max_basis=24,
                restart_keep=12, max_products=160, residual_tol=1e-3,
                stability_tol=1e-3, max_validation_attempts=3, seed=3)
        self.assertFalse(scalars['accepted'])
        self.assertEqual(scalars['validation_attempts'], 3)
        self.assertIn('validation_attempt_budget_exhausted',
                      table['failure_reasons'])
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 160)
        self.assertIsNotNone(scalars['lambda_1_est'])
        self.assertGreater(scalars['max_direct_residual'], 1e-3)
        self.assertIsNone(scalars['top10_condition_est'])

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_timeout_on_later_attempt_retains_completed_snapshot(self, _memory):
        diagonal = np.r_[np.linspace(10., 9., 5),
                         np.linspace(4., .1, 59)].astype(np.float32)
        calls = 0
        now = [0.]
        gram_calls = 0
        incremental = ms._incremental_gram
        def apply(vector):
            nonlocal calls
            calls += 1
            image = diagonal * vector
            if calls in (19, 20):
                image = image + .1 * np.roll(vector, 1)
            return image
        def expire_second_validation(*args, **kwargs):
            nonlocal gram_calls
            gram_calls += 1
            if gram_calls == 2:
                now[0] = 2.
            return incremental(*args, **kwargs)
        with mock.patch.object(ms, '_incremental_gram',
                               side_effect=expire_second_validation):
            scalars, table = ms.estimate_top_spectrum(
                apply, 64, top_k=5, check_every=2, max_basis=48,
                restart_keep=24, max_products=120, residual_tol=1e-3,
                stability_tol=1e-3, seed=3, max_seconds=1.,
                clock=lambda: now[0])
        self.assertFalse(scalars['accepted'])
        self.assertEqual(scalars['validation_attempts'], 2)
        self.assertIn('time_budget_exhausted', table['failure_reasons'])
        self.assertEqual(calls, scalars['gn_products'])
        self.assertGreater(max(table['direct_residuals'].values()), 1e-3)
        self.assertEqual(scalars['max_direct_residual'],
                         max(table['direct_residuals'].values()))
        self.assertEqual(scalars['lambda_1_est'], table['values'][0])
        self.assertEqual(scalars['max_relative_ritz_residual'],
                         max(table['residuals']))
        self.assertIsNone(scalars['top10_condition_est'])

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_deadline_during_incomplete_validation_publishes_nothing(self, _memory):
        diagonal = np.arange(8., 0., -1, dtype=np.float32)
        now = [0.]
        incremental = ms._incremental_gram
        def expire_in_gram(*args, **kwargs):
            now[0] = 2.
            return incremental(*args, **kwargs)
        with mock.patch.object(ms, '_incremental_gram',
                               side_effect=expire_in_gram):
            scalars, table = ms.estimate_top_spectrum(
                lambda v: diagonal * v, 8, top_k=4, check_every=2,
                max_basis=8, restart_keep=6, max_products=30,
                residual_tol=1e-4, max_seconds=1., clock=lambda: now[0])
        self.assertFalse(scalars['accepted'])
        self.assertIn('time_budget_exhausted', table['failure_reasons'])
        self.assertEqual(table['direct_residuals'], {})
        self.assertIsNone(scalars['lambda_1_est'])

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_invalid_basis_does_not_publish_estimates(self, _memory):
        diagonal = np.arange(8., 0., -1, dtype=np.float32)
        incremental = ms._incremental_gram
        def invalidate_basis(*args, **kwargs):
            cache, count = incremental(*args, **kwargs)
            cache[0, 0] += 1.
            return cache, count
        with mock.patch.object(ms, '_incremental_gram',
                               side_effect=invalidate_basis):
            scalars, table = ms.estimate_top_spectrum(
                lambda vector: diagonal * vector, 8, top_k=4,
                check_every=2, max_basis=8, restart_keep=6,
                max_products=30, residual_tol=1e-4)
        self.assertFalse(scalars['accepted'])
        self.assertIn('invalid_orthonormal_basis', table['failure_reasons'])
        self.assertEqual(table['values'], [])
        self.assertEqual(table['direct_residuals'], {})
        self.assertIsNone(scalars['lambda_1_est'])

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_fake_deadline_inside_restart_transform_is_controlled(self, _memory):
        diagonal = np.r_[np.linspace(20., 16., 5),
                         np.linspace(4., .1, 187)].astype(np.float32)
        now = [0.]
        transform = ms._transform_in_place
        def expire_in_restart(*args, **kwargs):
            now[0] = 2.
            return transform(*args, **kwargs)
        with mock.patch.object(ms, '_transform_in_place',
                               side_effect=expire_in_restart):
            scalars, table = ms.estimate_top_spectrum(
                lambda v: diagonal * v, 192, top_k=5, check_every=4,
                max_basis=12, restart_keep=8, max_products=100,
                residual_tol=1e-8, stability_tol=1e-8,
                max_seconds=1., clock=lambda: now[0])
        self.assertFalse(scalars['accepted'])
        self.assertEqual(scalars['restart_count'], 0)
        self.assertIn('time_budget_exhausted', table['failure_reasons'])
        self.assertIsNone(scalars['lambda_1_est'])

    def test_fake_deadline_interrupts_expansion_and_chunked_cpu_work(self):
        class Clock:
            def __init__(self): self.value = 0.
            def __call__(self):
                self.value += .1
                return self.value

        clock = Clock()
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            scalars, table = ms.estimate_top_spectrum(
                lambda v: v, 8, top_k=3, check_every=2, max_basis=8,
                restart_keep=5, max_products=30, max_seconds=.5, clock=clock)
        self.assertFalse(scalars['accepted'])
        self.assertIn('time_budget_exhausted', table['failure_reasons'])

        checks = 0
        def deadline():
            nonlocal checks
            checks += 1
            if checks == 3:
                raise ms._TimeBudgetExceeded
        q = np.ones((4, 16), np.float32)
        with self.assertRaises(ms._TimeBudgetExceeded):
            ms._incremental_gram(q, 4, chunk=2, check_deadline=deadline)
        checks = 0
        with self.assertRaises(ms._TimeBudgetExceeded):
            ms._transform_in_place(
                q, 4, np.eye(4, 2), chunk=2, check_deadline=deadline)

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_zero_seconds_disables_deadline(self, _memory):
        diagonal = np.arange(8., 0., -1, dtype=np.float32)
        scalars, table = ms.estimate_top_spectrum(
            lambda v: diagonal * v, 8, top_k=4, check_every=2,
            max_basis=8, restart_keep=6, max_products=30,
            residual_tol=1e-4, max_seconds=0, clock=lambda: 123.)
        self.assertTrue(scalars['accepted'], table)

    @mock.patch.object(ms, 'available_host_memory', return_value=10**15)
    def test_nonfinite_product_is_unresolved(self, _memory):
        scalars, table = ms.estimate_top_spectrum(
            lambda v: v * np.nan, 8, top_k=3, max_basis=8,
            restart_keep=5, max_products=20)
        self.assertFalse(scalars['accepted'])
        self.assertEqual(scalars['gn_products'], 1)
        self.assertIn('invalid_operator_product', table['failure_reasons'])
        self.assertEqual(table['values'], [])
        self.assertIsNone(scalars['lambda_1_est'])

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
