import unittest
from unittest import mock

import numpy as np

from EasyLM import matrix_spectrum as ms


class MatrixSpectrumTest(unittest.TestCase):
    def test_memory_preflight_counts_two_basis_buffers(self):
        with mock.patch.object(ms, 'available_host_memory', return_value=10**15):
            report = ms.memory_preflight(1000, 160, 4, reserve_bytes=0)
        self.assertEqual(report['basis_buffer_bytes'], 2 * 160 * 1000 * 4)

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
        # A repeated leading eigenspace requires more independent directions
        # than fit alongside the trailing space in the 160-vector basis.
        diagonal = np.ones(192, np.float32)
        diagonal[:100] = 2.
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
        np.testing.assert_allclose(table['values'][:100], 2., rtol=.01, atol=.01)
        self.assertGreater(table['restart_count'], 0)
        self.assertEqual(calls, scalars['gn_products'])
        self.assertLessEqual(calls, 600)
        self.assertAlmostEqual(scalars['top100_condition_est'], 1., delta=.01)


if __name__ == '__main__':
    unittest.main()
