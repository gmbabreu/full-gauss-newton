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
                max_basis=5, restart_keep=4, max_products=30,
                residual_tol=.05, stability_tol=.05, seed=3)
        self.assertGreaterEqual(len(table['values']), 3)
        self.assertTrue(all(a >= b for a, b in zip(
            table['values'], table['values'][1:])))
        self.assertLessEqual(scalars['gn_products'], 30)


if __name__ == '__main__':
    unittest.main()
