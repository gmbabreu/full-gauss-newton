import itertools
import unittest

import jax.numpy as jnp
import numpy as np

from EasyLM import matrix_condition as mc


class MatrixConditionTest(unittest.TestCase):
    def test_identity_maximum(self):
        report = mc.condition_diagnostic(lambda x: x, jnp.ones(5),
            top_maxiter=24, agreement_tol=.01, residual_tol=.01)
        self.assertTrue(report['resolved'])
        self.assertAlmostEqual(report['lambda_max_est'], 1., places=5)

    def test_underbudget_preserves_rayleigh_but_withholds_estimate(self):
        matrix = jnp.diag(jnp.array([1., 100.]))
        report = mc.condition_diagnostic(lambda x: matrix @ x, jnp.ones(2),
            top_maxiter=1, residual_tol=1e-8)
        self.assertFalse(report['resolved'])
        self.assertIsNone(report['lambda_max_est'])
        self.assertIsNotNone(report['top']['rayleigh_quotient'])

    def test_zero_operator_is_unresolved(self):
        report = mc.condition_diagnostic(lambda x: jnp.zeros_like(x), jnp.ones(3))
        self.assertFalse(report['resolved'])
        self.assertIsNone(report['lambda_max_est'])

    def test_power_uses_one_product_per_iteration(self):
        endpoint = mc.power_iteration(lambda x: x, jnp.ones(3), maxiter=8)
        self.assertEqual(endpoint.operator_matvecs, endpoint.iterations * 2)

    def test_identity_dominated_p_uses_b_scale(self):
        b = jnp.diag(jnp.array([1., .001]))
        c = 100.
        report = mc.preconditioned_condition_diagnostic(
            lambda x: b @ x, lambda x: c * x + b @ x, jnp.ones(2), c,
            top_maxiter=100, agreement_tol=.01, residual_tol=.01)
        self.assertTrue(report['resolved'])
        self.assertAlmostEqual(report['lambda_max_est'], 101., delta=.01)
        self.assertAlmostEqual(report['damping_condition_proxy'], 1.01, delta=1e-4)

    def test_lambda_zero_p_special_case(self):
        report = mc.preconditioned_condition_diagnostic(
            lambda x: jnp.zeros_like(x), lambda x: 2 * x, jnp.ones(2), 2.,
            effective_lambda=0.)
        self.assertEqual(report['lambda_max_est'], 2.)
        self.assertEqual(report['damping_condition_proxy'], 1.)
        self.assertEqual(report['operator_matvecs'], 0)

    def test_exhaustive_rademacher_trace_identities_and_concentration(self):
        matrix = np.array([[2., .5], [.5, 3.]])
        traces, squares = [], []
        for signs in itertools.product((-1., 1.), repeat=2):
            z = np.asarray(signs)
            mz = matrix @ z
            traces.append(z @ mz)
            squares.append(mz @ mz)
        summary = mc.summarize_probe_samples(
            traces, squares, 3.20710678, dimension=2)
        self.assertAlmostEqual(summary['trace_est'], np.trace(matrix))
        self.assertAlmostEqual(summary['trace_square_est'], np.trace(matrix @ matrix))
        self.assertAlmostEqual(summary['spectral_concentration_est'],
                               2 * 3.20710678 / 5.)

    def test_reused_az_matches_explicit_damped_matrix(self):
        g = jnp.array([[2., .5], [.5, 3.]])
        diagonal = jnp.array([4., 5.])
        lam, c = .3, .7
        samples = mc.probe_operators(lambda z: g @ z, jnp.ones(2), num_probes=4,
            apply_a_from_g=lambda gz, z: lam * gz + c * diagonal * z)
        self.assertEqual(len(samples['G'][0]), 4)
        self.assertEqual(len(samples['A'][0]), 4)


if __name__ == '__main__':
    unittest.main()
