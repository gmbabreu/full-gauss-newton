import unittest

import jax
import jax.numpy as jnp
import numpy as np

from EasyLM import matrix_condition as mc


class MatrixConditionTest(unittest.TestCase):
    def test_identity_maximum(self):
        report = mc.condition_diagnostic(lambda x: x, jnp.ones(5),
            endpoint_maxiter=24, agreement_tol=.01, residual_tol=.01)
        self.assertTrue(report['resolved'])
        self.assertAlmostEqual(report['lambda_max_est'], 1., places=5)

    def test_underbudget_preserves_rayleigh_but_withholds_estimate(self):
        matrix = jnp.diag(jnp.array([1., 100.]))
        report = mc.condition_diagnostic(lambda x: matrix @ x, jnp.ones(2),
            endpoint_maxiter=1, residual_tol=1e-8)
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
            endpoint_maxiter=100, agreement_tol=.01, residual_tol=.01)
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

    def test_damped_endpoints_against_independent_jax_jacobian(self):
        weights = jnp.asarray(np.random.default_rng(14).normal(size=(8, 5)), jnp.float32)
        params = jnp.linspace(-.1, .1, 5)
        model = lambda p: jnp.tanh(weights @ p)
        jacobian = np.asarray(jax.jacfwd(model)(params), dtype=np.float64)
        diagonal = jnp.array([.5, 1., 2., 3., 4.])
        lam, eta = .3, .7
        matrix = lam * jacobian.T @ jacobian + (1-lam)/eta * np.diag(diagonal)
        def apply(v):
            _, jv = jax.jvp(model, (params,), (v,))
            gv = jax.vjp(model, params)[1](jv)[0]
            return lam * gv + (1-lam)/eta * diagonal * v
        report = mc.damped_condition_diagnostic(apply, params,
            preconditioner=lambda v: v / diagonal, endpoint_maxiter=150,
            inverse_cg_maxiter=20, inverse_cg_tol=1e-5,
            agreement_tol=1e-4, residual_tol=1e-4)
        truth = np.linalg.eigvalsh(matrix)
        self.assertTrue(report['resolved'], report)
        np.testing.assert_allclose(
            [report['lambda_min_est'], report['lambda_max_est']],
            truth[[0, -1]], rtol=2e-5, atol=1e-6)
        self.assertAlmostEqual(report['condition_est'], truth[-1]/truth[0], delta=1e-4)

    def test_inverse_counts_actual_products_including_solve_checks(self):
        calls = []
        def apply(v):
            jax.debug.callback(lambda: calls.append(1), ordered=True)
            return 2 * v
        report = mc.damped_condition_diagnostic(apply, jnp.ones(3))
        jax.effects_barrier()
        self.assertTrue(report['resolved'], report)
        self.assertEqual(report['lambda_min_est'], 2.)
        self.assertEqual(report['condition_est'], 1.)
        self.assertEqual(report['operator_matvecs'], len(calls))
        # Two starts: three maximum products and three (solve + check) pairs each.
        self.assertEqual(report['operator_matvecs'], 18)

    def test_inverse_underbudget_withholds_minimum_and_ratio(self):
        matrix = jnp.diag(jnp.array([1., 3., 20., 100.]))
        report = mc.damped_condition_diagnostic(lambda v: matrix @ v, jnp.ones(4),
            endpoint_maxiter=100, inverse_cg_maxiter=1, inverse_cg_tol=1e-6)
        self.assertIsNotNone(report['lambda_max_est'])
        self.assertIsNone(report['lambda_min_est'])
        self.assertIsNone(report['condition_est'])
        self.assertFalse(report['resolved'])
        self.assertIn('minimum_outer_not_converged', report['failure_reasons'])

    def test_singular_operator_does_not_publish_positive_minimum(self):
        diagonal = jnp.array([0., 1., 10.])
        report = mc.damped_condition_diagnostic(lambda v: diagonal * v, jnp.ones(3),
            endpoint_maxiter=100, inverse_cg_maxiter=20)
        self.assertIsNone(report['lambda_min_est'])
        self.assertIsNone(report['condition_est'])
        self.assertFalse(report['resolved'])

if __name__ == '__main__':
    unittest.main()
