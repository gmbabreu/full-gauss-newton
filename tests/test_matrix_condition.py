import itertools
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from EasyLM import matrix_condition as mc


class MatrixConditionTest(unittest.TestCase):
    def options(self, **updates):
        options = dict(top_maxiter=100, inverse_maxiter=20, inner_maxiter=100,
                       inner_tol=1e-6, agreement_tol=.01, residual_tol=.01)
        options.update(updates)
        return options

    def test_identity_has_condition_one_without_spurious_large_bound(self):
        report = mc.condition_diagnostic(lambda x: x, jnp.ones(5), **self.options())
        self.assertTrue(report['resolved'])
        self.assertAlmostEqual(report['condition_est'], 1., places=5)
        self.assertLess(report['condition_rayleigh_lower_bound_est'], 1.01)

    def test_spd_and_symmetric_preconditioning(self):
        matrix = jnp.diag(jnp.array([1., 10., 100.]))
        diagonal = jnp.array([1., 2., 4.])
        raw = mc.condition_diagnostic(lambda x: matrix @ x, jnp.ones(3),
                                      **self.options())
        preconditioned = mc.condition_diagnostic(mc.symmetric_diagonal_operator(
            lambda x: matrix @ x, diagonal), jnp.ones(3), **self.options())
        self.assertAlmostEqual(raw['condition_est'], 100., delta=1.)
        self.assertAlmostEqual(preconditioned['condition_est'], 25., delta=.3)

    def test_underbudget_is_unresolved_not_singular(self):
        matrix = jnp.diag(jnp.array([1., 1e4]))
        report = mc.condition_diagnostic(lambda x: matrix @ x, jnp.ones(2),
            **self.options(inverse_maxiter=3, inner_maxiter=1, inner_tol=1e-10))
        self.assertFalse(report['resolved'])
        self.assertIsNone(report['condition_est'])
        self.assertNotIn('numerically_singular', report)

    def test_singular_and_zero_operators_do_not_fabricate_condition(self):
        for matrix in (jnp.diag(jnp.array([0., 1., 2.])), jnp.zeros((3, 3))):
            report = mc.gauss_newton_diagnostics(
                lambda x, m=matrix: m @ x, jnp.ones(3), **self.options())
            self.assertFalse(report['G']['resolved'])
            self.assertIsNone(report['G']['condition_est'])

    def test_rayleigh_agreement_does_not_stop_with_large_residual(self):
        matrix = jnp.diag(jnp.array([1., .999, .5]))
        endpoint = mc.power_iteration(lambda x: matrix @ x, jnp.ones(3),
            maxiter=8, agreement_tol=.1, residual_tol=1e-10)
        self.assertEqual(endpoint.iterations, 8)
        self.assertFalse(endpoint.resolved)

    def test_calibration_does_extra_work_and_failure_is_preserved(self):
        matrix = jnp.diag(jnp.array([1., 2., 10.]))
        plain = mc.condition_diagnostic(lambda x: matrix @ x, jnp.ones(3),
                                        **self.options(inverse_maxiter=4))
        calibrated = mc.condition_diagnostic(lambda x: matrix @ x, jnp.ones(3),
            **self.options(inverse_maxiter=4, calibrate=True, residual_tol=1e-12))
        self.assertGreater(calibrated['operator_matvecs'], plain['operator_matvecs'])
        self.assertFalse(calibrated['resolved'])
        self.assertIn('calibration_failed', calibrated['failure_reasons'])

    def test_product_count_includes_bound_rayleigh_products(self):
        report = mc.condition_diagnostic(lambda x: x, jnp.ones(3), **self.options())
        self.assertEqual(report['operator_matvecs'],
            report['top']['operator_matvecs']
            + report['bottom']['operator_matvecs'] + 2)

    def test_reversed_endpoints_are_rejected(self):
        top = mc.Endpoint(1., 0., True, 3, 1, rayleigh_agreement=True,
                          starts_agree=True, vector=jnp.ones(2))
        bottom = mc.Endpoint(2., 0., True, 3, 1, rayleigh_agreement=True,
                             starts_agree=True, vector=jnp.ones(2))
        with mock.patch.object(mc, 'power_iteration', return_value=top), \
             mock.patch.object(mc, 'inverse_iteration', return_value=bottom):
            report = mc.condition_diagnostic(lambda x: x, jnp.ones(2))
        self.assertFalse(report['resolved'])
        self.assertIsNone(report['condition_est'])
        self.assertIn('endpoint_order', report['failure_reasons'])

    def test_calibration_status_is_independent_per_operator(self):
        calls = []
        def fake_condition(_operator, _template, **kwargs):
            calls.append(kwargs)
            return dict(lambda_max_est=1., operator_matvecs=0, inner_solves=0,
                        calibration_attempted=kwargs['calibrate'],
                        calibration_status=('passed' if kwargs['calibrate']
                                            else 'previously_passed'))
        with mock.patch.object(mc, 'condition_diagnostic', side_effect=fake_condition):
            result = mc.gauss_newton_diagnostics(lambda x: x, jnp.ones(2),
                shifts=(1e-2, 1e-4),
                calibrate_by_operator={'G': True, 'condition_shift_1e-2': True})
        self.assertEqual(result['G']['calibration_status'], 'previously_passed')
        self.assertEqual(result['shifted']['condition_shift_1e-2']
                         ['calibration_status'], 'previously_passed')
        self.assertTrue(result['shifted']['condition_shift_1e-4']
                        ['calibration_attempted'])

    def test_condition_reuses_one_compiled_pcg_callable(self):
        with mock.patch.object(jax, 'jit', wraps=jax.jit) as jit:
            mc.condition_diagnostic(lambda x: x, jnp.ones(3), **self.options())
        self.assertEqual(jit.call_count, 1)

    def test_exhaustive_rademacher_trace_identities(self):
        matrix = np.array([[2., .5], [.5, 3.]])
        traces, squares = [], []
        for signs in itertools.product((-1., 1.), repeat=2):
            z = np.asarray(signs)
            mz = matrix @ z
            traces.append(z @ mz)
            squares.append(mz @ mz)
        summary = mc.summarize_probe_samples(traces, squares, 3.20710678)
        self.assertAlmostEqual(summary['trace_est'], np.trace(matrix))
        self.assertAlmostEqual(summary['trace_square_est'], np.trace(matrix @ matrix))

    def test_reused_az_matches_explicit_damped_matrix(self):
        g = jnp.array([[2., .5], [.5, 3.]])
        diagonal = jnp.array([4., 5.])
        lam, c = .3, .7
        samples = mc.probe_operators(lambda z: g @ z, jnp.ones(2), num_probes=4,
            apply_a_from_g=lambda gz, z: lam * gz + c * diagonal * z)
        # The paired A samples use exactly the explicit A action for the same z.
        key = jax.random.PRNGKey(0)
        explicit = []
        a = lam * g + c * jnp.diag(diagonal)
        for index in range(4):
            z = mc.random_rademacher_pytree(jax.random.fold_in(key, index), jnp.ones(2))
            explicit.append(float(mc.tree_dot(z, a @ z)))
        np.testing.assert_allclose(samples['A'][0], explicit)


if __name__ == '__main__':
    unittest.main()
