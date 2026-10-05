"""Focused CPU tests for the damped matrix-free PCG helpers."""
import unittest

from jax import config
config.update('jax_enable_x64', True)

import jax
import jax.numpy as jnp
import numpy as np

from EasyLM import pcg


class DampedPCGTest(unittest.TestCase):
    def test_matrix_free_nonlinear_gauss_newton_matches_dense(self):
        theta = jnp.array([.3, -.7], dtype=jnp.float64)
        vector = jnp.array([.2, 1.1], dtype=jnp.float64)
        mu = .4

        def model(p):
            return jnp.array([jnp.sin(p[0]) + p[1] ** 2,
                              p[0] * jnp.exp(p[1])])

        def loss(output):
            return .5 * output @ jnp.array([[2., .3], [.3, 1.5]]) @ output

        output, jvp = jax.linearize(model, theta)
        transpose = jax.linear_transpose(jvp, theta)
        grad_loss = jax.grad(loss)

        def apply_g(v):
            _, hjv = jax.jvp(grad_loss, (output,), (jvp(v),))
            return transpose(hjv)[0]

        jacobian = jax.jacobian(model)(theta)
        loss_hessian = jax.hessian(loss)(output)
        dense_g = jacobian.T @ loss_hessian @ jacobian
        dense_gradient = jacobian.T @ jax.grad(loss)(output)
        np.testing.assert_allclose(
            pcg.damped_operator(apply_g, mu)(vector),
            (dense_g + mu * jnp.eye(2)) @ vector, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            -dense_gradient, -jax.grad(lambda p: loss(model(p)))(theta),
            rtol=1e-12, atol=1e-12)

    def test_both_preconditioners_converge_to_dense_solution(self):
        matrix = jnp.array([[4., 1., 0.], [1., 2., .25], [0., .25, .5]],
                           dtype=jnp.float64)
        gradient = jnp.array([1., -2., .5], dtype=jnp.float64)
        mu = .3
        operator = pcg.damped_operator(lambda value: matrix @ value, mu)
        adam_diagonal = jnp.array([2., 1.5, .7], dtype=jnp.float64)
        adam_inverse = lambda value: value / adam_diagonal
        estimate = pcg.estimate_gn_diagonal(
            lambda value: matrix @ value, gradient,
            jax.random.PRNGKey(9), 256)
        jacobi_inverse = pcg.gn_jacobi_preconditioner(estimate, mu)
        expected = np.linalg.solve(np.asarray(matrix + mu * jnp.eye(3)),
                                   -np.asarray(gradient))
        for preconditioner in (adam_inverse, jacobi_inverse):
            actual, _ = jax.scipy.sparse.linalg.cg(
                operator, -gradient, M=preconditioner, tol=1e-12,
                atol=1e-12, maxiter=40)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_jacobi_floor_inheritance_and_explicit_override(self):
        estimate = jnp.array([-2., .5, 3.])
        mu = .2
        inherited = pcg.effective_gn_jacobi_floor(mu, 0)
        explicit = pcg.effective_gn_jacobi_floor(mu, .7)
        np.testing.assert_allclose(
            pcg.gn_jacobi_preconditioner(estimate, inherited)(jnp.ones(3)),
            1 / (jnp.maximum(estimate, 0) + mu))
        np.testing.assert_allclose(
            pcg.gn_jacobi_preconditioner(estimate, explicit)(jnp.ones(3)),
            1 / (jnp.maximum(estimate, 0) + .7))

    def test_floor_does_not_change_system_rhs_residuals_or_solution(self):
        matrix = jnp.array([[5., 1.], [1., 1.]], dtype=jnp.float64)
        gradient = jnp.array([1.5, -.25], dtype=jnp.float64)
        mu = .1
        operator = pcg.damped_operator(lambda value: matrix @ value, mu)
        rhs = -gradient
        exact = np.linalg.solve(
            np.asarray(matrix + mu * jnp.eye(2)), np.asarray(rhs))
        probe = jnp.array([.3, -.8], dtype=jnp.float64)
        operator_value = operator(probe)
        residuals = pcg.damped_residuals(
            lambda value: matrix @ value, probe, gradient, mu)

        for floor in (.02, 3.0):
            preconditioner = pcg.gn_jacobi_preconditioner(
                jnp.diag(matrix), floor)
            actual, _ = jax.scipy.sparse.linalg.cg(
                operator, rhs, M=preconditioner, tol=1e-12,
                atol=1e-12, maxiter=40)
            np.testing.assert_allclose(actual, exact, rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(operator(probe), operator_value)
            np.testing.assert_array_equal(rhs, -gradient)
            for actual_residual, expected_residual in zip(
                    pcg.damped_residuals(
                        lambda value: matrix @ value, probe, gradient, mu),
                    residuals):
                np.testing.assert_array_equal(actual_residual,
                                              expected_residual)

    def test_damped_and_raw_residuals_share_one_g_product(self):
        matrix = jnp.array([[3., 1.], [1., 2.]], dtype=jnp.float64)
        gradient = jnp.array([.5, -1.25], dtype=jnp.float64)
        solution = jnp.array([.2, -.4], dtype=jnp.float64)
        mu = .3
        calls = []

        def apply_g(value):
            calls.append(1)
            return matrix @ value

        damped, raw = pcg.damped_residuals(
            apply_g, solution, gradient, mu)
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(raw, matrix @ solution + gradient,
                                   rtol=0, atol=0)
        np.testing.assert_allclose(
            damped, matrix @ solution + mu * solution + gradient,
            rtol=0, atol=0)
        expected_relative = (jnp.linalg.norm(matrix @ solution + gradient)
                             / (jnp.linalg.norm(gradient) + 1e-12))
        np.testing.assert_allclose(
            jnp.linalg.norm(raw) / (jnp.linalg.norm(gradient) + 1e-12),
            expected_relative, rtol=1e-14)

        exact = jnp.linalg.solve(matrix + mu * jnp.eye(2), -gradient)
        exact_damped, exact_raw = pcg.damped_residuals(
            lambda value: matrix @ value, exact, gradient, mu)
        self.assertLess(float(jnp.linalg.norm(exact_damped)), 1e-12)
        np.testing.assert_allclose(exact_raw, -mu * exact,
                                   rtol=1e-12, atol=1e-12)

    def test_diagonal_estimator_exact_preconditioner_and_deterministic(self):
        diagonal = jnp.array([.25, 2., 5.], dtype=jnp.float64)
        key = jax.random.PRNGKey(17)
        estimate = pcg.estimate_gn_diagonal(
            lambda value: diagonal * value, diagonal, key, 3)
        repeated = pcg.estimate_gn_diagonal(
            lambda value: diagonal * value, diagonal, key, 3)
        np.testing.assert_array_equal(estimate, repeated)
        np.testing.assert_allclose(estimate, diagonal, rtol=0, atol=0)
        inverse = pcg.gn_jacobi_preconditioner(estimate, .5)
        np.testing.assert_allclose(inverse(jnp.ones(3)), 1 / (diagonal + .5),
                                   rtol=1e-7)
        self.assertEqual(estimate.dtype, jnp.float32)

    def test_diagonal_estimator_jits_over_parameter_pytree(self):
        diagonal = {'a': jnp.array([1., 3.]), 'b': jnp.array([[2.]])}

        def apply_g(value):
            return jax.tree.map(lambda d, v: d * v, diagonal, value)

        estimate = jax.jit(lambda key: pcg.estimate_gn_diagonal(
            apply_g, diagonal, key, 2))(jax.random.PRNGKey(4))
        for actual, expected in zip(jax.tree.leaves(estimate),
                                    jax.tree.leaves(diagonal)):
            np.testing.assert_allclose(actual, expected)

    def test_damping_is_added_once_after_microbatch_sum(self):
        blocks = (jnp.array([[2., 0.], [0., 1.]]),
                  jnp.array([[4., 1.], [1., 3.]]))
        vector = jnp.array([.5, -1.])
        mu = .7
        full_g = sum(block @ vector for block in blocks) / len(blocks)
        one = pcg.damped_operator(lambda value: sum(
            block @ value for block in blocks) / len(blocks), mu)(vector)
        many = pcg.damped_operator(lambda value: sum(
            .5 * (block @ value) for block in blocks), mu)(vector)
        expected = full_g + mu * vector
        np.testing.assert_allclose(one, expected)
        np.testing.assert_allclose(many, expected)
        self.assertFalse(np.allclose(many, full_g + len(blocks) * mu * vector))

    def test_legacy_operator_rhs_preconditioner_and_solution_unchanged(self):
        matrix = jnp.array([[3., .2], [.2, 1.]])
        moment = jnp.array([.4, -1.])
        second = jnp.array([.25, 4.])
        beta1_correction, beta2_correction = .1, .2
        learning_rate, interpolation = .01, .35
        epsilon = 1e-8
        diagonal = jnp.sqrt(second / beta2_correction) + epsilon
        rhs = -moment / beta1_correction

        def legacy(value):
            return (interpolation * matrix @ value
                    + (1 - interpolation) / learning_rate * diagonal * value)

        preconditioner = lambda value: value / diagonal
        before, _ = jax.scipy.sparse.linalg.cg(
            legacy, rhs, M=preconditioner, tol=1e-10, maxiter=20)
        # cg_damping_mu == 0 selects this exact legacy callable in production.
        selected = legacy
        after, _ = jax.scipy.sparse.linalg.cg(
            selected, rhs, M=preconditioner, tol=1e-10, maxiter=20)
        np.testing.assert_array_equal(selected(jnp.ones(2)), legacy(jnp.ones(2)))
        np.testing.assert_array_equal(rhs, -moment / beta1_correction)
        np.testing.assert_array_equal(preconditioner(rhs), rhs / diagonal)
        np.testing.assert_array_equal(after, before)

    def test_flag_validation(self):
        valid = dict(interpolation_lambda=1, lambda_batch_denominator=0,
                     lambda_final=-1, lambda_ramp_steps=0, adam_b1=0,
                     condition_log=False)
        pcg.validate_damped_pcg(.1, 'adam_diag', 1, 0, **valid)
        pcg.validate_damped_pcg(.1, 'gn_jacobi', 1, 0, **valid)
        pcg.validate_damped_pcg(.1, 'gn_jacobi', 1, .03, **valid)
        invalid = [
            (dict(damping_mu=-1), 'finite and nonnegative'),
            (dict(damping_mu=float('nan')), 'finite and nonnegative'),
            (dict(preconditioner='identity'), 'must be one of'),
            (dict(probes=0), 'positive integer'),
            (dict(probes=1.5), 'positive integer'),
            (dict(jacobi_floor=-1), 'finite and nonnegative'),
            (dict(jacobi_floor=float('inf')), 'finite and nonnegative'),
            (dict(jacobi_floor=.1), 'requires gn_jacobi'),
            (dict(interpolation_lambda=.9), 'fixed pure-GN'),
            (dict(lambda_batch_denominator=10), 'fixed pure-GN'),
            (dict(lambda_final=.5), 'fixed pure-GN'),
            (dict(lambda_ramp_steps=2), 'fixed pure-GN'),
            (dict(adam_b1=.9), 'b1 == 0'),
            (dict(condition_log=True), 'condition_log'),
        ]
        base = dict(damping_mu=.1, preconditioner='adam_diag', probes=1,
                    jacobi_floor=0,
                    **valid)
        for changes, message in invalid:
            arguments = dict(base, **changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(
                    ValueError, message):
                pcg.validate_damped_pcg(
                    arguments.pop('damping_mu'),
                    arguments.pop('preconditioner'), arguments.pop('probes'),
                    arguments.pop('jacobi_floor'),
                    **arguments)
        with self.assertRaisesRegex(ValueError, 'requires cg_damping_mu > 0'):
            pcg.validate_damped_pcg(
                0, 'gn_jacobi', 1, .1, **dict(valid, adam_b1=.9))


if __name__ == '__main__':
    unittest.main()
