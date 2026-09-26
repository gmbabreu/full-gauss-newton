"""Host orchestration for observational spectrum diagnostics.

Numerical estimators live in matrix_condition and matrix_spectrum. This
controller owns their compiled-solver cache; changing parameters, batches,
diagonals and interpolation settings are passed to those solvers at runtime.
"""
import timeit

import jax
import jax.numpy as jnp
import numpy as np

from EasyLM import matrix_condition, matrix_spectrum


class ConditionDiagnostics:
    """Reusable controller with explicit trainer-provided dependencies."""

    def __init__(self, *, config, apply_g, param_shards):
        self.config = config
        self.apply_g = apply_g
        self.param_shards = param_shards
        self._power_solvers = {}

    def run(self, params, solve_batch, *, step, cg_diagonal=None,
            effective_lambda=None, safe_adam_lr=None):
        """Host controller over explicitly sharded immutable Gv kernels."""
        FLAGS = self.config
        sharded_condition_apply_g = self.apply_g
        diagnostic_param_shards = self.param_shards
        condition_power_solvers = self._power_solvers
        started = timeit.default_timer()
        key = jax.random.fold_in(jax.random.PRNGKey(0x434f4e44), step)
        def apply_g(vector, diagnostic_params, diagnostic_batch, *unused):
            return sharded_condition_apply_g(
                diagnostic_params, diagnostic_batch, vector,
                FLAGS.inner_loop_wd)
        g_operator_args = (params, solve_batch)
        metrics_out = {}
        spectrum_max = None
        spectrum_started = timeit.default_timer()
        spectrum_transfer_seconds = dict(upload=0., compute=0., download=0.)
        leaves, structure = jax.tree.flatten(params)
        shapes = [leaf.shape for leaf in leaves]
        sizes = [leaf.size for leaf in leaves]
        dimension = sum(sizes)
        def cpu_apply(flat_vector):
            boundary = timeit.default_timer()
            offset, vector_leaves = 0, []
            for shape, size, leaf in zip(shapes, sizes, leaves):
                vector_leaves.append(flat_vector[offset:offset + size].reshape(
                    shape).astype(np.float32, copy=False))
                offset += size
            vector_tree = jax.tree.unflatten(structure, vector_leaves)
            vector_tree = jax.tree.map(
                lambda value, shard: shard(value),
                vector_tree, diagnostic_param_shards)
            jax.block_until_ready(vector_tree)
            spectrum_transfer_seconds['upload'] += (
                timeit.default_timer() - boundary)
            boundary = timeit.default_timer()
            product = sharded_condition_apply_g(
                params, solve_batch, vector_tree, FLAGS.inner_loop_wd)
            jax.block_until_ready(product)
            spectrum_transfer_seconds['compute'] += (
                timeit.default_timer() - boundary)
            boundary = timeit.default_timer()
            host = jax.device_get(product)
            result = np.concatenate([
                np.asarray(leaf, np.float32).reshape(-1)
                for leaf in jax.tree.leaves(host)])
            spectrum_transfer_seconds['download'] += (
                timeit.default_timer() - boundary)
            del vector_tree, product, host
            return result
        print(f'[spectrum] G top-{FLAGS.spectrum_top_k}: start', flush=True)
        spectrum_scalars, spectrum_table = matrix_spectrum.estimate_top_spectrum(
            cpu_apply, dimension, top_k=FLAGS.spectrum_top_k,
            check_every=FLAGS.spectrum_check_every,
            max_basis=FLAGS.spectrum_max_basis,
            restart_keep=FLAGS.spectrum_restart_keep,
            max_products=FLAGS.spectrum_max_gn_products,
            residual_tol=FLAGS.spectrum_residual_tol,
            stability_tol=FLAGS.spectrum_stability_tol,
            seed=FLAGS.spectrum_seed,
            progress=lambda done, total: print(
                f'[spectrum] G products {done}/{total}; '
                f'elapsed={timeit.default_timer() - spectrum_started:.1f}s, '
                f'upload={spectrum_transfer_seconds["upload"]:.1f}s, '
                f'compute={spectrum_transfer_seconds["compute"]:.1f}s, '
                f'download={spectrum_transfer_seconds["download"]:.1f}s',
                flush=True))
        spectrum_scalars.update({
            f'seconds_{name}': value
            for name, value in spectrum_transfer_seconds.items()})
        if spectrum_scalars['accepted']:
            spectrum_max = spectrum_scalars['lambda_1_est']
        print(f'[spectrum] G top-{FLAGS.spectrum_top_k}: end ' +
              str(spectrum_scalars), flush=True)
        options = dict(endpoint_maxiter=FLAGS.spectrum_endpoint_maxiter,
            num_starts=FLAGS.spectrum_endpoint_num_starts,
            agreement_tol=FLAGS.spectrum_endpoint_agreement_tol,
            residual_tol=FLAGS.spectrum_endpoint_residual_tol, key=key)
        if spectrum_max is None:
            if 'G' not in condition_power_solvers:
                condition_power_solvers['G'] = matrix_condition.make_power_solver(
                    apply_g, maxiter=FLAGS.spectrum_endpoint_maxiter,
                    agreement_tol=FLAGS.spectrum_endpoint_agreement_tol,
                    residual_tol=FLAGS.spectrum_endpoint_residual_tol)
            print('[spectrum] G fallback maximum: start', flush=True)
            g_report = matrix_condition.condition_diagnostic(apply_g, params,
                compiled_power_solver=condition_power_solvers['G'],
                operator_args=g_operator_args, **options)
            print('[spectrum] G fallback maximum: end ' + str({name: g_report[name]
                for name in ('lambda_max_est', 'resolved', 'failure_reasons')}),
                flush=True)
        else:
            g_report = None
        total_products = (spectrum_scalars['gn_products']
                          + (g_report['operator_matvecs']
                             if g_report is not None else 0))
        direct_residuals = spectrum_table['direct_residuals']
        g_metrics = {
            'accepted': spectrum_scalars['accepted'],
            'gn_products': total_products,
            'seconds': timeit.default_timer() - spectrum_started,
            'basis_size': spectrum_scalars['basis_size'],
            'orthogonality_error': spectrum_scalars['orthogonality_error'],
            'max_relative_ritz_residual':
                spectrum_scalars['max_relative_ritz_residual'],
            'max_direct_residual': (max(direct_residuals.values())
                                    if direct_residuals else None),
            'top10_condition_est': spectrum_scalars['top10_condition_est'],
            'top100_condition_est': spectrum_scalars['top100_condition_est'],
        }
        g_metrics.update({name: value for name, value in spectrum_scalars.items()
                          if name.startswith('lambda_') and value is not None})
        if not spectrum_scalars['accepted']:
            g_metrics['failure_reasons'] = ','.join(
                spectrum_table['failure_reasons'])
            g_metrics.update({
                'fallback_lambda_max_est': g_report['lambda_max_est'],
                'fallback_lambda_max_residual':
                    g_report['lambda_max_residual'],
                'fallback_resolved': g_report['resolved'],
            })
            if not g_report['resolved']:
                g_metrics['fallback_failure_reasons'] = ','.join(
                    g_report['failure_reasons'])
        metrics_out.update({f'spectrum/G/{name}': value
                            for name, value in g_metrics.items()
                            if value is not None})

        if cg_diagonal is not None:
            a_started = timeit.default_timer()
            c = (1.0 - effective_lambda) / safe_adam_lr
            def apply_a(vector, diagnostic_params, diagnostic_batch,
                        diagonal, interpolation, learning_rate):
                coefficient = (1.0 - interpolation) / learning_rate
                gv = apply_g(vector, diagnostic_params, diagnostic_batch)
                return jax.tree.map(
                    lambda g, v, d: interpolation * g + coefficient * d * v,
                    gv, vector, diagonal)
            def apply_b(vector, diagnostic_params, diagnostic_batch,
                        diagonal, interpolation, _learning_rate):
                scaled = jax.tree.map(
                    lambda value, d: value / jnp.sqrt(d), vector, diagonal)
                g_scaled = apply_g(
                    scaled, diagnostic_params, diagnostic_batch)
                return jax.tree.map(
                    lambda value, d: interpolation * value / jnp.sqrt(d),
                    g_scaled, diagonal)
            a_operator_args = (params, solve_batch, cg_diagonal,
                               effective_lambda, safe_adam_lr)
            apply_p = matrix_condition.symmetric_diagonal_operator(
                apply_a, lambda *args: args[2])
            if 'A' not in condition_power_solvers:
                condition_power_solvers['A'] = matrix_condition.make_power_solver(
                    apply_a, maxiter=FLAGS.spectrum_endpoint_maxiter,
                    agreement_tol=FLAGS.spectrum_endpoint_agreement_tol,
                    residual_tol=FLAGS.spectrum_endpoint_residual_tol)
                condition_power_solvers['A_inverse'] = matrix_condition.make_inverse_solver(
                    apply_a, maxiter=FLAGS.spectrum_endpoint_maxiter,
                    inner_maxiter=FLAGS.spectrum_inverse_cg_maxiter,
                    inner_tol=FLAGS.spectrum_inverse_cg_tol,
                    agreement_tol=FLAGS.spectrum_endpoint_agreement_tol,
                    residual_tol=FLAGS.spectrum_endpoint_residual_tol,
                    preconditioner=lambda v, _params, _batch, d, *_: jax.tree.map(
                        lambda value, diagonal: value / diagonal, v, d))
            if 'A_preconditioned_B' not in condition_power_solvers:
                condition_power_solvers['A_preconditioned_B'] = \
                    matrix_condition.make_power_solver(
                        apply_b, maxiter=FLAGS.spectrum_endpoint_maxiter,
                        agreement_tol=FLAGS.spectrum_endpoint_agreement_tol,
                        residual_tol=FLAGS.spectrum_endpoint_residual_tol)
            print('[spectrum] A: start', flush=True)
            a_report = matrix_condition.damped_condition_diagnostic(apply_a, params,
                operator_args=a_operator_args,
                compiled_power_solver=condition_power_solvers['A'],
                compiled_inverse_solver=condition_power_solvers['A_inverse'], **options)
            print('[spectrum] A: end ' + str({name: a_report[name]
                for name in ('lambda_max_est', 'lambda_min_est',
                             'condition_est', 'resolved', 'failure_reasons')}), flush=True)
            print('[spectrum] A preconditioned: start', flush=True)
            p_report = matrix_condition.preconditioned_condition_diagnostic(
                apply_b, apply_p, params, c,
                operator_args=a_operator_args,
                compiled_power_solver=condition_power_solvers[
                    'A_preconditioned_B'],
                effective_lambda=effective_lambda, **options)
            print('[spectrum] A preconditioned: end ' + str({name: p_report[name]
                for name in ('lambda_max_est', 'resolved')}), flush=True)
            a_products = (a_report['operator_matvecs']
                          + p_report['operator_matvecs'])
            a_metrics = {
                name: a_report.get(name) for name in (
                    'lambda_max_est', 'lambda_max_residual',
                    'lambda_min_est', 'lambda_min_residual',
                    'condition_est', 'resolved')}
            a_metrics.update({
                'preconditioned_lambda_max_est': p_report['lambda_max_est'],
                'preconditioned_lambda_max_residual':
                    p_report['lambda_max_residual'],
                'preconditioned_damping_condition_proxy':
                    p_report['damping_condition_proxy'],
                'preconditioned_resolved': p_report['resolved'],
                'gn_products': a_products,
                'seconds': timeit.default_timer() - a_started,
            })
            if not a_report['resolved']:
                a_metrics['failure_reasons'] = ','.join(
                    a_report['failure_reasons'])
            if not p_report['resolved']:
                a_metrics['preconditioned_failure_reasons'] = ','.join(
                    p_report['failure_reasons'])
            metrics_out.update({f'spectrum/A/{name}': value
                                for name, value in a_metrics.items()
                                if value is not None})
            total_products += a_products

        print('[spectrum] total: ' + str({
            'seconds': timeit.default_timer() - started,
            'gn_products': total_products}), flush=True)
        return metrics_out
