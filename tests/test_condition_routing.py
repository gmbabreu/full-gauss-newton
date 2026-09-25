"""Exercise the diagnostic controller directly on small CPU operators."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from EasyLM import matrix_condition, matrix_spectrum, cg_resume
from EasyLM.condition_diagnostics import ConditionDiagnostics

TRAINER = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'


class ConditionRoutingTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.g = jnp.array([10., 7., 4., 3., 2., 1., .8, .6, .5, .4, .3, .2])
        def apply_g(params, batch, vector, wd):
            jax.debug.callback(lambda: self.calls.append(1), ordered=True)
            return {'w': self.g * vector['w']}
        flags = SimpleNamespace(inner_loop_wd=0., spectrum_top_k=2,
            spectrum_check_every=2, spectrum_max_basis=8, spectrum_restart_keep=4,
            spectrum_max_gn_products=100, spectrum_residual_tol=.01,
            spectrum_stability_tol=.02, spectrum_seed=0,
            spectrum_endpoint_maxiter=150, spectrum_endpoint_num_starts=2,
            spectrum_endpoint_agreement_tol=1e-4,
            spectrum_endpoint_residual_tol=1e-4,
            spectrum_inverse_cg_maxiter=24, spectrum_inverse_cg_tol=1e-5)
        self.diagnostics = ConditionDiagnostics(
            config=flags, apply_g=apply_g, param_shards={'w': jnp.asarray})
        self.params = {'w': jnp.ones(12)}

    def run_controller(self, **kwargs):
        with redirect_stdout(io.StringIO()), patch.object(
                matrix_spectrum, 'available_host_memory', return_value=10**15):
            result = self.diagnostics.run(self.params, {}, step=0, **kwargs)
        jax.effects_barrier()
        self.assertFalse(any(key.startswith('condition/') for key in result))
        return result

    def test_non_cg_reuses_spectrum_maximum_without_damped_work(self):
        with patch.object(matrix_condition, 'condition_diagnostic',
                          side_effect=AssertionError('duplicate G maximum')):
            result = self.run_controller()
        self.assertTrue(result['spectrum/G/accepted'])
        self.assertAlmostEqual(result['spectrum/G/lambda_1_est'], 10., delta=.01)
        self.assertFalse(any(k.startswith('spectrum/A/') for k in result))
        self.assertEqual(result['spectrum/G/gn_products'], len(self.calls))
        self.assertEqual(set(result), {
            'spectrum/G/accepted', 'spectrum/G/gn_products',
            'spectrum/G/seconds', 'spectrum/G/basis_size',
            'spectrum/G/orthogonality_error',
            'spectrum/G/max_relative_ritz_residual',
            'spectrum/G/max_direct_residual', 'spectrum/G/lambda_1_est',
            'spectrum/G/lambda_2_est'})

    def test_cg_damped_endpoints_and_compiled_reuse_with_new_diagonal(self):
        for scale in (1., 2.):
            self.calls.clear()
            diagonal = jnp.linspace(1., 2., 12).at[-1].set(.2) * scale
            result = self.run_controller(cg_diagonal={'w': diagonal},
                                         effective_lambda=.3, safe_adam_lr=.7)
            expected = .3 * np.asarray(self.g) + np.asarray(diagonal)
            self.assertTrue(result['spectrum/A/resolved'], result)
            np.testing.assert_allclose(
                [result['spectrum/A/lambda_min_est'],
                 result['spectrum/A/lambda_max_est']],
                [expected.min(), expected.max()], rtol=1e-4)
            self.assertAlmostEqual(result['spectrum/A/condition_est'],
                                   expected.max()/expected.min(), delta=1e-3)
            self.assertIn('spectrum/A/preconditioned_lambda_max_est', result)
            self.assertEqual(
                result['spectrum/G/gn_products']
                + result['spectrum/A/gn_products'], len(self.calls))

    def test_failed_spectrum_falls_back_for_g_maximum(self):
        self.diagnostics.config.spectrum_max_gn_products = 5
        result = self.run_controller()
        self.assertFalse(result['spectrum/G/accepted'])
        self.assertTrue(result['spectrum/G/fallback_resolved'])
        self.assertAlmostEqual(
            result['spectrum/G/fallback_lambda_max_est'], 10., delta=.01)
        self.assertEqual(result['spectrum/G/gn_products'], len(self.calls))
        self.assertIn('spectrum/G/failure_reasons', result)

    def test_cached_solvers_use_current_params_batch_and_step(self):
        def apply_g(params, batch, vector, wd):
            return {'w': (params['w'] + batch['shift']) * vector['w']}

        self.diagnostics.config.spectrum_max_gn_products = 5
        controller = ConditionDiagnostics(
            config=self.diagnostics.config, apply_g=apply_g,
            param_shards={'w': jnp.asarray})
        cached = None
        for step in (0, 17):
            params = {'w': self.g * (1. if step == 0 else 2.)}
            batch = {'shift': jnp.float32(step / 17)}
            diagonal = {'w': jnp.linspace(.2, 2., 12)}
            kwargs = dict(step=step, cg_diagonal=diagonal,
                          effective_lambda=.3, safe_adam_lr=.7)
            fresh = ConditionDiagnostics(
                config=controller.config, apply_g=apply_g,
                param_shards={'w': jnp.asarray})
            with redirect_stdout(io.StringIO()), patch.object(
                    matrix_spectrum, 'available_host_memory', return_value=10**15):
                actual = controller.run(params, batch, **kwargs)
                expected = fresh.run(params, batch, **kwargs)
            self.assertEqual(
                {k: v for k, v in actual.items() if not k.endswith('/seconds')},
                {k: v for k, v in expected.items() if not k.endswith('/seconds')})
            self.assertAlmostEqual(actual['spectrum/G/fallback_lambda_max_est'],
                                   10. if step == 0 else 21., delta=.01)
            if cached is not None:
                for name, solver in cached.items():
                    self.assertIs(controller._power_solvers[name], solver)
            cached = dict(controller._power_solvers)

    def test_single_switch_and_cadence_at_all_training_call_sites(self):
        tree = ast.parse(TRAINER.read_text())
        defaults = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == 'define_flags_with_default')
        names = {kw.arg for kw in defaults.keywords}
        self.assertEqual({name for name in names if name.startswith('condition_')},
                         {'condition_log', 'condition_every'})
        self.assertIn('spectrum_endpoint_maxiter', names)
        self.assertIn('spectrum_inverse_cg_maxiter', names)
        self.assertIn('spectrum_check_every', names)
        schedules = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'do_condition'
                             for t in n.targets)]
        self.assertEqual(len(schedules), 3)  # CG, adaptive inner, regular inner.
        for expression in schedules:
            compiled = compile(ast.Expression(expression), str(TRAINER), 'eval')
            for enabled, step, expected in ((False, 0, False), (True, 0, True),
                                            (True, 49, False), (True, 50, True)):
                flags = SimpleNamespace(condition_log=enabled, condition_every=50)
                self.assertEqual(eval(compiled, dict(FLAGS=flags, step=step)), expected)

    def test_spectrum_reporting_flags_are_resume_compatible(self):
        cg_resume.validate_flags(
            {'optimizer_type': 'cg', 'condition_log': False},
            {'optimizer_type': 'cg', 'condition_log': True, 'condition_every': 50,
             'spectrum_inverse_cg_maxiter': 100,
             'spectrum_inverse_cg_tol': .001})


if __name__ == '__main__':
    unittest.main()
