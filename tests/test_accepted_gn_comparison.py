"""CPU checks of real host update paths, without TPU/model-service imports.

Extract trainer functions/blocks as in the other host logging checks; only the
solver is replaced with a fixed displacement on a tiny masked data-loss model.
"""
import ast
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'EasyLM/models/llama/llama_train_gn.py'
TREE = ast.parse(TRAINER.read_text())


def execute(nodes, namespace, *, loop=False):
    if loop:  # The source blocks contain dataset-exhaustion breaks.
        nodes = [ast.For(target=ast.Name(id='_once', ctx=ast.Store()),
                         iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
                         body=nodes, orelse=[])]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(TRAINER), 'exec'), namespace)


def function(name, tree=TREE):
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


@dataclass(frozen=True)
class State:
    params: object
    step: int = 0
    opt_state: int = 0
    warmstart_params: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class AcceptedGNComparisonTest(unittest.TestCase):
    def run_update(self, solver='cg', *, enabled=True, step=0, fixed=0.,
                   linesearch=True, armijo=False, restricted=None, export=False,
                   exported=False):
        flags = SimpleNamespace(
            export_update_geometry=export,
            optimizer_type=solver, log_gn_quadratic=enabled, log_freq=2,
            normalize_step=False, outer_momentum_beta=0., outer_weight_decay=0.,
            single_batch_inner=True, inner_loop_iter=1, inner_loop_wd=0.,
            log_inner_steps=False, log_update_balance=False, weight_average=False,
            linesearch=linesearch, gauss_newton=True, cg_n_micro=3,
            cg_interpolation_lambda=1., cg_lambda_batch_denominator=0,
            cg_lambda_final=-1., fixed_step_size=fixed, armijo_linesearch=armijo,
            armijo_init_step=1., armijo_beta=.5, patience=1, ls_range=7,
            ls_eval_batches=2)
        if restricted:
            setattr(flags, *restricted)
        # Paired targets, unequal sequence mask lengths, and one all-masked row.
        solve = dict(input_tokens=jnp.array([[1., 2.]] * 5),
                     target_tokens=jnp.array([[0, 1], [1, 0], [0, 0], [1, 1], [0, 0]]),
                     loss_masks=jnp.array([[1, 0], [1, 0], [1, 1], [1, 1], [0, 0]]))
        ls = [dict(solve, input_tokens=solve['input_tokens'] * s) for s in (.5, 1.5)]
        evaluations, pulls, records, captures = [], [], [], []
        ns = dict(jax=jax, jnp=jnp, linearize=jax.linearize,
                  linear_transpose=jax.linear_transpose, FLAGS=flags,
                  warnings=warnings, data_shards=1)
        loss_tree = ast.parse((ROOT / 'EasyLM/jax_utils.py').read_text())
        execute([function('cross_entropy_loss_and_accuracy', loss_tree)], ns)
        ce = ns['cross_entropy_loss_and_accuracy']

        def logits(params, batch):
            z = params * batch['input_tokens']
            return jnp.stack((z, jnp.zeros_like(z)), axis=-1)

        def loss(params, batch, rng):
            # Read the key to ensure evaluations accept genuine local RNGs.
            jax.random.uniform(rng, ())
            return ce(logits(params, batch), batch['target_tokens'], batch['loss_masks'])

        def quadratic(base, candidate, rng, batch):
            return ns['measure_gn_quadratic'](
                lambda p: logits(p, batch),
                lambda z: ce(z, batch['target_tokens'], batch['loss_masks'])[0],
                base, candidate)

        for name in ('gn_quadratic_summary', 'measure_gn_quadratic',
                     'accepted_gn_comparison', 'microbatch_groups', 'microbatched_loss_fn'):
            execute([function(name)], ns)
        ns['parallel_loss_fn'] = jax.jit(loss)
        micro_loss = ns['microbatched_loss_fn']

        def counted_loss(params, batch, rng, n_micro):
            evaluations.append((float(params), np.array(rng)))
            return micro_loss(params, batch, rng, n_micro)

        initial_rng = jax.random.PRNGKey(17)
        candidate = jnp.array(-3.)  # raw displacement -4; best scale is .25

        def cg_solver(base, m, v, x, count, outer, rng, batch, wd, lam):
            return (candidate, m + 1, v + 2, x + 3, count + 1,
                    jax.random.split(rng)[0], quadratic(base, candidate, rng, batch))

        def gn_solver(state, base, rng, batch, wd, last):
            return state.replace(params=candidate, opt_state=9), jax.random.split(rng)[0], {}

        def pull(kind):
            pulls.append(kind)
            return ls[(len(pulls) - 1) % len(ls)], {}

        # Use the actual source's local helpers and scheduling gates.
        ns.update(microbatched_loss_fn=counted_loss,
                  geometry_exported=exported,
                  geometry_captures=captures,
                  capture_update_geometry=lambda *args: captures.append(args),
                  apply_outer_decay=lambda base, updated: updated,
                  pull_training_batch=pull, dataset=None, step=step,
                  train_state=State(jnp.array(1.)), inner_state=State(jnp.array(1.)),
                  sharded_rng=initial_rng, single_batch_=solve, single_dataset_metrics_={},
                  do_condition=False, cg_first_moment=jnp.array(0.),
                  cg_second_moment=jnp.array(0.), cg_x0=jnp.array(0.), cg_adam_step=0,
                  sharded_train_step_cg=cg_solver, sharded_train_step=gn_solver,
                  sharded_gn_quadratic_metrics=quadratic,
                  global_norm=lambda x: jnp.linalg.norm(x),
                  defer_wandb=lambda values, **kwargs: records.append(values),
                  PS=lambda *args: None)
        # Host sharding is irrelevant to this single-device CPU check.
        ns['jax'] = SimpleNamespace(**{**jax.__dict__, 'lax': SimpleNamespace(
            with_sharding_constraint=lambda x, spec: x)})
        main = function('main')
        gate_index = next(i for i, n in enumerate(main.body)
                          if isinstance(n, ast.Assign)
                          and ast.unparse(n.targets[0]) == 'log_accepted_gn')
        schedule = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == 'do_accepted_gn')
        branch = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "FLAGS.optimizer_type == 'cg'"
                      and any(isinstance(c, ast.Assign) and isinstance(c.targets[0], ast.Tuple)
                              and ast.unparse(c.targets[0].elts[0]) == 'candidate_params'
                              for c in n.body))
        with redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            execute(main.body[gate_index:gate_index + 2] + [schedule], ns)
            execute([function('run_linesearch'), function('pull_ls_batches_and_baseline')], ns)
            execute(branch.body if solver == 'cg' else branch.orelse[0].orelse, ns, loop=True)
        return ns, evaluations, pulls, records, caught, loss, solve

    def test_accepted_scale_and_actual_losses_reuse_candidate(self):
        for solver in ('cg', 'adamw'):
            for armijo in (False, True):
                with self.subTest(solver=solver, armijo=armijo):
                    ns, calls, pulls, _, _, loss, solve = self.run_update(solver, armijo=armijo)
                    alpha = float(ns['effective_step_size'])
                    self.assertAlmostEqual(alpha, .25, places=5)
                    self.assertNotEqual(float(ns['losses'][-1][0]), alpha)
                    metrics = ns['metrics']
                    expected_prediction = -alpha * (metrics['gn_quad/g_dot_d'] +
                                                   .5 * alpha * metrics['gn_quad/d_G_d'])
                    self.assertAlmostEqual(float(metrics['gn_quad/accepted_predicted_improvement']),
                                           float(expected_prediction), places=6)
                    base = float(loss(jnp.array(1.), solve, jax.random.PRNGKey(0))[0])
                    accepted = float(loss(ns['train_state'].params, solve, jax.random.PRNGKey(0))[0])
                    self.assertAlmostEqual(metrics['gn_quad/accepted_actual_improvement_solve'],
                                           base - accepted, places=6)
                    accepted_ls = next(float(l) for s, l in ns['losses'] if float(s) == alpha)
                    self.assertAlmostEqual(metrics['gn_quad/accepted_actual_improvement_linesearch'],
                                           ns['baseline_loss'] - accepted_ls, places=6)
                    # Baselines + tested candidates + exactly two solve evaluations.
                    self.assertEqual(len(calls), 2 * (1 + len(ns['losses'])) + 2)
                    np.testing.assert_array_equal(calls[-1][1], calls[-2][1])
                    np.testing.assert_array_equal(calls[-1][1], jax.random.PRNGKey(17))
                    self.assertEqual(pulls, ['linesearch', 'linesearch'])

    def test_fixed_override_evaluates_actual_accepted_params(self):
        for solver in ('cg', 'adamw'):
            # Same rounded tag as .25, distinct numeric scale.
            ns, calls, _, _, _, loss, _ = self.run_update(solver, fixed=.25004)
            expected = ns['baseline_loss'] - sum(
                float(loss(ns['train_state'].params, b, r)[0])
                for b, r in zip(ns['ls_batches'], ns['ls_rngs'])) / 2
            self.assertAlmostEqual(ns['metrics']['gn_quad/accepted_actual_improvement_linesearch'],
                                   expected, places=7)
            self.assertEqual(len(calls), 2 * (1 + len(ns['losses'])) + 4)

    def test_logging_preserves_updates_rng_state_and_data(self):
        for solver in ('cg', 'adamw'):
            # Grid arithmetic produces .50000006, so a literal .5 override
            # correctly needs evaluation too; 1.0 is an exact existing entry.
            for fixed in (0., 1., .5, .25004):
                for step in (0, 1):
                    with self.subTest(solver=solver, fixed=fixed, step=step):
                        on, calls_on, pulls_on, records_on, _, _, _ = self.run_update(
                            solver, fixed=fixed, step=step)
                        off, calls_off, pulls_off, records_off, _, _, _ = self.run_update(
                            solver, fixed=fixed, step=step, enabled=False)
                        for key in ('train_state', 'inner_state', 'cg_adam_step'):
                            self.assertEqual(on[key], off[key])
                        for key in ('sharded_rng', 'cg_first_moment', 'cg_second_moment', 'cg_x0'):
                            np.testing.assert_array_equal(on[key], off[key])
                        self.assertEqual(pulls_on, pulls_off)
                        self.assertEqual(records_on, records_off)
                        for key, value in off['metrics'].items():
                            np.testing.assert_array_equal(on['metrics'][key], value)
                        self.assertEqual(len(calls_on) - len(calls_off),
                                         (4 if fixed in (.5, .25004) else 2) if step == 0 else 0)
                        if step == 1:
                            self.assertFalse(any('/accepted_' in k for k in on['metrics']))

    def test_restrictions_warn_and_preserve_existing_diagnostics(self):
        for restriction in (('normalize_step', True), ('outer_momentum_beta', .5),
                            ('outer_weight_decay', .1)):
            # Ordinary GN does not implement outer momentum; the logging gate
            # must nevertheless conservatively skip it, as requested.
            ns, calls, _, _, caught, _, _ = self.run_update('adamw', restricted=restriction)
            self.assertEqual(len(caught), 1)
            self.assertIn('Skipping accepted GN comparison metrics', str(caught[0].message))
            self.assertIn('gn_quad/g_dot_d', ns['metrics'])
            self.assertFalse(any('/accepted_' in k for k in ns['metrics']))
            self.assertEqual(len(calls), 2 * (1 + len(ns['losses'])))

    def test_no_linesearch_logs_only_solve_comparison(self):
        ns, calls, pulls, _, _, _, _ = self.run_update('adamw', linesearch=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(pulls, [])
        self.assertIn('gn_quad/accepted_predicted_improvement', ns['metrics'])
        self.assertIn('gn_quad/accepted_actual_improvement_solve', ns['metrics'])
        self.assertNotIn('gn_quad/accepted_actual_improvement_linesearch', ns['metrics'])

    def test_export_gate_and_raw_candidate_for_both_solvers(self):
        for solver in ('cg', 'adamw'):
            for linesearch in (False, True):
                # CG always uses its existing line-search route.
                with self.subTest(solver=solver, linesearch=linesearch):
                    off, _, _, _, _, _, _ = self.run_update(solver, fixed=.125, linesearch=linesearch)
                    on, _, _, _, _, _, solve = self.run_update(
                        solver, fixed=.125, linesearch=linesearch, export=True)
                    self.assertEqual(off['geometry_captures'], [])
                    self.assertEqual(len(on['geometry_captures']), 1)
                    base, candidate, batch = on['geometry_captures'][0]
                    self.assertEqual(float(base), 1.)
                    self.assertEqual(float(candidate), -3.)
                    for key in solve:
                        np.testing.assert_array_equal(batch[key], solve[key])
                    self.assertEqual(on['train_state'], off['train_state'])
                    later, _, _, _, _, _, _ = self.run_update(
                        solver, fixed=.125, linesearch=linesearch, export=True, exported=True)
                    self.assertEqual(later['geometry_captures'], [])


if __name__ == '__main__':
    unittest.main()
