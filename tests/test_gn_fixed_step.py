"""Exercise the ordinary GN host update on CPU without importing the trainer."""
import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np


TRAINER = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'
TREE = ast.parse(TRAINER.read_text())


@dataclass(frozen=True)
class State:
    params: object
    step: int = 7
    opt_state: object = None
    warmstart_params: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class GNFixedStepTest(unittest.TestCase):
    def run_update(self, fixed, decay=0.):
        outer = {'layer': {'kernel': jnp.array([[1., -2.], [3., 4.]]),
                           'bias': jnp.array([.25, -.5])}}
        inner = {'layer': {'kernel': jnp.array([[-3., 6.], [5., -8.]]),
                           'bias': jnp.array([2., -3.])}}
        before = State(outer, opt_state=object(), warmstart_params=object())
        solved = State(inner, opt_state=object(), warmstart_params=object())
        comparison = Mock(return_value={'accepted': True})
        namespace = dict(
            jax=jax, train_state=before, inner_state=solved, step=7,
            FLAGS=SimpleNamespace(fixed_step_size=fixed, outer_weight_decay=decay,
                                  log_update_balance=False, cg_n_micro=1),
            do_accepted_gn=True, metrics={}, accepted_gn_comparison=comparison,
            batch=object(), gn_quadratic_rng=object(), microbatched_loss_fn=object())
        mask = next(n for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == 'outer_decay_mask')
        apply_decay = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)
                           and n.name == 'apply_outer_decay')
        branch = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'FLAGS.linesearch'
                      and any(isinstance(c, ast.Assign)
                              and ast.unparse(c.targets[0]) == 'effective_step_size'
                              for c in n.orelse))
        exec(compile(ast.Module(body=[mask, apply_decay] + branch.orelse,
                                type_ignores=[]), str(TRAINER), 'exec'), namespace)
        after = namespace['train_state']
        self.assertEqual(after.step, before.step + 1)
        self.assertIs(after.opt_state, solved.opt_state)
        self.assertIs(after.warmstart_params, before.warmstart_params)
        self.assertIs(solved.params, inner)
        self.assertIs(comparison.call_args.args[2], after.params)
        self.assertEqual(comparison.call_args.args[5], namespace['effective_step_size'])
        self.assertTrue(namespace['metrics']['accepted'])
        return namespace, outer, inner

    def test_positive_fixed_step_scales_outer_displacement(self):
        ns, outer, inner = self.run_update(.046875)
        self.assertEqual(ns['effective_step_size'], .046875)
        for actual, base, solved in zip(jax.tree.leaves(ns['train_state'].params),
                                        jax.tree.leaves(outer), jax.tree.leaves(inner)):
            np.testing.assert_array_equal(actual, base + .046875 * (solved - base))

    def test_nonpositive_fixed_step_preserves_direct_full_step(self):
        for fixed in (0., -1.):
            with self.subTest(fixed=fixed):
                ns, _, inner = self.run_update(fixed)
                self.assertEqual(ns['effective_step_size'], 1.)
                self.assertIs(ns['train_state'].params, inner)

    def test_decay_is_applied_after_scaling_to_matrix_leaves_only(self):
        for fixed in (0., .046875):
            with self.subTest(fixed=fixed):
                ns, outer, inner = self.run_update(fixed, decay=.125)
                for actual, base, solved in zip(jax.tree.leaves(ns['train_state'].params),
                                                jax.tree.leaves(outer), jax.tree.leaves(inner)):
                    expected = base + fixed * (solved - base) if fixed > 0 else solved
                    if base.ndim == 2:
                        expected = expected - .125 * base
                    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
