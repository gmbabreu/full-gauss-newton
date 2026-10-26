"""Exercise the trainer's host logging blocks without importing JAX or TPU code."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


TRAINER = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'


def logging_blocks():
    tree = ast.parse(TRAINER.read_text())
    setup = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == 'wandb.define_metric':
                setup.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'FLAGS.log_inner_steps':
            block = node
    return setup, block


def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(TRAINER), 'exec'), namespace)


def emit(wandb, *, rank=0, enabled=True, count=25, outer_steps=(0, 1)):
    setup, block = logging_blocks()
    namespace = dict(
        wandb=wandb, jax=SimpleNamespace(process_index=lambda: rank),
        FLAGS=SimpleNamespace(log_inner_steps=enabled, inner_loop_iter=count),
        metrics=dict(relative_residual=0.25, linear_model_loss=1.0,
                     gradient_norm=0.1, param_norm=2.0, gpu_memory=0,
                     learning_rate=0.01), defer_wandb=Mock(),
    )
    execute(setup, namespace)
    output = io.StringIO()
    with redirect_stdout(output):
        for step in outer_steps:
            for i in range(count):
                namespace.update(step=step, i=i)
                execute([block], namespace)
    return output.getvalue(), namespace['defer_wandb']


class InnerResidualLoggingTest(unittest.TestCase):
    def test_residual_prints_share_scalar_cadence_and_keep_deferred_logging(self):
        for count in (1, 25, 50):
            with self.subTest(count=count):
                wandb = Mock()
                output, deferred = emit(wandb, count=count, outer_steps=(0,))
                expected = sorted({0, count - 1, *range(9, count, 10)})
                self.assertEqual(output.splitlines(), [
                    f'inner_step={i} relative_residual=2.500000e-01' for i in expected])
                self.assertEqual(
                    [c.args[0]['inner_step'] for c in wandb.log.call_args_list], expected)
                self.assertEqual(
                    [c.args[0]['inner_step'] for c in deferred.call_args_list],
                    list(range(count)))

    def test_each_workers_existing_run_receives_the_complete_scalar_trajectory(self):
        for rank in (0, 1, 3):
            with self.subTest(rank=rank):
                wandb = Mock()
                emit(wandb, rank=rank)
                self.assertEqual(
                    [call.args[0]['inner_step'] for call in wandb.log.call_args_list],
                    [0, 9, 19, 24, 25, 34, 44, 49])
                for call in wandb.log.call_args_list:
                    self.assertEqual(call.args[0]['inner_relative_residual'], 0.25)
                    self.assertEqual(call.kwargs, {'commit': True})
                wandb.define_metric.assert_any_call('inner_step', hidden=True)
                wandb.define_metric.assert_any_call(
                    'inner_relative_residual', step_metric='inner_step', hidden=False)

    def test_disabled_logging_emits_no_inner_scalars(self):
        wandb = Mock()
        emit(wandb, enabled=False)
        wandb.log.assert_not_called()

    def test_one_iteration_is_not_logged_twice(self):
        wandb = Mock()
        emit(wandb, count=1)
        self.assertEqual([c.args[0]['inner_step'] for c in wandb.log.call_args_list], [0, 1])

    def test_thousand_iterations_include_initial_point_and_every_tenth_iteration(self):
        wandb = Mock()
        emit(wandb, count=1000, outer_steps=(4,))
        self.assertEqual(
            [c.args[0]['inner_step'] for c in wandb.log.call_args_list],
            [4000] + list(range(4009, 5000, 10)))


if __name__ == '__main__':
    unittest.main()
