"""Small CPU continuation through the trainer's actual save function and serializer.

Extracting the nested function avoids initializing LLaMA or downloading C4.
The numerical model is a two-parameter quadratic, not a TPU training run.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from ml_collections import ConfigDict
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.data import HuggingfaceDataset
from EasyLM.training_progress import TrainingProgress, ProcessTiming
from EasyLM import cg_resume

TRAINER = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'


def trainer_function(name, namespace, nonlocals):
    tree = ast.parse(TRAINER.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    wrapper = ast.parse('def factory():\n    pass').body[0]
    wrapper.body = ast.parse('\n'.join(f'{key} = {value!r}' for key, value in nonlocals.items())).body
    wrapper.body += [node, ast.Return(value=ast.Name(id=name, ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(TRAINER), 'exec'), namespace)
    return namespace['factory']()


class Tokenizer:
    name_or_path = 'synthetic'
    def __len__(self):
        return 10000


class Processor:
    config = ConfigDict({})
    def __call__(self, example):
        return example, [1.] * len(example)


def loader():
    with patch('EasyLM.data.load_dataset', return_value=[list(range(1000))]):
        return HuggingfaceDataset(dict(batch_size=2, seq_length=2, shuffle_data=False),
                                  Tokenizer(), Processor())


def update(state, batch):
    """Independent tiny PCG problem with moments and a warm start."""
    tokens = jnp.asarray(batch['input_tokens'], dtype=jnp.float32).reshape(-1)
    features = jnp.stack([jnp.ones_like(tokens), tokens / 100], axis=1)
    gram = features.T @ features / len(tokens)
    grad = gram @ state['params'] - jnp.array([1., .2])
    m = .9 * state['cg_first_moment'] + .1 * grad
    v = .95 * state['cg_second_moment'] + .05 * grad**2
    t = state['step'] + 1
    lam = cg_resume.batch_lambda(batch['input_tokens'].shape[0], 20.)
    diag = (jnp.sqrt(v / (1 - .95**t)) + 1e-8) / .01
    matrix = lam * gram + (1-lam) * jnp.diag(diag)
    rhs = -m / (1 - .9**t)
    direction, _ = jax.scipy.sparse.linalg.cg(lambda x: matrix @ x, rhs,
        x0=state['cg_x0'], M=lambda x: x / jnp.diag(matrix), tol=1e-7, maxiter=20)
    np.testing.assert_allclose(direction, np.linalg.solve(np.asarray(matrix), np.asarray(rhs)), rtol=2e-5, atol=1e-7)
    rng, _ = jax.random.split(state['sharded_rng'])
    return dict(params=state['params'] + direction, step=t, cg_adam_step=t,
                cg_first_moment=m, cg_second_moment=v, cg_x0=direction, sharded_rng=rng)


class SerializationTests(unittest.TestCase):
    def test_trainer_save_and_next_update(self):
        data = loader()
        iterator = iter(data)
        progress = TrainingProgress()
        state = dict(params=jnp.zeros(2), step=jnp.array(0, dtype=jnp.int32), cg_adam_step=jnp.array(0, dtype=jnp.int32),
                     cg_first_moment=jnp.zeros(2), cg_second_moment=jnp.zeros(2),
                     cg_x0=jnp.zeros(2), sharded_rng=jax.random.PRNGKey(42))
        for _ in range(2):
            batch, metrics = next(iterator)
            progress.charge('solve', batch, metrics)
            state = update(state, batch)
            progress.complete_update()
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(jax=jax, np=np, flatten_dict=flatten_dict, cg_resume=cg_resume,
                StreamingCheckpointer=StreamingCheckpointer,
                FLAGS=NS(optimizer_type='cg', outer_momentum_beta=0., weight_average=False),
                variant={}, flags_config_dict={'cg_lambda_batch_denominator': 20.},
                llama_config=ConfigDict({}), progress=progress, dataset_object=data,
                output_dir=directory, checkpointer=NS(enable=True), timing=ProcessTiming(),
                cg_param_gathers=lambda x: np.asarray(x), gather_fns=NS(step=lambda x: np.asarray(x)),
                multihost_utils=NS(sync_global_devices=lambda label: None), **state)
            save = trainer_function('save_checkpoint', namespace, {'cg_generation': -1})
            save(NS(params=state['params'], step=state['step']), milestone=True)
            path = cg_resume.resolve_resume(directory, exists=True)
            metadata, cursor, marker = cg_resume.read_bundle(path)
            restored = StreamingCheckpointer.load_checkpoint(path)
            restored = unflatten_dict({key: value.astype(metadata['state_dtypes'][key])
                                       for key, value in flatten_dict(restored).items()})
            # Production shard functions put restored host arrays back on devices.
            restored = jax.tree.map(jnp.asarray, restored)
            for key in state:
                np.testing.assert_array_equal(restored[key], state[key])
            resumed_data = loader()
            resumed_data.load_state_dict(cursor)
            # Cross a batch-growth boundary immediately after restoring the cursor.
            data.config.batch_size = resumed_data.config.batch_size = 3
            expected_batch, _ = next(iterator)
            actual_batch, _ = next(iter(resumed_data))
            for key in expected_batch:
                np.testing.assert_array_equal(actual_batch[key], expected_batch[key])
            expected = update(state, expected_batch)
            actual = update(restored, actual_batch)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(TrainingProgress.from_state_dict(metadata['training_progress']).state_dict(), progress.state_dict())
            self.assertEqual(progress.phase_skipped_tokens, 0)
            self.assertEqual(marker['generation'], 0)
            self.assertEqual(cg_resume.read_bundle(directory + '/milestones/step_2/cg_state_0')[0]['step'], 2)

    def test_actual_fetch_helper_counts_and_growth(self):
        data = loader()
        progress = TrainingProgress()
        namespace = dict(FLAGS=NS(train_batch_growth_interval=1), dataset_object=data,
                         dataset=iter(data), solve_batch_size=3, progress=progress)
        pull = trainer_function('pull_training_batch', namespace, {'actual_solve_batch_size': None})
        solve, _ = pull('solve')
        search, _ = pull('linesearch')
        np.testing.assert_array_equal(solve['input_tokens'].reshape(-1), np.arange(6))
        np.testing.assert_array_equal(search['input_tokens'].reshape(-1), np.arange(6, 10))
        self.assertEqual((progress.phase_solve_tokens, progress.phase_linesearch_tokens,
                          progress.phase_skipped_tokens, progress.dataset_total_tokens), (6, 4, 0, 10))
        # Ensure no outer-loop branch reintroduces an unused fetch.
        calls = [n for n in ast.walk(ast.parse(TRAINER.read_text()))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'pull_training_batch']
        self.assertFalse(any(n.args and isinstance(n.args[0], ast.Constant)
                             and n.args[0].value == 'skipped' for n in calls))


if __name__ == '__main__':
    unittest.main()
