"""Execute real GN routing with real packed HF loaders, without importing libtpu."""
import ast
import copy
from itertools import islice
from pathlib import Path
from types import SimpleNamespace as NS

from datasets import IterableDataset
import jax
import numpy as np
import pytest

from EasyLM import data
from EasyLM.training_progress import TrainingProgress
from test_streaming_replay import Tokenizer, assert_batch


TRAINER = Path(__file__).parents[1] / 'EasyLM/models/llama/llama_train_gn.py'
TREE = ast.parse(TRAINER.read_text())


def execute(nodes, namespace):
    module = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(TRAINER), 'exec'), namespace)


def function(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.fixture
def flags(monkeypatch):
    docs = [dict(prefix=str(i + 10), text=' '.join(str(40 + i + j)
            for j in range(4 + i % 5))) for i in range(80)]
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k:
        IterableDataset.from_generator(lambda: iter(copy.deepcopy(docs))))
    config = data.DatasetFactory.get_default_config()
    config.text_processor.fields = '[prefix],text'
    config.huggingface_dataset.update(dict(batch_size=2, seq_length=3,
        repeat_first_n_batches=3, shuffle_seed=42, shuffle_buffer_size=5))
    return NS(train_dataset=config, eval_dataset=data.DatasetFactory.get_default_config(),
        optimizer_type='adamw', gauss_newton=True, single_batch_inner=True,
        adaptive_inner_loop=False, inner_loop_iter=4, train_batch_growth_interval=0,
        train_dataset_batch_size=2, start_tokens=0, ls_eval_batches=1, cg_n_micro=1,
        load_dataset_state='', cg_resume_state='', log_step_offset=0, log_token_offset=0,
        load_checkpoint='trainstate_params::gs://gfn-soo/checkpoints/warmup_150m/1_0/streaming_train_state',
        save_model_freq=0, save_milestone_freq=0)


def helpers():
    namespace = dict(copy=copy, DatasetFactory=data.DatasetFactory)
    execute([function('validate_gn_replay'), function('build_gn_replay_auxiliary_dataset')], namespace)
    return namespace


def make_router(dataset, auxiliary, progress, flags, enabled):
    # Extract the trainer's actual iterator setup and nested pull function.
    body = next(n.body for n in ast.walk(TREE) if isinstance(getattr(n, 'body', None), list)
                and any(isinstance(c, ast.FunctionDef) and c.name == 'pull_training_batch' for c in n.body))
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'dataset_object')
    end = next(i for i, n in enumerate(body) if isinstance(n, ast.FunctionDef) and n.name == 'pull_training_batch')
    wrapper = ast.parse('def make(dataset, linesearch_dataset, progress, FLAGS, replay_enabled):\n    pass').body[0]
    wrapper.body = copy.deepcopy(body[start:end + 1]) + [ast.Return(ast.Name('pull_training_batch', ast.Load()))]
    namespace = {}
    execute([wrapper], namespace)
    return namespace['make'](dataset, auxiliary, progress, flags, enabled)


def streams(flags):
    namespace = helpers()
    enabled = namespace['validate_gn_replay'](flags)
    tokenizer = Tokenizer()
    solve = data.DatasetFactory.load_dataset(flags.train_dataset, tokenizer, adam_gn_replay=enabled)
    auxiliary = (namespace['build_gn_replay_auxiliary_dataset'](flags.train_dataset, tokenizer)
                 if enabled else None)
    progress = TrainingProgress()
    return solve, auxiliary, progress, make_router(solve, auxiliary, progress, flags, enabled)


@pytest.mark.parametrize('ls_count', [1, 2])
def test_actual_loader_routing_two_epochs_and_ls_independence(flags, ls_count):
    flags.ls_eval_batches = ls_count
    original = copy.deepcopy(flags.train_dataset.to_dict())
    solve, auxiliary, progress, pull = streams(flags)
    expected = list(islice(data.DatasetFactory.load_dataset(flags.train_dataset,
        Tokenizer(), first_order_adam_replay=True), 3))
    assert flags.train_dataset.to_dict() == original
    auxiliary_expected = copy.deepcopy(original)
    auxiliary_expected['huggingface_dataset'].update(repeat_first_n_batches=0, shuffle_seed=43)
    assert auxiliary.config.to_dict() == auxiliary_expected['huggingface_dataset']
    assert auxiliary.text_processor.config == solve.text_processor.config
    ls_namespace = dict(FLAGS=flags, pull_training_batch=pull, jax=jax,
        microbatched_loss_fn=lambda params, batch, rng, n: (np.mean(batch['input_tokens']), None))
    execute([function('pull_ls_batches_and_baseline')], ls_namespace)
    rng = jax.random.PRNGKey(7)
    for update in range(1, 7):
        batch, metadata = pull('solve')
        assert_batch(batch, expected[(update - 1) % 3][0])
        assert metadata['dataset_total_tokens'] == update * 6
        assert metadata['unique_solve_tokens'] == min(update * 6, 18)
        assert metadata['replay_completed_epochs'] == update // 3
        assert metadata['replay_next_batch_index'] == update % 3
        ls_batches, ls_rngs, rng, _, exhausted = ls_namespace['pull_ls_batches_and_baseline'](rng, None, None)
        assert not exhausted and len(ls_batches) == len(ls_rngs) == ls_count
        assert progress.dataset_total_tokens == update * 6
        progress.complete_update()
        record = progress.record(update - 1, **metadata)
        assert record['total_tokens'] == update * 6
        assert record['phase_linesearch_tokens'] == update * ls_count * 6
    assert auxiliary.metadata['dataset_total_tokens'] == 6 * ls_count * 6


def test_disabled_preserves_single_shared_cursor_and_metadata(flags):
    flags.train_dataset.huggingface_dataset.repeat_first_n_batches = 0
    flags.ls_eval_batches = 0  # Legacy fallback remains supported.
    solve, auxiliary, progress, pull = streams(flags)
    assert auxiliary is None
    reference = iter(data.DatasetFactory.load_dataset(flags.train_dataset, Tokenizer()))
    for role in ['solve', 'linesearch', 'linesearch', 'solve', 'linesearch']:
        batch, metadata = pull(role)
        expected, expected_metadata = next(reference)
        assert_batch(batch, expected)
        assert metadata == expected_metadata
        assert progress.dataset_total_tokens == metadata['dataset_total_tokens']
        assert 'unique_solve_tokens' not in metadata
    assert progress.phase_solve_tokens == 12
    assert progress.phase_linesearch_tokens == 18


@pytest.mark.parametrize('key,value', [('optimizer_type', 'muon'), ('gauss_newton', False),
    ('single_batch_inner', False), ('adaptive_inner_loop', True), ('inner_loop_iter', 0),
    ('train_batch_growth_interval', 1), ('train_dataset_batch_size', 3),
    ('ls_eval_batches', 0), ('load_dataset_state', 'dataset.pkl'),
    ('cg_resume_state', 'resume'), ('log_step_offset', 1), ('log_token_offset', 1),
    ('start_tokens', 1), ('load_checkpoint', 'trainstate::parent'),
    ('save_model_freq', 1), ('save_milestone_freq', 1)])
def test_unsupported_gn_replay_rejected(flags, key, value):
    setattr(flags, key, value)
    with pytest.raises(ValueError):
        helpers()['validate_gn_replay'](flags)


@pytest.mark.parametrize('key,value', [('streaming', False), ('pretokenized_dataset_dir', 'disk'),
    ('tokens_count_at_start', 1)])
def test_unsupported_source_rejected(flags, key, value):
    flags.train_dataset.huggingface_dataset[key] = value
    with pytest.raises(ValueError):
        helpers()['validate_gn_replay'](flags)


def test_validation_replay_and_unauthorized_factory_rejected(flags):
    flags.eval_dataset.huggingface_dataset.repeat_first_n_batches = 3
    with pytest.raises(ValueError, match='Validation'):
        helpers()['validate_gn_replay'](flags)
    with pytest.raises(ValueError, match='explicitly authorized'):
        data.DatasetFactory.load_dataset(flags.train_dataset, Tokenizer())
    flags.train_dataset.huggingface_dataset.streaming = False
    with pytest.raises(ValueError, match='streaming'):
        data.DatasetFactory.load_dataset(flags.train_dataset, Tokenizer(), adam_gn_replay=True)


def test_actual_outer_fetch_and_inner_reuse(flags):
    _, _, progress, pull = streams(flags)
    single_fetch = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'FLAGS.single_batch_inner'
        and 'single_batch_, single_dataset_metrics_ = pull_training_batch' in ast.unparse(n))
    inner_loop = next(n for n in ast.walk(TREE) if isinstance(n, ast.For)
        and ast.unparse(n.iter) == 'range(FLAGS.inner_loop_iter)'
        and any(isinstance(c, ast.If) and ast.unparse(c.test) == 'FLAGS.single_batch_inner' for c in n.body))
    selection = next(n for n in inner_loop.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'FLAGS.single_batch_inner')
    instrumented = copy.deepcopy(inner_loop)
    instrumented.body = [copy.deepcopy(selection), ast.parse('seen.append(batch_)').body[0]]
    seen = []
    namespace = dict(FLAGS=flags, pull_training_batch=pull, seen=seen)
    for _ in range(6):
        execute([single_fetch, instrumented], namespace)
    assert progress.phase_solve_tokens == 6 * 6
    for start in range(0, len(seen), flags.inner_loop_iter):
        assert all(batch is seen[start] for batch in seen[start:start + flags.inner_loop_iter])
    for i in range(3 * flags.inner_loop_iter):
        assert_batch(seen[i], seen[i + 3 * flags.inner_loop_iter])


def test_solve_metadata_reaches_outer_record(flags):
    _, _, progress, pull = streams(flags)
    _, metadata = pull('solve')
    pull('linesearch')
    logging = next(n for n in ast.walk(TREE) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'replay_enabled'
        and 'defer_wandb(single_dataset_metrics_' in ast.unparse(n))
    pending = {}
    execute([logging], dict(replay_enabled=True, single_dataset_metrics_=metadata, step=0,
        defer_wandb=lambda values, **kwargs: pending.update(values)))
    progress.complete_update()
    record = progress.record(0, **pending)
    assert record['unique_solve_tokens'] == 6
    assert record['replay_next_batch_index'] == 1
    assert record['dataset_total_tokens'] == 6
    assert record['phase_linesearch_tokens'] == 6


def test_update501_real_loader_boundary_at_experiment_dimensions(flags, monkeypatch):
    hf = flags.train_dataset.huggingface_dataset
    hf.update(batch_size=720, seq_length=1024, repeat_first_n_batches=500, shuffle_data=False)
    flags.train_dataset_batch_size = 720
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k:
        IterableDataset.from_generator(lambda: iter([dict(prefix='9', text='10')])) )
    tokenizer = Tokenizer()
    tokenizer.encode = lambda text, **kwargs: [int(text)] * (720 * 1024 + 2)
    solve = data.DatasetFactory.load_dataset(flags.train_dataset, tokenizer, adam_gn_replay=True)
    original, _ = next(iter(solve))
    state = solve.get_state_dict()
    state['metadata']['dataset_total_tokens'] = 500 * 720 * 1024
    solve.load_state_dict(state)
    progress = TrainingProgress(phase_completed_updates=500,
        phase_solve_tokens=368640000, dataset_total_tokens=368640000)
    pull = make_router(solve, solve, progress, flags, True)
    batch, metadata = pull('solve')
    assert_batch(batch, original)
    progress.complete_update()
    record = progress.record(500, **metadata)
    assert record['total_tokens'] == 369377280
    assert record['unique_solve_tokens'] == 368640000
    assert record['replay_completed_epochs'] == 1
    assert record['replay_next_batch_index'] == 1
