"""Offline tests of the real HF packer, resume validator and Adam checkpointer."""
from copy import deepcopy
from itertools import islice
from pathlib import Path

import jax
import jax.numpy as jnp
import mlxu
import numpy as np
import pytest
from datasets import IterableDataset
from flax.training.train_state import TrainState

from EasyLM import data
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.training_resume import (
    validate_dataset_snapshot, validate_replay_config, validate_resume_metadata,
)


class Tokenizer:
    name_or_path = 'synthetic'
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self):
        return 512

    def encode(self, text, add_special_tokens=False):
        return [int(token) for token in text.split()]


@pytest.fixture
def loader(monkeypatch):
    documents = [dict(prefix=str(10 + i), text=' '.join(
        str(30 + i * 9 + j) for j in range(3 + i % 5))) for i in range(14)]

    def make(repeat=0, **updates):
        monkeypatch.setattr(data, 'load_dataset', lambda *a, **k:
            IterableDataset.from_generator(lambda: iter(deepcopy(documents))))
        tokenizer = Tokenizer()
        processor = data.TextProcessor(dict(fields='[prefix],text'), tokenizer)
        return data.HuggingfaceDataset(dict(batch_size=2, seq_length=3,
            repeat_first_n_batches=repeat, shuffle_buffer_size=5, **updates),
            tokenizer, processor)
    return make


def assert_batch(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def original_packer(loader):
    """Pre-replay packing contract, including its original exhaustion rule."""
    tokens, masks = [], []
    total = loader.config.tokens_count_at_start
    chunk = loader.config.batch_size * loader.config.seq_length
    for index, example in enumerate(loader.dataset):
        t, m = loader.text_processor(example)
        tokens.extend(t)
        masks.extend(m)
        while len(tokens) > chunk + 1:
            total += chunk
            shape = (loader.config.batch_size, loader.config.seq_length)
            batch = dict(input_tokens=np.array(tokens[:chunk], dtype='i4').reshape(shape),
                target_tokens=np.array(tokens[1:chunk + 1], dtype='i4').reshape(shape),
                loss_masks=np.array(masks[1:chunk + 1], dtype=np.float32).reshape(shape))
            if loader.config.always_start_with_bos:
                batch['input_tokens'][:, 0] = loader.tokenizer.bos_token_id
            yield batch, dict(dataset_example_index=index, dataset_total_tokens=total)
            tokens, masks = tokens[chunk:], masks[chunk:]


@pytest.mark.parametrize('bos', [False, True])
@pytest.mark.parametrize('origin', [0, 12])
def test_disabled_matches_original_and_exhaustion(loader, bos, origin):
    stream = loader(always_start_with_bos=bos, tokens_count_at_start=origin)
    expected = list(original_packer(stream))
    actual = list(stream)
    assert len(actual) == len(expected) > 4
    for (batch, metrics), (reference, metadata) in zip(actual, expected):
        assert_batch(batch, reference)
        assert metrics == metadata


@pytest.mark.parametrize('bos', [False, True])
def test_epochs_repeat_shifted_targets_masks_and_counters(loader, bos):
    expected = list(islice(iter(loader(always_start_with_bos=bos)), 4))
    stream = loader(4, always_start_with_bos=bos)
    for i, (batch, metrics) in enumerate(islice(stream, 17), 1):
        assert_batch(batch, expected[(i - 1) % 4][0])
        assert metrics['dataset_total_tokens'] == i * 6
        assert metrics['unique_solve_tokens'] == min(i * 6, 24)
        assert metrics['replay_completed_epochs'] == i // 4
        assert metrics['replay_next_batch_index'] == i % 4


@pytest.mark.parametrize('stop', [1, 3, 4, 5, 7, 8, 9])
def test_resume_before_at_after_boundary(loader, stop):
    expected = list(islice(loader(4), 16))
    parent = loader(4)
    list(islice(parent, stop))
    restored = loader(4)
    restored.load_state_dict(parent.get_state_dict())
    for (batch, metrics), (reference, metadata) in zip(
            islice(restored, 16 - stop), expected[stop:]):
        assert_batch(batch, reference)
        assert metrics == metadata
    # A new iterator still starts at the saved cursor, so epoch resets must
    # explicitly use the underlying HF source rather than iter(restored).
    assert_batch(next(iter(restored))[0], expected[stop][0])


def flags_for(stream, total_steps=4):
    return dict(seed=42, dtype='fp32', param_dtype='fp32', total_steps=total_steps,
        tokenizer='synthetic', llama={'hidden_size': 2},
        optimizer=OptimizerFactory.get_default_config().to_dict(),
        train_dataset=dict(type='huggingface', huggingface_dataset=stream.config.to_dict()),
        eval_dataset=data.DatasetFactory.get_default_config().to_dict(),
        train_dataset_batch_size=2, eval_steps=0, target_loss=0.1,
        output_dir='experiments', experiment_id='parent', wandb_run_id='',
        load_checkpoint='', load_dataset_state='', start_step=0)


def bundle(stream, steps=4, historical=False):
    list(islice(stream, steps))
    flags = flags_for(stream)
    state = stream.get_state_dict()
    state.update(training_step=steps, snapshot_id='snapshot')
    metadata = dict(resume_version=1, optimizer_saved=True,
        checkpoint_float_dtype='fp32', train_rng=np.array([0, 42], np.uint32),
        step=steps, snapshot_id='snapshot', flags=flags)
    complete = dict(resume_version=1, step=steps, snapshot_id='snapshot')
    if historical:
        for name in ('train_dataset', 'eval_dataset'):
            flags[name]['huggingface_dataset'].pop('repeat_first_n_batches')
        state['config'].pop('repeat_first_n_batches')
    return metadata, complete, state


def requested(metadata, repeat=4):
    flags = deepcopy(metadata['flags'])
    flags['train_dataset']['huggingface_dataset']['repeat_first_n_batches'] = repeat
    flags['eval_dataset']['huggingface_dataset']['repeat_first_n_batches'] = 0
    flags.update(total_steps=12, target_loss=0., experiment_id='replay',
        load_checkpoint='trainstate::experiments/parent/streaming_train_state_4')
    return flags


@pytest.mark.parametrize('steps', [2, 4])
def test_legacy_transition_requires_authorization_and_restarts_boundary(loader, steps):
    metadata, complete, state = bundle(loader(), steps, historical=True)
    flags = requested(metadata)
    stream = loader(4)
    with pytest.raises(ValueError, match='validated initial transition'):
        stream.load_state_dict(state)
    authorized = validate_resume_metadata(metadata, complete, flags, state)
    assert authorized is True
    stream.load_state_dict(state, allow_replay_transition=authorized)
    reference = list(islice(loader(4), 12))
    for (batch, _), (expected, _) in zip(islice(stream, 8), reference[steps:]):
        assert_batch(batch, expected)


def test_old_configs_normalized_for_ordinary_resume(loader):
    metadata, complete, state = bundle(loader(), historical=True)
    flags = deepcopy(metadata['flags'])
    for name in ('train_dataset', 'eval_dataset'):
        flags[name]['huggingface_dataset']['repeat_first_n_batches'] = 0
    assert validate_resume_metadata(metadata, complete, flags, state) is False
    validate_dataset_snapshot(state, metadata)
    stream = loader()
    stream.load_state_dict(state)
    expected = list(islice(loader(), 5))[-1][0]
    assert_batch(next(iter(stream))[0], expected)


def test_over_budget_parent_rejected(loader):
    metadata, complete, state = bundle(loader(), steps=6)
    with pytest.raises(ValueError, match='more than the first replay epoch'):
        validate_resume_metadata(metadata, complete, requested(metadata), state)


def test_single_pass_step1500_parent_rejected_for_k1000(loader):
    metadata, complete, state = bundle(loader())
    metadata['step'] = complete['step'] = state['training_step'] = 1500
    state['metadata']['dataset_total_tokens'] = 1500 * 6
    with pytest.raises(ValueError, match='more than the first replay epoch'):
        validate_resume_metadata(metadata, complete, requested(metadata, 1000), state)


@pytest.mark.parametrize('setting,value', [
    ('seed', 5), ('tokenizer', 'other'), ('llama', {'hidden_size': 3}),
    ('train_dataset_batch_size', 3), ('total_steps', 3), ('target_loss', 0.2),
    ('experiment_id', 'parent'), ('wandb_run_id', 'parent-run'),
    ('dtype', 'bf16'), ('start_step', 1), ('update_llama_config', 'other'),
])
def test_unrelated_transition_changes_rejected(loader, setting, value):
    metadata, complete, state = bundle(loader())
    flags = requested(metadata)
    flags[setting] = value
    with pytest.raises(ValueError):
        validate_resume_metadata(metadata, complete, flags, state)


@pytest.mark.parametrize('key,value', [('batch_size', 3), ('seq_length', 4),
    ('shuffle_seed', 43), ('shuffle_data', False), ('path', 'other'),
    ('tokens_count_at_start', 6)])
def test_dataset_transition_changes_rejected(loader, key, value):
    metadata, complete, state = bundle(loader())
    flags = requested(metadata)
    flags['train_dataset']['huggingface_dataset'][key] = value
    with pytest.raises(ValueError):
        validate_resume_metadata(metadata, complete, flags, state)


def test_optimizer_changes_and_validation_replay_rejected(loader):
    metadata, complete, state = bundle(loader())
    for edit in ('optimizer', 'eval'):
        flags = requested(metadata)
        if edit == 'optimizer':
            flags['optimizer']['adamw_optimizer']['lr'] *= 2
        else:
            flags['eval_dataset']['huggingface_dataset']['repeat_first_n_batches'] = 4
        with pytest.raises(ValueError):
            validate_resume_metadata(metadata, complete, flags, state)


@pytest.mark.parametrize('part', ['metadata', 'complete', 'dataset', 'config', 'rng', 'optimizer', 'buffers', 'progress'])
def test_incomplete_or_mismatched_transition_rejected(loader, part):
    metadata, complete, state = bundle(loader())
    flags = requested(metadata)
    if part == 'metadata':
        metadata['resume_version'] = 0
    elif part == 'complete':
        complete['snapshot_id'] = 'other'
    elif part == 'dataset':
        state['snapshot_id'] = 'other'
    elif part == 'config':
        state['config']['shuffle_seed'] = 99
    elif part == 'rng':
        metadata.pop('train_rng')
    elif part == 'buffers':
        state.pop('token_buffer')
    elif part == 'progress':
        metadata['training_progress'] = dict(step_offset=5, token_offset=30)
    else:
        metadata['optimizer_saved'] = False
    with pytest.raises(ValueError):
        validate_resume_metadata(metadata, complete, flags, state)


def test_replay_recovery_remains_strict(loader):
    metadata, complete, state = bundle(loader(4))
    flags = deepcopy(metadata['flags'])
    assert validate_resume_metadata(metadata, complete, flags, state) is False
    for name, value in [('total_steps', 12), ('target_loss', 0.)]:
        changed = deepcopy(flags)
        changed[name] = value
        with pytest.raises(ValueError, match='trajectory'):
            validate_resume_metadata(metadata, complete, changed, state)
    for repeat in (0, 3, 5):
        with pytest.raises(ValueError):
            loader(repeat).load_state_dict(state)


def test_short_source_and_unsupported_configs(loader):
    with pytest.raises(ValueError, match='cannot supply'):
        list(loader(100))
    with pytest.raises(ValueError, match='tokens_count_at_start'):
        loader(4, tokens_count_at_start=6)
    for repeat in (-1, 1.5, True):
        with pytest.raises((ValueError, TypeError)):
            loader(repeat)
    flags = flags_for(loader(4))
    flags['optimizer']['accumulate_gradient_steps'] = 2
    with pytest.raises(ValueError, match='accumulation=1'):
        validate_replay_config(flags)
    cfg = data.DatasetFactory.get_default_config()
    assert cfg.huggingface_dataset.repeat_first_n_batches == 0
    cfg.huggingface_dataset.repeat_first_n_batches = 4
    with pytest.raises(ValueError, match='first-order Adam'):
        data.DatasetFactory.load_dataset(cfg, Tokenizer())
    cfg.huggingface_dataset.pretokenized_dataset_dir = 'unused'
    with pytest.raises(ValueError, match='first-order Adam'):
        data.DatasetFactory.load_dataset(cfg, Tokenizer(), first_order_adam_replay=True)
    with pytest.raises(ValueError, match='raw Hugging Face'):
        data.OptHuggingfaceDataset(cfg.huggingface_dataset, Tokenizer(), None)
    with pytest.raises(ValueError, match='complete packed'):
        loader(4).load_state_dict(dict(config={}))


def test_boundary_does_not_fetch_next_document_or_keep_remainder(monkeypatch):
    visits = []
    def source():
        visits.append('start')
        yield dict(prefix='9', text=' '.join(str(i) for i in range(20, 60)))
        pytest.fail('Replay read beyond the first two packed batches')
    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: IterableDataset.from_generator(source))
    tokenizer = Tokenizer()
    stream = data.HuggingfaceDataset(dict(batch_size=2, seq_length=3,
        repeat_first_n_batches=2, shuffle_data=False), tokenizer,
        data.TextProcessor(dict(fields='[prefix],text'), tokenizer))
    batches = list(islice(stream, 6))
    assert len(visits) == 3
    for i in range(2, 6):
        assert_batch(batches[i][0], batches[i % 2][0])


def assert_tree(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a, b)


def test_cpu_adam_full_state_legacy_boundary_continuation(loader, tmp_path):
    assert jax.default_backend() == 'cpu'
    optimizer_config = OptimizerFactory.get_default_config()
    optimizer_config.adamw_optimizer.lr_warmup_steps = 2
    optimizer_config.adamw_optimizer.lr_decay_steps = 32
    tx, _ = OptimizerFactory.get_optimizer(optimizer_config)
    apply_fn = lambda: None

    def initial():
        return TrainState.create(apply_fn=apply_fn,
            params={'params': jnp.array([0.3, -0.2])}, tx=tx)

    def update(state, rng, batch):
        rng, noise_rng = jax.random.split(rng)
        x = jnp.asarray(batch['input_tokens'], jnp.float32) / 100
        y = jnp.asarray(batch['target_tokens'], jnp.float32) / 100
        mask = jnp.asarray(batch['loss_masks'])
        noise = jax.random.normal(noise_rng, x.shape) * 0.01
        def loss(params):
            w = params['params']
            return jnp.mean(mask * (w[0] * x + w[1] + noise - y) ** 2)
        return state.apply_gradients(grads=jax.grad(loss)(state.params)), rng

    reference, reference_rng = initial(), jax.random.PRNGKey(13)
    for batch, _ in islice(loader(4), 12):
        reference, reference_rng = update(reference, reference_rng, batch)

    parent, rng = initial(), jax.random.PRNGKey(13)
    parent_loader = loader()
    for batch, _ in islice(parent_loader, 4):
        parent, rng = update(parent, rng, batch)
    metadata, complete, snapshot = bundle(parent_loader, 4, historical=True)
    metadata['train_rng'] = np.asarray(rng)
    metadata['flags']['optimizer'] = optimizer_config.to_dict()
    checkpointer = StreamingCheckpointer(dict(float_dtype='fp32', save_optimizer_state=True), str(tmp_path))
    checkpointer.save_all(parent, None, metadata=metadata, dataset=snapshot)
    checkpointer.save_pickle(complete, 'complete.pkl')
    metadata = mlxu.load_pickle(str(tmp_path / 'metadata.pkl'))
    complete = mlxu.load_pickle(str(tmp_path / 'complete.pkl'))
    snapshot = mlxu.load_pickle(str(tmp_path / 'dataset.pkl'))
    flags = requested(metadata)
    flags['load_checkpoint'] = 'trainstate::' + str(tmp_path / 'streaming_train_state')
    authorized = validate_resume_metadata(metadata, complete, flags, snapshot)
    restored, params = checkpointer.load_trainstate_checkpoint(flags['load_checkpoint'], initial())
    assert params is None
    assert int(restored.step) == 4
    assert_tree(restored, parent)  # Includes both Adam moments and schedule counts.
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree_util.tree_leaves(restored.opt_state))
    restored_rng = jax.device_put(metadata['train_rng'])
    assert_tree(restored_rng, rng)
    stream = loader(4)
    stream.load_state_dict(snapshot, allow_replay_transition=authorized)
    for batch, _ in islice(stream, 8):
        restored, restored_rng = update(restored, restored_rng, batch)
    assert_tree(restored, reference)
    assert_tree(restored_rng, reference_rng)


def test_trainer_preserves_full_state_and_rng_wiring():
    source = (Path(__file__).parents[1] / 'EasyLM/models/llama/llama_train.py').read_text()
    assert "load_companion('dataset')" in source
    assert 'allow_replay_transition=replay_transition' in source
    assert "jax.device_put(resume_metadata['train_rng'])" in source
    assert 'load_trainstate_checkpoint(' in source
