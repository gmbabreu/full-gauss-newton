"""CPU native AdamW math, donated trainer integration, routing, and I/O gates."""
import ast
import copy
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
import numpy as np
import pytest
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from EasyLM import batch_transfer as probe
from EasyLM.data import DatasetFactory
from EasyLM.jax_utils import (JaxRNG, cross_entropy_loss_and_accuracy, global_norm,
                              make_shard_and_gather_fns, with_sharding_constraint)
from EasyLM.optimizers import OptimizerFactory
from EasyLM.training_resume import validate_resume_metadata


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'EasyLM/models/llama/llama_train.py'
TREE = ast.parse(TRAINER.read_text())


def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(TRAINER), 'exec'), namespace)


def trainer_function(name, namespace):
    node = next(node for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == name)
    execute([node], namespace)
    return namespace[name]


def configuration():
    return dict(optimizer=OptimizerFactory.get_default_config().to_dict(),
                weight_average=False, outer_loop_method='None', wandb_run_id='', start_step=0,
                train_dataset=DatasetFactory.get_default_config().to_dict())


MODEL = dict.fromkeys(('embedding_dropout', 'feedforward_dropout', 'attention_dropout',
                      'residue_dropout', 'fcm_min_ratio', 'fcm_max_ratio'), 0.)


@pytest.mark.parametrize('path,value,message', [
    (('optimizer', 'type'), 'muon', 'adamw'),
    (('optimizer', 'accumulate_gradient_steps'), 2, 'accumulation'),
    (('optimizer', 'adamw_optimizer', 'multiply_by_parameter_scale'), True, 'ordinary AdamW'),
    (('weight_average',), True, 'weight_average'),
    (('outer_loop_method',), 'gn', 'outer_loop_method'),
    (('wandb_run_id',), 'old-run', 'replay disabled'),
    (('start_step',), 5, 'start_step'),
    (('train_dataset', 'type'), 'json', 'raw Hugging Face'),
    (('train_dataset', 'huggingface_dataset', 'streaming'), False, 'streaming'),
    (('train_dataset', 'huggingface_dataset', 'pretokenized_dataset_dir'), '/tokens', 'raw'),
    (('train_dataset', 'huggingface_dataset', 'tokens_count_at_start'), 20, 'tokens_count'),
])
def test_invalid_config(path, value, message):
    config = configuration()
    parent = config
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        probe.validate_config(config, MODEL)


@pytest.mark.parametrize('name', list(MODEL))
def test_finalized_stochasticity(name):
    probe.validate_config(configuration(), MODEL)
    with pytest.raises(ValueError, match=name):
        probe.validate_config(configuration(), dict(MODEL, **{name: .1}))


def batches():
    a = dict(input_tokens=np.array([[0, 1, 0], [1, 1, 0]], np.int32),
             target_tokens=np.array([[0, 1, 0], [1, 1, 0]], np.int32),
             loss_masks=np.array([[1, 0, 0], [1, 1, 1]], np.float32))
    b = dict(a, target_tokens=1 - a['target_tokens'])
    return a, b


def numpy_loss_grad(w, batch):
    logits = w[batch['input_tokens']].astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    targets = np.eye(2)[batch['target_tokens']]
    mask = batch['loss_masks']
    weights = mask / mask.sum(axis=-1, keepdims=True) / mask.shape[0]
    loss = -np.sum(np.log(probabilities) * targets * weights[..., None])
    gradient = np.zeros_like(w, dtype=np.float64)
    np.add.at(gradient, batch['input_tokens'], (probabilities - targets) * weights[..., None])
    return loss, gradient


def optimizer(lr=.04):
    return OptimizerFactory.get_optimizer(dict(type='adamw', adamw_optimizer=dict(
        lr_sched='constant_with_warmup', init_lr=lr / 4, lr=lr, lr_warmup_steps=10,
        b1=.6, b2=.8, clip_gradient=.2, weight_decay=.07)))


def warm_state(tx, count=7):
    state = TrainState.create(params={'w': jnp.array([[.2, -.2], [-.1, .1]])},
                              tx=tx, apply_fn=None)
    a, _ = batches()
    for _ in range(count):
        _, gradient = numpy_loss_grad(np.asarray(state.params['w']), a)
        state = state.apply_gradients(grads={'w': jnp.asarray(gradient, jnp.float32)})
    return state


def trainer_namespace(info):
    class SoftmaxModel:
        def apply(self, params, inputs, **kwargs):
            return NS(logits=params['w'][inputs])

    return dict(jax=jax, jnp=jnp, PS=PS, JaxRNG=JaxRNG, model=SoftmaxModel(),
        LLaMAConfigurator=NS(rng_keys=lambda: ('params', 'dropout', 'fcm')),
        with_sharding_constraint=with_sharding_constraint,
        cross_entropy_loss_and_accuracy=cross_entropy_loss_and_accuracy,
        global_norm=global_norm, optimizer_info=info, get_gpu_memory=lambda: [0])


def compiled_trainer(info):
    namespace = trainer_namespace(info)
    train = trainer_function('train_step', namespace)
    evaluate = trainer_function('eval_step', namespace)
    # Same donation contract as production, with replicated CPU state.
    train = pjit(train, in_shardings=(PS(), PS(), PS()),
                 out_shardings=(PS(), PS(), PS()), donate_argnums=(0, 1))
    evaluate = pjit(evaluate, in_shardings=(PS(), PS(), PS()),
                    out_shardings=(PS(), PS()), donate_argnums=(1,))
    return train, evaluate


def test_four_cpu_devices_partitioned_probe_evaluation():
    # A subprocess ensures device count is set before JAX initializes, even
    # when the surrounding suite has already used a single CPU device.
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=4',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env['PYTHONPATH'] = str(ROOT)
    result = subprocess.run([sys.executable, '-c',
        "import runpy; runpy.run_path('tests/test_batch_transfer.py')['partitioned_eval_worker']()"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PARTITIONED_PROBE_EVAL_OK' in result.stdout


def partitioned_eval_worker():
    assert jax.device_count() >= 4
    with Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('dp', 'fsdp')) as mesh:
        tx, info = optimizer()
        state = warm_state(tx)
        state_partition = jax.tree.map(lambda _: PS(), state)
        batch_partition = {name: PS(('dp', 'fsdp')) for name in batches()[0]}
        namespace = trainer_namespace(info)
        namespace.update(pjit=pjit, train_state_partition=state_partition,
                         batch_partition=batch_partition, FLAGS=NS(batch_transfer_probe=True))
        trainer_function('train_step', namespace)
        trainer_function('eval_step', namespace)
        # Compile the trainer's actual pjit declarations, including its input
        # contracts. Copying the contracts here would conceal the regression.
        declarations = {}
        for name in ('sharded_train_step', 'sharded_eval_step'):
            assignment = next(node for node in ast.walk(TREE) if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == name
                                      for target in node.targets))
            declarations[name] = assignment
            execute([assignment], namespace)
        sharding = NamedSharding(mesh, PS(('dp', 'fsdp')))
        pair = tuple(jax.tree.map(lambda x: jax.device_put(np.tile(x, (4, 1)), sharding), batch)
                     for batch in batches())
        for batch in pair:
            for leaf in batch.values():
                assert not leaf.is_fully_replicated
                assert len(leaf.addressable_shards) == 4
                assert {shard.data.shape for shard in leaf.addressable_shards} == {(2, 3)}
                assert len({str(shard.index) for shard in leaf.addressable_shards}) == 4
        before = np.asarray(state.params['w']).copy()
        _, param_gathers = make_shard_and_gather_fns(state_partition.params)
        _, batch_gathers = make_shard_and_gather_fns(batch_partition)
        updated, _, result = probe.measure(state, jax.random.PRNGKey(31), *pair,
            train_step=namespace['sharded_train_step'], eval_step=namespace['sharded_eval_step'],
            param_gathers=param_gathers, batch_gathers=batch_gathers, seed=42)
        after = np.asarray(updated.params['w'])
        for role, batch in zip(('update', 'transfer'), pair):
            host_batch = jax.device_get(batch)
            for when, weights in (('before', before), ('after', after)):
                assert result['metrics'][f'batch_transfer/loss_{role}_{when}'] == pytest.approx(
                    numpy_loss_grad(weights, host_batch)[0], abs=2e-7)
        assert result['optimizer_step_after'] == result['optimizer_step_before'] + 1
        # Ordinary evaluation still receives replicated inputs with its PS()
        # contract, on the same four-device mesh.
        namespace['FLAGS'].batch_transfer_probe = False
        execute([declarations['sharded_eval_step']], namespace)
        replicated = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, PS())),
                                  jax.device_get(pair[0]))
        _, metrics = namespace['sharded_eval_step'](updated.params, jax.random.PRNGKey(9), replicated)
        assert float(metrics['eval_loss']) == pytest.approx(
            numpy_loss_grad(after, jax.device_get(replicated))[0], abs=2e-7)
    print('PARTITIONED_PROBE_EVAL_OK')


def measure(state, rng, train, evaluate):
    _, param_gathers = make_shard_and_gather_fns(jax.tree.map(lambda _: PS(), state.params))
    _, batch_gathers = make_shard_and_gather_fns({name: PS(('dp', 'fsdp')) for name in batches()[0]})
    return probe.measure(state, rng, *batches(), train_step=train, eval_step=evaluate,
                         param_gathers=param_gathers, batch_gathers=batch_gathers, seed=42)


@pytest.fixture
def mesh():
    with Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('dp', 'fsdp')):
        yield


def test_native_adamw_against_independent_numpy_and_donation(mesh):
    tx, info = optimizer()
    state = warm_state(tx)
    # Independent NumPy moments, bias corrections, clipping, decay and warmup.
    w = np.array([[.2, -.2], [-.1, .1]], np.float64)
    m, v = np.zeros_like(w), np.zeros_like(w)
    a, b = batches()
    for count in range(1, 9):
        if count == 8:
            loss_a, _ = numpy_loss_grad(w, a)
            loss_b, _ = numpy_loss_grad(w, b)
        _, gradient = numpy_loss_grad(w, a)
        gradient *= min(1., .2 / np.linalg.norm(gradient))
        m = .6 * m + .4 * gradient
        v = .8 * v + .2 * gradient**2
        rate = .01 + .03 * (count - 1) / 10
        w -= rate * ((m / (1 - .6**count)) / (np.sqrt(v / (1 - .8**count)) + 1e-8) + .07 * w)
    old_parameter = state.params['w']
    old_rng = jax.random.PRNGKey(19)
    train, evaluate = compiled_trainer(info)
    updated, rng, result = measure(state, old_rng, train, evaluate)
    np.testing.assert_allclose(updated.params['w'], w, rtol=2e-6, atol=1e-7)
    metrics = result['metrics']
    expected = dict(loss_update_before=loss_a, loss_transfer_before=loss_b,
                    loss_update_after=numpy_loss_grad(w, a)[0],
                    loss_transfer_after=numpy_loss_grad(w, b)[0])
    expected.update(improvement_update=loss_a - expected['loss_update_after'],
                    improvement_transfer=loss_b - expected['loss_transfer_after'])
    for name, value in expected.items():
        assert metrics['batch_transfer/' + name] == pytest.approx(value, abs=2e-7)
    assert metrics['batch_transfer/improvement_update'] > 0
    assert metrics['batch_transfer/improvement_transfer'] < 0
    assert (result['optimizer_step_before'], result['optimizer_step_after']) == (7, 8)
    assert old_parameter.is_deleted() and old_rng.is_deleted()
    assert 'completed' not in result


def test_probe_preserves_native_state_and_training_rng(mesh):
    tx, info = optimizer()
    state = warm_state(tx)
    control = jax.tree.map(lambda x: jnp.array(x, copy=True), state)
    train, evaluate = compiled_trainer(info)
    calls = []

    def native(s, r, a):
        calls.append(('train', a))
        return train(s, r, a)

    def evaluation(p, r, batch):
        calls.append(('eval', batch))
        return evaluate(p, r, batch)

    updated, returned_rng, result = measure(state, jax.random.PRNGKey(31), native, evaluation)
    normal, normal_rng, _ = train(control, jax.random.PRNGKey(31), batches()[0])
    jax.block_until_ready((normal, normal_rng))
    for actual, expected in zip(jax.tree.leaves((updated, returned_rng)),
                                jax.tree.leaves((normal, normal_rng))):
        np.testing.assert_array_equal(actual, expected)
    assert [kind for kind, _ in calls] == ['eval', 'train', 'eval', 'eval']
    np.testing.assert_array_equal(calls[1][1]['target_tokens'], batches()[0]['target_tokens'])
    np.testing.assert_array_equal(calls[0][1]['target_tokens'], batches()[1]['target_tokens'])


def test_zero_update(mesh):
    tx, info = optimizer(lr=0.)
    train, evaluate = compiled_trainer(info)
    _, _, result = measure(warm_state(tx), jax.random.PRNGKey(0), train, evaluate)
    assert result['metrics']['batch_transfer/improvement_update'] == 0.
    assert result['metrics']['batch_transfer/improvement_transfer'] == 0.


def test_nonfinite_never_completes(mesh):
    tx, info = optimizer()
    train, _ = compiled_trainer(info)
    evaluate = lambda p, r, b: (r, {'eval_loss': jnp.nan})
    with pytest.raises(ValueError, match='nonfinite'):
        measure(warm_state(tx), jax.random.PRNGKey(0), train, evaluate)


def test_fresh_raw_hf_pair_leaves_saved_cursor_untouched(monkeypatch):
    from EasyLM import data

    class RawStream:
        def shuffle(self, **kwargs):
            assert kwargs == dict(seed=42, buffer_size=100000)
            return self

        def __iter__(self):
            yield {'text': 'x'}

    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: RawStream())
    tokenizer = NS(bos_token_id=0, eos_token_id=1, encode=lambda *a, **k: list(range(2, 80)))
    config = DatasetFactory.get_default_config(dict(text_processor=dict(fields='text'),
        huggingface_dataset=dict(batch_size=2, seq_length=3)))
    saved = DatasetFactory.load_dataset(config, tokenizer)
    saved_iterator = iter(saved)
    next(saved_iterator)
    cursor = copy.deepcopy(saved._live_state)
    original_config = config.to_dict()
    loads = []

    def load(cfg, tok):
        assert cfg is not config and tok is tokenizer
        dataset = DatasetFactory.load_dataset(cfg, tok)
        loads.append(dataset)
        return dataset

    a, b = probe.fresh_pair(NS(load_dataset=load), config, tokenizer)
    np.testing.assert_array_equal(a['input_tokens'], [[0, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(b['input_tokens'], [[7, 8, 9], [10, 11, 12]])
    assert loads[0]._live_state['metadata']['dataset_total_tokens'] == 12
    assert saved._live_state == cursor
    assert config.to_dict() == original_config
    continued, _ = next(saved_iterator)
    np.testing.assert_array_equal(continued['input_tokens'], b['input_tokens'])


def test_exhausted_pair():
    with pytest.raises(ValueError, match='two complete'):
        probe.fresh_pair(NS(load_dataset=lambda *a: []), {}, None)


def test_fingerprint_gathers_before_host_access_in_sorted_native_order():
    class RemoteLeaf:
        def __array__(self, *args, **kwargs):
            raise AssertionError('non-addressable shard was transferred before gathering')

    arrays = dict(z=np.array([3, 4], np.int32), a=np.array([1, 2], np.float32))
    calls = []

    def gather(name):
        def call(value):
            assert isinstance(value, RemoteLeaf)
            calls.append(name)
            return arrays[name]
        return call

    result = probe.fingerprint({name: RemoteLeaf() for name in arrays},
                               {name: gather(name) for name in arrays})
    assert calls == ['a', 'z']
    assert result == probe.fingerprint_leaves([((name,), arrays[name]) for name in sorted(arrays)])


def test_gcs_writer_uses_existing_file_abstraction(monkeypatch):
    import io
    stream = io.StringIO()
    opened = []

    class RemoteFile:
        def __enter__(self):
            return stream

        def __exit__(self, *args):
            pass

    def open_file(path, mode):
        opened.append((path, mode))
        return RemoteFile()

    monkeypatch.setattr(probe.os, 'makedirs', Mock(side_effect=AssertionError('local GCS mkdir')))
    probe.complete(dict(metrics={}, optimizer_step_after=501), 'gs://bucket/run', writer=True,
        open_file=open_file, barrier=Mock(), wandb=NS(log=Mock(), finish=Mock()))
    assert opened == [('gs://bucket/run/batch_transfer/metadata.json', 'w')]
    assert json.loads(stream.getvalue())['completed']


@pytest.mark.parametrize('writer', [False, True])
def test_writer_gating_and_completion_order(tmp_path, writer, capsys):
    events = []
    result = dict(metrics={'batch_transfer/improvement_transfer': -.01}, optimizer_step_after=501)
    wandb = NS(log=lambda *a, **k: events.append('log'), finish=lambda: events.append('finish'))

    def open_file(path, mode):
        events.append('write')
        return open(path, mode)

    def barrier(name):
        events.append('barrier')
        if writer:
            assert json.loads((tmp_path / 'batch_transfer/metadata.json').read_text())['completed']

    probe.complete(result, str(tmp_path), writer=writer, open_file=open_file,
                   barrier=barrier, wandb=wandb)
    assert events == (['log', 'write', 'barrier', 'finish'] if writer else ['barrier', 'finish'])
    assert ('BATCH_TRANSFER_COMPLETE' in capsys.readouterr().out) == writer
    assert (tmp_path / 'batch_transfer/metadata.json').exists() == writer


def test_strict_resume_does_not_validate_diagnostic_flag():
    flags = configuration()
    flags.update(dtype='fp32', param_dtype='fp32', total_steps=2000)
    metadata = dict(resume_version=1, optimizer_saved=True, checkpoint_float_dtype='fp32',
                    train_rng=np.array([0, 1], np.uint32), step=500, snapshot_id='s', flags=flags)
    complete = dict(resume_version=1, step=500, snapshot_id='s')
    validate_resume_metadata(metadata, complete, dict(flags, batch_transfer_probe=True))
    with pytest.raises(ValueError, match='trajectory settings'):
        validate_resume_metadata(metadata, complete, dict(flags, batch_transfer_probe=True, total_steps=1))
    with pytest.raises(ValueError, match='trajectory settings'):
        validate_resume_metadata(metadata, complete, dict(flags, dtype='bf16'))


def test_disabled_flag_has_no_probe_side_effects_and_branch_returns():
    main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    blocks = [node for node in ast.walk(main) if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'FLAGS.batch_transfer_probe']
    assert len(blocks) == 2
    namespace = dict(FLAGS=NS(batch_transfer_probe=False))
    wrapper = ast.parse('def disabled():\n    pass').body[0]
    wrapper.body = blocks
    execute([wrapper], namespace)
    namespace['disabled']()  # No datasets, model, W&B, gather, or file APIs supplied.
    branch = blocks[1]
    assert isinstance(branch.body[-1], ast.Return)
    # Bypass the initial checkpoint writes, old_params copy, initial eval and loop.
    first_save = next(node for node in main.body if isinstance(node, ast.With))
    body = first_save.body
    branch_index = body.index(branch)
    assert all('save_checkpoint(' not in ast.unparse(node) for node in body[:branch_index])
    assert any('save_checkpoint(' in ast.unparse(node) for node in body[branch_index + 1:])
    assert 'old_params' not in ast.unparse(branch)


@pytest.mark.parametrize('restore_optimizer', [True, False])
def test_main_with_tiny_checkpoint_and_real_compiled_llama(tmp_path, monkeypatch, restore_optimizer):
    """Real checkpoint load, native pjit train/eval, raw packing and sidecar write."""
    from EasyLM.models.llama import llama_train as trainer
    from EasyLM import data
    from EasyLM.checkpoint import StreamingCheckpointer
    from EasyLM.models.llama.llama_model import LLaMAConfigurator, FlaxLLaMAForCausalLMModule

    class Tokenizer:
        name_or_path = 'cpu-tokenizer'
        bos_token_id, eos_token_id = 0, 1

        def __len__(self):
            return 16

        def encode(self, *args, **kwargs):
            return [2 + i % 14 for i in range(300)]

    class RawStream:
        def shuffle(self, **kwargs):
            return self

        def __iter__(self):
            yield {'text': 'cpu sample'}

    monkeypatch.setattr(data, 'load_dataset', lambda *a, **k: RawStream())
    tokenizer = Tokenizer()
    flags = ConfigDict(copy.deepcopy(trainer.FLAGS_DEF))
    flags.update(dict(batch_transfer_probe=True, mesh_dim='1,1,1', total_steps=2000,
        output_dir=str(tmp_path / 'results'), experiment_id='probe', wandb_dir=str(tmp_path),
        save_model_freq=1, save_milestone_freq=1, eval_steps=2, eval_freq=2,
        log_initial_eval=True, train_dataset_batch_size=2, tokenizer='cpu-tokenizer'))
    flags.train_dataset.text_processor.fields = 'text'
    flags.train_dataset.huggingface_dataset.update(dict(batch_size=2, seq_length=4))
    flags.checkpointer.update(dict(save_optimizer_state=True, float_dtype='fp32'))
    flags.llama.update(dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=16, max_position_embeddings=8))
    config = LLaMAConfigurator.finalize_config(flags.llama)
    model = FlaxLLaMAForCausalLMModule(config, dtype=jnp.float32, param_dtype=jnp.float32)
    params = model.init(input_ids=jnp.zeros((2, 4), jnp.int32),
        rngs=JaxRNG(jax.random.PRNGKey(3))(LLaMAConfigurator.rng_keys()))
    tx, _ = OptimizerFactory.get_optimizer(flags.optimizer)
    state = TrainState.create(params=params, tx=tx, apply_fn=None)
    for _ in range(7):
        state = state.apply_gradients(grads=jax.tree.map(lambda p: jnp.full_like(p, .01), params))
    parent = tmp_path / 'parent'
    parent.mkdir()
    checkpoint = parent / 'streaming_train_state_7'
    StreamingCheckpointer.save_train_state_to_file(state, str(checkpoint), float_dtype='fp32')
    flags.load_checkpoint = ('trainstate::' if restore_optimizer else 'trainstate_params::') + str(checkpoint)
    parent_dataset = DatasetFactory.load_dataset(flags.train_dataset, tokenizer)
    parent_iterator = iter(parent_dataset)
    for _ in range(7):
        next(parent_iterator)
    snapshot = parent_dataset.get_state_dict()
    snapshot.update(training_step=7, snapshot_id='cpu-snapshot')
    native_rng = np.array([123, 456], np.uint32)
    metadata = dict(step=7, resume_version=1, snapshot_id='cpu-snapshot',
        flags=ConfigDict(flags.to_dict()), optimizer_saved=True, checkpoint_float_dtype='fp32', train_rng=native_rng)
    for name, value in dict(metadata=metadata, dataset=snapshot,
            complete=dict(step=7, resume_version=1, snapshot_id='cpu-snapshot')).items():
        with open(parent / f'{name}_7.pkl', 'wb') as stream:
            pickle.dump(value, stream)

    monkeypatch.setattr(trainer, 'FLAGS', flags)
    monkeypatch.setattr(trainer.mlxu, 'get_user_flags', lambda *a: {})
    monkeypatch.setattr(trainer.mlxu, 'user_flags_to_config_dict', lambda *a: ConfigDict(flags.to_dict()))
    monkeypatch.setattr(trainer.AutoTokenizer, 'from_pretrained', lambda *a: tokenizer)
    forbidden_save = Mock(side_effect=AssertionError('probe attempted a training checkpoint write'))
    monkeypatch.setattr(StreamingCheckpointer, 'save_all', forbidden_save)
    monkeypatch.setattr(StreamingCheckpointer, 'save_pickle', forbidden_save)
    run = NS(config=NS(update=Mock()), id='cpu-run')
    fake_wandb = NS(init=Mock(), config=run.config, run=run, log=Mock(), finish=Mock())
    monkeypatch.setattr(trainer, 'wandb', fake_wandb)
    datasets = []
    original_load = DatasetFactory.load_dataset

    def load_dataset(*args, **kwargs):
        dataset = original_load(*args, **kwargs)
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(DatasetFactory, 'load_dataset', load_dataset)
    computations = []
    original_pjit = trainer.pjit

    def record_pjit(function, **kwargs):
        compiled = original_pjit(function, **kwargs)

        def call(*args):
            computations.append(function.__name__)
            if restore_optimizer and function.__name__ == 'train_step':
                np.testing.assert_array_equal(args[1], native_rng)
                assert int(args[0].step) == 7
                for restored, expected in zip(jax.tree.leaves(args[0].opt_state),
                                              jax.tree.leaves(state.opt_state)):
                    np.testing.assert_array_equal(restored, expected)
            return compiled(*args)
        return call

    monkeypatch.setattr(trainer, 'pjit', record_pjit)
    trainer.main([])
    result = json.loads((tmp_path / 'results/probe/batch_transfer/metadata.json').read_text())
    assert result['completed'] and result['method'] == 'first_order_adamw'
    assert result['checkpoint_uri'] == flags.load_checkpoint
    assert result['optimizer_state_source'] == ('restored' if restore_optimizer else 'fresh')
    assert result['optimizer_step_before'] == (7 if restore_optimizer else 0)
    assert result['optimizer_step_after'] == (8 if restore_optimizer else 1)
    assert result['total_steps'] == 2000
    assert computations[-4:] == ['eval_step', 'train_step', 'eval_step', 'eval_step']
    assert len(datasets) == 2  # restored training loader plus independent diagnostic loader
    if restore_optimizer:
        assert datasets[0].get_state_dict() == {k: v for k, v in snapshot.items()
                                               if k not in ('training_step', 'snapshot_id')}
    assert datasets[1].metadata['dataset_total_tokens'] == 16
    assert [p.name for p in (tmp_path / 'results/probe').rglob('*') if p.is_file()] == ['metadata.json']
    forbidden_save.assert_not_called()
    fake_wandb.log.assert_called_once_with(result['metrics'], step=result['optimizer_step_after'], commit=True)
    fake_wandb.finish.assert_called_once()
