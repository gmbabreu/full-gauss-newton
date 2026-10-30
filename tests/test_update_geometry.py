"""CPU SPD ground truth, stream compatibility, and trainer integration gates."""
import ast
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from ml_collections import ConfigDict

from EasyLM import update_geometry as geometry
from EasyLM.checkpoint import StreamingCheckpointer


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'EasyLM/models/llama/llama_train_gn.py'
TREE = ast.parse(TRAINER.read_text())


def function(name):
    return next(node for node in ast.walk(TREE)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(TRAINER), 'exec'), namespace)


def parameter_tree(vector):
    # Includes a matrix and a non-matrix, with nonzero cross-layer coupling in G.
    return {'params': {'matrix': np.asarray(vector[:4], np.float32).reshape(2, 2),
                       'bias': np.asarray(vector[4:], np.float32)}}


def write_bundle(directory, vector, product):
    directory.mkdir(parents=True, exist_ok=True)
    tree, gtree = parameter_tree(vector), parameter_tree(product)
    gathers = jax.tree.map(lambda _: lambda value: np.asarray(value), tree)
    for values, name in ((tree, geometry.DIRECTION_FILE), (gtree, geometry.PRODUCT_FILE)):
        geometry.save_parameter_tree(values, str(directory / name), gathers, writer=True)
    base = parameter_tree(np.arange(6))
    metadata = dict(
        format_version=geometry.FORMAT_VERSION,
        base_parameters=geometry.fingerprint_leaves(sorted(flatten_dict(base).items())),
        solve_batch=geometry.fingerprint_leaves([
            ((name,), np.ones((2, 3), np.int32 if name != 'loss_masks' else np.float32))
            for name in ('input_tokens', 'target_tokens', 'loss_masks')]),
        parameters=[dict(path=list(key), shape=list(value.shape), dtype='float32')
                    for key, value in flatten_dict(tree).items()],
        gn_operator=dict(model_config={'name': 'explicit_spd'}, compute_dtype='fp32',
                         parameter_dtype='fp32', microbatch_groups=[[0, 1, 2]]),
        solver=dict(name='arbitrary', iteration_budget=3))
    (directory / geometry.METADATA_FILE).write_text(json.dumps(metadata))
    return metadata


@pytest.fixture
def spd_pair(tmp_path):
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(6, 6)).astype(np.float32)
    matrix = matrix.T @ matrix + np.eye(6, dtype=np.float32)
    a = np.array([1, -2, 3, -1, .5, 4], np.float32)
    c = np.array([2, 1, -1, 2, 3, -.5], np.float32)
    ga, gc = matrix @ a, matrix @ c
    left, right = tmp_path / 'a', tmp_path / 'c'
    write_bundle(left, a, ga)
    write_bundle(right, c, gc)
    return left, right, matrix.astype(np.float64), a.astype(np.float64), c.astype(np.float64)


def test_spd_cosines_cross_terms_and_difference(spd_pair):
    left, right, matrix, a, c = spd_pair
    result, matrices = geometry.compare_bundles(left, right)
    assert result['euclidean_cosine'] == pytest.approx(a @ c / np.linalg.norm(a) / np.linalg.norm(c))
    assert result['gn_cosine'] == pytest.approx(
        (a @ matrix @ c) / np.sqrt((a @ matrix @ a) * (c @ matrix @ c)))
    assert result['a_Gc'] == pytest.approx(a @ matrix @ c)
    assert result['c_Ga'] == pytest.approx(c @ matrix @ a)
    assert result['cross_symmetry_relative_discrepancy'] < 1e-6
    assert result['rayleigh_a'] == pytest.approx(a @ matrix @ a / (a @ a))
    assert result['rayleigh_c'] == pytest.approx(c @ matrix @ c / (c @ c))
    e = c - (a @ c / (a @ a)) * a
    assert result['difference_norm'] == pytest.approx(np.linalg.norm(e))
    assert result['difference_e_G_e'] == pytest.approx(e @ matrix @ e)
    assert result['difference_rayleigh'] == pytest.approx(e @ matrix @ e / (e @ e))
    assert len(matrices) == 1
    assert matrices[0]['path'] == 'params/matrix'
    assert matrices[0]['euclidean_cosine'] == pytest.approx(
        a[:4] @ c[:4] / np.linalg.norm(a[:4]) / np.linalg.norm(c[:4]))
    assert not any('gn_cosine' in key for key in matrices[0])


@pytest.mark.parametrize('case', ['zero_a', 'zero_c', 'parallel', 'negative_g', 'zero_g'])
def test_degenerate_metrics_are_explicit(tmp_path, case):
    a, c = np.arange(1, 7, dtype=np.float32), np.arange(6, 0, -1, dtype=np.float32)
    if case == 'zero_a':
        a[:] = 0
    if case == 'zero_c':
        c[:] = 0
    if case == 'parallel':
        c = 2 * a
    ga, gc = a.copy(), c.copy()
    if case == 'negative_g':
        ga, gc = -a, -c
    if case == 'zero_g':
        ga[:], gc[:] = 0, 0
    left, right = tmp_path / 'a', tmp_path / 'c'
    write_bundle(left, a, ga)
    write_bundle(right, c, gc)
    result, matrices = geometry.compare_bundles(left, right)
    json.dumps(result, allow_nan=False)
    if case in ('zero_a', 'zero_c'):
        assert result['euclidean_cosine'] is None
        assert result['gn_cosine'] is None
        assert matrices[0]['euclidean_cosine'] is None
    if case == 'zero_a':
        assert result['difference_norm'] is None
        assert result['difference_rayleigh'] is None
        assert result['rayleigh_a'] is None
    if case in ('zero_c', 'parallel'):
        assert result['difference_norm'] == 0
        assert result['difference_rayleigh'] is None
    if case in ('negative_g', 'zero_g'):
        assert result['gn_cosine'] is None


@pytest.mark.parametrize('key', ['base_parameters', 'solve_batch', 'parameters', 'gn_operator'])
def test_incompatible_bundles_rejected(spd_pair, key):
    left, right, *_ = spd_pair
    path = right / geometry.METADATA_FILE
    metadata = json.loads(path.read_text())
    if key in ('base_parameters', 'solve_batch'):
        metadata[key]['sha256'] = 'different contents'
    elif key == 'parameters':
        metadata[key][0]['shape'] = [100]
    else:
        metadata[key]['compute_dtype'] = 'bf16'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=key):
        geometry.compare_bundles(left, right)


def test_solver_differences_allowed_but_missing_identity_rejected(spd_pair):
    left, right, *_ = spd_pair
    path = right / geometry.METADATA_FILE
    metadata = json.loads(path.read_text())
    metadata['solver'] = {'optimizer': 'cg', 'damping': 100., 'maxiter': 99}
    path.write_text(json.dumps(metadata))
    geometry.compare_bundles(left, right)
    del metadata['solve_batch']['sha256']
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='fingerprint'):
        geometry.compare_bundles(left, right)


def test_payload_structure_verified_against_sidecar(spd_pair):
    left, right, *_ = spd_pair
    path = right / geometry.DIRECTION_FILE
    StreamingCheckpointer.save_train_state_to_file(
        {'wrong_path': np.ones((2, 2), np.float32)}, str(path), float_dtype='fp32')
    with pytest.raises(ValueError, match='mismatch'):
        geometry.compare_bundles(left, right)


def test_stream_roundtrip_and_nonwriter_collective_order(tmp_path):
    tree = {'params': {'z': np.array([2, 3], np.float64),
                       'a': np.arange(6, dtype=np.float32).reshape(2, 3),
                       'bf16': np.asarray(jnp.array([1., 2.], dtype=jnp.bfloat16))}}
    order = []
    def gather(name):
        def collect(value):
            order.append(name)
            return np.asarray(value)
        return collect
    gathers = {'params': {name: gather(name) for name in tree['params']}}
    path = tmp_path / 'tree.msgpack'
    geometry.save_parameter_tree(tree, str(path), gathers, writer=True)
    writer_order = order.copy()
    order.clear()
    nonwriter_path = tmp_path / 'nonwriter' / 'must_not_exist.msgpack'
    geometry.save_parameter_tree(tree, str(nonwriter_path), gathers, writer=False)
    assert not nonwriter_path.parent.exists()
    assert order == writer_order == ['z', 'a', 'bf16']
    restored = dict(geometry.iter_parameter_leaves(path))
    for key, value in flatten_dict(tree).items():
        np.testing.assert_array_equal(restored[key], value)
        assert restored[key].dtype == np.float32
    # The repository's loader must read exactly the same serialized tree.
    restored_existing = flatten_dict(StreamingCheckpointer.load_checkpoint(str(path)))
    assert restored.keys() == restored_existing.keys()
    for key in restored:
        np.testing.assert_array_equal(restored[key], restored_existing[key])


def test_fingerprints_cover_paths_values_shapes_dtypes_and_all_batch_fields():
    leaves = [((name,), np.arange(6, dtype=np.int32).reshape(2, 3))
              for name in ('input_tokens', 'target_tokens', 'loss_masks')]
    original = geometry.fingerprint_leaves(leaves)
    assert original == geometry.fingerprint_leaves(leaves)
    for index in range(3):
        changed = copy.deepcopy(leaves)
        changed[index][1][0, 0] += 1
        assert original['sha256'] != geometry.fingerprint_leaves(changed)['sha256']
    for value in (leaves[0][1].reshape(3, 2), leaves[0][1].astype(np.int64)):
        assert original['sha256'] != geometry.fingerprint_leaves(
            [(leaves[0][0], value), *leaves[1:]])['sha256']
    assert original['sha256'] != geometry.fingerprint_leaves(
        [(('other',), leaves[0][1]), *leaves[1:]])['sha256']
    bf16 = np.asarray(jnp.array([1., 2.], dtype=jnp.bfloat16))
    assert geometry.fingerprint_leaves([(('w',), bf16)])['leaves'][0]['dtype'] == 'bfloat16'


def test_cli_needs_only_cpu_numpy_and_msgpack(spd_pair, tmp_path):
    left, right, *_ = spd_pair
    prefix = tmp_path / 'report'
    code = ("import runpy, sys; sys.modules['jax'] = None; sys.modules['flax'] = None; "
            f"sys.argv = ['compare', {str(left)!r}, {str(right)!r}, "
            f"'--output-prefix', {str(prefix)!r}]; "
            f"runpy.run_path({str(ROOT / 'scripts/compare_update_geometry.py')!r}, run_name='__main__')")
    process = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'gn_cosine:' in process.stdout
    assert json.loads(prefix.with_suffix('.json').read_text())['gn_cosine'] is not None
    with open(str(prefix) + '_per_matrix.csv') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['path'] for row in rows] == ['params/matrix']


def test_actual_trainer_capture_one_raw_product_and_original_base(tmp_path):
    base = parameter_tree(np.arange(6))
    candidate = parameter_tree(np.arange(6) + 2)
    solve = dict(input_tokens=np.ones((2, 3), np.int32),
                 target_tokens=np.zeros((2, 3), np.int32), loss_masks=np.ones((2, 3), np.float32))
    calls = []
    def product(params, batch, direction, wd):
        calls.append((params, batch, direction, wd))
        return jax.tree.map(lambda value: value * 3, direction)
    flags_node = next(node for node in ast.walk(TREE) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) == 'mlxu.define_flags_with_default')
    flags_dict = {}
    for keyword in flags_node.keywords:
        try:
            flags_dict[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            flags_dict[keyword.arg] = {}
    flags_dict.update(optimizer_type='adamw', cg_n_micro=1)
    gathers = jax.tree.map(lambda _: lambda value: np.asarray(value), base)
    ns = dict(jax=jax, jnp=jnp, np=np, os=os, sp=subprocess, __file__=str(TRAINER),
              update_geometry=geometry, flatten_dict=flatten_dict,
              sharded_condition_apply_g=product, geometry_param_gathers=gathers,
              geometry_batch_gathers={key: np.asarray for key in solve},
              output_dir=str(tmp_path), checkpointer=NS(enable=True), data_shards=1,
              FLAGS=NS(**flags_dict), flags_config_dict=ConfigDict(flags_dict),
              llama_config=ConfigDict({'hidden_size': 2}),
              init_checkpoint_path='trainstate_params::gs://original/checkpoint')
    execute([function('microbatch_groups'), function('capture_update_geometry')], ns)
    metadata = ns['capture_update_geometry'](base, candidate, solve)
    assert len(calls) == 1
    assert calls[0][0] is base and calls[0][1] is solve
    assert float(calls[0][3]) == 0  # no damping/weight decay
    assert metadata['checkpoint_uri'] == 'trainstate_params::gs://original/checkpoint'
    assert metadata['base_parameters'] == geometry.fingerprint_leaves(sorted(flatten_dict(base).items()))
    assert metadata['solve_batch'] == geometry.fingerprint_leaves(
        sorted(((key,), value) for key, value in solve.items()))
    # In particular the nested optimizer ConfigDict is JSON serializable.
    json.dumps(metadata, allow_nan=False)
    for name, expected in ((geometry.DIRECTION_FILE, 2.), (geometry.PRODUCT_FILE, 6.)):
        for _, leaf in geometry.iter_parameter_leaves(tmp_path / 'update_geometry' / name):
            np.testing.assert_array_equal(leaf, np.full_like(leaf, expected))


@pytest.mark.parametrize('override', [
    {'single_batch_inner': False}, {'adaptive_inner_loop': True},
    {'normalize_step': True}, {'outer_momentum_beta': .1}, {'outer_weight_decay': .1},
    {'gauss_newton': False}, {'optimizer_type': 'soap'}, {'inner_loop_iter': 0},
    {'load_checkpoint': 'trainstate::checkpoint'}, {'cg_resume_state': 'resume'},
])
def test_export_restrictions(override):
    flags = NS(export_update_geometry=True, single_batch_inner=True,
               adaptive_inner_loop=False, normalize_step=False, outer_momentum_beta=0.,
               outer_weight_decay=0., gauss_newton=True, optimizer_type='adamw',
               inner_loop_iter=3, cg_maxiter=5, log_freq=50,
               load_checkpoint='trainstate_params::checkpoint', cg_resume_state='')
    for key, value in override.items():
        setattr(flags, key, value)
    ns = dict(FLAGS=flags)
    execute([function('supports_condition_diagnostics')], ns)
    guard = next(node for node in function('main').body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'FLAGS.export_update_geometry')
    with pytest.raises(ValueError):
        execute([guard], ns)


@pytest.mark.parametrize('field', ['embedding_dropout', 'feedforward_dropout',
                                 'attention_dropout', 'residue_dropout',
                                 'fcm_min_ratio', 'fcm_max_ratio'])
def test_zero_dropout_required_even_when_quadratic_logging_disabled(field):
    guard = next(node for node in ast.walk(function('main')) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'FLAGS.log_gn_quadratic or FLAGS.export_update_geometry')
    with pytest.raises(ValueError, match='zero dropout and FCM'):
        execute([guard], dict(FLAGS=NS(log_gn_quadratic=False, export_update_geometry=True),
                              llama_config=NS(**{field: .1})))


def test_disabled_export_does_not_create_gather_functions():
    guards = [node for node in function('main').body if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'FLAGS.export_update_geometry']
    factory = Mock(side_effect=AssertionError('Unexpected gather setup'))
    execute(guards, dict(FLAGS=NS(export_update_geometry=False),
                         make_shard_and_gather_fns=factory))
    factory.assert_not_called()


@pytest.mark.parametrize('enabled,writer', [(True, True), (True, False), (False, True)])
def test_sidecar_after_acceptance_and_disabled_write_gate(tmp_path, enabled, writer):
    block = next(node for node in ast.walk(function('main')) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == 'geometry_capture is not None')
    bundle = tmp_path / 'update_geometry'
    bundle.mkdir()
    opened, barriers = [], []
    def open_file(path, mode):
        opened.append(path)
        return open(path, mode)
    ns = dict(geometry_capture={'format_version': 1} if enabled else None,
              jax=jax, np=np, json=json, os=os, step=0, effective_step_size=.125,
              metrics={'gn_quad/g_dot_d': jnp.float32(-2.),
                       'gn_quad/d_G_d': jnp.float32(3.),
                       'gn_quad/quadratic_fidelity': jnp.float32(jnp.nan), 'other': 7},
              checkpointer=NS(enable=writer), output_dir=str(tmp_path),
              update_geometry=geometry, mlxu=NS(open_file=open_file),
              multihost_utils=NS(sync_global_devices=barriers.append),
              geometry_exported=False)
    execute([block], ns)
    assert len(opened) == int(enabled and writer)
    assert len(barriers) == int(enabled)
    assert ns['geometry_exported'] == enabled
    if enabled and writer:
        metadata = geometry.read_metadata(bundle)
        assert metadata['accepted_scalar_step'] == .125
        assert metadata['outer_step'] == 0
        assert metadata['metrics'] == {'gn_quad/g_dot_d': -2., 'gn_quad/d_G_d': 3.,
                                       'gn_quad/quadratic_fidelity': None}


def test_four_device_streamed_export(tmp_path):
    # Actual replication gathers on partitioned arrays, in an isolated CPU
    # runtime. This checks the existing checkpoint gather functions used above.
    code = '''
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
from pathlib import Path
import sys
import jax
import numpy as np
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh, PartitionSpec as P
from EasyLM.jax_utils import make_shard_and_gather_fns
from EasyLM import update_geometry as g
directory = Path(sys.argv[1])
tree = {'params': {'matrix': np.arange(32, dtype=np.float32).reshape(8, 4),
                   'bias': np.arange(8, dtype=np.float32)}}
spec = jax.tree.map(lambda _: P('x'), tree)
with Mesh(np.array(jax.devices()), ('x',)):
    shards, gathers = make_shard_and_gather_fns(spec)
    distributed = jax.tree.map(lambda x, put: put(x), tree, shards)
    assert all(len(x.addressable_shards) == 4 for x in jax.tree.leaves(distributed))
    fingerprint = g.fingerprint_leaves(
        (key, flatten_dict(gathers)[key](value))
        for key, value in sorted(flatten_dict(distributed).items()))
    assert fingerprint == g.fingerprint_leaves(sorted(flatten_dict(tree).items()))
    g.save_parameter_tree(distributed, str(directory / 'd.msgpack'), gathers, writer=True)
    loaded = dict(g.iter_parameter_leaves(directory / 'd.msgpack'))
    for key, expected in flatten_dict(tree).items():
        np.testing.assert_array_equal(loaded[key], expected)
print('FOUR_DEVICE_EXPORT_OK')
'''
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=4',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    process = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env,
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'FOUR_DEVICE_EXPORT_OK' in process.stdout


def test_stop_flag_forces_log_before_break_off_cadence():
    should_log = next(node for node in ast.walk(TREE) if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == 'should_log')
    log_block = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == 'should_log')
    logs = []
    ns = dict(FLAGS=NS(log_freq=50, train_batch_growth_interval=0, optimizer_type='adamw',
                       stop_after_first_outer_step=True, eval_freq=0, eval_steps=0),
              step=7, do_condition=False, metrics={'completed': 1},
              get_tpu_metrics=lambda: {}, global_norm=lambda _: 2.,
              train_state=NS(params=None), jax=jax, pending_record={},
              timing=NS(metrics=lambda: {}), progress=NS(record=lambda step, **kwargs: kwargs),
              inner_diagnostic_rows=[], wandb=NS(log=lambda row, **kwargs: logs.append(row)),
              tqdm=NS(write=lambda _: None), pprint=NS(pformat=str))
    # A second iteration must never run; the actual trainer log block breaks.
    loop = ast.For(target=ast.Name(id='unused', ctx=ast.Store()),
                   iter=ast.List(elts=[ast.Constant(0), ast.Constant(1)], ctx=ast.Load()),
                   body=[should_log, log_block], orelse=[])
    execute([loop], ns)
    assert len(logs) == 1 and logs[0]['completed'] == 1
    # No mutation of the LR schedule's horizon is used to implement stopping.
    assert not any(isinstance(node, (ast.Assign, ast.AugAssign))
                   and any(isinstance(target, ast.Attribute) and target.attr == 'total_steps'
                           for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
                   for node in ast.walk(TREE))
