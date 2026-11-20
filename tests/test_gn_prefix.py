"""CPU execution of actual trainer helpers with a tiny nonlinear model.

AST extraction avoids TPU SDK, tokenizer and cloud-service imports; optimizer,
GN inner step, quadratic, loss, host commit and line search are production code.
"""
import ast
import copy
from contextlib import redirect_stdout
import io
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
import weakref
from unittest import mock

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict

from EasyLM import gn_prefix
from EasyLM.training_progress import ProcessTiming

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'EasyLM/models/llama/llama_train_gn.py'
TREE = ast.parse(SOURCE.read_text())
UTILS = ast.parse((ROOT / 'EasyLM/jax_utils.py').read_text())


def node(name, tree=TREE):
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name == name)


def execute(nodes, ns, loop=False):
    if loop:
        nodes = [ast.For(target=ast.Name(id='_once', ctx=ast.Store()),
                         iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
                         body=nodes, orelse=[])]
    # Lift the existing closure variable to module scope in this isolated harness.
    nodes = copy.deepcopy(nodes)
    for function in nodes:
        if isinstance(function, ast.FunctionDef):
            function.body = [ast.Global(names=n.names) if isinstance(n, ast.Nonlocal) else n
                             for n in function.body]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(SOURCE), 'exec'), ns)


class State(TrainState):
    warmstart_params: object = None


class Model:
    def apply(self, p, inputs, **kwargs):
        p = p['params']
        return SimpleNamespace(logits=jnp.tanh(inputs @ p['block']['kernel']) @
                               p['lm_head']['kernel'])


def flags(**kwargs):
    result = dict(gn_prefix_probe_steps='0', gn_prefix_ks='10,20,50,100',
                  gn_prefix_holdout_batches=2, gauss_newton=True, optimizer_type='adamw',
                  inner_loop_iter=10, single_batch_inner=True, adaptive_inner_loop=False,
                  reset_start=False, linesearch=True, ls_eval_batches=1, weight_average=False,
                  normalize_step=False, outer_momentum_beta=0, outer_weight_decay=0,
                  fixed_step_size=0, ls_reuse_lag=0, inner_loop_wd=0, lr_sched='global_cosine',
                  n_micro=1, log_inner_steps=False, log_gn_quadratic=False,
                  log_update_balance=False, export_update_geometry=False, log_freq=1,
                  armijo_linesearch=False, armijo_init_step=1., armijo_beta=.5,
                  patience=1, ls_range=5)
    result.update(kwargs)
    return SimpleNamespace(**result)


def setup(optimizer='adamw', armijo=False, n_micro=1):
    f = flags(optimizer_type=optimizer, armijo_linesearch=armijo, n_micro=n_micro)
    ns = dict(jax=jax, jnp=jnp, optax=optax, FLAGS=f, model=Model(),
              linearize=jax.linearize, linear_transpose=jax.linear_transpose,
              flatten_dict=flatten_dict, unflatten_dict=unflatten_dict,
              with_sharding_constraint=lambda x, spec: x, PS=lambda *args: None,
              data_shards=1, get_gpu_memory=lambda: [0],
              global_norm=gn_prefix.norm, LLaMAConfigurator=SimpleNamespace(rng_keys=lambda: ('dropout',)))
    for name in ('cross_entropy_loss_and_accuracy',
                 'cross_entropy_loss_and_accuracy_with_weight_decay', 'JaxRNG'):
        execute([node(name, UTILS)], ns)
    for name in ('get_global_lr_sched', 'build_optimizer', 'train_step_gauss_newton',
                 'gn_quadratic_summary', 'measure_gn_quadratic', 'gn_quadratic_metrics',
                 'loss_fn', 'microbatch_groups', 'weighted_microbatch_sum',
                 'iterative_gn_microbatch', 'microbatched_loss_fn', 'run_linesearch'):
        execute([node(name)], ns)
    ns['lr_sched'] = ns['get_global_lr_sched']('global_cosine', .03, 200, 10, .1, 0, .001)
    tx = ns['build_optimizer'](ns['lr_sched'], .9, .99, 1., .01, optimizer)
    p = {'params': {'block': {'kernel': jnp.array([[.6, -.2], [.3, .7]])},
                    'lm_head': {'kernel': jnp.array([[.5, -.3], [-.2, .6]])}}}
    state = State.create(params=p, tx=tx, apply_fn=None, warmstart_params=p)
    ns['parallel_loss_fn'] = jax.jit(ns['loss_fn'])
    ns['apply_outer_decay'] = lambda base, updated: updated
    ns['inner'] = jax.jit(ns['train_step_gauss_newton'])
    ns['quad'] = jax.jit(lambda p, q, r, b: ns['gn_quadratic_metrics'](p, q, r, b, prefix=True))
    batch = dict(input_tokens=jnp.array([[[1., .2], [.3, -.4]], [[-.8, .7], [.2, .5]]]),
                 target_tokens=jnp.array([[0, 1], [1, 0]]), loss_masks=jnp.array([[1., 0.], [1., 1.]]))
    return ns, state, batch


def arrays_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


class PrefixTest(unittest.TestCase):
    def test_standalone_preset_uses_existing_main_launcher(self):
        spec = importlib.util.spec_from_file_location('prefix_run_experiment',
                                                    ROOT / 'scripts/run_experiment.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        plan = launcher.resolve('gn_prefix_probe', [])
        f = flags(**{k: v for k, v in plan['flags'].items() if hasattr(flags(), k)})
        steps, ks = gn_prefix.validate(f)
        self.assertEqual(ks, (10, 20, 50, 100))
        self.assertEqual(steps, {0, 10, 100, 1000, 2500})
        self.assertEqual(plan['program'], 'EasyLM.models.llama.llama_train_gn')
        self.assertEqual(plan['route'], 'sweep')
        ordinary = launcher.resolve('gn_adam', [])['flags']
        self.assertEqual(int(ordinary['inner_loop_iter']), 50)
        self.assertEqual(float(ordinary['inner_loop_wd']), .01)
        self.assertFalse(any(k.startswith('gn_prefix') for k in ordinary))

    def test_validation_and_disabled_noop(self):
        self.assertEqual(gn_prefix.validate(flags(gn_prefix_probe_steps='', inner_loop_iter=99)),
                         (frozenset(), ()))
        for changes in (dict(gn_prefix_ks='20,50'), dict(gn_prefix_ks='10,10'),
                        dict(gn_prefix_probe_steps='-1'), dict(gn_prefix_holdout_batches=0),
                        dict(optimizer_type='cg'), dict(inner_loop_wd=.1), dict(lr_sched='constant'),
                        dict(reset_start=True), dict(normalize_step=True), dict(ls_reuse_lag=1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gn_prefix.validate(flags(**changes))
        self.assertEqual(gn_prefix.validate(flags())[1], (10, 20, 50, 100))

    def test_off_cadence_probe_row_does_not_trigger_normal_evaluation(self):
        assignments = {ast.unparse(n.targets[0]): n for n in ast.walk(node('main'))
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and ast.unparse(n.targets[0]) in ('should_log', 'do_eval')}
        f = flags(log_freq=10, eval_freq=1, eval_steps=2,
                  train_batch_growth_interval=0, stop_after_first_outer_step=False)
        for enabled in (False, True):
            ns = dict(FLAGS=f, step=3, schedule_steps=100, do_condition=False, do_prefix=enabled)
            execute([assignments['should_log'], assignments['do_eval']], ns)
            self.assertEqual(ns['should_log'], enabled)
            self.assertFalse(ns['do_eval'])

    def test_prefixes_dense_ground_truth_and_measurements(self):
        for optimizer in ('adamw', 'muon'):
            for armijo in (False, True):
                with self.subTest(optimizer=optimizer, armijo=armijo), redirect_stdout(io.StringIO()):
                    ns, initial, batch = setup(optimizer, armijo)
                    base = initial.params
                    flat, unravel = ravel_pytree(base)
                    # Independent dense J and logits-space Hessian (NOT parameter Hessian).
                    z = lambda v: ns['model'].apply(unravel(v), batch['input_tokens']).logits
                    loss_z = lambda logits: ns['cross_entropy_loss_and_accuracy'](
                        logits, batch['target_tokens'], batch['loss_masks'])[0]
                    logits = z(flat)
                    jac = jax.jacrev(z)(flat).reshape((-1, len(flat)))
                    h = jax.hessian(lambda zz: loss_z(zz.reshape(logits.shape)))(logits.ravel())
                    g = jac.T @ jax.grad(loss_z)(logits).ravel()
                    G = jac.T @ h @ jac
                    rng = jax.random.PRNGKey(37)
                    state = initial
                    expected, used = {}, []
                    for i in range(100):
                        counter, lr = gn_prefix.optimizer_usage(state, ns['lr_sched'])
                        self.assertEqual(counter, i)
                        self.assertEqual(lr, float(ns['lr_sched'](i)))
                        # Independently use real Optax with the dense residual.
                        residual = unravel(g + G @ (ravel_pytree(state.params)[0] - flat))
                        independent = state.apply_gradients(grads=residual)
                        state, rng, _ = ns['inner'](state, base, rng, batch, 0., jnp.bool_(False))
                        np.testing.assert_allclose(ravel_pytree(state.params)[0],
                                                   ravel_pytree(independent.params)[0], atol=2e-6)
                        used.append((counter, lr))
                        if i + 1 in (10, 20, 50, 100):
                            expected[i + 1] = (state, rng)
                    k10, key10 = expected[10]
                    direction = jax.tree.map(lambda p, b: p - b, k10.params, base)
                    ls_rngs = [jax.random.PRNGKey(12)]
                    alpha, losses = ns['run_linesearch'](base, direction, [batch], ls_rngs)
                    holdout = [dict(batch, input_tokens=batch['input_tokens'] * scale)
                               for scale in (.6, 1.4)]
                    measured_states = []
                    def quadratic(p, q, r, b):
                        k = (10, 20, 50, 100)[len(measured_states)]
                        arrays_equal(q, expected[k][0].params)
                        measured_states.append(weakref.ref(q['params']['block']['kernel']))
                        return ns['quad'](p, q, r, b)
                    baseline_ls = ns['microbatched_loss_fn'](base, batch, ls_rngs[0], 1)[0]
                    result = gn_prefix.run_probe(
                        step=0, ks=(10, 20, 50, 100), base=base, state=k10, rng=key10,
                        solve_batch=batch, solve_rng=key10, ls_batches=[batch], ls_rngs=ls_rngs,
                        baseline_loss=baseline_ls, selected_step=alpha, losses=losses,
                        holdout_iterator=iter((b, {}) for b in holdout), holdout_batches=2,
                        inner_step=ns['inner'], quadratic=quadratic,
                        loss_fn=lambda p, b, r: ns['microbatched_loss_fn'](p, b, r, 1),
                        search=ns['run_linesearch'], schedule=ns['lr_sched'], used_updates=used[:10])
                    R = result['gn_prefix/k10/accepted_update_norm']
                    baseline = np.mean([float(ns['loss_fn'](base, b, key10)[0]) for b in holdout])
                    for k, (prefix, _) in expected.items():
                        tag = f'gn_prefix/k{k}/'
                        d = ravel_pytree(prefix.params)[0] - flat
                        a = result[tag + 'selected_alpha']
                        expected_alpha, expected_candidates = ns['run_linesearch'](
                            base, unravel(d), [batch], ls_rngs)
                        self.assertEqual(a, float(expected_alpha))
                        self.assertEqual(result[tag + 'candidate_count'], len(expected_candidates))
                        for i, (scale, loss) in enumerate(expected_candidates):
                            self.assertEqual(result[tag + f'candidate{i}/alpha'], float(scale))
                            self.assertEqual(result[tag + f'candidate{i}/loss'], float(loss))
                        predicted = -g @ d - .5 * d @ G @ d
                        np.testing.assert_allclose(result[tag + 'predicted_raw_improvement'], predicted, atol=2e-6)
                        np.testing.assert_allclose(result[tag + 'predicted_accepted_improvement'],
                                                   -a * g @ d - .5 * a*a * d @ G @ d, atol=2e-6)
                        np.testing.assert_allclose(result[tag + 'residual_norm'], jnp.linalg.norm(g + G @ d), atol=2e-6)
                        np.testing.assert_allclose(result[tag + 'relative_residual'],
                                                   jnp.linalg.norm(g + G @ d)/(jnp.linalg.norm(g)+1e-12), atol=2e-6)
                        for metric, update in [('heldout_improvement', a*d),
                                               ('matched_norm_heldout_improvement', R*d/jnp.linalg.norm(d))]:
                            candidate = unravel(flat + update)
                            actual = baseline - np.mean([float(ns['loss_fn'](candidate, b, key10)[0]) for b in holdout])
                            self.assertAlmostEqual(result[tag + metric], actual, places=6)
                        self.assertAlmostEqual(float(gn_prefix.norm(gn_prefix.matched_direction(unravel(d), R))), R, places=6)
                        if not armijo:
                            self.assertEqual(result[tag + 'candidate_count'], ns['FLAGS'].ls_range)
                        for i in range(1, k+1):
                            self.assertEqual(result[f'gn_prefix/update{i}/optimizer_counter'], i-1)
                            self.assertEqual(result[f'gn_prefix/update{i}/learning_rate_used'], used[i-1][1])
                    self.assertGreater(result['gn_prefix/runtime_s'], 0.)
                    # Measurements retain weak references only; run_probe returns scalars/strings.
                    self.assertTrue(all(isinstance(v, (float, int, str)) for v in result.values()))
                    self.assertTrue(all(ref() is None for ref in measured_states[1:]))

    def test_zero_controls(self):
        for d, radius in [(jnp.zeros(3), 2.), (jnp.ones(3), 0.), (jnp.zeros(3), 0.)]:
            result = gn_prefix.matched_direction(d, radius)
            np.testing.assert_array_equal(result, jnp.zeros(3))
            self.assertTrue(bool(jnp.all(jnp.isfinite(result))))

    def test_packed_loader_isolation_uses_actual_factory_and_constructor(self):
        data_tree = ast.parse((ROOT / 'EasyLM/data.py').read_text())
        from contextlib import contextmanager
        import time
        ns = dict(np=np, time=time, copy=copy.copy, deepcopy=copy.deepcopy,
                  contextmanager=contextmanager, TextProcessor=lambda *args: None)
        execute([node('DatasetFactory', data_tree), node('OptHuggingfaceDataset', data_tree)], ns)
        factory, packed = ns['DatasetFactory'], ns['OptHuggingfaceDataset']
        records = [dict(input_tokens=list(range(30)), loss_masks=[1.] * 30)] * 20
        config = SimpleNamespace(type='huggingface', text_processor=None,
            huggingface_dataset=SimpleNamespace(pretokenized_dataset_dir='/unused',
                seq_length=2, batch_size=2, always_start_with_bos=False,
                batch_token_dtype='i4', tokens_count_at_start=0,
                throughput_average_window_size=20))
        tokenizer = SimpleNamespace(bos_token_id=1)
        with mock.patch.object(factory, 'get_default_config', side_effect=lambda c: c), \
             mock.patch.object(packed, 'get_default_config', side_effect=lambda c: c), \
             mock.patch.object(packed, 'load_dataset', return_value=records):
            ordinary = factory.load_dataset(copy.deepcopy(config), tokenizer)
            reference = factory.load_dataset(copy.deepcopy(config), tokenizer)
            train = factory.load_dataset(copy.deepcopy(config), tokenizer)
            train_ref = factory.load_dataset(copy.deepcopy(config), tokenizer)
            ordinary_iter, reference_iter = iter(ordinary), iter(reference)
            train_iter, train_ref_iter = iter(train), iter(train_ref)
            arrays_equal(next(ordinary_iter)[0], next(reference_iter)[0])
            arrays_equal(next(train_iter)[0], next(train_ref_iter)[0])
            # Execute exactly the production loader construction block.
            construction = next(n for n in node('main').body if isinstance(n, ast.If)
                                and ast.unparse(n.test) == 'prefix_steps')
            host = dict(prefix_steps={0}, DatasetFactory=factory, copy=copy,
                        FLAGS=SimpleNamespace(eval_dataset=config),
                        dataset=SimpleNamespace(tokenizer=tokenizer))
            execute([construction], host)
            probe = host['prefix_dataset']
            self.assertIsNot(probe, ordinary)
            self.assertIsNot(probe.config, ordinary.config)
            self.assertIsNot(probe._token_buffer, ordinary._token_buffer)
            snapshot = copy.deepcopy((ordinary._token_buffer, ordinary._loss_mask_buffer,
                                      ordinary._total_tokens, ordinary.config.tokens_count_at_start))
            next(host['prefix_iterator'])
            next(host['prefix_iterator'])
            self.assertEqual(snapshot, (ordinary._token_buffer, ordinary._loss_mask_buffer,
                                        ordinary._total_tokens, ordinary.config.tokens_count_at_start))
            arrays_equal(next(ordinary_iter)[0], next(reference_iter)[0])
            arrays_equal(next(train_iter)[0], next(train_ref_iter)[0])

    def test_synchronization_and_zero_candidate_baselines(self):
        ns, state, batch = setup()
        key = jax.random.PRNGKey(3)
        events = []
        real_block = jax.block_until_ready
        def block(value):
            events.append('ready')
            return real_block(value)
        def clock():
            events.append('clock')
            return 1. if events.count('clock') == 1 else 8.
        base_loss = float(ns['loss_fn'](state.params, batch, key)[0])
        with mock.patch.object(jax, 'block_until_ready', side_effect=block), \
             mock.patch.object(gn_prefix.time, 'perf_counter', side_effect=clock):
            measured = gn_prefix.run_probe(
                step=0, ks=(10,), base=state.params, state=state, rng=key,
                solve_batch=batch, solve_rng=key, ls_batches=[batch], ls_rngs=[key],
                baseline_loss=base_loss, selected_step=1., losses=[(1.,base_loss)],
                holdout_iterator=iter([(batch,{})]), holdout_batches=1,
                inner_step=None, quadratic=ns['quad'], loss_fn=ns['loss_fn'],
                search=None, schedule=None, used_updates=[])
        self.assertEqual(events, ['ready', 'clock', 'ready', 'clock'])
        self.assertEqual(measured['gn_prefix/runtime_s'], 7.)
        self.assertEqual(measured['gn_prefix/k10/heldout_improvement'], 0.)
        self.assertEqual(measured['gn_prefix/k10/matched_norm_heldout_improvement'], 0.)
        self.assertTrue(np.isfinite(measured['gn_prefix/k10/residual_norm']))

    def test_schedule_counter_is_not_trainstate_step(self):
        for optimizer in ('adamw', 'muon'):
            ns, state, batch = setup(optimizer)
            state = state.replace(step=999)
            self.assertEqual(gn_prefix.optimizer_usage(state, ns['lr_sched'])[0], 0)
            state, _, _ = ns['inner'](state, state.params, jax.random.PRNGKey(1), batch, 0., jnp.bool_(False))
            self.assertEqual(gn_prefix.optimizer_usage(state, ns['lr_sched'])[0], 1)

    def test_microbatch_continuation_matches_full_solve(self):
        for optimizer in ('adamw', 'muon'):
            full, state, batch = setup(optimizer)
            micro, split_state, _ = setup(optimizer, n_micro=3)
            base, key = state.params, jax.random.PRNGKey(22)
            split_key = key
            for _ in range(20):
                state, key, _ = full['inner'](state, base, key, batch, 0., jnp.bool_(False))
                split_state, split_key, _ = micro['inner'](
                    split_state, base, split_key, batch, 0., jnp.bool_(False))
            for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(split_state)):
                np.testing.assert_allclose(a, b, atol=2e-6)
            q_full = full['quad'](base, state.params, key, batch)
            q_micro = micro['quad'](base, split_state.params, split_key, batch)
            for name in q_full:
                np.testing.assert_allclose(q_full[name], q_micro[name], atol=2e-6)

    def test_untouched_main_baseline_equivalence(self):
        ref = os.environ.get('GN_PREFIX_BASELINE_REF', 'origin/main')
        path = 'EasyLM/models/llama/llama_train_gn.py'
        try:
            baseline = ast.parse(subprocess.check_output(
                ['git', 'show', f'{ref}:{path}'], cwd=ROOT, text=True, stderr=subprocess.PIPE))
        except subprocess.CalledProcessError:
            self.skipTest('Fetch origin/main or set GN_PREFIX_BASELINE_REF for baseline verification')
        for name in ('train_step_gauss_newton', 'build_optimizer', 'get_global_lr_sched',
                     'run_linesearch', 'pull_ls_batches_and_baseline', 'loss_fn'):
            self.assertEqual(ast.dump(node(name)), ast.dump(node(name, baseline)))
        self.test_host_commit_and_next_update_equivalence(baseline)

    def test_host_commit_and_next_update_equivalence(self, baseline_tree=None):
        # Execute the trainer's entire ordinary host branch, including real inner
        # updates, LS pull, LS selection and warm-start/optimizer commit.
        main = node('main')
        ordinary = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "FLAGS.optimizer_type == 'cg'"
                        and any(isinstance(c, ast.Assign) and isinstance(c.targets[0], ast.Tuple)
                                and ast.unparse(c.targets[0].elts[0]) == 'candidate_params' for c in n.body))
        current_body = ordinary.orelse[0].orelse
        baseline_body = current_body
        if baseline_tree is not None:
            original = next(n for n in ast.walk(node('main', baseline_tree)) if isinstance(n, ast.If)
                            and ast.unparse(n.test) == "FLAGS.optimizer_type == 'cg'"
                            and any(isinstance(c, ast.Assign) and isinstance(c.targets[0], ast.Tuple)
                                    and ast.unparse(c.targets[0].elts[0]) == 'candidate_params'
                                    for c in n.body))
            baseline_body = original.orelse[0].orelse
        timing = node('main')
        outer = next(n for n in ast.walk(timing) if isinstance(n, ast.For)
                     and ast.unparse(n.target) == 'step' and ast.unparse(n.iter) == 'step_counter')
        stop_index = next(i for i, n in enumerate(outer.body) if isinstance(n, ast.Expr)
                          and ast.unparse(n.value) == 'timing.stop_train_interval(completed_update=True)')
        post_probe = outer.body[stop_index + 1]
        for optimizer in ('adamw', 'muon'):
            runs = []
            for enabled in (False, True):
                body = current_body if enabled else baseline_body
                ns, initial, batch = setup(optimizer)
                records, pulls = [], []
                rng = jax.random.PRNGKey(4)
                clock = [0.]
                ns.update(train_state=initial, inner_state=initial, sharded_rng=rng,
                          do_condition=False, do_accepted_gn=False,
                          sharded_train_step=ns['inner'], sharded_gn_quadratic_metrics=ns['quad'],
                          sharded_prefix_quadratic=ns['quad'], gn_prefix=gn_prefix,
                          prefix_ks=(10,20,50,100), prefix_iterator=iter([(batch,{})]*2),
                          timeit=__import__('timeit'), timing=ProcessTiming(clock=lambda: clock[0]),
                          dataset=None, geometry_exported=False, outer_update_metrics=lambda *args: {},
                          defer_wandb=lambda values, **kw: records.append(values))
                # Replace only host mesh constraint; actual inner-step JIT retains JAX.
                ns['jax'] = SimpleNamespace(**{**jax.__dict__, 'lax': SimpleNamespace(**{**jax.lax.__dict__,
                    'with_sharding_constraint': lambda x, spec: x})})
                def pull(kind):
                    pulls.append(kind)
                    return batch, {}
                def timed_probe(**kwargs):
                    measured = gn_prefix.run_probe(**kwargs)
                    clock[0] += 100.  # Simulated diagnostic wall time must be excluded.
                    return measured
                ns['gn_prefix'] = SimpleNamespace(optimizer_usage=gn_prefix.optimizer_usage,
                                                  run_probe=timed_probe)
                ns['pull_training_batch'] = pull
                execute([node('pull_ls_batches_and_baseline')], ns)
                checkpoints = []
                with redirect_stdout(io.StringIO()):
                    for step in range(2):
                        ns.update(step=step, do_prefix=enabled and step==0,
                                  prefix_capture=None, prefix_used_updates=[], prefix_seconds=0.,
                                  single_batch_=batch, single_dataset_metrics_={})
                        ns['timing'].start()
                        execute(body, ns, loop=True)
                        jax.block_until_ready((ns['train_state'],ns['inner_state'],ns['sharded_rng']))
                        ns['timing'].stop_train_interval(completed_update=True)
                        execute([post_probe], ns)
                        self.assertEqual(ns['timing'].train_time_s, 0.)
                        checkpoints.append((serialization.to_bytes(ns['train_state']),
                                            serialization.to_bytes(ns['inner_state']), np.array(ns['sharded_rng'])))
                        # The normal warm-start continuation remains the committed K10 state.
                ordinary_logs = [{k:v for k,v in row.items() if not k.startswith('gn_prefix/')}
                                 for row in records]
                ordinary_logs = [r for r in ordinary_logs if r]
                runs.append((checkpoints, pulls, ordinary_logs))
            self.assertEqual(runs[0][1:], runs[1][1:])
            for a,b in zip(runs[0][0], runs[1][0]):
                self.assertEqual(a[:2],b[:2])
                np.testing.assert_array_equal(a[2], b[2])


if __name__ == '__main__':
    unittest.main()
