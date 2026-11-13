"""CPU-only resident queue integration tests. Never call GCP, TPU, W&B or JAX."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tpu_resident_queue as queue


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()
        self.fail_read = False

    def create(self, key, value):
        with self.lock:
            if key in self.objects:
                raise RuntimeError('GCS if_generation_match=0 conflict')
            self.objects[key] = json.loads(queue.canonical(value))

    def read(self, key):
        with self.lock:
            if self.fail_read:
                raise PermissionError('GCS auth error')
            return self.objects.get(key)


def fixture(workers=(0, 1, 2, 3), count=2):
    return queue.finish_plan(dict(
        route='queue', run_id='testqueue', workers=[], plan_sha256='',
        trials=[dict(index=i, name='t'+str(i), plan=dict(run_id='run'+str(i)))
                for i in range(1, count + 1)]), list(workers))


class Finished:
    def __init__(self, rc=0):
        self.rc = rc

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc

    def terminate(self):
        self.rc = 143

    def kill(self):
        self.rc = -9


class QueueTests(unittest.TestCase):
    def test_spec_rejects_unsafe_duplicate_and_invalid(self):
        good = dict(preset='gn_adam', default_solve_tokens=150000000,
                    schedule_tokens=1000000000,
                    trials=[dict(name='B8192_lr0.001', batch=8192,
                                 inner_lr=.001, n_micro=16)])
        queue.parse_spec(good)
        for change in (dict(name='../bad'), dict(inner_lr=float('nan')),
                       dict(inner_lr=0), dict(batch=15), dict(n_micro=0),
                       dict(solve_tokens=100)):
            bad = dict(good, trials=[dict(good['trials'][0], **change)])
            with self.assertRaises(ValueError):
                queue.parse_spec(bad)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            queue.parse_spec(dict(good, trials=good['trials'] * 2))
        with self.assertRaisesRegex(ValueError, 'inner_loop_iter'):
            queue.parse_spec(dict(good, trials=[
                dict(good['trials'][0], inner_loop_iter=0)]))
        queue.parse_spec(dict(good, trials=[
            dict(good['trials'][0], inner_loop_iter=25)]))

    def test_queue_spec_real_followup_order_and_exact_parameters(self):
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'sweeps' / 'gn_batch_k_followup.json'
        spec = json.loads(path.read_text())
        plan = queue.make_queue_plan(spec, queue_id='controlled')
        expectation = [
            (8192, .002, 50, 150_000_000, 16),
            (4096, .002, 25, 150_000_000, 8),
            (4096, .002, 100, 75_000_000, 8),
            (4096, .004, 50, 200_000_000, 8),
            (2048, .002, 50, 200_000_000, 4),
            (3072, .002, 50, 200_000_000, 6),
        ]
        self.assertEqual(len(plan['trials']), len(expectation))
        for t, (b, lr, k, requested, micro) in zip(plan['trials'], expectation):
            f = t['plan']['flags']
            self.assertEqual(int(f['train_dataset_batch_size']), b)
            self.assertEqual(int(f['train_dataset.huggingface_dataset.batch_size']), b)
            self.assertEqual(float(f['inner_loop_lr']), lr)
            self.assertEqual(int(f['inner_loop_iter']), k)
            self.assertEqual(int(f['n_micro']), micro)
            self.assertEqual(t['requested_solve_tokens'], requested)
            self.assertEqual(int(f['total_steps']), requested // (b * 1024))
            self.assertEqual(int(f['schedule_steps']), -(-1_000_000_000 // (b * 1024)))
            self.assertEqual(t['actual_solve_tokens'], requested // (b * 1024) * b * 1024)
            self.assertIn(f'_K{k}_', f['notes'])
            self.assertEqual(f['load_checkpoint'], 'trainstate_params::gs://gfn-soo/checkpoints/warmup_150m/1_0/streaming_train_state')
        self.assertEqual(len({t['plan']['run_id'] for t in plan['trials']}), len(expectation))

    def test_adam_spec_strict_validation(self):
        p = {'preset': 'adam', 'default_solve_tokens': 1200000000,
             'schedule_tokens': 2000000000, 'target_loss': 3.25,
             'trials': [{'name': 'Adam_B128', 'batch': 128, 'lr': .01,
                         'b2': .95, 'weight_decay': 0}]}
        queue.parse_spec(p)
        invalid = [
            {'batch': 0}, {'batch': 127}, {'lr': float('nan')},
            {'lr': 0}, {'b2': 1}, {'b2': 0}, {'weight_decay': -1},
            {'name': '../escape'}, {'solve_tokens': 100}
        ]
        for patch in invalid:
            bad = dict(p, trials=[dict(p['trials'][0], **patch)])
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                queue.parse_spec(bad)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            queue.parse_spec(dict(p, trials=p['trials'] * 2))
        with self.assertRaisesRegex(ValueError, 'target_loss'):
            queue.parse_spec(dict(p, target_loss=float('nan')))

    def test_adam_followup_real_config_and_short_notes(self):
        path = Path(__file__).resolve().parents[1] / 'sweeps' / 'adamw_followup.json'
        spec = json.loads(path.read_text())
        plan = queue.make_queue_plan(spec, queue_id='adam-controlled')
        expected = [(128,.01,.95,0), (128,.015,.95,0),
                    (128,.02,.95,0), (64,.015,.95,0),
                    (192,.01,.95,0), (256,.015,.95,0),
                    (128,.01,.98,0), (128,.01,.99,0),
                    (128,.01,.95,.001), (128,.01,.95,.01)]
        self.assertEqual(len(plan['trials']), len(expected))
        run_ids = set()
        for item, (b, lr, b2, wd) in zip(plan['trials'], expected):
            p = item['plan']
            f = p['flags']
            self.assertEqual(p['preset'], 'adam')
            self.assertEqual(p['route'], 'direct')
            self.assertEqual(p['program'], 'EasyLM.models.llama.llama_train')
            self.assertEqual(int(f['train_dataset_batch_size']), b)
            self.assertEqual(int(f['train_dataset.huggingface_dataset.batch_size']), b)
            self.assertEqual(int(f['total_steps']), 1200000000 // (b*1024))
            self.assertEqual(int(f['optimizer.adamw_optimizer.lr_decay_steps']),
                             -(-2000000000 // (b*1024)))
            self.assertEqual(int(f['optimizer.adamw_optimizer.lr_warmup_steps']),
                             max(1, __import__('math').ceil(.01 * (-(-2000000000 // (b*1024))))))
            self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.lr']), lr)
            self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.b2']), b2)
            self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.weight_decay']), wd)
            self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.init_lr']), lr*.1)
            self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.end_lr']), lr*.1)
            self.assertEqual(f['optimizer.adamw_optimizer.lr_sched'], 'cosine')
            self.assertEqual(float(f['target_loss']), 3.25)
            self.assertEqual(int(f['eval_steps']), 50)
            self.assertEqual(int(f['optimizer.accumulate_gradient_steps']), 1)
            self.assertIn('150m', f['load_checkpoint'])
            self.assertEqual(f['notes'], f'Adam_B{b}_lr{lr:g}_b2{b2:g}_wd{wd:g}')
            self.assertNotIn('_T', f['notes'])
            self.assertNotIn('seed42', f['notes'])
            self.assertNotIn('_0', p['experiment_id'])
            self.assertEqual(item['actual_solve_tokens'], b * 1024 * int(f['total_steps']))
            run_ids.add(p['run_id'])
        self.assertEqual(len(run_ids), len(expected))
        self.assertEqual(plan['preset'], 'adam')

    def test_adam_legacy_eight_batch_comparability(self):
        path = Path(__file__).resolve().parents[1] / 'sweeps' / 'adamw_followup_comparable.json'
        spec = json.loads(path.read_text())
        self.assertEqual(spec['protocol'], 'adam_followup_600m')
        self.assertEqual(spec['schedule_tokens'], 599982080)
        plan = queue.make_queue_plan(spec, queue_id='legacy-comparable')
        self.assertEqual(len(plan['trials']), 7)
        self.assertEqual(plan['trials'][0]['plan']['flags']['notes'],
                         'Adam_B128_lr0.015_b20.95_wd0')
        omitted = {'Adam_B128_lr0.02_b20.95_wd0',
                   'Adam_B128_lr0.01_b20.95_wd0',
                   'Adam_B64_lr0.015_b20.95_wd0'}
        self.assertTrue(omitted.isdisjoint(
            {t['plan']['flags']['notes'] for t in plan['trials']}))
        for t in plan['trials']:
            f = t['plan']['flags']
            batch = int(f['train_dataset_batch_size'])
            unit = batch * 1024
            old_horizon = 599982080 // unit
            old_eval = -(-old_horizon // 30)
            self.assertEqual(int(f['eval_steps']), 8)
            self.assertEqual(f['log_initial_eval'], 'False')
            self.assertEqual(int(f['log_freq']), old_eval)
            self.assertEqual(int(f['eval_freq']), old_eval)
            self.assertEqual(int(f['optimizer.adamw_optimizer.lr_decay_steps']),
                             old_horizon)
            self.assertEqual(int(f['optimizer.adamw_optimizer.lr_warmup_steps']), 10)
            self.assertEqual(float(f['optimizer.adamw_optimizer.init_lr']), 0)
            self.assertEqual(float(f['optimizer.adamw_optimizer.end_lr']),
                             float(f['optimizer.adamw_optimizer.lr']) * .1)
            self.assertEqual(f['checkpointer.save_optimizer_state'], 'True')
            self.assertEqual(int(f['total_steps']), 1200000000 // unit)
            self.assertEqual(float(f['target_loss']), 3.25)
        # Original pasted B32 command (600M solve tokens).
        historical = queue.build_adam_plan(
            1, {'batch': 32, 'lr': .005, 'b2': .95, 'weight_decay': 0},
            599982080, 599982080, 3.25, protocol='adam_followup_600m')['flags']
        for key, expected in {
            'total_steps': 18310,
            'log_freq': 611,
            'eval_freq': 611,
            'eval_steps': 8,
            'optimizer.adamw_optimizer.lr_warmup_steps': 10,
            'optimizer.adamw_optimizer.lr_decay_steps': 18310,
            'optimizer.adamw_optimizer.init_lr': 0.0,
            'optimizer.adamw_optimizer.end_lr': .0005,
            'checkpointer.save_optimizer_state': 'True',
            'log_initial_eval': 'False',
        }.items():
            if isinstance(expected, (int, float)):
                self.assertEqual(float(historical[key]), float(expected))
            else:
                self.assertEqual(historical[key], expected)
        # B128 keeps the historical ~20M-token cadence: 153 steps.
        self.assertEqual(int(plan['trials'][0]['plan']['flags']['eval_freq']), 153)
        old = queue.make_queue_plan(json.loads(
            (Path(__file__).resolve().parents[1] / 'sweeps' / 'adamw_followup.json').read_text()),
            queue_id='original-new')
        self.assertEqual(int(old['trials'][0]['plan']['flags']['eval_steps']), 50)
        self.assertEqual(old['trials'][0]['plan']['flags']['log_initial_eval'], 'True')
        # Both versions intentionally evaluate around every 20M tokens,
        # but differ in validation batch count and warmup.
        self.assertEqual(old['trials'][0]['plan']['flags']['eval_freq'],
                         plan['trials'][0]['plan']['flags']['eval_freq'])

    def test_adam_long_horizon_queues_exact_protocol(self):
        from math import ceil
        root = Path(__file__).resolve().parents[1] / 'sweeps'
        expected = {
            'adamw_long_horizon_primary.json': [
                (128, .01, .95, 0),
                (256, .01, .95, 0),
                (384, .01, .95, 0),
                (720, .01, .95, .01),
                (128, .015, .95, 0),
            ],
            'adamw_long_horizon_optional.json': [
                (128, .01, .95, .01),
                (512, .01, .95, 0),
                (192, .0125, .95, 0),
            ],
        }
        for filename, rows in expected.items():
            with self.subTest(filename=filename):
                spec = json.loads((root / filename).read_text())
                self.assertEqual(spec['target_loss'], 0)
                self.assertEqual(spec['schedule_tokens'], 2_211_840_000)
                self.assertEqual(spec['default_solve_tokens'], 2_211_840_000)
                plan = queue.make_queue_plan(spec, queue_id='long-horizon')
                self.assertEqual(len(plan['trials']), len(rows))
                for trial, (batch, lr, beta2, wd) in zip(plan['trials'], rows):
                    f = trial['plan']['flags']
                    unit = batch * 1024
                    horizon = 2_211_840_000 // unit
                    self.assertEqual(int(f['train_dataset_batch_size']), batch)
                    self.assertEqual(int(f['train_dataset.huggingface_dataset.batch_size']), batch)
                    self.assertEqual(int(f['total_steps']), horizon)
                    self.assertEqual(int(f['optimizer.adamw_optimizer.lr_decay_steps']), horizon)
                    self.assertEqual(int(f['optimizer.adamw_optimizer.lr_warmup_steps']),
                                     ceil(7_372_800 / unit))
                    self.assertEqual(int(f['eval_steps']), 8)
                    self.assertEqual(f['log_initial_eval'], 'False')
                    self.assertEqual(int(f['log_freq']), ceil(horizon / 30))
                    self.assertEqual(int(f['eval_freq']), ceil(horizon / 30))
                    self.assertEqual(float(f['target_loss']), 0)
                    self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.lr']), lr)
                    self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.init_lr']), lr*.01)
                    self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.end_lr']), lr*.1)
                    self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.b2']), beta2)
                    self.assertAlmostEqual(float(f['optimizer.adamw_optimizer.weight_decay']), wd)
                    self.assertEqual(f['optimizer.adamw_optimizer.lr_sched'], 'cosine')
                    self.assertIn('150m', f['load_checkpoint'])
                    self.assertEqual(f['checkpointer.save_optimizer_state'], 'True')
                    self.assertEqual(int(trial['actual_solve_tokens']), horizon*unit)
        # Historical B720 reference is 3000 updates at 2,211,840,000 tokens.
        b720 = queue.build_adam_plan(
            1, {'batch': 720, 'lr': .01, 'b2': .95, 'weight_decay': 0},
            2_211_840_000, 2_211_840_000, 0,
            protocol='adam_followup_600m',
            init_lr_fraction=.01, warmup_tokens=7_372_800)['flags']
        self.assertEqual(int(b720['total_steps']), 3000)
        self.assertEqual(int(b720['optimizer.adamw_optimizer.lr_decay_steps']), 3000)
        self.assertEqual(int(b720['optimizer.adamw_optimizer.lr_warmup_steps']), 10)
        self.assertEqual(float(b720['optimizer.adamw_optimizer.init_lr']), .0001)
        self.assertEqual(float(b720['optimizer.adamw_optimizer.end_lr']), .001)

    def test_adam_long_horizon_rejects_invalid_overrides(self):
        spec = {'preset': 'adam', 'default_solve_tokens': 2_211_840_000,
                'schedule_tokens': 2_211_840_000, 'target_loss': 0,
                'protocol': 'adam_followup_600m',
                'init_lr_fraction': .01, 'warmup_tokens': 7_372_800,
                'trials': [{'name': 'test', 'batch': 128, 'lr': .01,
                            'b2': .95, 'weight_decay': 0}]}
        queue.parse_spec(spec)
        for override in ({'init_lr_fraction': -1}, {'init_lr_fraction': 1.1},
                         {'init_lr_fraction': float('nan')}, {'warmup_tokens': 0},
                         {'target_loss': -1}, {'target_loss': float('nan')},
                         {'protocol': 'unknown'}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                queue.parse_spec(dict(spec, **override))
        with self.assertRaisesRegex(ValueError, 'require'):
            queue.parse_spec(dict(spec, protocol=None))

    def test_adam_smoke_evaluates_each_single_update(self):
        path = Path(__file__).resolve().parents[1] / 'sweeps' / 'adamw_smoke.json'
        plan = queue.make_queue_plan(json.loads(path.read_text()), queue_id='smoke-controlled')
        self.assertEqual(len(plan['trials']), 2)
        for item in plan['trials']:
            f = item['plan']['flags']
            self.assertEqual(int(f['total_steps']), 1)
            self.assertEqual(int(f['eval_freq']), 1)
            self.assertEqual(int(f['log_freq']), 1)
            self.assertEqual(int(f['eval_steps']), 50)
            self.assertEqual(float(f['target_loss']), 3.25)
            self.assertEqual(item['requested_solve_tokens'], 131072)

    def test_gn_queue_still_uses_existing_builder(self):
        p = queue.make_queue_plan({
            'preset': 'gn_adam', 'default_solve_tokens': 150000000,
            'schedule_tokens': 1000000000,
            'trials': [{'name': 'GN_B4096', 'batch': 4096,
                        'inner_lr': .002, 'n_micro': 8}]},
            queue_id='gn-controlled')
        self.assertEqual(p['preset'], 'gn_adam')
        self.assertEqual(p['trials'][0]['plan']['route'], 'sweep')
        self.assertEqual(int(p['trials'][0]['plan']['flags']['inner_loop_iter']), 50)

    def test_plan_hash_and_worker_count(self):
        p = fixture((2, 3), count=1)
        queue.verify_plan(p, 2)
        with self.assertRaises(ValueError):
            queue.verify_plan(p, 0)
        p['trials'][0]['plan']['run_id'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'hash'):
            queue.verify_plan(p, 2)

    def test_barrier_validates_identity_and_conditional_create(self):
        p, s = fixture((0, 1), 1), MemoryStore()
        for w in p['workers']:
            s.create(f'ready/{w}.json', queue.record(p, w, 0, p['run_id']))
        queue.barrier(s, p, 'ready', 0, p['run_id'], .01)
        with self.assertRaisesRegex(RuntimeError, 'conflict'):
            s.create('ready/0.json', {})
        s.create('result/1/0.json', queue.record(p, 0, 1, 'run1', 0))
        s.create('result/1/1.json', queue.record(p, 1, 1, 'stale', 0))
        with self.assertRaisesRegex(RuntimeError, 'Invalid result'):
            queue.barrier(s, p, 'result', 1, 'run1', .01)

    def test_absent_worker_timeout_and_gcs_failure(self):
        p = fixture((0, 1), 1)
        with self.assertRaises(TimeoutError):
            queue.barrier(MemoryStore(), p, 'ready', 0, p['run_id'],
                          .005, poll_s=.001)
        s = MemoryStore()
        s.fail_read = True
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(PermissionError):
                queue.ResidentRunner(fixture((0,), 1), 0, s).run(root, install_signals=False)

    def test_success_and_failure_stop_every_worker_before_next_trial(self):
        def simulate(bad_worker=None):
            p, store = fixture(), MemoryStore()
            started, errors = [], []
            lock = threading.Lock()
            def fake_popen(args, env=None):
                directory = Path(args[-1]).parent.name
                worker = int(directory[len('worker'):])
                trial = int(Path(args[-1]).stem.rsplit('_', 1)[1])
                with lock:
                    started.append((worker, trial))
                return Finished(1 if worker == bad_worker and trial == 1 else 0)
            def run(w, path):
                try:
                    queue.ResidentRunner(p, w, store, sleep=lambda _: time.sleep(.001)).run(
                        path / ('worker'+str(w)), startup_s=1, barrier_s=1,
                        install_signals=False)
                except Exception as error:
                    with lock:
                        errors.append(w)
            with tempfile.TemporaryDirectory() as root, patch.object(
                    queue.subprocess, 'Popen', side_effect=fake_popen):
                threads = [threading.Thread(target=run, args=(w, Path(root))) for w in p['workers']]
                for t in threads: t.start()
                for t in threads: t.join(4)
                self.assertFalse(any(t.is_alive() for t in threads), 'deadlocked queue')
            return sorted(started), sorted(errors)
        self.assertEqual(simulate(), ([(w, i) for w in range(4) for i in (1, 2)], []))
        started, errors = simulate(2)
        self.assertEqual(started, [(w, 1) for w in range(4)])
        self.assertEqual(errors, list(range(4)))

    def test_signal_terminates_owned_child(self):
        p = fixture((0,), 1)
        runner = queue.ResidentRunner(p, 0, MemoryStore())
        child = Finished()
        child.poll = lambda: None
        runner.child = child
        runner.terminate(15, None)
        self.assertTrue(runner.cancelled)
        self.assertEqual(child.rc, 143)

    def test_new_queue_plan_preserves_existing_gn_science_settings(self):
        from gn_sweep_queue import build_plan, build_plan_for_config, TRIALS
        known = TRIALS[8]  # trial 9: B2048, LR .001, micro 4
        old = build_plan(known[0], 150000000, 1000000000)
        new = build_plan_for_config(*known, 150000000, 1000000000)
        scientific = lambda plan: {k:v for k,v in plan['flags'].items()
                                   if k not in ('experiment_id', 'wandb_run_id')}
        self.assertEqual(scientific(old), scientific(new))
        self.assertEqual(new['flags']['train_dataset.huggingface_dataset.batch_size'],
                         new['flags']['train_dataset_batch_size'])
        for b, lr, micro in ((4096, .002, 8), (8192, .001, 16)):
            plan = build_plan_for_config(1, b, lr, micro, 150000000, 1000000000)
            f = plan['flags']
            self.assertEqual(int(f['total_steps']), 150000000 // (b * 1024))
            self.assertEqual(int(f['schedule_steps']), -(-1000000000 // (b * 1024)))
            self.assertEqual(int(f['train_dataset.huggingface_dataset.batch_size']), b)
            self.assertEqual(int(f['n_micro']), micro)
            self.assertEqual(float(f['inner_loop_lr']), lr)
            self.assertEqual(int(f['inner_loop_iter']), 50)


if __name__ == '__main__':
    unittest.main()
