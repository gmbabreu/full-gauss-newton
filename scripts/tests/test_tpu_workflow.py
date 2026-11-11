"""Local-only workflow validation. Never uses real gcloud, tmux, or trainers."""
import ast
import base64
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import run_experiment as experiment
import tpu_control as control
import tpu_worker as worker


class PresetTests(unittest.TestCase):
    def test_all_presets_and_solver_settings(self):
        self.assertEqual(set(experiment.choices()), {'adam', 'gn_adam', 'gn_muon', 'gn_rmsprop',
                          'gn_sgd', 'pcg_jacobi', 'pcg_interpolated'})
        for name in experiment.choices():
            with self.subTest(name=name):
                p = experiment.make_plan(name, [], 'shared')
                f = p['flags']
                self.assertEqual(f['train_dataset_batch_size'], f[experiment.BATCH])
                self.assertEqual(f['train_dataset_batch_size'], 720)
                self.assertEqual(f['eval_dataset.huggingface_dataset.batch_size'], 128)
                self.assertEqual(f['llama.base_model'], '150M')
                self.assertFalse(f['eval_dataset.huggingface_dataset.shuffle_data'])
                self.assertFalse(any('repeat_first_n_batches' in k for k in f))
                if name == 'adam':
                    self.assertNotIn('gauss_newton', f)
                    self.assertNotIn('optimizer_type', f)
                    self.assertEqual(f['dtype'], 'fp32')
                    self.assertEqual(p['route'], 'direct')
                else:
                    self.assertTrue(f['gauss_newton'])
                    self.assertEqual(p['experiment_id'], 'shared_0')
        self.assertEqual(experiment.resolve('gn_muon', [])['flags']['optimizer_type'], 'muon')
        self.assertEqual(experiment.resolve('gn_rmsprop', [])['flags']['inner_b1'], 0)
        self.assertEqual(experiment.resolve('gn_rmsprop', [])['flags']['optimizer_type'], 'adamw')
        self.assertEqual(experiment.resolve('gn_sgd', [])['flags']['inner_loop_iter'], 200)
        for name in ('pcg_jacobi', 'pcg_interpolated'):
            f = experiment.resolve(name, [])['flags']
            self.assertEqual(f['cg_lambda_final'], -1)
            self.assertEqual(f['cg_lambda_ramp_steps'], 0)
            self.assertEqual(f['cg_lambda_batch_denominator'], 0)
            self.assertFalse(f['condition_log'])

    def test_flags_exist_in_current_branch_source_without_imports(self):
        def tree(path):
            return ast.parse((experiment.ROOT / path).read_text())
        def attrs(path, klass):
            cls = next(n for n in tree(path).body if isinstance(n, ast.ClassDef) and n.name == klass)
            method = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and n.name == 'get_default_config')
            return {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name) and n.value.id == 'config'}
        nested = {
            'llama': attrs('EasyLM/models/llama/llama_model.py', 'LLaMAConfigurator'),
            'checkpointer': attrs('EasyLM/checkpoint.py', 'StreamingCheckpointer'),
            'optimizer': attrs('EasyLM/optimizers.py', 'OptimizerFactory'),
            'optimizer.adamw_optimizer': attrs('EasyLM/optimizers.py', 'AdamWOptimizerFactory'),
        }
        for kind in ('train', 'eval'):
            nested[kind + '_dataset'] = attrs('EasyLM/data.py', 'DatasetFactory')
            nested[kind + '_dataset.text_processor'] = attrs('EasyLM/data.py', 'TextProcessor')
            nested[kind + '_dataset.huggingface_dataset'] = attrs('EasyLM/data.py', 'OptHuggingfaceDataset')
        for name in experiment.choices():
            p = experiment.resolve(name, [])
            trainer = tree(p['program'].replace('.', '/') + '.py')
            call = next(n for n in ast.walk(trainer) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == 'define_flags_with_default')
            top = {k.arg for k in call.keywords}
            for key in p['flags']:
                with self.subTest(preset=name, flag=key):
                    if '.' in key:
                        prefix, leaf = key.rsplit('.', 1)
                        self.assertIn(leaf, nested[prefix])
                        self.assertIn(key.split('.')[0], top)
                    else:
                        self.assertIn(key, top)

    def test_overrides_preserve_text_and_last_wins(self):
        values = ['--inner_loop_iter=7', '--inner_loop_iter=100', '--notes=a b=c&d "q" $(touch /tmp/nope)',
                  '--load_dataset_state=', '--llama.initializer_range=2', '--linesearch=False']
        p = experiment.make_plan('gn_adam', values)
        self.assertEqual(p['flags']['inner_loop_iter'], '100')
        self.assertEqual(p['flags']['notes'], values[2].split('=', 1)[1])
        self.assertEqual(p['flags']['load_dataset_state'], '')
        self.assertEqual(p['flags']['linesearch'], 'False')
        self.assertEqual(p['explicit_overrides'], values)
        args = shlex.split(experiment.preview(p))
        self.assertEqual(sum(a.startswith('--inner_loop_iter=') for a in args), 1)
        self.assertIn(values[2], args)

    def test_batch_synchronization_and_conflict(self):
        for key in ('train_dataset_batch_size', experiment.BATCH):
            f = experiment.resolve('gn_adam', [f'--{key}=1440'])['flags']
            self.assertEqual(f['train_dataset_batch_size'], '1440')
            self.assertEqual(f[experiment.BATCH], '1440')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            experiment.resolve('adam', ['--train_dataset_batch_size=10', f'--{experiment.BATCH}=11'])
        f = experiment.resolve('adam', ['--train_dataset_batch_size=5', '--train_dataset_batch_size=11',
                                      f'--{experiment.BATCH}=11'])['flags']
        self.assertEqual(f['train_dataset_batch_size'], '11')

    def test_missing_preset_and_bad_arguments(self):
        for argv in (['1', 'run'], ['2', 'run', '--dry-run']):
            with self.assertRaisesRegex(ValueError, 'Choose:'):
                control.main(argv)
        for arg in ('--inner_loop_iter', 'hello', '--resume_from=old', '--wandb_run_id=old', '--program=bad'):
            with self.assertRaises(ValueError):
                experiment.resolve('gn_adam', [arg])

    def test_new_identity_and_checkpoint_override(self):
        a = experiment.make_plan('adam', ['--load_checkpoint=trainstate::/explicit'])
        b = experiment.make_plan('adam', [])
        self.assertNotEqual(a['run_id'], b['run_id'])
        self.assertEqual(a['flags']['wandb_run_id'], '')
        self.assertEqual(a['flags']['load_checkpoint'], 'trainstate::/explicit')

    def test_sweep_route_preserves_main_and_final_flags_once(self):
        p = experiment.make_plan('gn_adam', ['--notes=space a=b&c'], 'shared')
        gcs = types.ModuleType('EasyLM.gcs_utils')
        gcs.gcs_path_exists = lambda path: False
        gcs.read_from_gcs = lambda path: None
        with patch.dict(sys.modules, {'EasyLM.gcs_utils': gcs}), patch.dict(os.environ), \
                patch('os.execvp') as execute:
            with redirect_stdout(io.StringIO()):
                experiment.execute(p)
            args = execute.call_args.args[1]
            self.assertEqual(args[:3], ['python', '-m', p['program']])
            self.assertIn('--experiment_id=shared_0', args)
            self.assertIn('--notes=space a=b&c', args)
            keys = [a.split('=', 1)[0] for a in args[3:]]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(os.environ['SLURM_ARRAY_TASK_ID'], '1')
            self.assertEqual(os.environ['SLURM_ARRAY_JOB_ID'], 'shared')
            self.assertEqual(os.environ['PYTHONUNBUFFERED'], '1')
        # execute() intentionally installs a parser only in the worker process.
        import sweep_launcher
        importlib.reload(sweep_launcher)

    def test_adam_executes_module_directly(self):
        p = experiment.make_plan('adam', [], 'shared')
        with patch('os.execv') as execute, patch.dict(os.environ):
            experiment.execute(p)
        self.assertEqual(execute.call_args.args[1][1:3], ['-m', p['program']])

    def test_dry_run_has_no_external_commands(self):
        with patch.object(subprocess, 'run', side_effect=AssertionError('external command')), \
                patch.object(subprocess, 'check_output', side_effect=AssertionError('external command')):
            for target in ('1', '2'):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(control.main([target, 'run', 'gn_adam', '--dry-run']), 0)
                self.assertIn(control.DEFAULTS['targets'][target], output.getvalue())
                self.assertIn('DRY RUN', output.getvalue())
                self.assertIn('Output:', output.getvalue())

    def test_shell_entrypoint_cwd_independent_without_gcloud(self):
        for cwd, script in ((experiment.ROOT, 'scripts/tpu.sh'), (SCRIPTS, 'tpu.sh')):
            for target in ('1', '2'):
                result = subprocess.run(['/bin/bash', script, target, 'run', 'pcg_jacobi', '--dry-run'],
                                        cwd=cwd, capture_output=True, text=True,
                                        env={**os.environ, 'PATH': '/usr/bin:/bin'})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(control.DEFAULTS['targets'][target], result.stdout)


# An executable fake gcloud exercises argv/shell transport as real subprocesses.
# It records decoded RPCs but NEVER executes the remote shell.
FAKE_GCLOUD = '''#!/usr/bin/python3
import base64, json, os, shlex, sys
from pathlib import Path
args=sys.argv[1:]
log=Path(os.environ['FAKE_GCLOUD_LOG'])
if 'describe' in args:
    print(json.dumps({'networkEndpoints': [{}, {}, {}]})); sys.exit(0)
wid=int(next(a.split('=',1)[1] for a in args if a.startswith('--worker=')))
remote=next(a.split('=',1)[1] for a in args if a.startswith('--command='))
setup=shlex.split(remote)[2]
tokens=shlex.split(setup)
action=tokens[-2]
p=json.loads(base64.urlsafe_b64decode(tokens[-1]))
with log.open('a') as f: f.write(json.dumps({'worker':wid,'action':action,'payload':p})+'\\n')
fail=os.environ.get('FAKE_FAIL','')
if action+':'+str(wid) in fail.split(','):
    print('GFN_RESULT='+json.dumps({'ok':False,'error':'injected '+fail})); sys.exit(1)
result={'branch':'tooling','sha':'source','dirty':'','tmux':[],'managed':[],
        'training_processes':[],'unmanaged':[],'remaining_tmux':[]}
if action=='launch': result={'message':'process started','log':'/outside/worker.log'}
if action=='fetch': result={'target_sha':'target'}
if action=='status' and os.environ.get('FAKE_UPDATE'):
    result.update(branch='epoch',sha='target')
print('GFN_RESULT='+json.dumps({'ok':True,'result':result}))
'''


class MockedCloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'gcloud').write_text(FAKE_GCLOUD)
        (self.path / 'gcloud').chmod(0o755)
        self.log = self.path / 'calls.jsonl'
        env = patch.dict(os.environ, PATH=str(self.path) + os.pathsep + os.environ['PATH'],
                         FAKE_GCLOUD_LOG=str(self.log))
        env.start()
        self.addCleanup(env.stop)
        self.git = patch.object(control, 'local_git', side_effect=lambda *a:
                                '' if a[0] == 'status' else 'tooling' if a[0] == 'branch' else 'source')
        self.git.start()
        self.addCleanup(self.git.stop)
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_shared_identity_and_all_preflights_before_launch(self):
        c = control.Controller('2')
        c.operate('run', experiment.make_plan('gn_adam', ['--notes=a b=c&d'], 'one-id'))
        calls = self.calls()
        pre = [i for i, c in enumerate(calls) if c['action'] == 'preflight']
        launch = [(i, c) for i, c in enumerate(calls) if c['action'] == 'launch']
        self.assertEqual(len(pre), 3)
        self.assertLess(max(pre), min(i for i, c in launch))
        self.assertEqual({c['payload']['plan']['run_id'] for i, c in launch}, {'one-id'})
        self.assertEqual({c['payload']['plan']['flags']['notes'] for i, c in launch}, {'a b=c&d'})
        self.assertEqual(len([c for c in calls if c['action'] == 'release']), 3)

    def test_preflight_failure_launches_nothing_and_checks_every_worker(self):
        with patch.dict(os.environ, FAKE_FAIL='preflight:1'):
            with self.assertRaisesRegex(RuntimeError, 'worker 1'):
                control.Controller('1').operate('run', experiment.make_plan('adam', []))
        calls = self.calls()
        self.assertEqual([c['worker'] for c in calls if c['action'] == 'preflight'], [0, 1, 2])
        self.assertFalse(any(c['action'] == 'launch' for c in calls))

    def test_partial_failure_cleans_only_shared_launch_on_all_workers(self):
        with patch.dict(os.environ, FAKE_FAIL='launch:1'):
            with self.assertRaisesRegex(RuntimeError, 'Partial launch failure'):
                control.Controller('1').operate('run', experiment.make_plan('gn_adam', [], 'failed-id'))
        stops = [c for c in self.calls() if c['action'] == 'stop']
        self.assertEqual([c['worker'] for c in stops], [0, 1, 2])
        self.assertTrue(all(c['payload']['stop_run_id'] == 'failed-id' for c in stops))

    def test_cleanup_error_preserves_original_launch_error(self):
        with patch.dict(os.environ, FAKE_FAIL='launch:1,release:1'):
            with self.assertRaises(RuntimeError) as caught:
                control.Controller('1').operate('run', experiment.make_plan('adam', [], 'cleanup-id'))
        self.assertIn('Partial launch failure', str(caught.exception))
        self.assertIn('Reservation cleanup incomplete', str(caught.exception))

    def test_lock_failure_never_launches(self):
        with patch.dict(os.environ, FAKE_FAIL='acquire:2'):
            with self.assertRaises(RuntimeError):
                control.Controller('1').operate('run', experiment.make_plan('adam', []))
        self.assertFalse(any(c['action'] == 'launch' for c in self.calls()))

    def test_dirty_controller_refuses_launch(self):
        with patch.object(control, 'local_git', return_value='dirty'):
            with self.assertRaisesRegex(RuntimeError, 'Controller working tree is dirty'):
                control.Controller('1').operate('run', experiment.make_plan('adam', []))
        self.assertFalse(any(c['action'] == 'launch' for c in self.calls()))

    def test_update_refusal_before_fetch(self):
        for action in ('pull', 'switch'):
            self.log.write_text('')
            with patch.dict(os.environ, FAKE_FAIL='preflight_update:2'):
                with self.assertRaises(RuntimeError):
                    control.Controller('1').operate(action, branch='epoch' if action == 'switch' else None)
            self.assertFalse(any(c['action'] in ('fetch', 'update') for c in self.calls()))

    def test_updates_have_one_commit_and_verify_convergence(self):
        with patch.dict(os.environ, FAKE_UPDATE='1'):
            control.Controller('1').operate('switch', branch='epoch')
        calls = self.calls()
        self.assertEqual([c['worker'] for c in calls if c['action'] == 'preflight_update'], [0, 1, 2])
        self.assertFalse(any(c['action'] == 'preflight' for c in calls))
        fetch = [i for i, c in enumerate(calls) if c['action'] == 'fetch']
        update = [(i, c) for i, c in enumerate(calls) if c['action'] == 'update']
        self.assertLess(max(fetch), min(i for i, c in update))
        self.assertEqual({c['payload']['target_sha'] for i, c in update}, {'target'})
        self.assertEqual(len([c for c in calls if c['action'] == 'status']), 3)

    def test_partial_update_failure_is_reported(self):
        with patch.dict(os.environ, FAKE_FAIL='update:1'):
            with self.assertRaisesRegex(RuntimeError, 'Update incomplete; workers may differ'):
                control.Controller('1').operate('switch', branch='epoch')


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.old_cwd)
        self.payload = dict(repo=str(experiment.ROOT), state=str(self.root / 'records'),
                            environment=str(self.root / 'env.sh'), token='ours', target='fake', worker=0,
                            fingerprint=worker.fingerprint(experiment.ROOT),
                            expected_sha='sha', expected_branch='branch')
        Path(self.payload['environment']).touch()
        self.w = worker.Worker(self.payload)
        self.w.acquire()
        self.clean = dict(branch='branch', sha='sha', dirty='', tmux=[], training_processes=[], managed=[], unmanaged=[])

    def test_preflight_rejects_dirty_running_and_mismatched_workers(self):
        for change in ({'dirty': 'M file'}, {'training_processes': [{'pid': 99}]},
                       {'tmux': ['gfn-old']}, {'managed': [{'run_id': 'old'}]},
                       {'sha': 'wrong'}, {'branch': 'wrong'}):
            with self.subTest(change=change), patch.object(self.w, 'report', return_value={**self.clean, **change}), patch('shutil.which', return_value='/bin/fake'):
                with self.assertRaises(RuntimeError):
                    self.w.preflight()
        with patch.object(self.w, 'report', return_value=self.clean), patch('shutil.which', return_value='/bin/mock'):
            self.assertEqual(self.w.preflight(), self.clean)
            self.w.p['fingerprint'] = {}
            with self.assertRaisesRegex(RuntimeError, 'tooling or presets differ'):
                self.w.preflight()

    def test_update_preflight_accepts_preset_differences_but_run_refuses(self):
        self.w.p['fingerprint']['scripts/presets/adam.json'] = 'new-preset'
        self.w.p['fingerprint']['scripts/presets/new.json'] = 'added-preset'
        with patch.object(self.w, 'report', return_value=self.clean), patch('shutil.which', return_value='/bin/mock'):
            self.assertEqual(self.w.preflight_update(), self.clean)
            with self.assertRaisesRegex(RuntimeError, 'tooling or presets differ'):
                self.w.preflight()
            self.w.p['fingerprint']['scripts/tpu_worker.py'] = 'changed-helper'
            with self.assertRaisesRegex(RuntimeError, 'executable helpers differ; manually bootstrap'):
                self.w.preflight_update()

    def test_missing_environment_and_tool(self):
        with patch.object(self.w, 'report', return_value=self.clean):
            Path(self.payload['environment']).unlink()
            with self.assertRaisesRegex(RuntimeError, 'Missing environment'):
                self.w.preflight()
            Path(self.payload['environment']).touch()
            with patch('shutil.which', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'must provide'):
                    self.w.preflight()

    def test_tmux_inspection_fails_closed_on_permission_error(self):
        with patch.object(subprocess, 'run', return_value=Mock(returncode=1, stdout='',
                          stderr='error connecting to /socket (Permission denied)')):
            with self.assertRaisesRegex(RuntimeError, 'Cannot inspect tmux'):
                worker.sessions()
        with patch.object(subprocess, 'run', return_value=Mock(returncode=1, stdout='',
                          stderr='error connecting to /socket (No such file or directory)')):
            self.assertEqual(worker.sessions(), [])

    def test_status_queries_tpu_usage_without_importing_trainer(self):
        with patch.object(self.w, 'report', return_value=dict(self.clean)), \
                patch('shutil.which', return_value=None), \
                patch.object(subprocess, 'run', return_value=Mock(returncode=0, stdout='usage stats')) as run:
            self.assertEqual(self.w.status()['tpu_usage'], 'usage stats')
            self.assertIn('libtpu.sdk', run.call_args.args[0][2])
        with patch.object(self.w, 'report', return_value=dict(self.clean)), \
                patch('shutil.which', return_value='/bin/tpu-info'), \
                patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired('tpu-info', 15)):
            self.assertIn('timed out', self.w.status()['tpu_usage'])

    def test_records_cannot_be_in_git_tree(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            worker.Worker({**self.payload, 'state': str(experiment.ROOT / 'run-records')})

    def test_lock_reservation_owner(self):
        other = worker.Worker({**self.payload, 'token': 'other'})
        with self.assertRaisesRegex(RuntimeError, 'reserved'):
            other.acquire()
        self.assertFalse(other.release()['released'])
        self.assertTrue(self.w.lock.exists())
        self.w.release()
        self.assertFalse(self.w.lock.exists())

    def test_training_detection_includes_adam_and_legacy_sweeps(self):
        table = [dict(pid=i, argv=argv) for i, argv in enumerate([
            ['python', '-m', 'EasyLM.models.llama.llama_train'],
            ['python', '-m', 'EasyLM.models.llama.llama_train_gn'],
            ['python', '/repo/sweep_launcher.py', '--program=other'],
            ['python', 'unrelated.py'], ['python3', 'scripts/tpu_worker.py', 'status', 'payload']])]
        with patch.object(worker, 'process_table', return_value=table):
            self.assertEqual([p['pid'] for p in worker.training_processes()], [0, 1, 2])

    def test_stop_only_signals_owned_group_and_preserves_files(self):
        env = {**os.environ, 'GFN_MANAGED_RUN_TOKEN': 'ours'}
        managed = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(45)'], env=env, start_new_session=True)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(45)'], start_new_session=True)
        def cleanup():
            for process in (managed, unrelated):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
        self.addCleanup(cleanup)
        directory = self.w.state / 'test-run'
        directory.mkdir()
        log = directory / 'worker.log'
        log.write_text('preserve log')
        checkpoint = self.root / 'checkpoint'
        checkpoint.write_text('preserve checkpoint')
        manifest = dict(run_id='test-run', pgid=managed.pid, token='ours', session='gfn-test-run', state='started')
        worker.atomic_json(directory / 'manifest.json', manifest)
        with patch.object(worker, 'sessions', return_value=[]), patch.object(self.w, 'report', return_value=self.clean):
            result = self.w.stop()
        self.assertIn('test-run', result['stopped'])
        managed.wait(timeout=5)
        self.assertIsNone(unrelated.poll())
        self.assertEqual(log.read_text(), 'preserve log')
        self.assertEqual(checkpoint.read_text(), 'preserve checkpoint')

    def test_wrong_process_ownership_refuses_signal(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(45)'], start_new_session=True)
        self.addCleanup(lambda: (process.terminate(), process.wait(timeout=5)))
        with self.assertRaisesRegex(RuntimeError, 'ownership changed'):
            self.w.terminate_group(dict(pgid=process.pid, token='not-its-token'))
        self.assertIsNone(process.poll())

    def test_status_reports_unmanaged_without_killing(self):
        unmanaged = [dict(pid=99, pgrp=99, argv=['python', '-m', 'EasyLM.models.llama.llama_train'])]
        with patch.object(worker, 'training_processes', return_value=unmanaged), \
                patch.object(self.w, 'active_records', return_value=[]), \
                patch.object(worker, 'sessions', return_value=[]), patch.object(worker, 'command', return_value='branch'):
            report = self.w.report()
        self.assertEqual(report['unmanaged'], unmanaged)

    def test_supervisor_clears_inherited_disabled_wandb_mode_and_records_exit(self):
        for wid in (0, 2):
            with self.subTest(worker=wid):
                p = {**self.payload, 'worker': wid, 'plan': experiment.make_plan('gn_adam', [], f'run-{wid}')}
                w = worker.Worker(p)
                directory = w.run_dir()
                directory.mkdir()
                worker.atomic_json(directory / 'manifest.json', dict(run_id=f'run-{wid}', token='ours', state='prepared'))
                child = Mock(pid=1234567)
                child.wait.return_value = 7
                table = [dict(pid=os.getpid(), start='birth', pgrp=1, argv=[])]
                with patch.object(worker, 'process_table', return_value=table), \
                        patch.object(worker.subprocess, 'Popen', return_value=child) as popen, \
                        patch.object(worker.signal, 'signal'), \
                        patch.dict(os.environ, WANDB_RUN_ID='old', WANDB_RESUME='must', WANDB_MODE='disabled'):
                    self.assertEqual(w.supervise()['exit_status'], 7)
                env = popen.call_args.kwargs['env']
                self.assertNotIn('WANDB_RUN_ID', env)
                self.assertNotIn('WANDB_RESUME', env)
                self.assertEqual(env['PYTHONUNBUFFERED'], '1')
                self.assertEqual(env['SLURM_ARRAY_JOB_ID'], f'run-{wid}')
                self.assertEqual(env['SLURM_ARRAY_TASK_ID'], '1')
                if wid:
                    self.assertEqual(env['WANDB_MODE'], 'disabled')
                else:
                    self.assertNotIn('WANDB_MODE', env)
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
                self.assertEqual(json.loads((directory / 'manifest.json').read_text())['exit_status'], 7)

    def test_worker_update_refuses_dirty_and_running_before_fetch(self):
        for change in ({'dirty': 'M file'}, {'training_processes': [{'pid': 10}]}):
            with patch.object(self.w, 'report', return_value={**self.clean, **change}), \
                    patch.object(worker, 'command') as cmd, patch('shutil.which', return_value='/bin/fake'):
                with self.assertRaises(RuntimeError):
                    self.w.fetch()
                cmd.assert_not_called()


    def test_launch_records_complete_manifest_and_sources_env_inside_tmux(self):
        self.w.p['plan'] = experiment.make_plan('gn_adam', ['--notes=a b=c'], 'launch-test')
        def tmux(argv, **kwargs):
            self.assertEqual(argv[:3], ['tmux', 'new-session', '-d'])
            setup = shlex.split(argv[-1])[2]
            self.assertIn('source ' + self.w.p['environment'], setup)
            path = self.w.run_dir() / 'manifest.json'
            value = json.loads(path.read_text())
            value.update(state='started', pgid=99999)
            worker.atomic_json(path, value)
            return ''
        with patch.object(self.w, 'preflight', return_value=self.clean), \
                patch.object(worker, 'command', side_effect=tmux), patch.object(self.w, 'group', return_value=[{}]):
            result = self.w.launch()
        self.assertIn('initialization is NOT verified', result['message'])
        value = json.loads((self.w.run_dir() / 'manifest.json').read_text())
        for key in ('run_id', 'preset', 'explicit_overrides', 'resolved_arguments', 'git_sha',
                    'branch', 'target', 'worker', 'exit_status', 'token', 'session', 'pane_command'):
            self.assertIn(key, value)
        self.assertEqual(value['explicit_overrides'], ['--notes=a b=c'])

    def test_stop_cancels_pre_acknowledgement_tmux_owned_by_launch(self):
        directory = self.w.state / 'pending'
        directory.mkdir()
        worker.atomic_json(directory / 'manifest.json',
                           dict(run_id='pending', state='prepared', token='ours',
                                session='gfn-pending', pane_command='the exact startup command'))
        live = ['gfn-pending']
        def tmux(argv, **kwargs):
            if argv[1] == 'display-message':
                return 'the exact startup command'
            self.assertEqual(argv[:2], ['tmux', 'kill-session'])
            live.clear()
            return ''
        with patch.object(worker, 'sessions', side_effect=lambda: list(live)), \
                patch.object(worker, 'command', side_effect=tmux), patch.object(self.w, 'report', return_value=self.clean):
            result = self.w.stop()
        self.assertEqual(result['stopped'], ['pending'])
        self.assertTrue((directory / 'cancelled').exists())
        self.assertEqual(json.loads((directory / 'manifest.json').read_text())['exit_status'], 143)

    def test_stop_escalates_after_graceful_timeout(self):
        manifest = dict(pgid=999999, token='ours')
        members = [dict(pid=999999)]
        state = {'killed': False, 'time': 0}
        def kill(pgid, sig):
            self.assertEqual(pgid, manifest['pgid'])
            if sig == signal.SIGKILL:
                state['killed'] = True
        def tick():
            state['time'] += 1
            return state['time']
        with patch.object(self.w, 'group', side_effect=lambda m: [] if state['killed'] else members), \
                patch.object(Path, 'read_bytes', return_value=b'GFN_MANAGED_RUN_TOKEN=ours\0'), \
                patch('os.killpg', side_effect=kill) as calls, patch('time.monotonic', side_effect=tick), patch('time.sleep'):
            self.w.terminate_group(manifest)
        self.assertEqual([c.args[1] for c in calls.call_args_list], [signal.SIGTERM, signal.SIGKILL])

    def test_logs_tail_uses_latest_outside_repo(self):
        (self.w.state / 'latest').write_text('last-run')
        with patch('os.execvp') as execute:
            self.w.logs()
        self.assertEqual(execute.call_args.args, ('tail', ['tail', '-n', '100', '-F',
                         str(self.w.state / 'last-run' / 'worker.log')]))


class LocalGitUpdateTests(unittest.TestCase):
    """Exercise fetch/switch/ff against a local-only origin, with mock processes."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.old_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.old_cwd)
        self.origin = self.path / 'origin'
        self.clone = self.path / 'clone'
        self.origin.mkdir()
        def git(where, *args):
            return subprocess.check_output(['git', '-C', str(where), *args], stderr=subprocess.PIPE, text=True).strip()
        self.git = git
        git(self.origin, 'init', '-b', 'tooling')
        git(self.origin, 'config', 'user.email', 'test@example.invalid')
        git(self.origin, 'config', 'user.name', 'Local test')
        for name in (Path(path).name for path in worker.HELPER_PATHS):
            (self.origin / 'scripts').mkdir(exist_ok=True)
            shutil.copy2(SCRIPTS / name, self.origin / 'scripts' / name)
        shutil.copytree(SCRIPTS / 'presets', self.origin / 'scripts' / 'presets')
        git(self.origin, 'add', '.')
        git(self.origin, 'commit', '-m', 'tooling fixture')
        git(self.origin, 'switch', '-c', 'epoch')
        (self.origin / 'training-marker').write_text('epoch-specific behavior')
        git(self.origin, 'add', '.')
        git(self.origin, 'commit', '-m', 'epoch fixture')
        self.target = git(self.origin, 'rev-parse', 'HEAD')
        subprocess.check_output(['git', 'clone', '-b', 'tooling', str(self.origin), str(self.clone)], stderr=subprocess.PIPE)
        environment = self.path / 'env.sh'
        environment.touch()
        self.w = worker.Worker(dict(repo=str(self.clone), state=str(self.path / 'records'),
                                    environment=str(environment), token='test', branch='epoch',
                                    fingerprint=worker.fingerprint(self.clone)))
        self.w.acquire()
        for target, name, value in ((worker, 'sessions', []), (worker, 'training_processes', []),
                                    (self.w, 'active_records', [])):
            mock = patch.object(target, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch('shutil.which', return_value='/bin/mock')
        mock.start()
        self.addCleanup(mock.stop)

    def test_switch_keeps_epoch_content_and_converges(self):
        self.assertEqual(self.w.fetch()['target_sha'], self.target)
        self.w.p['target_sha'] = self.target
        report = self.w.update()
        self.assertEqual((report['branch'], report['sha'], report['dirty']), ('epoch', self.target, ''))
        self.assertEqual((self.clone / 'training-marker').read_text(), 'epoch-specific behavior')

    def test_missing_target_tooling_refuses_before_switch(self):
        self.git(self.origin, 'rm', 'scripts/tpu_worker.py')
        self.git(self.origin, 'commit', '-m', 'remove tooling fixture')
        with self.assertRaisesRegex(RuntimeError, 'lacks scripts/tpu_worker.py'):
            self.w.fetch()
        self.assertEqual(self.git(self.clone, 'branch', '--show-current'), 'tooling')

    def test_changed_target_preset_is_delivered_by_switch(self):
        preset = self.origin / 'scripts/presets/adam.json'
        value = json.loads(preset.read_text())
        value['flags']['total_steps'] = 3100
        preset.write_text(json.dumps(value, indent=2) + '\n')
        self.git(self.origin, 'add', '.')
        self.git(self.origin, 'commit', '-m', 'change preset fixture')
        target = self.git(self.origin, 'rev-parse', 'HEAD')
        self.assertEqual(self.w.fetch()['target_sha'], target)
        self.w.p['target_sha'] = target
        report = self.w.update()
        self.assertEqual((report['branch'], report['sha'], report['dirty']), ('epoch', target, ''))
        self.assertEqual((self.clone / 'scripts/presets/adam.json').read_text(), preset.read_text())

    def test_changed_target_helper_requires_manual_bootstrap(self):
        with (self.origin / 'scripts/tpu_worker.py').open('a') as helper:
            helper.write('\n# changed helper revision\n')
        self.git(self.origin, 'add', '.')
        self.git(self.origin, 'commit', '-m', 'change helper fixture')
        with self.assertRaisesRegex(RuntimeError, 'executable helpers differ; manually bootstrap'):
            self.w.fetch()
        self.assertEqual(self.git(self.clone, 'branch', '--show-current'), 'tooling')

    def test_controller_delivers_preset_only_changes_to_all_workers(self):
        # Exercise the controller's reservations/preflight/fetch/apply/convergence
        # path against two real local Git checkouts; no SSH/cloud calls occur.
        second_clone = self.path / 'clone-2'
        subprocess.check_output(['git', 'clone', '-b', 'tooling', str(self.origin), str(second_clone)],
                                stderr=subprocess.PIPE)
        second = worker.Worker({**self.w.p, 'repo': str(second_clone), 'state': str(self.path / 'records-2')})
        self.w.release()
        workers = {0: self.w, 1: second}
        for operation in ('pull', 'switch'):
            with self.subTest(operation=operation):
                # First fast-forward tooling, then switch to epoch. Both target
                # revisions alter only presets, and controllers already see them.
                branch = 'tooling' if operation == 'pull' else 'epoch'
                self.git(self.origin, 'switch', branch)
                preset = self.origin / 'scripts/presets/adam.json'
                value = json.loads(preset.read_text())
                value['flags']['total_steps'] = 3100 if operation == 'pull' else 3200
                preset.write_text(json.dumps(value, indent=2) + '\n')
                self.git(self.origin, 'add', '.')
                self.git(self.origin, 'commit', '-m', 'preset-only ' + operation)
                target = self.git(self.origin, 'rev-parse', 'HEAD')
                controller = control.Controller('1')
                controller.base['fingerprint'] = worker.fingerprint(self.origin)

                def discover():
                    controller.workers = list(workers)

                def rpc(wid, action, extra=None):
                    w = workers[wid]
                    os.chdir(w.repo)  # Each real RPC starts in its worker's checkout.
                    w.p.update(token=controller.token, fingerprint=controller.base['fingerprint'])
                    w.p.update(extra or {})
                    # Monitor usage is irrelevant to Git convergence; report Git
                    # and mocked process/tmux state without querying libtpu.
                    return w.report() if action == 'status' else getattr(w, action)()

                with patch.object(control.Controller, 'discover', side_effect=discover), \
                        patch.object(controller, 'rpc', side_effect=rpc), \
                        patch.object(second, 'active_records', return_value=[]), \
                        redirect_stdout(io.StringIO()):
                    controller.operate(operation, branch='epoch' if operation == 'switch' else None)

                for w in workers.values():
                    self.assertEqual(self.git(w.repo, 'rev-parse', 'HEAD'), target)
                    self.assertEqual(self.git(w.repo, 'branch', '--show-current'), branch)
                    self.assertEqual(self.git(w.repo, 'status', '--porcelain'), '')
                    self.assertEqual((w.repo / 'scripts/presets/adam.json').read_text(), preset.read_text())
                    self.assertFalse(w.lock.exists())

    def test_diverged_local_branch_refuses_fast_forward(self):
        self.git(self.clone, 'config', 'user.email', 'test@example.invalid')
        self.git(self.clone, 'config', 'user.name', 'Local test')
        self.git(self.clone, 'switch', '-c', 'epoch')
        (self.clone / 'local-marker').write_text('local')
        self.git(self.clone, 'add', '.')
        self.git(self.clone, 'commit', '-m', 'divergent local fixture')
        with self.assertRaises(RuntimeError):
            self.w.fetch()
        self.assertTrue((self.clone / 'local-marker').exists())



if __name__ == '__main__':
    unittest.main()
