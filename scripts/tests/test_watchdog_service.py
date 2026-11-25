"""Linux CPU tests only; all cloud and installation subprocesses are mocked."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TPU = Path(__file__).resolve().parents[1] / 'tpu'
sys.path.insert(0, str(TPU))
import spot_queue as queue
import watchdog_service as service
import install_watchdog as installer


class ServiceTests(unittest.TestCase):
    def setUp(self):
        executable = patch.object(queue, 'GCLOUD_EXECUTABLE', 'gcloud')
        executable.start()
        self.addCleanup(executable.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.startup = self.root / 'startup.sh'
        self.body = b'#!/bin/bash\n# captured bytes\nprintf \"prepared\"\n'
        self.startup.write_bytes(self.body)
        self.legacy = dict(runtime_version='runtime', service_account='resource@example.invalid',
                           startup_script=str(self.startup))
        self.configuration = queue.queue_config.normalize(self.legacy)
        self.config_path = self.root / 'queue.json'
        self.config_path.write_text(json.dumps(self.configuration))

    def test_ordinary_gcloud_discovery_uses_fixed_path(self):
        for executable in ('/usr/bin/gcloud', '/usr/local/bin/gcloud'):
            with self.subTest(executable=executable), \
                    patch.dict(os.environ, {'PATH': '/untrusted/bin'}), \
                    patch.object(service.shutil, 'which', return_value=executable) as which, \
                    patch.object(service.Path, 'is_file') as is_file:
                self.assertEqual(service.find_gcloud(), executable)
                which.assert_called_once_with('gcloud', path=service.SERVICE_PATH)
                is_file.assert_not_called()

    @patch.object(service.shutil, 'which', return_value=None)
    @patch.object(service.Path, 'is_file', return_value=True)
    @patch.object(service.os, 'access', return_value=True)
    def test_snap_discovery_uses_direct_sdk_entry_point(self, access, is_file, which):
        self.assertEqual(service.find_gcloud(), '/snap/google-cloud-cli/current/bin/gcloud')
        access.assert_called_once_with(service.SNAP_GCLOUD, os.X_OK)

    @patch.object(service.shutil, 'which', return_value=None)
    def test_missing_or_nonexecutable_gcloud_fails_closed(self, which):
        for exists, executable in ((False, True), (True, False)):
            with self.subTest(exists=exists, executable=executable), \
                    patch.object(service.Path, 'is_file', return_value=exists), \
                    patch.object(service.os, 'access', return_value=executable), \
                    patch.object(service, 'metadata_identity', return_value=service.IDENTITY), \
                    patch.object(queue, 'inventory') as inventory:
                with self.assertRaisesRegex(ValueError, 'No executable gcloud'):
                    service.preflight(self.configuration)
                inventory.assert_not_called()

    @patch.object(service, 'metadata_identity', return_value=service.IDENTITY)
    def test_selected_executable_is_reported_and_used_for_all_commands(self, identity):
        for executable in ('/usr/bin/gcloud', service.SNAP_GCLOUD):
            with self.subTest(executable=executable), \
                    patch.object(service, 'find_gcloud', return_value=executable), \
                    patch.object(queue, 'log') as log, \
                    patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='[]')) as run:
                with tempfile.TemporaryDirectory() as state, service.isolated_environment(Path(state)) as env:
                    service.preflight(self.configuration)
                    self.assertEqual(len(run.call_args_list), 2)
                    self.assertEqual([c.args[0][3:5] for c in run.call_args_list],
                                     [['queued-resources', 'list'], ['tpu-vm', 'list']])
                    for call in run.call_args_list:
                        self.assertEqual(call.args[0][0], executable)
                        self.assertEqual(call.kwargs['env'], env)
                    for action in ('create', 'delete'):
                        self.assertEqual(queue.gcloud('queued-resources', action)[0], executable)
                log.assert_any_call('Using gcloud executable: ' + executable)
                log.assert_any_call('Attached identity: ' + service.IDENTITY)

    @patch.object(service, 'metadata_identity', return_value=service.IDENTITY)
    @patch.object(service, 'find_gcloud', return_value='/usr/bin/gcloud')
    def test_preflight_logs_only_bounded_status_fields(self, executable, identity):
        sensitive = {'metadata': {'startup-script': 'STARTUP_SECRET', 'ssh-keys': 'SSH_SECRET'},
                     'description': 'DESCRIPTION_SECRET', 'unknown': 'RAW_SECRET'}
        requests = {'failed-request': dict(sensitive, state={
            'state': 'FAILED', 'failedData': {'error': {
                'code': 8, 'message': 'capacity unavailable\nretry', 'details': ['DETAIL_SECRET']}},
            'suspendedData': {'reason': 'USER_INITIATED'}, 'unknown': 'STATE_SECRET'})}
        nodes = {'failed-node': dict(sensitive, state='STOPPED', health='UNHEALTHY',
                                    healthDescription='network failure'),
                 'unknown-node': dict(sensitive)}
        with patch.object(queue, 'inventory', side_effect=[requests, nodes]) as inventory, \
                patch.object(queue, 'reconcile') as reconcile, patch.object(queue, 'log') as log:
            service.preflight(self.configuration)
        output = '\n'.join(c.args[0] for c in log.call_args_list)
        for key in ('3', '4'):
            target = queue.REGISTRY[key]
            self.assertIn(f'Managed target {key}: {target.name} ({target.project}/{target.zone})', output)
        for text in ('failed-request', 'FAILED', 'capacity unavailable\\nretry', 'USER_INITIATED',
                     'failed-node', 'STOPPED', 'UNHEALTHY', 'network failure', 'unknown-node', 'UNKNOWN'):
            self.assertIn(text, output)
        self.assertNotIn('SECRET', output)
        self.assertNotIn('nodeSpec', output)
        self.assertEqual([c.args[0] for c in inventory.call_args_list], ['queued-resources', 'tpu-vm'])
        self.assertTrue(all('\n' not in c.args[0] for c in log.call_args_list))
        self.assertLess(len(service.resource_status({'state': 'X' * 1000,
                                                    'healthDescription': 'Y' * 1000})), 400)
        reconcile.assert_not_called()

    def test_environment_is_fresh_allowlist_and_restored(self):
        hostile = {'CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE': '/user/key',
                   'CLOUDSDK_AUTH_ACCESS_TOKEN': 'token',
                   'CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT': 'other',
                   'CLOUDSDK_CORE_ACCOUNT': 'user', 'CLOUDSDK_CONFIG': '/user/cache',
                   'GOOGLE_APPLICATION_CREDENTIALS': '/user/adc',
                   'CLOUDSDK_AUTH_ACCESS_TOKEN_FILE': '/token',
                   'CLOUDSDK_ACTIVE_CONFIG_NAME': 'user', 'HOME': '/user',
                   'HTTPS_PROXY': 'proxy', 'PYTHONPATH': '/user/code',
                   'PATH': '/user/bin:/snap/bin'}
        with tempfile.TemporaryDirectory() as state, patch.dict(os.environ, hostile, clear=True):
            with service.isolated_environment(Path(state)) as env:
                self.assertEqual(dict(os.environ), env)
                for key in hostile:
                    self.assertNotEqual(os.environ.get(key), hostile[key])
                first = Path(env['CLOUDSDK_CONFIG'])
                self.assertTrue((first / 'configurations/config_watchdog').is_file())
                (first / 'credentials.db').touch()
                with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='[]')) as call:
                    queue.inventory('queued-resources')
                self.assertEqual(call.call_args.kwargs['env'], env)
            self.assertFalse(first.exists())
            with service.isolated_environment(Path(state)) as env:
                self.assertFalse((Path(env['CLOUDSDK_CONFIG']) / 'credentials.db').exists())
            self.assertEqual(dict(os.environ), hostile)

    @patch.object(service, 'metadata_identity', return_value='wrong@example.invalid')
    @patch.object(queue, 'inventory')
    def test_metadata_mismatch_prevents_any_cloud_call(self, inventory, identity):
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            service.preflight(self.configuration)
        inventory.assert_not_called()

    def test_metadata_request_is_direct_and_requires_google_header(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = {'Metadata-Flavor': 'Google'}
        response.read.return_value = service.IDENTITY.encode()
        opener = unittest.mock.Mock()
        opener.open.return_value = response
        with patch.object(service.urllib.request, 'build_opener', return_value=opener):
            self.assertEqual(service.metadata_identity(), service.IDENTITY)
            request = opener.open.call_args.args[0]
            self.assertEqual(request.get_header('Metadata-flavor'), 'Google')
            self.assertEqual(opener.open.call_args.kwargs['timeout'], 10)
            response.headers = {}
            with self.assertRaisesRegex(ValueError, 'metadata response header'):
                service.metadata_identity()

    @patch.object(service, 'metadata_identity', return_value=service.IDENTITY)
    @patch.object(service, 'find_gcloud', return_value='/usr/bin/gcloud')
    @patch.object(queue, 'main')
    def test_preflight_failure_never_enters_watchdog(self, main, executable, identity):
        for response, exception, message in ((RuntimeError('permission denied'), RuntimeError, 'permission denied'),
                                            ({'gn-v4-32-spot-2': {}}, ValueError, 'naming conflict')):
            with self.subTest(message=message), \
                    patch.object(queue, 'inventory', side_effect=[response, response]), \
                    patch.object(queue, 'log') as log, \
                    patch.object(service, 'validate_files', return_value=self.configuration), \
                    patch.object(sys, 'argv', ['service']), \
                    patch.object(service, 'isolated_environment',
                                 return_value=service.isolated_environment(self.root)):
                with self.assertRaisesRegex(exception, message):
                    service.main()
                main.assert_not_called()
                self.assertFalse(any('preflight passed' in c.args[0] for c in log.call_args_list))

    def test_copies_bytes_rewrites_only_path_and_checks_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, startup = root / 'source.json', root / 'source.sh'
            original = dict(runtime_version='captured-runtime', service_account='captured@example.invalid',
                            startup_script='/box248/prepared/startup.sh')
            config.write_text(json.dumps(original))
            payload = b'#!/bin/bash\n# captured bytes\nprintf "prepared"\n'
            startup.write_bytes(payload)
            destination = root / 'installed'
            destination.mkdir()
            with patch.object(installer.subprocess, 'run') as run:
                copied = installer.copy_configuration(config, startup, destination)
                self.assertEqual(copied['preparations']['v4'],
                                 dict(original, accelerator_type='v4-32',
                                      startup_script=str(destination / 'startup-v4.sh')))
            self.assertEqual((destination / 'startup-v4.sh').read_bytes(), payload)
            self.assertEqual(json.loads(config.read_text()), original)
            service.validate_files(destination / 'queue.json')
            (destination / 'startup-v4.sh').chmod(0o666)
            with self.assertRaisesRegex(ValueError, 'group/other write'):
                service.validate_files(destination / 'queue.json')
            self.assertEqual(run.call_args.args[0][:2], ['bash', '-n'])

    def test_installation_never_activates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            account = SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid(),
                                      pw_dir=str(root / 'state'), pw_shell='/usr/sbin/nologin')
            account.pw_uid = 12345
            with patch.multiple(installer, CODE=root / 'code', STATE=root / 'state', UNIT=root / 'unit'), \
                    patch.object(installer.os, 'geteuid', return_value=0), \
                    patch.object(installer.os, 'chown'), \
                    patch.object(installer.pwd, 'getpwnam', return_value=account), \
                    patch.object(installer.grp, 'getgrnam', return_value=SimpleNamespace(gr_gid=account.pw_gid)), \
                    patch.object(installer, 'require_inactive') as inactive, \
                    patch.object(installer, 'copy_configuration') as copy, \
                    patch.object(installer, 'run') as run:
                installer.install('prepared.json', 'prepared.sh')
            self.assertEqual(inactive.call_count, 2)
            commands = [x.args for x in run.call_args_list]
            self.assertIn(('systemctl', 'disable', installer.SERVICE), commands)
            self.assertFalse(any(x[0] == 'systemctl' and any(v in x for v in ('enable', 'start', 'restart')) for x in commands))
            self.assertEqual((root / 'code/spot_queue.py').read_bytes(), (TPU / 'spot_queue.py').read_bytes())
            for name, source in installer.BUNDLE.items():
                self.assertEqual((root / 'code' / name).read_bytes(), source.read_bytes())
            self.assertEqual(set(installer.BUNDLE),
                             {'spot_queue.py', 'watchdog_service.py', 'queue_config.py', 'tpu_targets.py'})
            subprocess.run([sys.executable, '-I', '-c',
                            'import sys; sys.path.insert(0, sys.argv[1]); '
                            'import watchdog_service; '
                            'assert len(watchdog_service.queue.REGISTRY) == 8', str(root / 'code')], check=True)
            for entry in ('spot_queue.py', 'watchdog_service.py', 'queue_config.py'):
                result = subprocess.run([sys.executable, '-I', str(root / 'code' / entry), '--help'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            unit = (root / 'unit').read_text()
            for setting in ['User=gfn-spot', 'Restart=on-failure', 'RestartSec=60',
                            'StandardOutput=journal', 'WantedBy=multi-user.target',
                            'NoNewPrivileges=true', 'ProtectSystem=strict', 'ProtectHome=true',
                            'PrivateTmp=true', 'ReadWritePaths=/var/lib/gfn-spot']:
                self.assertIn(setting, unit)

    @patch.object(installer.subprocess, 'run')
    def test_installer_refuses_active_or_enabled_service(self, run):
        for responses in ([SimpleNamespace(returncode=0, stdout='active', stderr='')],
                          [SimpleNamespace(returncode=3, stdout='inactive', stderr=''),
                           SimpleNamespace(returncode=0, stdout='enabled', stderr='')]):
            run.side_effect = responses
            with self.assertRaisesRegex(ValueError, 'stopped and disabled'):
                installer.require_inactive()

    def test_targets_and_poll_interval(self):
        with patch.object(queue.Path, 'home', return_value=self.root), \
                patch.object(queue, 'inventory', return_value={}), \
                patch.object(queue, 'reconcile') as reconcile, \
                patch.object(queue.time, 'sleep', side_effect=InterruptedError('stop after one poll')) as sleep:
            with self.assertRaises(InterruptedError):
                queue.main(['--config', str(self.config_path), '--targets', '3', '4', '--watch'])
            self.assertEqual([c.args[0] for c in reconcile.call_args_list], ['3', '4'])
            sleep.assert_called_once_with(300)

    @patch.object(service, 'validate_files')
    @patch.object(service, 'preflight')
    @patch.object(queue, 'main')
    def test_service_runs_watchdog_unless_preflight_only(self, main, preflight, validate):
        for options in ([], ['--preflight']):
            main.reset_mock()
            preflight.reset_mock()
            with self.subTest(options=options), patch.object(sys, 'argv', ['service', *options]), \
                    patch.object(service, 'isolated_environment',
                                 return_value=service.isolated_environment(self.root)):
                service.main()
            preflight.assert_called_once_with(validate.return_value)
            if options:
                main.assert_not_called()
            else:
                main.assert_called_once_with(['--config', str(service.CONFIG), '--apply', '--watch'])

    def test_staged_bundle_entry_points_work_with_isolated_python(self):
        import tarfile
        bundle = self.root / 'bundle.tgz'
        paths = ('tpu/spot_queue.py', 'tpu/watchdog_service.py', 'tpu/install_watchdog.py',
                 'tpu/queue_config.py', 'tpu/gfn-spot-watchdog.service', 'tpu_targets.py')
        with tarfile.open(bundle, 'w:gz') as archive:
            for name in paths:
                archive.add(TPU.parent / name, arcname=name)
        stage = self.root / 'stage'
        stage.mkdir()
        with tarfile.open(bundle) as archive:
            archive.extractall(stage)
        # --help parses and exits before any installer or cloud operation.
        for entry in ('install_watchdog.py', 'watchdog_service.py', 'queue_config.py', 'spot_queue.py'):
            result = subprocess.run([sys.executable, '-I', str(stage / 'tpu' / entry), '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_separate_preparations_and_payload_directory_relocation(self):
        for key in ('5', '7'):
            accelerator = queue.REGISTRY[key].accelerator
            preparation = accelerator.split('-')[0]
            self.configuration['preparations'][preparation] = dict(self.legacy, accelerator_type=accelerator)
            self.configuration['targets'][key] = dict(managed=True, preparation=preparation)
        staging = self.root / 'payloads'
        staging.mkdir()
        for name, prep in self.configuration['preparations'].items():
            prep['startup_script'] = '/original-host/' + name + '.sh'
            (staging / (name + '.sh')).write_bytes(self.body + name.encode() + b'\n')
        source = self.root / 'version2.json'
        source.write_text(json.dumps(self.configuration))
        destination = self.root / 'installed'
        destination.mkdir()
        with patch.object(installer.subprocess, 'run'):
            installed = installer.copy_configuration(source, None, destination, staging)
        for name in ('v4', 'v5e', 'v6e'):
            self.assertEqual(Path(installed['preparations'][name]['startup_script']).read_bytes(),
                             (staging / (name + '.sh')).read_bytes())
        self.assertFalse(installed['targets']['6']['managed'])
        self.assertFalse(installed['targets']['8']['managed'])


if __name__ == '__main__':
    unittest.main()
