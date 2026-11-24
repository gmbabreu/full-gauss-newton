"""Linux CPU tests only; all cloud and installation subprocesses are mocked."""
import json
import os
from pathlib import Path
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
                    service.preflight()
                inventory.assert_not_called()

    @patch.object(service, 'metadata_identity', return_value=service.IDENTITY)
    def test_selected_executable_is_reported_and_used_for_all_commands(self, identity):
        for executable in ('/usr/bin/gcloud', service.SNAP_GCLOUD):
            with self.subTest(executable=executable), \
                    patch.object(service, 'find_gcloud', return_value=executable), \
                    patch.object(queue, 'log') as log, \
                    patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='[]')) as run:
                with tempfile.TemporaryDirectory() as state, service.isolated_environment(Path(state)) as env:
                    service.preflight()
                    self.assertEqual(len(run.call_args_list), 2)
                    for call in run.call_args_list:
                        self.assertEqual(call.args[0][0], executable)
                        self.assertEqual(call.kwargs['env'], env)
                    for action in ('create', 'delete'):
                        self.assertEqual(queue.gcloud('queued-resources', action)[0], executable)
                log.assert_any_call('Using gcloud executable: ' + executable)
                snapshot = next(json.loads(call.args[0]) for call in log.call_args_list
                                if call.args[0].startswith('{'))
                self.assertEqual(snapshot['gcloud_executable'], executable)

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
            service.preflight()
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
    @patch.object(queue, 'inventory', return_value={})
    @patch.object(queue, 'reconcile')
    def test_preflight_reads_both_inventories_only(self, reconcile, inventory, executable, identity):
        service.preflight()
        self.assertEqual([x.args[0] for x in inventory.call_args_list], ['queued-resources', 'tpu-vm'])
        reconcile.assert_not_called()

    @patch.object(service, 'metadata_identity', return_value=service.IDENTITY)
    @patch.object(service, 'find_gcloud', return_value='/usr/bin/gcloud')
    @patch.object(queue, 'inventory', side_effect=RuntimeError('permission denied'))
    @patch.object(queue, 'main')
    def test_preflight_failure_never_enters_watchdog(self, main, inventory, executable, identity):
        with tempfile.TemporaryDirectory() as directory, patch.object(service, 'STATE', Path(directory)), \
                patch.object(service, 'validate_files'), patch.object(sys, 'argv', ['service']):
            # Explicit context keeps this test independent of the production state path.
            with patch.object(service, 'isolated_environment', return_value=service.isolated_environment(Path(directory))):
                with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                    service.main()
        main.assert_not_called()

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
            self.assertEqual(copied, dict(original, startup_script=str(destination / 'startup.sh')))
            self.assertEqual((destination / 'startup.sh').read_bytes(), payload)
            self.assertEqual(json.loads(config.read_text()), original)
            service.validate_files(destination / 'queue.json')
            (destination / 'startup.sh').chmod(0o666)
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
            unit = (root / 'unit').read_text()
            for setting in ['User=gfn-spot', 'Restart=on-failure', 'RestartSec=60',
                            'StandardOutput=journal', 'WantedBy=multi-user.target',
                            'NoNewPrivileges=true', 'ProtectSystem=strict', 'ProtectHome=true',
                            'PrivateTmp=true', 'ReadWritePaths=/var/lib/gfn-spot']:
                self.assertIn(setting, unit)

    @patch.object(installer.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='active', stderr=''))
    def test_installer_refuses_active_or_enabled_service(self, run):
        with self.assertRaisesRegex(ValueError, 'stopped and disabled'):
            installer.require_inactive()

    def test_targets_and_poll_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'queue.json'
            startup = Path(directory) / 'startup.sh'
            startup.write_text('#!/bin/bash\n')
            config.write_text(json.dumps(dict(runtime_version='runtime', service_account='resource@example.invalid',
                                             startup_script=str(startup))))
            with patch.object(queue.Path, 'home', return_value=Path(directory)), \
                    patch.object(queue, 'inventory', return_value={}), \
                    patch.object(queue, 'reconcile') as reconcile, \
                    patch.object(queue.time, 'sleep', side_effect=InterruptedError('stop after one poll')) as sleep:
                with self.assertRaises(InterruptedError):
                    queue.main(['--config', str(config), '--targets', '3', '4', '--watch'])
                self.assertEqual([c.args[0] for c in reconcile.call_args_list], ['3', '4'])
                sleep.assert_called_once_with(300)

    @patch.object(service, 'validate_files')
    @patch.object(service, 'preflight')
    @patch.object(queue, 'main')
    def test_service_selects_both_targets(self, main, preflight, validate):
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, 'argv', ['service']):
            with patch.object(service, 'isolated_environment', return_value=service.isolated_environment(Path(directory))):
                service.main()
        self.assertEqual(main.call_args.args[0], ['--config', str(service.CONFIG), '--targets', '3', '4', '--apply', '--watch'])

    @patch.object(service, 'validate_files')
    @patch.object(service, 'preflight')
    @patch.object(queue, 'main')
    def test_preflight_option_does_not_reconcile(self, main, preflight, validate):
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, 'argv', ['service', '--preflight']):
            with patch.object(service, 'isolated_environment', return_value=service.isolated_environment(Path(directory))):
                service.main()
        preflight.assert_called_once()
        main.assert_not_called()

    def test_either_inventory_failure_prevents_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'queue.json'
            startup = Path(directory) / 'startup.sh'
            startup.write_text('#!/bin/bash\n')
            config.write_text(json.dumps(dict(runtime_version='runtime', service_account='resource@example.invalid',
                                             startup_script=str(startup))))
            for responses in ([RuntimeError('auth failed')], [{}, RuntimeError('auth failed')]):
                with patch.object(queue.Path, 'home', return_value=Path(directory)), \
                        patch.object(queue, 'inventory', side_effect=responses), \
                        patch.object(queue, 'reconcile') as reconcile:
                    with self.assertRaisesRegex(RuntimeError, 'auth failed'):
                        queue.main(['--config', str(config), '--targets', '3', '4', '--apply'])
                    reconcile.assert_not_called()


if __name__ == '__main__':
    unittest.main()
