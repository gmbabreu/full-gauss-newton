"""Registry, multi-location watchdog and routing checks. No real cloud calls."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / 'tpu')]
import tpu_targets as registry
import tpu_control as control
import spot_queue as queue
import queue_config as config
import install_watchdog as installer
import watchdog_service as service


class RoutingTests(unittest.TestCase):
    def test_stable_legacy_names_and_numeric_suffixes_for_new_targets(self):
        expected = {'1': 'gn-v4-32', '2': 'gn-v4-32-2', '3': 'gn-v4-32-spot-1',
                    '4': 'gn-v4-32-spot-2', '5': 'gn-v6e-16-spot-5', '6': 'gn-v6e-16-spot-6',
                    '7': 'gn-v5e-16-spot-7', '8': 'gn-v5e-16-spot-8'}
        self.assertEqual({key: value.name for key, value in registry.TARGETS.items()}, expected)
        for key in ('5', '6', '7', '8'):
            self.assertTrue(registry.TARGETS[key].name.endswith('-' + key))
            self.assertFalse(registry.TARGETS[key].managed)

    def test_numeric_routing_and_dynamic_worker_counts(self):
        for key, target in registry.TARGETS.items():
            with self.subTest(key=key):
                controller = control.Controller(key)
                for action in ('describe', 'ssh'):
                    command = controller.gcloud(action, target.name)
                    self.assertIn('--project=' + target.project, command)
                    self.assertIn('--zone=' + target.zone, command)
                    self.assertIn(target.name, command)
                self.assertEqual(controller.cfg['accelerator'], target.accelerator)
                for count in (1, 2, 4, 8):
                    response = SimpleNamespace(returncode=0, stdout=json.dumps({'networkEndpoints': [{}] * count}))
                    with patch.object(control.subprocess, 'run', return_value=response):
                        controller.discover()
                    self.assertEqual(controller.workers, list(range(count)))

    def test_all_numeric_commands_are_available_even_when_watchdog_disabled(self):
        def discover(controller):
            controller.workers = [0, 1]
        for key in registry.TARGETS:
            with self.subTest(key=key), redirect_stdout(io.StringIO()), \
                    patch.object(control.Controller, 'discover', discover), \
                    patch.object(control.Controller, 'rpc', return_value={}) as rpc, \
                    patch.object(control.Controller, 'ssh', return_value=SimpleNamespace(returncode=0)) as ssh, \
                    patch.object(control.Controller, 'operate') as operate:
                for args in (['status'], ['ssh', '0'], ['logs', '0'], ['run', 'gn_adam'],
                             ['stop'], ['pull'], ['switch', 'main']):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(control.main([key, *args]), 0)
                    self.assertTrue(output.getvalue().startswith(f'Target {key}: {registry.TARGETS[key].name} '))
                self.assertEqual(operate.call_count, 4)
                self.assertEqual([c.args[0] for c in operate.call_args_list], ['run', 'stop', 'pull', 'switch'])
                self.assertEqual(operate.call_args.kwargs['branch'], 'main')
                self.assertEqual(ssh.call_args.args[0], 0)
                self.assertEqual(rpc.call_args.args[:2], (0, 'logs'))

    def test_previews_show_type_and_location_without_cloud_or_flag_changes(self):
        for key, target in registry.TARGETS.items():
            output = io.StringIO()
            with redirect_stdout(output), patch.object(control.subprocess, 'run', side_effect=AssertionError('cloud')):
                control.main([key, 'run', 'gn_adam', '--dry-run', '--mesh_dim=1,2,4', '--inner_loop_iter=17'])
            for value in (f'Target {key}:', target.name, target.zone, target.accelerator, target.allocation,
                          '--mesh_dim=1,2,4', '--inner_loop_iter=17'):
                self.assertIn(value, output.getvalue())

    def test_environment_overrides_win_only_for_manual_controller(self):
        with patch.dict(os.environ, GFN_PROJECT='other', GFN_ZONE='other-zone', GFN_TPU_NAME='manual'):
            controller = control.Controller('5')
            self.assertEqual((controller.cfg['project'], controller.cfg['zone'], controller.cfg['target']),
                             ('other', 'other-zone', 'manual'))
            self.assertEqual(registry.TARGETS['5'].zone, 'us-east1-d')

    def test_unallocated_or_unreadable_target_is_useful_error_not_empty_workers(self):
        controller = control.Controller('5')
        with patch.object(control.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='NOT_FOUND')):
            with self.assertRaisesRegex(RuntimeError, 'us-east1-d.*may not be allocated'):
                controller.discover()
        with patch.object(control.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='{}')):
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                controller.discover()


class RegistryWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.startup = self.root / 'captured.sh'
        self.body = b'#!/bin/bash\n# existing captured payload\nprintf "unchanged"\n'
        self.startup.write_bytes(self.body)
        self.legacy = dict(runtime_version='operator-captured-runtime',
                           service_account='captured@example.invalid', startup_script=str(self.startup))
        self.configuration = config.normalize(self.legacy)
        self.args = SimpleNamespace(apply=True, configuration=self.configuration)

    def enable(self, key):
        target = registry.TARGETS[key]
        prep = target.accelerator.split('-')[0]
        self.configuration['preparations'][prep] = dict(self.legacy, accelerator_type=target.accelerator)
        self.configuration['targets'][key] = dict(managed=True, preparation=prep)
        config.normalize(self.configuration)

    def resource(self, key, state):
        target = registry.TARGETS[key]
        return {'name': f'projects/{target.project}/locations/{target.zone}/queuedResources/{target.name}',
                'spot': {}, 'state': {'state': state}, 'tpu': {'nodeSpec': [{
                    'nodeId': target.name, 'parent': f'projects/{target.project}/locations/{target.zone}',
                    'node': {'acceleratorType': target.api_accelerator, 'labels': {'managed-by': queue.OWNER}}}]}}

    def test_legacy_migration_preserves_settings_and_only_enables_3_4(self):
        self.assertEqual(config.enabled(self.configuration), ['3', '4'])
        self.assertEqual(self.configuration['preparations']['v4'], dict(self.legacy, accelerator_type='v4-32'))
        self.assertEqual(self.startup.read_bytes(), self.body)
        for key in ('1', '2', '5', '6', '7', '8'):
            self.assertFalse(self.configuration['targets'][key]['managed'])

    def test_copy_migrates_without_changing_payload_bytes(self):
        source = self.root / 'legacy.json'
        source.write_text(json.dumps(self.legacy))
        destination = self.root / 'installed'
        destination.mkdir()
        with patch.object(installer.subprocess, 'run'):
            copied = installer.copy_configuration(source, None, destination)
        self.assertEqual(config.enabled(copied), ['3', '4'])
        self.assertEqual((destination / 'startup-v4.sh').read_bytes(), self.body)
        self.assertEqual(copied['preparations']['v4']['runtime_version'], self.legacy['runtime_version'])
        self.assertEqual(copied['preparations']['v4']['service_account'], self.legacy['service_account'])
        self.assertEqual(json.loads(source.read_text()), self.legacy)
        service.validate_files(destination / 'queue.json')

    def test_staged_bundle_entry_points_work_with_isolated_python(self):
        import tarfile
        bundle = self.root / 'bundle.tgz'
        paths = ('tpu/spot_queue.py', 'tpu/watchdog_service.py', 'tpu/install_watchdog.py',
                 'tpu/queue_config.py', 'tpu/gfn-spot-watchdog.service', 'tpu_targets.py')
        with tarfile.open(bundle, 'w:gz') as archive:
            for name in paths:
                archive.add(SCRIPTS / name, arcname=name)
        stage = self.root / 'stage'
        stage.mkdir()
        with tarfile.open(bundle) as archive:
            archive.extractall(stage)
        # --help parses and exits before any installer or cloud operation.
        for entry in ('install_watchdog.py', 'watchdog_service.py', 'queue_config.py'):
            result = subprocess.run([sys.executable, '-I', str(stage / 'tpu' / entry), '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_separate_preparations_and_payload_directory_relocation(self):
        for key in ('5', '7'):
            self.enable(key)
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

    def test_missing_preparation_or_wrong_accelerator_blocks_enabling(self):
        self.configuration['targets']['5'] = dict(managed=True, preparation='missing')
        with self.assertRaisesRegex(ValueError, 'explicit prepared'):
            config.normalize(self.configuration)
        self.configuration['targets']['5']['preparation'] = 'v4'
        with self.assertRaisesRegex(ValueError, 'accelerator'):
            config.normalize(self.configuration)

    def test_config_cli_enables_only_selected_target_and_preserves_input(self):
        source, output = self.root / 'source.json', self.root / 'enabled.json'
        source.write_text(json.dumps(self.legacy))
        argv = ['queue_config', '--config', str(source), '--output', str(output), '--enable', '5',
                '--preparation', 'v6e-us', '--accelerator-type', 'v6e-16', '--runtime-version', 'reviewed-runtime',
                '--service-account', 'new-resource@example.invalid', '--startup-script', str(self.startup)]
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            config.main()
        changed = config.load(output)
        self.assertEqual(config.enabled(changed), ['3', '4', '5'])
        self.assertEqual(config.preparation(changed, '3'), dict(self.legacy, accelerator_type='v4-32'))
        self.assertEqual(json.loads(source.read_text()), self.legacy)
        with patch.object(sys, 'argv', argv), self.assertRaises(FileExistsError):
            config.main()
        disabled = self.root / 'disabled.json'
        with patch.object(sys, 'argv', ['queue_config', '--config', str(output), '--output', str(disabled),
                                      '--disable', '5']), redirect_stdout(io.StringIO()):
            config.main()
        self.assertEqual(config.enabled(config.load(disabled)), ['3', '4'])

    def test_config_cli_refuses_enabling_without_output_or_complete_preparation(self):
        source = self.root / 'source.json'
        source.write_text(json.dumps(self.legacy))
        for options in (['--enable', '5'], ['--output', str(self.root / 'bad.json'), '--enable', '5',
                                         '--preparation', 'v6e', '--runtime-version', 'only-one-field']):
            with patch.object(sys, 'argv', ['queue_config', '--config', str(source), *options]), \
                    redirect_stdout(io.StringIO()), patch.object(config.argparse.ArgumentParser, 'error',
                                                                  side_effect=ValueError('bad config arguments')):
                with self.assertRaisesRegex(ValueError, 'bad config arguments'):
                    config.main()

    def test_disabled_targets_need_no_payload_and_cause_no_actions(self):
        self.configuration['targets']['5'] = dict(managed=False, preparation='not-prepared')
        config.normalize(self.configuration)
        with patch.object(queue, 'call') as call:
            queue.reconcile('5', {}, {}, self.args)
            queue.reconcile('5', {registry.TARGETS['5'].name: self.resource('5', 'FAILED')}, {}, self.args)
        call.assert_not_called()
        self.assertEqual(config.locations(self.configuration), [('fast-second-order', 'us-central2-b')])

    def test_disabling_existing_managed_target_leaves_resources_untouched(self):
        for key in ('3', '4'):
            self.configuration['targets'][key]['managed'] = False
            config.normalize(self.configuration)
            with patch.object(queue, 'call') as call:
                queue.reconcile(key, {registry.TARGETS[key].name: self.resource(key, 'FAILED')}, {}, self.args)
                queue.reconcile(key, {}, {}, self.args)
                call.assert_not_called()

    def test_fixed_targets_and_unregistered_configuration_are_rejected(self):
        for key in ('1', '2'):
            changed = deepcopy(self.configuration)
            changed['targets'][key] = dict(managed=True, preparation='v4')
            with self.assertRaisesRegex(ValueError, 'never be watchdog managed'):
                config.normalize(changed)
            with self.assertRaisesRegex(ValueError, 'cannot be watchdog managed'):
                queue.reconcile(key, {}, {}, self.args)
        self.configuration['targets']['9'] = dict(managed=False)
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            config.normalize(self.configuration)

    def test_entire_configuration_is_validated_before_any_cloud_call(self):
        self.configuration['targets']['8'] = dict(managed=True, preparation='absent')
        path = self.root / 'invalid.json'
        path.write_text(json.dumps(self.configuration))
        with patch.object(queue, 'call') as call:
            with self.assertRaises(ValueError):
                queue.main(['--config', str(path), '--targets', '3', '--apply'])
        call.assert_not_called()

    def test_grouped_inventories_once_per_location(self):
        for key in ('5', '6', '7', '8'):
            self.enable(key)
        with patch.object(queue, 'inventory', return_value={}) as inventory:
            observed = queue.collect_inventory(self.configuration)
        self.assertEqual(len(observed), 5)
        self.assertEqual(inventory.call_count, 10)
        self.assertEqual(set(c.args[1] for c in inventory.call_args_list), set(config.locations(self.configuration)))

    def test_failure_in_any_location_prevents_all_reconciliation(self):
        self.enable('5')
        path = self.root / 'queue.json'
        path.write_text(json.dumps(self.configuration))
        for failed_group in ('queued-resources', 'tpu-vm'):
            for failed_location in config.locations(self.configuration):
                def inventory(group, location):
                    if (group, location) == (failed_group, failed_location):
                        raise RuntimeError('auth/API failure')
                    return {}
                with patch.object(queue.Path, 'home', return_value=self.root), \
                        patch.object(queue, 'inventory', side_effect=inventory), \
                        patch.object(queue, 'reconcile') as reconcile:
                    with self.assertRaisesRegex(RuntimeError, 'auth/API'):
                        queue.main(['--config', str(path), '--apply'])
                    reconcile.assert_not_called()

    def test_per_target_create_and_ownership_checks(self):
        for key, target in registry.TARGETS.items():
            if key in {'1', '2'}:
                continue
            self.enable(key)
            resource = self.resource(key, 'FAILED')
            queue.validate_owned(resource, target)
            command = queue.create_command(key, config.preparation(self.configuration, key))
            for flag in ('--zone=' + target.zone, '--project=' + target.project,
                         '--accelerator-type=' + target.api_accelerator, '--node-id=' + target.name, '--spot',
                         '--labels=managed-by=' + queue.OWNER + ',gfn-target=' + key):
                self.assertIn(flag, command)
            for field in ('nodeId', 'parent', 'acceleratorType', 'labels', 'allocation', 'resource_name'):
                changed = deepcopy(resource)
                spec = changed['tpu']['nodeSpec'][0]
                if field in ('nodeId', 'parent'):
                    spec[field] = 'foreign'
                elif field == 'acceleratorType':
                    spec['node'][field] = 'v4-8'
                elif field == 'labels':
                    spec['node'][field] = {}
                elif field == 'allocation':
                    del changed['spot']
                else:
                    changed['name'] = changed['name'].replace(target.zone, 'wrong-zone')
                with self.subTest(key=key, field=field), self.assertRaises(ValueError):
                    queue.validate_owned(changed, target)

    def test_target_label_is_optional_for_legacy_ownership(self):
        for key in ('3', '4'):
            target = registry.TARGETS[key]
            resource = self.resource(key, 'ACTIVE')
            self.assertNotIn('gfn-target', resource['tpu']['nodeSpec'][0]['node']['labels'])
            queue.validate_owned(resource, target)
            resource['tpu']['nodeSpec'][0]['node']['labels']['gfn-target'] = key
            queue.validate_owned(resource, target)
            del resource['tpu']['nodeSpec'][0]['node']['labels']['managed-by']
            with self.assertRaisesRegex(ValueError, 'not owned'):
                queue.validate_owned(resource, target)

    def test_numeric_id_in_watchdog_state_apply_and_preview_logs(self):
        for key in ('3', '4', '5', '6', '7', '8'):
            self.enable(key)
            name = registry.TARGETS[key].name
            prefix = f'Target {key} [{name}]: '
            with patch.object(queue, 'log') as log, patch.object(queue, 'call') as call:
                queue.reconcile(key, {name: self.resource(key, 'WAITING_FOR_RESOURCES')}, {}, self.args)
                log.assert_called_once_with(prefix + 'WAITING_FOR_RESOURCES')
                call.assert_not_called()
                self.args.apply = True
                queue.reconcile(key, {}, {}, self.args)
                self.assertTrue(log.call_args.args[0].startswith(prefix + 'APPLY '))
                self.assertIn('gfn-target=' + key, log.call_args.args[0])
                self.args.apply = False
                queue.reconcile(key, {name: self.resource(key, 'FAILED')}, {}, self.args)
                self.assertTrue(log.call_args.args[0].startswith(prefix + 'PREVIEW '))
            self.args.apply = True

    def test_all_spot_targets_preserve_pending_and_terminal_recovery(self):
        for key, target in registry.TARGETS.items():
            if key in {'1', '2'}:
                continue
            self.enable(key)
            with patch.object(queue, 'call') as call:
                for state in ('WAITING_FOR_RESOURCES', 'ACCEPTED', 'PROVISIONING', 'ACTIVE', 'DELETING', 'UNKNOWN'):
                    queue.reconcile(key, {target.name: self.resource(key, state)}, {}, self.args)
                call.assert_not_called()
                # Unrelated foreign resources are never traversed or deleted.
                requests = {target.name: self.resource(key, 'FAILED'), 'unregistered': {'state': {'state': 'FAILED'}}}
                queue.reconcile(key, requests, {}, self.args)
                command = call.call_args.args[0]
                self.assertIn('delete', command)
                self.assertIn('--zone=' + target.zone, command)
                self.assertNotIn('--force', command)
                self.assertEqual(call.call_count, 1)
                call.reset_mock()
                with self.assertRaises(ValueError):
                    queue.reconcile(key, requests, {target.name: {}}, self.args)
                call.assert_not_called()
                # Only a later inventory showing absence permits recreation.
                queue.reconcile(key, {}, {}, self.args)
                self.assertIn('create', call.call_args.args[0])

    def test_declared_quota_counts_all_slots_and_refuses_overflow(self):
        usage = registry.validate_registry()
        self.assertEqual(usage[(registry.PROJECT, 'us-central2-b', 'v4-32', 'spot')], 32)
        self.assertEqual(usage[(registry.PROJECT, 'us-central2-b', 'v4-32', 'on-demand')], 32)
        self.assertTrue(all(chips == 16 for pool, chips in usage.items() if pool[2] != 'v4-32'))
        overflow = dict(registry.TARGETS, **{'9': replace(registry.TARGETS['3'], name='extra')})
        with self.assertRaisesRegex(ValueError, 'exceed'):
            registry.validate_registry(overflow)
        with self.assertRaisesRegex(ValueError, 'chip accounting'):
            registry.validate_registry({'3': replace(registry.TARGETS['3'], chips=32)})

    def test_v5e_uses_documented_api_accelerator_and_logical_quota_pool(self):
        for key in ('7', '8'):
            target = registry.TARGETS[key]
            self.assertEqual(target.accelerator, 'v5e-16')
            self.assertEqual(target.api_accelerator, 'v5litepod-16')
            self.enable(key)
            self.assertIn('--accelerator-type=v5litepod-16',
                          queue.create_command(key, config.preparation(self.configuration, key)))
            queue.validate_owned(self.resource(key, 'ACTIVE'), target)
            self.assertIn((target.project, target.zone, 'v5e-16', 'spot'), registry.validate_registry())


if __name__ == '__main__':
    unittest.main()
