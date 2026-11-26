"""CPU-only configuration, registry and capacity state-machine checks."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
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
import queue_config as config
import spot_queue as queue
import prepare_startup as startup


class SpotQueueTests(unittest.TestCase):
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
        self.name = registry.TARGETS['3'].name

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
        self.assertEqual(set(queue.TARGETS), {'3', '4', '5', '6', '7', '8'})
        self.assertNotIn('gn-v4-32', queue.TARGETS.values())
        self.assertNotIn('gn-v4-32-2', queue.TARGETS.values())
        self.assertEqual(self.configuration['preparations']['v4'], dict(self.legacy, accelerator_type='v4-32'))
        self.assertEqual(self.startup.read_bytes(), self.body)
        for key in ('1', '2', '5', '6', '7', '8'):
            self.assertFalse(self.configuration['targets'][key]['managed'])

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
        self.assertEqual(changed['preparations']['v4'], dict(self.legacy, accelerator_type='v4-32'))
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

    def test_either_legacy_request_or_node_blocks_both_renamed_targets(self):
        for key in ('3', '4'):
            for old_name in ('gn-v4-32-spot-1', 'gn-v4-32-spot-2'):
                for kind in ('request', 'node'):
                    for state in ('FAILED', 'SUSPENDED', 'WAITING_FOR_RESOURCES', 'ACTIVE', 'DELETING', 'UNKNOWN'):
                        old = {old_name: {'state': {'state': state}}}
                        requests, nodes = (old, {}) if kind == 'request' else ({}, old)
                        for new_state in (None, 'FAILED', 'SUSPENDED', 'ACTIVE'):
                            current = dict(requests)
                            if new_state:
                                current[registry.TARGETS[key].name] = self.resource(key, new_state)
                            with self.subTest(key=key, old_name=old_name, kind=kind,
                                              state=state, new_state=new_state), \
                                    patch.object(queue, 'call') as call:
                                with self.assertRaisesRegex(ValueError, 'naming conflict.*' + old_name) as caught:
                                    queue.reconcile(key, current, nodes, self.args)
                                self.assertIn('resolve legacy', str(caught.exception))
                                call.assert_not_called()

    def test_legacy_guard_reports_all_requests_and_nodes_without_ownership_assumptions(self):
        old = {name: {} for name in registry.LEGACY_SPOT_NAMES.values()}
        with self.assertRaises(ValueError) as caught:
            queue.refuse_legacy_spot_conflicts(old, old)
        message = str(caught.exception)
        for name in old:
            self.assertIn('queued request ' + name, message)
            self.assertIn('node ' + name, message)
        self.assertIn('fast-second-order/us-central2-b', message)
        self.assertIn('service stopped', message)

    def test_legacy_conflict_stops_poll_before_any_location_or_selected_target_reconciles(self):
        self.enable('5')
        path = self.root / 'queue.json'
        path.write_text(json.dumps(self.configuration))
        for selection in ([], ['--targets', '3'], ['--targets', '4']):
            def inventory(group, location):
                if group == 'tpu-vm' and location == (registry.PROJECT, 'us-central2-b'):
                    return {'gn-v4-32-spot-1': {}}
                return {}
            with patch.object(queue.Path, 'home', return_value=self.root), \
                    patch.object(queue, 'inventory', side_effect=inventory), \
                    patch.object(queue, 'reconcile') as reconcile:
                with self.assertRaisesRegex(ValueError, 'naming conflict'):
                    queue.main(['--config', str(path), '--apply', *selection])
                reconcile.assert_not_called()

    def test_unmanaged_renamed_targets_do_not_block_independent_locations(self):
        for key in ('3', '4'):
            self.configuration['targets'][key]['managed'] = False
            with patch.object(queue, 'call') as call:
                queue.reconcile(key, {'gn-v4-32-spot-1': {}}, {}, self.args)
                call.assert_not_called()
        self.enable('5')
        with patch.object(queue, 'inventory', return_value={}) as inventory:
            queue.collect_inventory(self.configuration)
        self.assertEqual({call.args[1] for call in inventory.call_args_list},
                         {(registry.PROJECT, 'us-east1-d')})

    def test_per_target_create_and_ownership_checks(self):
        for key, target in registry.TARGETS.items():
            if key in {'1', '2'}:
                continue
            self.enable(key)
            resource = self.resource(key, 'FAILED')
            labels = resource['tpu']['nodeSpec'][0]['node']['labels']
            self.assertNotIn('gfn-target', labels)
            queue.validate_owned(resource, target)
            labels['gfn-target'] = key
            queue.validate_owned(resource, target)
            if key in ('7', '8'):
                self.assertEqual(target.accelerator, 'v5e-16')
                self.assertEqual(target.api_accelerator, 'v5litepod-16')
            preparation = self.configuration['targets'][key]['preparation']
            command = queue.create_command(key, self.configuration['preparations'][preparation])
            for flag in ('--zone=' + target.zone, '--project=' + target.project,
                         '--accelerator-type=' + target.api_accelerator, '--node-id=' + target.name, '--spot',
                         '--labels=managed-by=' + queue.OWNER + ',gfn-target=' + key, '--async'):
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
                with self.subTest(key=key, field=field), patch.object(queue, 'call') as call:
                    with self.assertRaises(ValueError):
                        queue.reconcile(key, {target.name: changed}, {}, self.args)
                    call.assert_not_called()

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
                for state in ('WAITING_FOR_RESOURCES', 'ACCEPTED', 'PROVISIONING', 'ACTIVE',
                              'SUSPENDING', 'CREATING', 'DELETING', 'UNKNOWN', 'FUTURE_STATE'):
                    with self.subTest(key=key, state=state):
                        queue.reconcile(key, {target.name: self.resource(key, state)}, {}, self.args)
                call.assert_not_called()
                # Unrelated foreign resources are never traversed or deleted.
                for state in ('FAILED', 'SUSPENDED'):
                    with self.subTest(key=key, state=state):
                        requests = {target.name: self.resource(key, state),
                                    'unregistered': {'state': {'state': 'FAILED'}}}
                        queue.reconcile(key, requests, {}, self.args)
                        command = call.call_args.args[0]
                        self.assertIn('delete', command)
                        self.assertIn('--zone=' + target.zone, command)
                        self.assertNotIn('--force', command)
                        self.assertNotIn('create', command)
                        call.assert_called_once()
                        call.reset_mock()
                        with self.assertRaises(ValueError):
                            queue.reconcile(key, requests, {target.name: {}}, self.args)
                        call.assert_not_called()
                # Only a later inventory showing absence permits recreation.
                queue.reconcile(key, {}, {}, self.args)
                self.assertIn('create', call.call_args.args[0])

    def preempted_node(self, key):
        target = registry.TARGETS[key]
        parent = f'projects/{target.project_number}/locations/{target.zone}'
        return dict(name=parent + '/nodes/' + target.name,
                    queuedResource=parent + '/queuedResources/' + target.name,
                    acceleratorType=target.api_accelerator, state='PREEMPTED',
                    labels={'managed-by': queue.OWNER, 'gfn-target': key},
                    schedulingConfig={'spot': True})

    def test_preempted_recovery_across_successive_inventories(self):
        for key in ('3', '4'):
            name = registry.TARGETS[key].name
            for state in ('FAILED', 'SUSPENDED'):
                requests = {name: self.resource(key, state)}
                node = self.preempted_node(key)
                with self.subTest(key=key, state=state), patch.object(queue, 'call') as call:
                    queue.reconcile(key, requests, {name: node}, self.args)
                    command = call.call_args.args[0]
                    self.assertEqual(command[3:6], ['tpu-vm', 'delete', name])
                    self.assertIn('--async', command)
                    self.assertNotIn('--force', command)
                    call.assert_called_once()
                    call.reset_mock()
                    node['state'] = 'DELETING'
                    queue.reconcile(key, requests, {name: node}, self.args)
                    call.assert_not_called()
                    queue.reconcile(key, requests, {}, self.args)
                    self.assertEqual(call.call_args.args[0][3:6], ['queued-resources', 'delete', name])
                    call.assert_called_once()
                    call.reset_mock()
                    queue.reconcile(key, {}, {}, self.args)
                    self.assertEqual(call.call_args.args[0][3:6], ['queued-resources', 'create', name])
                    call.assert_called_once()

    def test_preempted_node_preview_and_refusals(self):
        requests = {self.name: self.resource('3', 'SUSPENDED')}
        node = self.preempted_node('3')
        self.args.apply = False
        with patch.object(queue, 'call') as call, patch.object(queue, 'log') as log:
            queue.reconcile('3', requests, {self.name: node}, self.args)
            call.assert_not_called()
            self.assertIn('PREVIEW gcloud compute tpus tpu-vm delete', log.call_args.args[0])
        self.args.apply = True
        changes = [dict(state=state) for state in ('READY', 'CREATING', 'UNKNOWN')]
        changes += [dict(name=node['name'].replace('us-central2-b', 'wrong-zone')),
                    dict(queuedResource=node['queuedResource'] + '-foreign'),
                    dict(acceleratorType='v4-8'), dict(labels={}),
                    dict(labels={'managed-by': queue.OWNER, 'gfn-target': '4'}),
                    dict(schedulingConfig={})]
        for change in changes:
            with self.subTest(change=change), patch.object(queue, 'call') as call:
                with self.assertRaises(ValueError):
                    queue.reconcile('3', requests, {self.name: dict(node, **change)}, self.args)
                call.assert_not_called()
        # An active or pending request never authorizes node cleanup.
        for state in ('ACTIVE', 'WAITING_FOR_RESOURCES', 'PROVISIONING'):
            with patch.object(queue, 'call') as call:
                queue.reconcile('3', {self.name: self.resource('3', state)},
                                {self.name: node}, self.args)
                call.assert_not_called()

    def test_registry_validates_identity_and_accelerator_without_static_grants(self):
        expanded = dict(registry.TARGETS, **{'9': replace(registry.TARGETS['3'], name='extra', zone='other-zone')})
        registry.validate_registry(expanded)
        with self.assertRaisesRegex(ValueError, 'Duplicate target identity'):
            registry.validate_registry(dict(registry.TARGETS, **{'9': registry.TARGETS['3']}))
        with self.assertRaisesRegex(ValueError, 'Unsupported accelerator'):
            registry.validate_registry({'3': replace(registry.TARGETS['3'], accelerator='unreviewed')})
        for key in ('1', '2'):
            with self.assertRaisesRegex(ValueError, 'cannot be watchdog managed'):
                registry.validate_registry({key: replace(registry.TARGETS[key], managed=True)})

    @patch.object(queue, 'call')
    def test_orphan_node_refused(self, call):
        with self.assertRaises(ValueError):
            queue.reconcile('3', {}, {self.name:{}}, self.args)
        call.assert_not_called()

    @patch.object(queue, 'call')
    def test_preview_no_mutations(self, call):
        self.args.apply=False
        queue.reconcile('3', {}, {}, self.args)
        queue.reconcile('3', {self.name:self.resource('3', 'FAILED')}, {}, self.args)
        call.assert_not_called()

    @patch.object(queue, 'call', side_effect=RuntimeError('reauth required'))
    def test_auth_failure_not_absence(self, call):
        with self.assertRaisesRegex(RuntimeError, 'reauth'):
            queue.inventory('queued-resources')


class StartupTests(unittest.TestCase):
    def test_unreviewed_requirements_rejected(self):
        for requirements in ['', 'jax', 'jax @ https://example.invalid/file', '-e /local/repo']:
            with self.assertRaises(ValueError):
                startup.render(requirements, 'main', 'a'*40, '3.10', 'wandb-api-key')

    def test_package_install_wait_option_and_failure_propagation(self):
        body = Path(startup.__file__).with_name('setup_vm.sh').read_text()
        packages = body.split('export DEBIAN_FRONTEND=noninteractive\n', 1)[1].split('actual_python=', 1)[0]
        # Execute the real package setup fragment with a shell stub, never APT.
        for failure, expected_calls in (('', 2), ('update', 1), ('install', 2)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                calls = Path(directory) / 'calls'
                stub = """
set -euo pipefail
apt-get() {
    printf '%s\\n' "$*" >> CALLS
    case " $* " in
        *" FAILURE "*) return 100 ;;
    esac
}
""".replace('CALLS', "'" + str(calls) + "'").replace('FAILURE', failure or 'never-match')
                result = subprocess.run(['bash'], input=stub + packages + '\necho READY',
                                        text=True, capture_output=True)
                commands = calls.read_text().splitlines()
                self.assertEqual(len(commands), expected_calls)
                self.assertEqual(result.returncode, 100 if failure else 0)
                self.assertEqual('READY' in result.stdout, not failure)
                if len(commands) == 2:
                    self.assertEqual(commands[1],
                                     '-o DPkg::Lock::Timeout=3600 install -y git tmux python3-venv')

    def test_payload_quoting_and_syntax(self):
        requirements='jax==0.4.34\njaxlib==0.4.34\nlibtpu==0.0.1\nflax==0.8.3\noptax==0.2.2\n'
        result=startup.render(requirements, 'codex/spot-tpu-recovery', 'a'*40, '3.10', "quote'$(false)")
        subprocess.run(['bash', '-n'], input=result, text=True, check=True)
        self.assertNotIn('WANDB_API_KEY=quote', result)


if __name__ == '__main__':
    unittest.main()
