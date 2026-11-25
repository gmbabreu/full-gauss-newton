#!/usr/bin/env python3
"""Registry-driven Spot capacity watchdog; never runs training."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_config
from tpu_targets import TARGETS as REGISTRY

# Compatibility names for the original v4 helpers; routing uses REGISTRY below.
PROJECT = REGISTRY['3'].project
ZONE = REGISTRY['3'].zone
TARGETS = {key: target.name for key, target in REGISTRY.items() if target.allocation == 'spot'}
OWNER = 'gfn-spot-queue'
GCLOUD_EXECUTABLE = 'gcloud'


def log(message, *, file=None):
    for line in str(message).splitlines() or ['']:
        timestamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')
        print(f'[{timestamp}] {line}', file=file, flush=True)


def gcloud(group, action, *args, location=None):
    project, zone = location or (PROJECT, ZONE)
    return [GCLOUD_EXECUTABLE, 'compute', 'tpus', group, action, *args,
            '--project=' + project, '--zone=' + zone, '--quiet']


def call(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=180,
                            env=dict(os.environ, CLOUDSDK_CORE_DISABLE_PROMPTS='1'))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'gcloud failed')
    return result.stdout


def inventory(group, location=None):
    # A failed list is never interpreted as an absent resource.
    data = json.loads(call(gcloud(group, 'list', '--format=json', location=location)))
    if not isinstance(data, list):
        raise ValueError('Unexpected inventory response')
    return {x['name'].rsplit('/', 1)[-1]: x for x in data}


def collect_inventory(configuration, selected=None):
    # Fail the entire poll before ANY reconciliation if any required list fails.
    return {location: (inventory('queued-resources', location), inventory('tpu-vm', location))
            for location in queue_config.locations(configuration, selected)}


def validate_owned(resource, target):
    name = target.name
    specs = resource.get('tpu', {}).get('nodeSpec', [])
    if len(specs) != 1:
        raise ValueError(name + ': expected exactly one node specification')
    spec = specs[0]
    node = spec.get('node', {})
    if (spec.get('nodeId') != name or
            spec.get('parent') not in {f'projects/{target.project}/locations/{target.zone}',
                                       f'projects/{target.project_number}/locations/{target.zone}'} or
            resource.get('name') not in {None, name,
                f'projects/{target.project}/locations/{target.zone}/queuedResources/{name}',
                f'projects/{target.project_number}/locations/{target.zone}/queuedResources/{name}'} or
            node.get('acceleratorType') != target.api_accelerator or
            node.get('labels', {}).get('managed-by') != OWNER or
            'spot' not in resource or 'guaranteed' in resource or target.allocation != 'spot'):
        raise ValueError(name + ': existing request is not owned by this Spot watchdog')


def create_command(target_id, preparation):
    target = REGISTRY[target_id]
    name = target.name
    return gcloud('queued-resources', 'create', name, '--node-id=' + name,
                  '--accelerator-type=' + target.api_accelerator, '--spot', '--async',
                  '--runtime-version=' + preparation['runtime_version'],
                  '--service-account=' + preparation['service_account'],
                  '--scopes=https://www.googleapis.com/auth/cloud-platform',
                  '--labels=managed-by=' + OWNER + ',gfn-target=' + target_id,
                  '--metadata-from-file=startup-script=' + str(Path(preparation['startup_script']).resolve()),
                  location=(target.project, target.zone))


def reconcile(target, requests, nodes, args):
    if target in {'1', '2'} or REGISTRY[target].allocation != 'spot':
        raise ValueError('Fixed/on-demand targets cannot be watchdog managed')
    if not args.configuration['targets'][target]['managed']:
        return
    specification = REGISTRY[target]
    name = specification.name
    identity = f'Target {target} [{name}]'
    resource = requests.get(name)
    if resource is None:
        if name in nodes:
            raise ValueError(name + ': node exists without our queued request; inspect manually')
        command = create_command(target, queue_config.preparation(args.configuration, target))
    else:
        validate_owned(resource, specification)
        state = resource.get('state', {}).get('state', 'UNKNOWN')
        log(identity + ': ' + state)
        if state not in {'SUSPENDED', 'FAILED'}:
            # Includes pending/active/deleting and unknown future states. Never reset a wait.
            return
        if name in nodes:
            raise ValueError(name + ': terminal request still has a node; refusing deletion')
        # No --force: server must also confirm that there are no nodes to delete.
        command = gcloud('queued-resources', 'delete', name, '--async',
                         location=(specification.project, specification.zone))
    log(identity + ': ' + ('APPLY ' if args.apply else 'PREVIEW ') + shlex.join(command))
    if args.apply:
        call(command)
    # Recreation follows only after a later successful inventory confirms absence.


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets', nargs='+', choices=sorted(TARGETS),
                   help='Optional subset of managed targets; cannot enable a disabled target')
    p.add_argument('--config', required=True, help='Version 2 queue.json or legacy captured configuration')
    p.add_argument('--apply', action='store_true', help='Allow creates and terminal request cleanup')
    p.add_argument('--watch', action='store_true', help='Poll every 300 seconds')
    a = p.parse_args(argv)
    a.configuration = queue_config.load(a.config)
    # This lock is local to this user's home on this host, not cross-host coordination.
    state = Path.home() / '.local/state/gfn-spot'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            inventories = collect_inventory(a.configuration, a.targets)
            for target in queue_config.enabled(a.configuration, a.targets):
                specification = REGISTRY[target]
                requests, nodes = inventories[(specification.project, specification.zone)]
                reconcile(target, requests, nodes, a)
            if not a.watch:
                break
            time.sleep(300)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        log('Watchdog stopped; no further actions: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
