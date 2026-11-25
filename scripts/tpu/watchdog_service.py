#!/usr/bin/env python3
"""Isolated attached-identity entry point; preflight never reconciles capacity."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import urllib.request

# -I ignores caller Python settings; import only the root-owned installed sibling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spot_queue as queue

IDENTITY = '598447672993-compute@developer.gserviceaccount.com'
STATE = Path('/var/lib/gfn-spot')
CONFIG = STATE / 'queue.json'
SERVICE_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
SNAP_GCLOUD = '/snap/google-cloud-cli/current/bin/gcloud'


def find_gcloud():
    # Prefer ordinary installations using only the service's fixed search path.
    # Invoke Snap's SDK entry point directly, avoiding the /snap/bin launcher
    # and its additional confinement machinery inside the service sandbox.
    executable = shutil.which('gcloud', path=SERVICE_PATH)
    if executable:
        return executable
    if Path(SNAP_GCLOUD).is_file() and os.access(SNAP_GCLOUD, os.X_OK):
        return SNAP_GCLOUD
    raise ValueError('No executable gcloud in service PATH or at ' + SNAP_GCLOUD)


@contextmanager
def isolated_environment(state=STATE):
    # Fresh SDK storage on every start: even credentials cached by an earlier
    # process cannot be selected. Never inherit CLOUDSDK_*, GOOGLE_*, BOTO_*,
    # token, proxy, credential-file, impersonation or Python path overrides.
    with tempfile.TemporaryDirectory(prefix='gcloud-', dir=state) as directory:
        configurations = Path(directory) / 'configurations'
        configurations.mkdir()
        (configurations / 'config_watchdog').write_text('[core]\ndisable_prompts = true\n')
        (Path(directory) / 'active_config').write_text('watchdog')
        environment = {
            'PATH': SERVICE_PATH,
            'HOME': str(state), 'LANG': 'C.UTF-8',
            'CLOUDSDK_CONFIG': directory,
            'CLOUDSDK_ACTIVE_CONFIG_NAME': 'watchdog',
            'CLOUDSDK_CORE_DISABLE_PROMPTS': '1',
            'CLOUDSDK_CORE_PROJECT': queue.PROJECT,
            'CLOUDSDK_COMPUTE_ZONE': queue.ZONE,
        }
        previous = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(environment)
            yield environment
        finally:
            os.environ.clear()
            os.environ.update(previous)


def metadata_identity():
    request = urllib.request.Request(
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email',
        headers={'Metadata-Flavor': 'Google'})
    # Do not use inherited proxies, even when called outside the service.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        if response.headers.get('Metadata-Flavor') != 'Google':
            raise ValueError('Missing Google metadata response header')
        return response.read(1024).decode().strip()


def validate_files(config=CONFIG):
    config = Path(config)
    data = queue.queue_config.load(config)
    paths = {config}
    for key in queue.queue_config.enabled(data):
        name = data['targets'][key]['preparation']
        startup = Path(data['preparations'][name]['startup_script'])
        if startup.parent != config.parent or startup.name not in {'startup.sh', f'startup-{name}.sh'}:
            raise ValueError('Startup script must be the installed absolute path')
        paths.add(startup)
    for path in paths:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or mode & 0o022:
            raise ValueError(f'Expected regular file without group/other write: {path}')
        with path.open('rb') as handle:
            if not handle.read(1):
                raise ValueError(f'Empty file: {path}')
    return data


def preflight(configuration):
    actual = metadata_identity()
    if actual != IDENTITY:
        raise ValueError(f'Attached identity mismatch: {actual!r}; expected {IDENTITY}')
    queue.GCLOUD_EXECUTABLE = find_gcloud()
    queue.log('Attached identity: ' + actual)
    queue.log('Using gcloud executable: ' + queue.GCLOUD_EXECUTABLE)
    inventories = queue.collect_inventory(configuration)
    for key in queue.queue_config.enabled(configuration):
        target = queue.REGISTRY[key]
        queue.log(f'Managed target {key}: {target.name} ({target.project}/{target.zone})')
    # Log only selected status fields, never node descriptions or nested payloads.
    for (project, zone), (requests, nodes) in inventories.items():
        queue.log(f'Inventory {project}/{zone}: {len(requests)} queued requests, {len(nodes)} nodes')
        for kind, resources in (('queued request', requests), ('node', nodes)):
            for name, resource in sorted(resources.items()):
                queue.log(f'{kind} {name}: ' + resource_status(resource))
    queue.log('Read-only preflight passed; create/delete permissions are NOT verified')


def resource_status(resource):
    """Bounded, single-line status summary from an explicit field allowlist."""
    state = resource.get('state', 'UNKNOWN')
    reasons = []
    if isinstance(state, dict):
        error = state.get('failedData', {}).get('error', {})
        reasons.extend((error.get('code'), error.get('message'),
                        state.get('suspendedData', {}).get('reason')))
        state = state.get('state', 'UNKNOWN')
    reasons.extend((resource.get('health'), resource.get('healthDescription')))
    fields = [f'state={json.dumps(str(state)[:80])}']
    fields.extend('reason=' + json.dumps(str(reason)[:240]) for reason in reasons if reason is not None)
    return ' '.join(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    configuration = validate_files()
    with isolated_environment():
        preflight(configuration)
        if not args.preflight:
            queue.main(['--config', str(CONFIG), '--apply', '--watch'])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        queue.log('Service stopped: ' + str(error), file=__import__('sys').stderr)
        raise SystemExit(1)
