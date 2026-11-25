#!/usr/bin/env python3
"""Install an independent watchdog copy, stopped and disabled. No cloud calls."""
import argparse
import grp
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import queue_config

CODE = Path('/opt/gfn-spot')
STATE = Path('/var/lib/gfn-spot')
UNIT = Path('/etc/systemd/system/gfn-spot-watchdog.service')
SERVICE = UNIT.name
BUNDLE = {name: SOURCE / name for name in ('spot_queue.py', 'watchdog_service.py', 'queue_config.py')}
BUNDLE['tpu_targets.py'] = SOURCE.parent / 'tpu_targets.py'


def run(*command):
    return subprocess.run(command, check=True)


def require_inactive():
    for command in ('is-active', 'is-enabled'):
        result = subprocess.run(['systemctl', command, SERVICE], capture_output=True, text=True)
        if result.returncode == 0:
            raise ValueError('Service must already be stopped and disabled before installation')
        if result.returncode not in (1, 3, 4) or result.stdout.strip() not in (
                'inactive', 'failed', 'disabled', 'not-found', ''):
            raise ValueError('Cannot confirm service is stopped and disabled: ' + result.stdout + result.stderr)


def copy_configuration(config, startup, destination, payload_dir=None):
    """Migrate legacy settings and copy exact prepared bytes; never execute setup."""
    data = json.loads(Path(config).read_text())
    if startup is not None:
        if set(data) != queue_config.LEGACY_KEYS:
            raise ValueError('--startup-script override is only for legacy configurations')
        data['startup_script'] = str(Path(startup).resolve())
    data = queue_config.normalize(data, check_files=False)
    destination = Path(destination).resolve()
    used = {data['targets'][key]['preparation'] for key in queue_config.enabled(data)}
    sources = [Path(prep['startup_script']).name for prep in data['preparations'].values()]
    if payload_dir and len(sources) != len(set(sources)):
        raise ValueError('Payload basenames must be unique when using --payload-dir')
    for name, prep in data['preparations'].items():
        source = Path(prep['startup_script'])
        if payload_dir:
            source = Path(payload_dir) / source.name
        output = destination / f'startup-{name}.sh'
        if source.is_file():
            body = source.read_bytes()
            if not body.startswith(b'#!/') or b'\x00' in body:
                raise ValueError('Expected existing prepared startup script')
            subprocess.run(['bash', '-n', str(source.resolve())], check=True)
            output.write_bytes(body)
            output.chmod(0o640)
        elif name in used:
            raise ValueError('Missing enabled preparation payload: ' + str(source))
        prep['startup_script'] = str(output)
    queue_config.normalize(data)  # Entire installed config validated before activation.
    (destination / 'queue.json').write_text(json.dumps(data, indent=2) + '\n')
    (destination / 'queue.json').chmod(0o640)
    return data


def install(config, startup=None, payload_dir=None):
    if os.geteuid() != 0:
        raise ValueError('Run installer using sudo')
    require_inactive()
    # Validate inputs before changing any installed files.
    with tempfile.TemporaryDirectory() as temporary:
        copy_configuration(config, startup, temporary, payload_dir)
    try:
        account = pwd.getpwnam('gfn-spot')
    except KeyError:
        run('useradd', '--system', '--user-group', '--home-dir', str(STATE),
            '--no-create-home', '--shell', '/usr/sbin/nologin', 'gfn-spot')
        account = pwd.getpwnam('gfn-spot')
    if account.pw_uid == 0 or account.pw_dir != str(STATE) or account.pw_shell != '/usr/sbin/nologin':
        raise ValueError('Existing gfn-spot account must be dedicated with expected home and nologin shell')
    group = grp.getgrnam('gfn-spot').gr_gid
    if account.pw_gid != group:
        raise ValueError('Unexpected gfn-spot primary group')
    CODE.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    os.chown(CODE, 0, 0)
    CODE.chmod(0o755)
    os.chown(STATE, account.pw_uid, group)
    STATE.chmod(0o700)
    for name, source in BUNDLE.items():
        path = CODE / name
        shutil.copyfile(source, path)
        os.chown(path, 0, 0)
        path.chmod(0o644)
    installed = copy_configuration(config, startup, STATE, payload_dir)
    for path in [STATE / 'queue.json'] + [Path(prep['startup_script'])
                                        for prep in installed['preparations'].values()]:
        if path.is_file():
            os.chown(path, 0, group)
    shutil.copyfile(SOURCE / SERVICE, UNIT)
    os.chown(UNIT, 0, 0)
    UNIT.chmod(0o644)
    run('systemd-analyze', 'verify', str(UNIT))
    run('systemctl', 'daemon-reload')
    run('systemctl', 'disable', SERVICE)
    require_inactive()
    run('runuser', '-u', 'gfn-spot', '--', '/usr/bin/python3', '-I', '-c',
        'import sys; sys.path.insert(0, "/opt/gfn-spot"); '
        'import watchdog_service; watchdog_service.validate_files()')
    print('Installed stopped and disabled. Run the documented preflight and handoff before activation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--startup-script', help='Override legacy payload source (copied byte for byte)')
    parser.add_argument('--payload-dir', help='Read preparation payloads by unique basename from this directory')
    args = parser.parse_args()
    install(args.config, args.startup_script, args.payload_dir)


if __name__ == '__main__':
    main()
