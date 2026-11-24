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
import tempfile

SOURCE = Path(__file__).resolve().parent
CODE = Path('/opt/gfn-spot')
STATE = Path('/var/lib/gfn-spot')
UNIT = Path('/etc/systemd/system/gfn-spot-watchdog.service')
SERVICE = UNIT.name


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


def copy_configuration(config, startup, destination):
    """Copy existing bytes, changing only the startup path in queue.json.

    Startup is explicit because the captured path usually refers to box248.
    """
    data = json.loads(Path(config).read_text())
    if set(data) != {'runtime_version', 'service_account', 'startup_script'}:
        raise ValueError('Invalid captured queue configuration')
    if any(not isinstance(v, str) or not v for v in data.values()) or '@' not in data['service_account']:
        raise ValueError('Invalid captured runtime/service account/startup path')
    body = Path(startup).read_bytes()
    if not body.startswith(b'#!/') or b'\x00' in body:
        raise ValueError('Expected existing prepared startup script')
    # Parse syntax, never execute the prepared training setup.
    subprocess.run(['bash', '-n', str(Path(startup).resolve())], check=True)
    destination = Path(destination).resolve()
    data['startup_script'] = str(destination / 'startup.sh')
    (destination / 'startup.sh').write_bytes(body)
    (destination / 'queue.json').write_text(json.dumps(data, indent=2) + '\n')
    for name in ('startup.sh', 'queue.json'):
        (destination / name).chmod(0o640)
    return data


def install(config, startup):
    if os.geteuid() != 0:
        raise ValueError('Run installer using sudo')
    require_inactive()
    # Validate inputs before changing any installed files.
    with tempfile.TemporaryDirectory() as temporary:
        copy_configuration(config, startup, temporary)
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
    for name in ('spot_queue.py', 'watchdog_service.py'):
        path = CODE / name
        shutil.copyfile(SOURCE / name, path)
        os.chown(path, 0, 0)
        path.chmod(0o644)
    copy_configuration(config, startup, STATE)
    for name in ('startup.sh', 'queue.json'):
        os.chown(STATE / name, 0, group)
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
    parser.add_argument('--startup-script', required=True)
    args = parser.parse_args()
    install(args.config, args.startup_script)


if __name__ == '__main__':
    main()
