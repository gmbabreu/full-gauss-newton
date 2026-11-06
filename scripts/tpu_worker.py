#!/usr/bin/env python3
"""Worker RPC and tmux supervisor. Only invoked by tpu_control.py."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time


HELPER_PATHS = tuple('scripts/' + name for name in
                     ('tpu.sh', 'tpu_control.py', 'tpu_worker.py', 'run_experiment.py'))


def command(argv, *, check=True):
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f'{argv[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def fingerprint(repo, *, include_presets=True):
    paths = [repo / path for path in HELPER_PATHS]
    if include_presets:
        paths += sorted((repo / 'scripts' / 'presets').glob('*.json'))
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def process_table():
    """Read Linux process identities. Never expose process environments."""
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
            processes.append(dict(pid=int(entry.name), pgrp=int(stat[2]),
                                  start=stat[19], argv=argv))
        except (OSError, ValueError, IndexError):
            continue
    return processes


def training_processes():
    modules = {'EasyLM.models.llama.llama_train', 'EasyLM.models.llama.llama_train_gn',
               'EasyLM.models.llama.llama_train_gn_layerwise'}
    names = {'sweep_launcher.py', 'llama_train.py', 'llama_train_gn.py',
             'llama_train_gn_layerwise.py'}
    return [p for p in process_table() if any(a in modules or Path(a).name in names
                                             for a in p['argv'])]


def owned_process(pid, start):
    return next((p for p in process_table() if p['pid'] == pid and p['start'] == start), None)


def sessions():
    result = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'],
                            text=True, capture_output=True)
    if result.returncode and not any(
            s in result.stderr.lower() for s in ('no server running', 'connection refused', 'no such file')):
        raise RuntimeError('Cannot inspect tmux: ' + result.stderr.strip())
    return [s for s in result.stdout.splitlines() if s.startswith('gfn-')]


class Worker:
    def __init__(self, payload):
        self.p = payload
        self.repo = Path(payload['repo']).resolve()
        self.state = Path(payload['state']).expanduser().resolve()
        if self.state == self.repo or self.repo in self.state.parents:
            raise ValueError('Run records must be outside the Git working tree')
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = self.state / 'operation.lock'
        os.chdir(self.repo)

    def require_lock(self):
        if not self.lock.is_dir() or (self.lock / 'token').read_text() != self.p['token']:
            raise RuntimeError('Operation reservation missing or belongs to another controller')

    def acquire(self):
        try:
            self.lock.mkdir()
        except FileExistsError:
            raise RuntimeError(f'Worker reserved by another operation: {self.lock}. '
                               'If its controller was lost, inspect managed runs before manually removing the reservation.')
        (self.lock / 'token').write_text(self.p['token'])
        return {'reserved': True}

    def release(self):
        if not self.lock.is_dir() or (self.lock / 'token').read_text() != self.p['token']:
            return {'released': False}
        self.require_lock()
        (self.lock / 'token').unlink()
        self.lock.rmdir()
        return {'released': True}

    def records(self):
        return [json.loads(p.read_text()) for p in sorted(self.state.glob('*/manifest.json'))]

    def active_records(self):
        return [m for m in self.records() if m['state'] not in ('exited', 'cancelled') and
                (self.group(m) or owned_process(m.get('supervisor_pid'), m.get('supervisor_start')))]

    def group(self, manifest):
        pgid = manifest.get('pgid')
        return [p for p in process_table() if p['pgrp'] == pgid]

    def report(self):
        processes = training_processes()
        managed = self.active_records()
        managed_groups = {m.get('pgid') for m in managed}
        return dict(branch=command(['git', 'branch', '--show-current']),
                    sha=command(['git', 'rev-parse', 'HEAD']),
                    dirty=command(['git', 'status', '--porcelain', '--untracked-files=all']),
                    tmux=sessions(), training_processes=processes, managed=managed,
                    unmanaged=[p for p in processes if p['pgrp'] not in managed_groups])

    def idle_clean(self, report):
        problems = []
        if report['dirty']:
            problems.append('dirty working tree; commit or move local edits first')
        if report['tmux'] or report['training_processes'] or report['managed']:
            problems.append('training or managed tmux active; inspect status and stop the owning run first')
        if problems:
            raise RuntimeError('; '.join(problems))

    def preflight(self, *, for_update=False):
        self.require_lock()
        if not Path(self.p['environment']).is_file():
            raise RuntimeError('Missing environment file: ' + self.p['environment'])
        if not shutil.which('tmux') or not shutil.which('python'):
            raise RuntimeError('The sourced environment must provide tmux, python and python3')
        report = self.report()
        self.idle_clean(report)
        actual = fingerprint(self.repo, include_presets=not for_update)
        expected = self.p.get('fingerprint')
        if for_update:
            expected = {path: (expected or {}).get(path) for path in HELPER_PATHS}
            if expected != actual:
                raise RuntimeError('Controller/worker executable helpers differ; manually bootstrap '
                                   'the same helper revision on every worker before updating')
        elif expected != actual:
            raise RuntimeError('Controller/worker tooling or presets differ; sync and commit the tooling on every worker')
        if not for_update and self.p.get('expected_sha') and (report['sha'], report['branch']) != (
                self.p['expected_sha'], self.p['expected_branch']):
            raise RuntimeError('Controller/worker branch or SHA differs; explicitly sync the chosen branch on every worker')
        return report

    def preflight_update(self):
        return self.preflight(for_update=True)

    def status(self):
        report = self.report()
        try:
            if shutil.which('tpu-info'):
                query = ['tpu-info']
            else:
                query = [sys.executable, '-c',
                    'import json; from libtpu.sdk import monitoring; '
                    'print(json.dumps({n:monitoring.get_metric(n).data() for n in '
                    '("duty_cycle_pct","tensorcore_util","hbm_capacity_usage","hbm_capacity_total")}))']
            result = subprocess.run(query, capture_output=True, text=True, timeout=15)
            report['tpu_usage'] = (result.stdout.strip() if result.returncode == 0
                                   else 'unavailable: install tpu-info or check libtpu monitoring in the environment')
        except subprocess.TimeoutExpired:
            report['tpu_usage'] = 'TPU usage query timed out'
        return report

    def run_dir(self):
        run_id = self.p['plan']['run_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
            raise ValueError('Invalid run ID')
        return self.state / run_id

    def launch(self):
        self.preflight()  # Reservation still held; catch external edits/processes.
        directory = self.run_dir()
        directory.mkdir()  # Identities are never reused.
        atomic_json(directory / 'plan.json', self.p['plan'])
        manifest = dict(run_id=self.p['plan']['run_id'], preset=self.p['plan']['preset'],
                        explicit_overrides=self.p['plan']['explicit_overrides'],
                        resolved_arguments=self.p['plan']['flags'], git_sha=self.p['expected_sha'],
                        branch=self.p['expected_branch'], target=self.p['target'], worker=self.p['worker'],
                        exit_status=None, state='prepared', token=self.p['token'],
                        session='gfn-' + self.p['plan']['run_id'])
        atomic_json(directory / 'manifest.json', manifest)
        runner = [sys.executable, str(self.repo / 'scripts' / 'tpu_worker.py'), 'supervise', encode(self.p)]
        # tmux accepts a single shell command; quote every argv element, never eval.
        shell = self.supervisor_command(runner)
        manifest['pane_command'] = shell
        atomic_json(directory / 'manifest.json', manifest)
        command(['tmux', 'new-session', '-d', '-s', manifest['session'], shell])
        (self.state / 'latest').write_text(self.p['plan']['run_id'])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            record = json.loads((directory / 'manifest.json').read_text())
            if record['state'] == 'exited':
                raise RuntimeError(f'Process exited during launch: {record["exit_status"]}; log: {directory}/worker.log')
            if record['state'] == 'started' and self.group(record):
                return dict(message='process started; training initialization is NOT verified',
                            run_id=record['run_id'], log=str(directory / 'worker.log'))
            time.sleep(0.1)
        raise RuntimeError('No process-start acknowledgement; inspect ' + str(directory))

    def supervisor_command(self, runner=None):
        if runner is None:
            runner = [sys.executable, str(self.repo / 'scripts' / 'tpu_worker.py'), 'supervise', encode(self.p)]
        setup = ('set -eo pipefail; source ' + shlex.quote(self.p['environment'])
                 + '; cd -- ' + shlex.quote(str(self.repo)) + '; exec ' + shlex.join(runner))
        return shlex.join(['bash', '-lc', setup])

    def supervise(self):
        directory = self.run_dir()
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        identity = next(p for p in process_table() if p['pid'] == os.getpid())
        manifest.update(supervisor_pid=identity['pid'], supervisor_start=identity['start'])
        atomic_json(manifest_path, manifest)
        env = dict(os.environ)
        for key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID'):
            env.pop(key, None)
        env.update(PYTHONUNBUFFERED='1', PYTHONPATH=str(self.repo) + os.pathsep + env.get('PYTHONPATH', ''),
                   GFN_MANAGED_RUN_TOKEN=manifest['token'])
        if self.p['worker'] == 0:
            env.pop('WANDB_MODE', None)
        else:
            env['WANDB_MODE'] = 'disabled'
        if self.p['plan']['route'] == 'sweep':
            env.update(SLURM_ARRAY_TASK_ID='1', SLURM_ARRAY_JOB_ID=manifest['run_id'])
        child = None
        interrupted = False

        def terminate(signum, frame):
            nonlocal interrupted
            interrupted = True
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        signal.signal(signal.SIGHUP, terminate)
        try:
            with (directory / 'worker.log').open('a', buffering=1) as log:
                if interrupted or (directory / 'cancelled').exists():
                    raise RuntimeError('Launch cancelled before process start')
                child = subprocess.Popen([sys.executable, str(self.repo / 'scripts' / 'run_experiment.py'),
                                          '--execute', str(directory / 'plan.json')],
                                         stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                identity = next((p for p in process_table() if p['pid'] == child.pid), None)
                manifest.update(pgid=child.pid, process_start=identity['start'] if identity else None,
                                state='started')
                atomic_json(manifest_path, manifest)
                if interrupted or (directory / 'cancelled').exists():
                    os.killpg(child.pid, signal.SIGTERM)
                result = child.wait()
                # Keep ownership supervision until descendants leave the group.
                if self.group(manifest):
                    self.terminate_group(manifest)
        except Exception as error:
            result = 1
            with (directory / 'worker.log').open('a') as log:
                log.write(str(error) + '\n')
            if child:
                self.terminate_group(manifest)
        manifest.update(state='exited', exit_status=result)
        atomic_json(manifest_path, manifest)
        return {'exit_status': result}

    def terminate_group(self, manifest):
        def verified_group():
            members = self.group(manifest)
            for p in members:
                try:
                    # Verify inherited ownership without ever printing environment data.
                    data = Path(f'/proc/{p["pid"]}/environ').read_bytes().split(b'\0')
                except FileNotFoundError:
                    continue
                if ('GFN_MANAGED_RUN_TOKEN=' + manifest['token']).encode() not in data:
                    raise RuntimeError('Process group ownership changed; refusing to signal it')
            return members
        for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            if not verified_group():
                return
            try:
                os.killpg(manifest['pgid'], sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self.group(manifest):
                    return
                time.sleep(0.1)
        if self.group(manifest):
            raise RuntimeError('Managed process group did not terminate')

    def stop(self):
        self.require_lock()
        run_id = self.p.get('stop_run_id')
        records = [m for m in self.records() if m['state'] not in ('exited', 'cancelled')]
        if run_id:
            records = [m for m in records if m['run_id'] == run_id]
        stopped = []
        for manifest in records:
            directory = self.state / manifest['run_id']
            (directory / 'cancelled').touch()
            # Only signal the recorded supervisor if its PID and birth time match.
            # Its handler cancels startup or gracefully forwards to the child group.
            supervisor = owned_process(manifest.get('supervisor_pid'), manifest.get('supervisor_start'))
            if supervisor:
                try:
                    os.kill(supervisor['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self.terminate_group(manifest)
            deadline = time.monotonic() + 5
            while supervisor and time.monotonic() < deadline:
                supervisor = owned_process(manifest.get('supervisor_pid'), manifest.get('supervisor_start'))
                if supervisor:
                    time.sleep(0.1)
            if supervisor:
                raise RuntimeError('Supervisor still active; inspect ' + manifest['session'])
            # A failed launch can leave a pane before its supervisor acknowledged.
            # Verify the exact original pane command (including this launch token)
            # before closing that session; cancellation prevents a late child start.
            if manifest['session'] in sessions():
                pane_command = command(['tmux', 'display-message', '-p', '-t',
                                        manifest['session'], '#{pane_start_command}'])
                if pane_command != manifest.get('pane_command'):
                    raise RuntimeError('Managed tmux ownership cannot be verified; inspect ' + manifest['session'])
                command(['tmux', 'kill-session', '-t', manifest['session']])
            # Reread: startup could have published the group during cancellation.
            latest = json.loads((directory / 'manifest.json').read_text())
            self.terminate_group(latest)
            deadline = time.monotonic() + 5
            while owned_process(latest.get('supervisor_pid'), latest.get('supervisor_start')):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Supervisor still active after closing managed session')
                time.sleep(0.1)
                latest = json.loads((directory / 'manifest.json').read_text())
                self.terminate_group(latest)
            if manifest['session'] in sessions() or self.group(latest):
                raise RuntimeError('Managed session/group still active')
            if latest['state'] not in ('exited', 'cancelled'):
                latest.update(state='cancelled', exit_status=143)
                atomic_json(directory / 'manifest.json', latest)
            stopped.append(manifest['run_id'])
        report = self.report()
        return dict(stopped=stopped, unmanaged=report['unmanaged'], remaining_tmux=report['tmux'])

    def logs(self):
        latest = (self.state / 'latest').read_text().strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]+', latest):
            raise RuntimeError('Invalid latest-run record')
        path = self.state / latest / 'worker.log'
        os.execvp('tail', ['tail', '-n', '100', '-F', str(path)])

    def fetch(self):
        self.preflight(for_update=True)
        branch = self.p['branch']
        command(['git', 'check-ref-format', '--branch', branch])
        command(['git', 'fetch', 'origin', f'refs/heads/{branch}:refs/remotes/origin/{branch}'])
        target = command(['git', 'rev-parse', '--verify', f'refs/remotes/origin/{branch}^{{commit}}'])
        self.verify_target(target)
        local = subprocess.run(['git', 'rev-parse', '--verify', f'refs/heads/{branch}'],
                               text=True, capture_output=True)
        if local.returncode == 0:
            command(['git', 'merge-base', '--is-ancestor', local.stdout.strip(), target])
        return {'target_sha': target}

    def verify_target(self, target):
        # The updater must retain compatible executable helpers across the
        # switch. Presets are data delivered by the update, not a prerequisite.
        for path in HELPER_PATHS:
            digest = self.p['fingerprint'].get(path)
            try:
                content = subprocess.check_output(['git', 'show', target + ':' + path], stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as error:
                raise RuntimeError('Target branch lacks ' + path + '; deliver the tooling commit there before switching') from error
            if hashlib.sha256(content).hexdigest() != digest:
                raise RuntimeError('Target branch executable helpers differ; manually bootstrap '
                                   'the same helper revision before switching')

    def update(self):
        self.preflight(for_update=True)
        branch, sha = self.p['branch'], self.p['target_sha']
        fetched = command(['git', 'rev-parse', f'refs/remotes/origin/{branch}'])
        if fetched != sha:
            raise RuntimeError('Fetched target changed; rerun update')
        self.verify_target(sha)
        local = subprocess.run(['git', 'show-ref', '--verify', '--quiet', f'refs/heads/{branch}'])
        if local.returncode == 0:
            command(['git', 'merge-base', '--is-ancestor', f'refs/heads/{branch}', sha])
            command(['git', 'switch', branch])
        else:
            command(['git', 'switch', '-c', branch, '--track', f'origin/{branch}'])
        command(['git', 'merge', '--ff-only', sha])
        return self.report()


def encode(value):
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def main():
    action, encoded = sys.argv[1:]
    payload = json.loads(base64.urlsafe_b64decode(encoded))
    try:
        worker = Worker(payload)
        if action not in {'acquire', 'release', 'preflight', 'preflight_update', 'status', 'launch', 'stop', 'logs',
                          'fetch', 'update', 'supervise'}:
            raise ValueError('Unknown worker action')
        result = getattr(worker, action)()
        print('GFN_RESULT=' + json.dumps(dict(ok=True, result=result)), flush=True)
    except Exception as error:
        print('GFN_RESULT=' + json.dumps(dict(ok=False, error=str(error))), flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
