# Spot capacity for targets 3 and 4

These helpers maintain **capacity**, not an experiment queue or training resume.
They never provision, stop, delete or update fixed targets 1 and 2. Keep using the
original checkout for those targets; use a separate Git worktree for this branch
because launchers require matching controller/worker branch, SHA and helper hashes.

| Target | Node and queued-resource name | Location | Type |
| --- | --- | --- | --- |
| 3 | gn-v4-32-spot-1 | fast-second-order / us-central2-b | v4-32 Spot |
| 4 | gn-v4-32-spot-2 | fast-second-order / us-central2-b | v4-32 Spot |

Verified user report on 2026-11-23: reference TPU1 uses tpu-ubuntu2204-base,
Python 3.10.12, JAX/jaxlib 0.6.2, libtpu 0.0.17, Flax 0.10.7, Optax 0.2.8,
and service account 598447672993-compute@developer.gserviceaccount.com.
Do not use the old scripts/setup_tpu_vm.sh pins for this recovery.

## One-time preparation on box248

From a clean, pushed checkout of this branch, capture TPU1's runtime and installed
training dependency closure. This is read-only on TPU1; it never initializes JAX,
installs packages or changes its checkout. Requires working gcloud SSH access.

```bash
python3 scripts/tpu/capture_setup.py --output-dir "$HOME/.config/gfn-spot/setup-20261123"
```

The snapshot stores exact package versions and a startup script outside Git.
No tokens, environment dump, service-account keys or W&B keys are captured.
The worker uses an isolated venv with access to the base image's system packages
and installs the captured pins with --no-deps, avoiding resolver-driven upgrades.
It checks every captured version and imports the training modules on CPU. Existing
package-metadata conflicts are logged, not silently repaired. This reproduces
versions, not an immutable image or wheel hashes; package availability, native
libraries and new runtime-image revisions still need a real-worker check.

The attached service account must already have permission to read secret
`wandb-api-key` and read the warmup checkpoint in `gfn-soo`; output/queue writes
need the corresponding bucket permissions. IAM is never automatically modified.
Startup fetches the secret at runtime, checks access as ga2740 and reports FAILED
if setup/import/access checks fail. No secret value goes into VM metadata.
The initial bootstrap clones the pinned branch/SHA into /full-gauss-newton on each
new worker. Rerunning startup refuses an existing checkout, protecting live work.

## Preview, then maintain target 3

```bash
python3 scripts/tpu/spot_queue.py \
  --config "$HOME/.config/gfn-spot/setup-20261123/queue.json" --targets 3

mkdir -p "$HOME/.local/state/gfn-spot"
nohup python3 -u scripts/tpu/spot_queue.py \
  --config "$HOME/.config/gfn-spot/setup-20261123/queue.json" \
  --targets 3 --apply --watch \
  > "$HOME/.local/state/gfn-spot/watchdog.log" 2>&1 &
echo $! > "$HOME/.local/state/gfn-spot/watchdog.pid"
tail -f "$HOME/.local/state/gfn-spot/watchdog.log"
```

Watchdog state, action and fatal-error messages include a New York timestamp
(EST in winter, EDT during daylight saving time), for example
`[2026-11-23 18:05:17 EST] gn-v4-32-spot-1: WAITING_FOR_RESOURCES`.
Timestamps are recorded when messages are emitted, so they remain accurate when
viewing historical logs. Update any installed copy of `spot_queue.py` and restart
that watchdog to load this change; `git pull` alone does not update installed
copies or running processes. Use `>>` instead of `>` above to retain earlier logs
when restarting. Existing log entries cannot be timestamped retroactively.

Preview reads cloud inventory but performs no mutation. Watch polls every 300s,
preserves pending/active/unknown states, and only deletes owned FAILED/SUSPENDED
requests when no node exists. Delete never uses --force. A later poll recreates an
absent request. Ownership checks require the exact node, project/zone, Spot tier,
v4-32 type and managed-by label. There is a local flock against duplicate watchdogs;
use exactly one controller (see the TPU1 migration below). API/auth/timeout errors stop with a visible log,
never masquerade as missing capacity. Review failed request details before retrying
if failures recur. Startup failure is not a capacity failure: an ACTIVE request
with FAILED setup is left in place for inspection.

**box248 credentials can still expire or require interactive reauthentication.**
nohup survives terminal disconnect, not host shutdown or authentication expiration.
An attached service account on workers does not authenticate box248. No claim of
permanent unattended recovery on box248 is made. The optional TPU1 service below
uses its existing attached identity; do not add JSON keys.

## TPU1 service deployment (pending operator deployment)

Deployment status as of 2026-11-24: implementation only, **not deployed**. Do not
mark deployed until the operator confirms installation, IAM review, handoff and
reboot validation. This service runs on **gn-v4-32, worker 0**, maintains targets
**3 and 4**, and polls every **300 seconds**. It depends on TPU1 remaining up;
TPU1 shutdown, replacement or failure also removes monitoring. It is not an
independent highly available controller.

The root-owned code lives in `/opt/gfn-spot`, independent of the active training
checkout `/full-gauss-newton`. State, lock, copied payload and fresh SDK config
directories live under `/var/lib/gfn-spot`; the dedicated `gfn-spot` user has no
login shell. No training environment is regenerated and no training dependencies
are installed on TPU1. Never execute the copied `startup.sh` on TPU1.

Each invocation creates a dedicated temporary gcloud configuration, discards
inherited environment settings (including credential files, access tokens,
impersonation, proxies and alternate account/config paths), and uses metadata
credentials. No user credential cache, interactive login or downloaded key is
used. The metadata email must be exactly
`598447672993-compute@developer.gserviceaccount.com`. Tokens are obtained and
refreshed by gcloud from the attached identity; this is not a promise that IAM,
OAuth scopes or host availability cannot change. The captured resource service
account in queue.json is preserved independently of controller authentication.

### Install while box248 is still monitoring

Run these **operator-only** commands on box248, from a separate clean checkout of
the reviewed PR commit, not the training checkout. Use the existing prepared
snapshot; do not run capture_setup.py or prepare_startup.py again. If the existing
watchdog uses a different snapshot, set `PREPARED` to that exact directory. Confirm
its startup bytes match the script referenced by the existing queue.json first.

```bash
PREPARED="$HOME/.config/gfn-spot/setup-20261123"
python3 -m json.tool "$PREPARED/queue.json"
sha256sum "$PREPARED/startup.sh"
tar -czf /tmp/gfn-watchdog-install.tgz -C scripts/tpu \
  spot_queue.py watchdog_service.py install_watchdog.py gfn-spot-watchdog.service
tar -czf /tmp/gfn-watchdog-prepared.tgz -C "$PREPARED" queue.json startup.sh
gcloud compute tpus tpu-vm scp /tmp/gfn-watchdog-install.tgz \
  /tmp/gfn-watchdog-prepared.tgz ga2740@gn-v4-32:/tmp/ \
  --project=fast-second-order --zone=us-central2-b --worker=0
gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32 \
  --project=fast-second-order --zone=us-central2-b --worker=0
```

In that worker-0 shell (the gcloud CLI, Python 3 with zoneinfo/tzdata, Bash and
systemd must already be available; stop and review if prerequisites are missing):

```bash
command -v gcloud python3 bash systemctl systemd-analyze
gcloud version
STAGE=$(mktemp -d /tmp/gfn-watchdog.XXXXXXXX)
mkdir "$STAGE/code" "$STAGE/prepared"
tar -xzf /tmp/gfn-watchdog-install.tgz -C "$STAGE/code"
tar -xzf /tmp/gfn-watchdog-prepared.tgz -C "$STAGE/prepared"
sha256sum "$STAGE/prepared/startup.sh"  # compare with box248
sudo /usr/bin/python3 -I "$STAGE/code/install_watchdog.py" \
  --config "$STAGE/prepared/queue.json" \
  --startup-script "$STAGE/prepared/startup.sh"
sudo systemctl is-active gfn-spot-watchdog.service   # expected inactive (exit 3)
sudo systemctl is-enabled gfn-spot-watchdog.service  # expected disabled (exit 1)
sudo stat -c '%U:%G %a %n' /opt/gfn-spot /opt/gfn-spot/*.py \
  /var/lib/gfn-spot /var/lib/gfn-spot/queue.json /var/lib/gfn-spot/startup.sh
sudo sha256sum /var/lib/gfn-spot/startup.sh
set -o pipefail
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py \
  --preflight | tee "$HOME/gfn-preflight-before.log" || { echo 'STOP: preflight failed'; exit 1; }
```

Expect code root:root 755/644; state gfn-spot:gfn-spot 700; configuration and
startup root:gfn-spot 640. Installation refuses an already enabled/active service,
validates startup syntax without executing it, validates readability as the
service user, reloads systemd, and leaves the unit stopped and disabled. On
updates, perform the same handoff and stop/disable before reinstalling; retain
the previous reviewed source and prepared snapshot for rollback.

Preflight uses the same entry point, environment isolation, state and identity as
the service. It verifies metadata identity and lists **both** queued resources and
TPU nodes, printing their full snapshots. It never reconciles, creates or deletes
anything. Run it while box248 remains active. Check snapshots include unchanged
targets 1–2 and the existing requests; natural cloud transitions may occur while
box248 is still operating. Any identity/auth/API error blocks handoff.

### Separate IAM verification before activation

Successful list calls prove only read access. They do **not** prove permission to
create or delete. Have the IAM operator review effective allow/deny policies,
conditions, OAuth scopes and relevant organization constraints for this attached
identity, without changing IAM or testing by creating/deleting capacity. Use
Policy Troubleshooter (or equivalent read-only policy evaluation) for
`tpu.nodes.create` on project `fast-second-order`, `tpu.nodes.delete` on the two
target resources, and `iam.serviceAccounts.actAs` on the **captured resource
service account**. Resolve conditional/unknown results before activation. Record
the reviewed principal, resources, permissions and results in the deployment
record. This policy review is separate from the automated list preflight and
does not promise a future API call will succeed.

The actual watchdog operations are queuedResources.list, nodes.list,
queuedResources.create and queuedResources.delete via the existing gcloud
commands, with explicit project/zone and asynchronous mutations. The API
references specify `tpu.nodes.create` for
[queuedResources.create](https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/create)
and `tpu.nodes.delete` for
[queuedResources.delete](https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/delete).
Review [service-account attachment permissions](https://docs.cloud.google.com/iam/docs/service-accounts-actas)
for the create command's captured `--service-account`. The service never changes
IAM. Keep the existing runtime and service account; no force deletion is used.

### Handoff: stop box248 first, then activate TPU1

The file lock is local to one user's home on one host. It does **not** coordinate
controllers on different hosts (or users). There must be no overlap between the
box248 watchdog and the TPU1 service.

On box248, inspect the recorded PID and all watchdog command lines, then check
automatic relaunch mechanisms before stopping anything:

```bash
OLD_PID=$(cat "$HOME/.local/state/gfn-spot/watchdog.pid")
ps -p "$OLD_PID" -o user,pid,ppid,lstart,args
pgrep -af '[s]pot_queue.py'
systemctl --user list-unit-files
systemctl --user list-timers --all
sudo systemctl list-unit-files
sudo systemctl list-timers --all
crontab -l
sudo crontab -l
tmux list-sessions
```

Also inspect any supervisor/container wrapper or site scheduler used for this
process, and `/etc/cron.d`, `/etc/crontab` and user login startup files if relevant.
Save the old exact command, config path, working directory and relaunch settings
for rollback. Verify `OLD_PID` is this watchdog, not a trainer or reused PID. If
managed by systemd, inspect and disable only the identified unit (`OLD_UNIT` must
be its real name):

```bash
# Run only for an identified user service; for a system service use sudo systemctl.
systemctl --user cat "$OLD_UNIT"
systemctl --user disable --now "$OLD_UNIT"
# If a dedicated timer launches it, disable that identified timer too.
# For cron, use crontab -e and remove only this watchdog's relaunch entry.
```

For the documented nohup process, after verifying the PID and disabling any
configured relaunch:

```bash
kill -TERM "$OLD_PID"
timeout 30 tail --pid="$OLD_PID" -f /dev/null
if kill -0 "$OLD_PID" 2>/dev/null; then echo 'STOP: old PID still exists'; false; fi
pgrep -af '[s]pot_queue.py'  # expected no watchdog; inspect any output
```

If a manager already stopped it, skip `kill` but still verify exit. Do not proceed
unless the watchdog has exited and no mechanism can restart it. Never use broad
`pkill python`, stop trainers, or cancel queued requests.

Back on TPU1 worker 0, rerun read-only preflight, compare resource snapshots, then
activate only after the exit and permission checks above:

```bash
set -o pipefail
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py \
  --preflight | tee "$HOME/gfn-preflight-handoff.log" || { echo 'STOP: preflight failed'; exit 1; }
sudo systemctl enable --now gfn-spot-watchdog.service
sudo systemctl status --no-pager gfn-spot-watchdog.service
sudo systemctl is-enabled gfn-spot-watchdog.service
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p SubState -p NRestarts
sudo journalctl -u gfn-spot-watchdog.service -n 100 --no-pager
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py \
  --preflight | tee "$HOME/gfn-preflight-after.log" || { echo 'STOP: inventory check failed'; exit 1; }
sudo journalctl -u gfn-spot-watchdog.service -f
```

The initial journal preflight snapshot precedes any reconciliation. Confirm
existing pending/active/deleting/unknown states remain untouched and targets 1–2
are unchanged. Owned FAILED/SUSPENDED requests with no node may be deleted;
recreation only occurs after a later successful inventory shows absence. An
existing node prevents deletion. API failures stop the process and systemd retries
after 60 seconds; failures never count as absent capacity. Inspect repeated errors
and stop the service if operator review is required.

Enabled units start after reboot. Actual reboot survival requires operator
validation at a maintenance window that permits TPU1 interruption: after a
separately approved reboot, run the same status commands and
`sudo journalctl -b -u gfn-spot-watchdog.service --no-pager`, and compare a new
`--preflight` snapshot. Do not reboot a training host just to validate this PR.

### Rollback without overlapping controllers

On TPU1 worker 0:

```bash
sudo systemctl disable --now gfn-spot-watchdog.service
sudo systemctl is-enabled gfn-spot-watchdog.service  # disabled
sudo systemctl is-active gfn-spot-watchdog.service   # inactive
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p ControlGroup -p SubState
pgrep -af '/opt/gfn-spot/[w]atchdog_service.py'  # expected no process
```

Require MainPID=0, inactive state and no remaining service process before
restarting box248. If TPU1 is unreachable, do not assume it has stopped: establish
that it cannot run before restarting another controller. Leave all requests and
nodes untouched. On box248, restore its saved command/config and authenticate
there as needed. For the original nohup deployment maintaining both targets:

```bash
# In the original controller checkout, after verifying no watchdog is running:
pgrep -af '[s]pot_queue.py'
nohup python3 -u scripts/tpu/spot_queue.py \
  --config "$HOME/.config/gfn-spot/setup-20261123/queue.json" \
  --targets 3 4 --apply --watch \
  >> "$HOME/.local/state/gfn-spot/watchdog.log" 2>&1 &
echo $! > "$HOME/.local/state/gfn-spot/watchdog.pid"
ps -p "$(cat "$HOME/.local/state/gfn-spot/watchdog.pid")" -o pid,args
tail -n 50 "$HOME/.local/state/gfn-spot/watchdog.log"
```

Use the previously saved target selection if it differs. Restore only the
identified relaunch mechanism if desired; do not also launch nohup when restoring
a systemd-managed watchdog. The TPU1 unit must remain disabled throughout.

## Check startup on all four workers

```bash
gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32-spot-1 \
  --project=fast-second-order --zone=us-central2-b --worker=all \
  --command='cat /tmp/gfn_setup_status; sudo tail -n 30 /var/log/gfn-setup.log'
bash scripts/tpu.sh 3 status
```

After every worker reports DONE, check distributed topology simultaneously on all
workers (only on this idle new target):

```bash
gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32-spot-1 \
  --project=fast-second-order --zone=us-central2-b --worker=all \
  --command="bash -lc 'source /etc/profile.d/gfn_env.sh && timeout 180 python3 -c \"import jax; jax.distributed.initialize(); print(jax.process_index(), jax.process_count(), jax.device_count(), jax.local_device_count()); assert (jax.process_count(), jax.device_count(), jax.local_device_count()) == (4,16,4); jax.distributed.shutdown()\"'"
```

Expect process indices 0..3, each reporting 4 processes, 16 global devices and 4
local devices. This is a topology check, not a training test. No training is
launched by the watchdog. GN parameter warm-start is not exact state/data resume.
Preempted runs need explicit operator decisions and available checkpoints.

## Expand to both targets / stop

In the controller shell, identify the watchdog before stopping it:

```bash
ps -p "$(cat "$HOME/.local/state/gfn-spot/watchdog.pid")" -o pid,args
```

After verifying it is this Python watchdog, stop that process:

```bash
kill -TERM "$(cat "$HOME/.local/state/gfn-spot/watchdog.pid")"
```

Stopping the watchdog leaves existing TPU VMs and pending requests running. It
does not cancel capacity charges. To maintain both, relaunch the same nohup command
with `--targets 3 4` after the old process exits. Do not delete a node while the
watchdog is running unless recreation is intended. Use target 4's exact name for
its startup/topology checks.

## Local validation

```bash
python3 -m unittest discover -s scripts/tests -v
bash -n scripts/tpu/setup_vm.sh
bash -n scripts/tpu.sh
systemd-analyze verify scripts/tpu/gfn-spot-watchdog.service
git diff --check
```

Tests mock gcloud and exercise pending/terminal states, orphan/foreign resource
refusal, authentication failure, nonmutating preview, payload quoting, isolated
credential environments, metadata identity/header checks, configuration copying,
permissions, installation without activation, both targets and 300-second polling. Cloud
provisioning, package downloads, IAM, actual startup and TPU execution require
operator validation. The scripts do not claim these local tests cover them.

Implementation validation on 2026-11-24: **66 CPU tests passed** in local Ubuntu
WSL (15.505s); both Bash syntax checks and systemd unit verification exited 0;
`git diff --check` passed. The initial Windows working-tree run had 62/63 tests
pass, with the existing shell entrypoint test failing on CRLF line endings;
setup_vm.sh syntax checking also hit existing CRLF. Final validation used a
temporary Linux snapshot of Git's stored files plus the changed files, leaving
unrelated checkout files untouched. The installer was tested with mocked system
commands, not installed into the local system. No cloud commands, live IAM
verification, TPU deployment, service activation or reboot test were performed.

API references:
- https://docs.cloud.google.com/tpu/docs/queued-resources
- https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/queued-resources/create
- https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/queued-resources/delete
