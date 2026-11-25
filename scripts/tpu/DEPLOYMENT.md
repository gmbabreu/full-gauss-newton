# Watchdog installation, update and rollback

These commands are for the operator; deployment of this branch remains pending.
Use TPU1 (`gn-v4-32`), worker 0. The service uses its attached account
`598447672993-compute@developer.gserviceaccount.com`, a fresh isolated gcloud
configuration and an allowlisted environment. Cached user credentials, inherited
credential/token/impersonation settings, interactive login and downloaded keys
are excluded. Ordinary gcloud installations on the fixed system PATH take priority
over direct `/snap/google-cloud-cli/current/bin/gcloud`.

The independent `/opt/gfn-spot` copy runs as unprivileged `gfn-spot`, with writable
state/config storage in `/var/lib/gfn-spot`, journal logging, a 60-second failure
restart and the existing systemd sandbox. Installation leaves it stopped and
disabled. Do not run setup/capture, install training dependencies or modify
`/full-gauss-newton` on TPU1. Bash, Python 3.10+ with zoneinfo/tzdata, systemd and
gcloud must already be available.

## Prepare the candidate

From a clean separate workstation checkout of the reviewed revision:

```bash
git status --short
git rev-parse HEAD
tar -czf /tmp/gfn-watchdog-update.tgz -C scripts \
  tpu/spot_queue.py tpu/watchdog_service.py tpu/install_watchdog.py \
  tpu/queue_config.py tpu/gfn-spot-watchdog.service tpu_targets.py
sha256sum /tmp/gfn-watchdog-update.tgz
gcloud compute tpus tpu-vm scp /tmp/gfn-watchdog-update.tgz \
  ga2740@gn-v4-32:/tmp/ --project=fast-second-order --zone=us-central2-b --worker=0
gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32 \
  --project=fast-second-order --zone=us-central2-b --worker=0
```

In that worker-0 shell, retain these variables for the procedure:

```bash
umask 077
sha256sum /tmp/gfn-watchdog-update.tgz # compare with workstation
STAGE=$(mktemp -d /tmp/gfn-watchdog-update.XXXXXXXX)
mkdir "$STAGE/code"
tar -xzf /tmp/gfn-watchdog-update.tgz -C "$STAGE/code"
SOURCE_CONFIG=/var/lib/gfn-spot/queue.json
# Initial installation: stage the existing prepared queue.json and payloads here
# instead; preserve their runtime/account and use actual local payload paths.
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" \
  --config "$SOURCE_CONFIG" --output "$STAGE/queue-v2.json"
NEXT_CONFIG="$STAGE/queue-v2.json"
sudo cat "$NEXT_CONFIG"
```

Legacy input enables only 3–4 and leaves 5–8 disabled. For relocated inputs use
the installer's `--payload-dir DIR` (unique source basenames), or
`--startup-script SOURCE` for legacy input only. Enabled payloads must exist.
Perform staging, local validation and read-only inventory checks while the old
controller remains active; do not install over an active service.

## Review permissions and readiness

Before activation, verify live accelerator and regional IP quota/headroom,
including other project allocations; the [supplied quotas](README.md#configuration)
are not enforced or queried by this code. Check metadata identity, OAuth scopes
and effective IAM allow/deny policies/conditions in every managed location.
List access proves only read permission. Separately evaluate the actual
[queuedResources.create](https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/create)
(`tpu.nodes.create`) and
[queuedResources.delete](https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/delete)
(`tpu.nodes.delete`) operations and `iam.serviceAccounts.actAs` on the captured
resource account, using read-only policy evaluation rather than test mutations.

For each newly enabled generation/location, review supported runtime, exact
dependency closure and secret-free startup payload separately. Verify the resource
account's data/checkpoint/secret reads and output writes. After allocation,
confirm setup completion on every discovered worker and separately validate
environment/topology before manual training. ACTIVE is not training readiness.
No new-generation runtime, package closure or mesh is assumed by this repository.

## Record processes and verify exit

Before any update, rollback or host handoff, save the old controller's exact
command, config, working directory and relaunch settings. A local file lock
does not coordinate hosts or different users. Inventory systemd user/system
units/timers, cron, supervisor/container wrappers and any old box248 watchdog;
disable only the mechanisms actually identified as belonging to this controller.
Never stop training or signal a broad Python/gcloud process group.

On TPU1 worker 0, record the service parent and descendants **before** stopping
anything. Use the recorded PID file instead only for a separately verified
nohup controller. Confirm the PID's start time, owner and command before signals:

```bash
OLD_PID=$(systemctl show gfn-spot-watchdog.service -p MainPID --value)
sudo ps -p "$OLD_PID" -o user,pid,ppid,pgid,sid,lstart,args
HANDOFF=$(mktemp -d "$HOME/gfn-handoff.XXXXXXXX")
sudo pstree -alp "$OLD_PID" | tee "$HANDOFF/tree-before.txt"
sudo ps -eo user,pid,ppid,pgid,sid,lstart,args > "$HANDOFF/processes-before.txt"
# Only after verifying OLD_PID is this controller, freeze it to prevent new polls.
sudo kill -STOP "$OLD_PID"
sudo pstree -alp "$OLD_PID" | tee "$HANDOFF/tree-paused.txt"
sudo ps -eo user,pid,ppid,pgid,sid,lstart,args > "$HANDOFF/processes-paused.txt"
```

The refreshed tree captures children launched between the first snapshot and
pause. Record **all descendant processes**, including gcloud's Python entry
point, wrappers, and their children; do not mistake pstree thread IDs in braces
for process PIDs. Record start times and commands and continue tracking any
further descendants. Reparenting after parent exit does not mean a cloud command
finished. If observation is incomplete or ownership is uncertain, block handoff.

```bash
OLD_DESCENDANT_PIDS=()  # fill with the verified process PIDs from BOTH saved trees
OLD_ALL_PIDS=("$OLD_PID" "${OLD_DESCENDANT_PIDS[@]}")
OLD_PID_LIST=$(IFS=,; echo "${OLD_ALL_PIDS[*]}")
printf '%s\n' "${OLD_ALL_PIDS[@]}" > "$HANDOFF/pids.txt"
sudo ps -p "$OLD_PID_LIST" -o user,pid,ppid,pgid,sid,lstart,args \
  | tee "$HANDOFF/identified-processes.txt"
sudo systemctl disable --now gfn-spot-watchdog.service
WAIT_ARGS=()
for pid in "${OLD_ALL_PIDS[@]}"; do WAIT_ARGS+=(--pid="$pid"); done
timeout 240 tail "${WAIT_ARGS[@]}" -f /dev/null
sudo ps -p "$OLD_PID_LIST" -o user,pid,ppid,pgid,sid,lstart,args
# Expected: header only, no recorded process remains (ps exits 1).
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p SubState -p ControlGroup
sudo systemctl status --no-pager gfn-spot-watchdog.service
pgrep -af '/opt/gfn-spot/[w]atchdog_service.py'
```

Require inactive/disabled state, MainPID=0, no remaining recorded process or
later descendant, and an empty service cgroup. A parent PID exiting alone is
insufficient. A timeout is a blocked handoff, not permission to proceed. If a
recorded PID was reused, inspect its saved identity; never kill it solely because
the number matches. Prefer letting commands finish; investigate stuck commands
individually using verified identities, not broad kills. Do not kill trainers.

For a verified **nohup** parent, disable its actual configured relaunch first.
After the same snapshots/pause, replace the systemctl stop above with
`kill -TERM "$OLD_PID"` followed by `kill -CONT "$OLD_PID"` (sudo if needed).
TERM must be pending before resuming a paused controller. Perform the same wait
and descendant checks. For a manager, stop only its identified unit; also disable
its identified dedicated timer if any. For cron, edit only that watchdog entry.
If aborting before any termination was requested, explicitly decide whether to
resume the verified old controller; do not leave it paused unintentionally or
start another controller while it can resume.

## Resolve in-flight mutations

Review saved cloud-command arguments and the journal at shutdown:

```bash
sudo journalctl -u gfn-spot-watchdog.service -n 200 --no-pager \
  | tee "$HANDOFF/journal-shutdown.txt"
```

If create/delete was in flight or its outcome is uncertain, local command exit
is still insufficient: asynchronous API work may continue server-side. After
every old local process exits, obtain **fresh** queued-resource and node
inventories for each affected registry location. Use these read-only commands for each affected location:

```bash
# Set to the affected registry location; repeat for each affected location.
: "${PROJECT:?}" "${ZONE:?}"
gcloud compute tpus queued-resources list --project="$PROJECT" --zone="$ZONE" --format=json
gcloud compute tpus tpu-vm list --project="$PROJECT" --zone="$ZONE" --format=json
```

Correlate the affected request/node with the original API operation and inspect
its status/errors (Cloud Console/audit record or the read-only Cloud TPU
operations API). Record resource/operation identities and resolved outcome in
`$HANDOFF`. Confirm accepted creation, confirmed deletion with no node, or an
understood completed failure; a stable accepted pending capacity request is not
permission to reset its wait. A still-running/ambiguous operation, failed reads,
one absent snapshot, or an `APPLY` log line is insufficient. Repeat read-only
checks until resolved, then take a new preflight snapshot before activation.
Do not create/delete requests merely to settle handoff uncertainty.

## Resolve the Spot resource name migration with the service stopped

After the handoff checks, keep all controllers/relaunch mechanisms stopped.
Numeric 3–4 now route to `gn-v4-32-spot-3`/`gn-v4-32-spot-4`; 1–2 are unchanged.
Either legacy `gn-v4-32-spot-1` or `gn-v4-32-spot-2` request or node, in any state
and regardless of labels, blocks both renamed targets in preflight and the
entire poll before mutations. Inspect old and new names with the inventory
commands above. The watchdog never deletes/adopts legacy resources or chooses
which capacity to retain when old and new names coexist.

Choose retention or retirement explicitly. Keep needed workloads/capacity until
their planned retirement. If retaining either legacy resource, leave the service
disabled or disable **both 3 and 4** in the candidate before managing independent
locations:

```bash
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" --config "$NEXT_CONFIG" \
  --disable 3 --output "$STAGE/queue-disable3.json"
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" --config "$STAGE/queue-disable3.json" \
  --disable 4 --output "$STAGE/queue-disable34.json"
NEXT_CONFIG="$STAGE/queue-disable34.json"
```

Retirement is a separate operator decision, not part of installation. Never force
delete a request with a node. Before managing renamed 3–4, require both legacy
names absent from both successful fresh inventories and completed operation
outcomes. Relabeling does not resolve the guard; retiring a pending request loses
its queue position. Inspect legacy resources by explicit name, not numeric 3–4.

## Back up and install the stopped copy

For updates, save the existing code, unit, config and payloads after handoff.
For first installation, retain the prepared inputs and recorded old host command
instead of trying to archive nonexistent installed files.

```bash
BACKUP="$HOME/gfn-watchdog-backups/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP"
# Existing installations only:
sudo tar -czpf "$BACKUP/code.tgz" -C /opt/gfn-spot .
sudo cp -p /etc/systemd/system/gfn-spot-watchdog.service "$BACKUP/unit.service"
sudo bash -c 'cd /var/lib/gfn-spot && tar -czpf "$1/state.tgz" queue.json startup*.sh' _ "$BACKUP"
sudo chmod 600 "$BACKUP/code.tgz" "$BACKUP/unit.service" "$BACKUP/state.tgz"
```

Use the same installation command for initial deployment and updates:

```bash
sudo /usr/bin/python3 -I "$STAGE/code/tpu/install_watchdog.py" --config "$NEXT_CONFIG"
sudo systemctl is-active gfn-spot-watchdog.service   # inactive
sudo systemctl is-enabled gfn-spot-watchdog.service  # disabled
sudo stat -c '%U:%G %a %n' /opt/gfn-spot /opt/gfn-spot/*.py /var/lib/gfn-spot
sudo bash -c 'stat -c "%U:%G %a %n" /var/lib/gfn-spot/queue.json /var/lib/gfn-spot/startup*.sh'
```

The installer refuses an active/enabled unit. Code is root:root 755/644, state
gfn-spot:gfn-spot 700, config/payloads root:gfn-spot 640. It checks startup syntax
and service-user readability without executing startup. Compare runtime/account
and payload hashes with the source/backup; only the installed absolute path
changes (legacy payload becomes `startup-v4.sh`).

## Read-only preflight and separate activation

Preflight uses the service's isolated authentication environment and reads both
resource kinds in every managed location. Before stopping the old controller,
the read-only inventory commands above can establish a comparison baseline;
after installation run the new preflight while the unit remains disabled:

```bash
set -o pipefail
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py \
  --preflight | tee "$HOME/gfn-preflight-handoff.log" || { echo 'STOP: preflight failed'; exit 1; }
```

Check identity, selected executable, managed IDs/locations and resource states.
Expect 3–4 only for an unchanged legacy config; compare fresh states with the
handoff baseline. Resolve errors/conflicts before proceeding. Only after all
checks pass, activate separately:

```bash
sudo systemctl enable --now gfn-spot-watchdog.service
sudo systemctl status --no-pager gfn-spot-watchdog.service
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p SubState -p NRestarts
sudo journalctl -u gfn-spot-watchdog.service -n 100 --no-pager
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py --preflight
```

Verify existing pending/active requests were not reset and targets 1–2 remain
unchanged; account for legitimate asynchronous state transitions. Enabled units
start after reboot, but reboot survival requires operator validation in a planned
maintenance window, followed by status and `journalctl -b`. Monitoring depends
on TPU1 availability.

## Enable or disable one additional target

Targets 5–8 stay disabled until explicitly reviewed. Select one row from the
[target table](README.md), supply that location's reviewed preparation, and use
this parameterized example (not a loop):

```bash
TARGET_ID=5
PREP_NAME=v6e-us
ACCELERATOR=v6e-16
: "${RUNTIME:?reviewed runtime}" "${RESOURCE_SA:?}" "${STARTUP:?absolute prepared path}"
NEXT_CONFIG="$STAGE/queue-enable${TARGET_ID}.json"
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" \
  --config /var/lib/gfn-spot/queue.json --output "$NEXT_CONFIG" \
  --enable "$TARGET_ID" --preparation "$PREP_NAME" --accelerator-type "$ACCELERATOR" \
  --runtime-version "$RUNTIME" --service-account "$RESOURCE_SA" --startup-script "$STARTUP"
```

Repeat the same readiness, handoff, backup, installation, preflight and activation
procedure with this candidate. To disable instead, use `--disable ID --output NEW`.
Disabling does not remove resources or stop charges. Activation can create absent
managed capacity but never launches or resumes experiments.

## Rollback

Apply the same process/operation handoff checks to the new controller before
restoring anything. If TPU1 is unreachable, establish that it cannot run before
starting another controller. Older code has no reverse-name guard: before
restoring old-name 3–4 management, both new `spot-3`/`spot-4` requests and nodes
must be absent from fresh inventory. Otherwise leave old management disabled or
use a compatible config with both 3–4 unmanaged; disposition is operator-controlled.

After verified handoff, restore the selected stopped backup:

```bash
sudo tar -xzpf "$BACKUP/code.tgz" -C /opt/gfn-spot
sudo tar -xzpf "$BACKUP/state.tgz" -C /var/lib/gfn-spot
sudo install -o root -g root -m 644 "$BACKUP/unit.service" /etc/systemd/system/gfn-spot-watchdog.service
sudo systemd-analyze verify /etc/systemd/system/gfn-spot-watchdog.service
sudo systemctl daemon-reload
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py --preflight
```

Review restored settings/states before the separate activation block. Rolling
back to v4-only leaves any 5–8 capacity unmanaged. To restore box248 instead,
leave TPU1 disabled and restart only the recorded old command/manager/config
after the same handoff gates; do not add nohup alongside a manager.

## Troubleshooting and local validation

- **Identity/executable failure:** check attached identity and fixed-path gcloud
  installation. Do not weaken the sandbox or pass user credential overrides.
- **Inventory failure:** inspect IAM/scopes/API errors; the entire poll fails
  before reconciliation and systemd retries after 60 seconds. Errors never mean
  absent capacity.
- **Naming conflict:** use the stopped-service migration above.
- **Repeated terminal failures:** inspect operation/runtime/setup errors and
  ownership. An existing node prevents deletion; force is never used.
- **Helper mismatch:** bootstrap the reviewed matching helpers, including the
  registry, on clean idle workers; do not reset active training checkouts.

```bash
python3 -m unittest discover -s scripts/tests -v
bash -n scripts/tpu/setup_vm.sh
bash -n scripts/tpu.sh
systemd-analyze verify scripts/tpu/gfn-spot-watchdog.service
git diff --check
```

Local CPU/mock checks do not validate live IAM, quota, provisioning, runtime,
training readiness or reboot behavior. Those require operator validation.
