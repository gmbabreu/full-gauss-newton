# TPU control and capacity watchdog

## Targets and routine control

The [registry](../tpu_targets.py) routes numeric IDs in project `fast-second-order`:

| Target | Name | Accelerator | Allocation | Zone | Default watchdog management |
| --- | --- | --- | --- | --- | --- |
| 1 | gn-v4-32 | v4-32 | On-demand | us-central2-b | Never |
| 2 | gn-v4-32-2 | v4-32 | On-demand | us-central2-b | Never |
| 3 | gn-v4-32-spot-3 | v4-32 | Spot | us-central2-b | Enabled after migration guard passes |
| 4 | gn-v4-32-spot-4 | v4-32 | Spot | us-central2-b | Enabled after migration guard passes |
| 5 | gn-v6e-16-spot-5 | v6e-16 | Spot | us-east1-d | Disabled |
| 6 | gn-v6e-16-spot-6 | v6e-16 | Spot | europe-west4-a | Disabled |
| 7 | gn-v5e-16-spot-7 | v5e-16 | Spot | us-central1-a | Disabled |
| 8 | gn-v5e-16-spot-8 | v5e-16 | Spot | europe-west4-b | Disabled |

Only 3–4 are managed by default; 5–8 require explicit preparation and enablement.
The deployed v4-only copy uses legacy Spot names; this branch's update is
**pending operator deployment**. Updating Git does not update `/opt/gfn-spot`.

From the controller repository root:

```bash
TARGET_ID=3
bash scripts/tpu.sh "$TARGET_ID" status
bash scripts/tpu.sh "$TARGET_ID" ssh 0
bash scripts/tpu.sh "$TARGET_ID" logs 0
bash scripts/tpu.sh "$TARGET_ID" run gn_adam --dry-run "${GN_ARGS[@]}"
bash scripts/tpu.sh "$TARGET_ID" stop
bash scripts/tpu.sh "$TARGET_ID" pull
bash scripts/tpu.sh "$TARGET_ID" switch main
```

Numeric IDs appear in controller/watchdog logs. Workers come from `networkEndpoints`,
including on manually allocated disabled targets. Supply reviewed topology in
`GN_ARGS`; see [experiment usage](../../docs/tpu_workflow.md) for matching helpers
and coordinated launches. Manual overrides `GFN_TPU_NAME`, `GFN_PROJECT`,
`GFN_ZONE`, `GFN_SSH_USER`, `GFN_REMOTE_REPO`, `GFN_REMOTE_ENV`, `GFN_STATE_DIR`
retain their meanings; unset identity overrides for normal routing. The watchdog
ignores these overrides and never launches or resumes experiments.

## Configuration and protections

Legacy three-key configs (`runtime_version`, `service_account`, `startup_script`)
normalize to version 2, enabling only 3–4. Version 2 has `preparations` and
`targets` maps: each preparation specifies `accelerator_type` plus the three
legacy fields; each target has boolean `managed` and, when enabled, `preparation`.
Omitted target entries use registry defaults. Enabled preparations require an
existing absolute payload path and matching accelerator; unused preparations
are structurally checked but need no existing file. The installer preserves
captured runtime/account and exact payload bytes, rewriting only local paths.

`queue_config.py --config INPUT --output NEW` validates/migrates without cloud
calls or overwriting output. To enable one additional target, choose its ID and
logical accelerator from the table and supply reviewed settings:

```bash
# On TPU1 worker 0, after staging source below; generated files stay outside Git.
TARGET_ID=5
PREP_NAME=v6e-us
ACCELERATOR=v6e-16
: "${RUNTIME:?reviewed runtime}" "${RESOURCE_SA:?}" "${STARTUP:?absolute prepared payload}"
NEXT_CONFIG="$STAGE/enable${TARGET_ID}.json"
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" \
  --config /var/lib/gfn-spot/queue.json --output "$NEXT_CONFIG" \
  --enable "$TARGET_ID" --preparation "$PREP_NAME" --accelerator-type "$ACCELERATOR" \
  --runtime-version "$RUNTIME" --service-account "$RESOURCE_SA" --startup-script "$STARTUP"
INSTALL_ARGS=(--config "$NEXT_CONFIG")
# To disable instead, generate a separate candidate:
# sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" --config INPUT --disable ID --output NEW
```

Use the installation/update procedure for the candidate. Disabling leaves
resources and charges untouched. `spot_queue.py --targets ID` only narrows
management; the entire configuration is still validated. Targets 1–2 cannot be
managed. Logical `v5e-16` maps to API accelerator `v5litepod-16`.

Every **300 seconds**, the watchdog successfully inventories all managed locations
before any reconciliation. API/auth/timeout/JSON failures stop the poll; errors
never mean absent capacity. Pending, active, deleting and unknown states remain
untouched. Ownership requires exact name/location/accelerator, Spot allocation
and `managed-by=gfn-spot-queue`. New nodes also get informational `gfn-target=ID`;
legacy ownership does not require it. Only owned FAILED/SUSPENDED requests with
no node may be deleted, without force; recreation waits for a later successful
inventory showing absence.

## Installation and updates

All following commands are **operator-run**. Use TPU1 (`gn-v4-32`), worker 0;
do not modify `/full-gauss-newton`, execute setup/capture or install training
dependencies there. Bash, Python 3.10+ with zoneinfo/tzdata, systemd and gcloud
must already exist. The independent service runs as `gfn-spot` from `/opt/gfn-spot`,
with writable state/config storage in `/var/lib/gfn-spot`. It uses attached account
`598447672993-compute@developer.gserviceaccount.com`, fresh isolated gcloud storage
and an allowlisted environment. User caches, inherited credentials/tokens/
impersonation, interactive login and downloaded keys are excluded. Discovery
prefers the fixed system PATH, then direct `/snap/google-cloud-cli/current/bin/gcloud`.
The systemd sandbox is retained, with journal logs and a 60-second failure retry.
Monitoring depends on TPU1 availability.

Before enabling capacity, check **live accelerator and regional networking/IP
quota and headroom**, including other project usage. Review supported runtime,
exact dependency closure and secret-free payload per generation/location;
no runtime or mesh is inferred. Verify resource-account data/checkpoint/secret
reads and output writes; after allocation, check startup on every discovered
worker and separately validate environment/topology before training.

Check controller metadata identity, OAuth scopes and effective IAM policies in
each location. List access proves only read permission. Separately evaluate
[queuedResources.create](https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/create)
(`tpu.nodes.create`),
[queuedResources.delete](https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/delete)
(`tpu.nodes.delete`) and `iam.serviceAccounts.actAs` on the captured resource
account through read-only policy evaluation, not test mutations or IAM changes.

### Stage the reviewed revision

From a clean, separate workstation checkout:

```bash
git status --short
git rev-parse HEAD
tar -czf /tmp/gfn-watchdog.tgz -C scripts \
  tpu/spot_queue.py tpu/watchdog_service.py tpu/install_watchdog.py \
  tpu/queue_config.py tpu/gfn-spot-watchdog.service tpu_targets.py
sha256sum /tmp/gfn-watchdog.tgz
gcloud compute tpus tpu-vm scp /tmp/gfn-watchdog.tgz \
  ga2740@gn-v4-32:/tmp/ --project=fast-second-order --zone=us-central2-b --worker=0
gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32 \
  --project=fast-second-order --zone=us-central2-b --worker=0
```

In that worker-0 shell, retain these variables throughout:

```bash
umask 077
set -o pipefail
sha256sum /tmp/gfn-watchdog.tgz # compare with workstation
STAGE=$(mktemp -d /tmp/gfn-watchdog.XXXXXXXX)
mkdir "$STAGE/code"
tar -xzf /tmp/gfn-watchdog.tgz -C "$STAGE/code"
NEXT_CONFIG=/var/lib/gfn-spot/queue.json # existing deployment, or reviewed candidate
INSTALL_ARGS=(--config "$NEXT_CONFIG")
```

For first installation, from the workstation copy the **existing prepared**
legacy config/payload (set the two source paths to the captured files):

```bash
: "${PREPARED_CONFIG:?}" "${PREPARED_STARTUP:?}"
gcloud compute tpus tpu-vm scp "$PREPARED_CONFIG" ga2740@gn-v4-32:/tmp/gfn-queue.json \
  --project=fast-second-order --zone=us-central2-b --worker=0
gcloud compute tpus tpu-vm scp "$PREPARED_STARTUP" ga2740@gn-v4-32:/tmp/gfn-startup.sh \
  --project=fast-second-order --zone=us-central2-b --worker=0
# Then in the TPU1 worker-0 shell:
INSTALL_ARGS=(--config /tmp/gfn-queue.json --startup-script /tmp/gfn-startup.sh)
```

For relocated version-2 inputs, use `--payload-dir DIR` with unique source
basenames instead of the legacy-only `--startup-script`. Do not regenerate
preparations on TPU1.

For an update, complete the handoff below **before replacing installed files**.
For initial installation alongside box248, install and run read-only preflight
while box248 remains active, then complete handoff on box248 before activation.

### Back up and install

After update handoff, save the stopped code/unit/config/payloads, excluding token
storage. Initial installation instead retains its prepared inputs and old command.

```bash
# Existing installations only, on TPU1 worker 0:
BACKUP="$HOME/gfn-watchdog-backups/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP"
sudo tar -czpf "$BACKUP/code.tgz" -C /opt/gfn-spot .
sudo cp -p /etc/systemd/system/gfn-spot-watchdog.service "$BACKUP/unit.service"
sudo bash -c 'cd /var/lib/gfn-spot && tar -czpf "$1/state.tgz" queue.json startup*.sh' _ "$BACKUP"
sudo chmod 600 "$BACKUP/code.tgz" "$BACKUP/unit.service" "$BACKUP/state.tgz"
```

The common installation command leaves the unit **stopped and disabled**:

```bash
sudo /usr/bin/python3 -I "$STAGE/code/tpu/install_watchdog.py" "${INSTALL_ARGS[@]}"
sudo systemctl is-active gfn-spot-watchdog.service   # inactive
sudo systemctl is-enabled gfn-spot-watchdog.service  # disabled
sudo stat -c '%U:%G %a %n' /opt/gfn-spot /opt/gfn-spot/*.py /var/lib/gfn-spot
sudo bash -c 'stat -c "%U:%G %a %n" /var/lib/gfn-spot/queue.json /var/lib/gfn-spot/startup*.sh'
```

The installer refuses active/enabled units, checks payload syntax/readability
without executing it, and copies exact bytes to `startup-PREPARATION.sh`
(`startup-v4.sh` for legacy input). Expect root:root code 755/644,
gfn-spot:gfn-spot state 700 and root:gfn-spot config/payloads 640. Compare
runtime/account and payload hashes with the source or backup before activation.

### Read-only preflight and routine service inspection

On TPU1 worker 0, use the same isolated identity/environment as the service:

```bash
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py \
  --preflight | tee "$HOME/gfn-preflight.log" || { echo 'STOP: preflight failed'; exit 1; }
sudo systemctl status --no-pager gfn-spot-watchdog.service
sudo journalctl -u gfn-spot-watchdog.service -n 100 --no-pager
```

Preflight only lists; it reports identity, executable, managed IDs/locations and
concise resource states/failure reasons without metadata/payloads/full node
descriptions. Compare with the old controller's final state. A conflict or failed
read blocks activation. Use these same commands for post-start/routine inspection.

### Handoff: processes and asynchronous operations

On the **old controller's host**, save its exact command, config, working directory
and relaunch settings. Identify only its systemd unit/timer, cron, supervisor or
nohup wrapper and disable automatic relaunch. A local file lock does not coordinate
different hosts/users. Record parent and descendants **before** stopping:

```bash
# TPU1 systemd case; for box248/nohup substitute its separately verified parent PID.
OLD_PID=$(systemctl show gfn-spot-watchdog.service -p MainPID --value)
[[ "$OLD_PID" =~ ^[1-9][0-9]*$ ]] || exit 1
HANDOFF=$(mktemp -d "$HOME/gfn-handoff.XXXXXXXX")
sudo ps -p "$OLD_PID" -o user,pid,ppid,lstart,args
sudo pstree -alp "$OLD_PID" | tee "$HANDOFF/tree-before.txt"
sudo ps -eo user,pid,ppid,lstart,args > "$HANDOFF/processes-before.txt"
# Verify this is the controller before pausing new polls.
sudo kill -STOP "$OLD_PID"
sudo pstree -alp "$OLD_PID" | tee "$HANDOFF/tree-paused.txt"
sudo ps -eo user,pid,ppid,lstart,args > "$HANDOFF/processes-paused.txt"
OLD_DESCENDANT_PIDS=() # fill with ALL verified process PIDs from both trees, not thread IDs
OLD_ALL_PIDS=("$OLD_PID" "${OLD_DESCENDANT_PIDS[@]}")
OLD_PID_LIST=$(IFS=,; echo "${OLD_ALL_PIDS[*]}")
sudo ps -p "$OLD_PID_LIST" -o user,pid,ppid,lstart,args | tee "$HANDOFF/identities.txt"
sudo systemctl disable --now gfn-spot-watchdog.service
WAIT_ARGS=()
for pid in "${OLD_ALL_PIDS[@]}"; do WAIT_ARGS+=(--pid="$pid"); done
timeout 240 tail "${WAIT_ARGS[@]}" -f /dev/null || { echo 'STOP: exit unverified'; exit 1; }
sudo ps -p "$OLD_PID_LIST" -o user,pid,ppid,lstart,args # header only; exit 1
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p SubState -p ControlGroup
sudo systemctl is-enabled gfn-spot-watchdog.service # disabled
```

For nohup, replace systemctl stop with `kill -TERM "$OLD_PID"` then
`kill -CONT "$OLD_PID"` after disabling its identified relaunch mechanism.
For other managers, stop only the verified watchdog unit/wrapper. Require parent,
all cloud-command descendants (including wrappers/Python and reparented or later
children) exited, and an inactive/disabled unit with MainPID=0 and empty cgroup
when applicable. Parent exit alone is insufficient. Check saved start times/
commands for PID reuse; never use broad kills or signal training. Timeout,
incomplete observation or ambiguous identity blocks handoff. If aborting before
termination, explicitly decide whether to resume the paused old controller.

Review shutdown logs and cloud-command arguments. If create/delete was in flight,
local exit does not resolve server-side async work. On that host, take fresh
inventories for each affected location (also useful for the pre-handoff baseline):

```bash
PROJECT=fast-second-order
ZONE=us-central2-b # affected location from the target table
gcloud compute tpus queued-resources list --project="$PROJECT" --zone="$ZONE" --format=json
gcloud compute tpus tpu-vm list --project="$PROJECT" --zone="$ZONE" --format=json
```

Correlate request/node identities with the original operation and inspect status/
errors through read-only Cloud TPU operations or Console/audit records. Record
the completed outcome in `$HANDOFF`: accepted creation, confirmed deletion with
no node, or understood completed failure. An ambiguous/running operation, failed
read, single absent snapshot or APPLY log is insufficient. Repeat reads until
resolved, then rerun preflight; never mutate merely to settle uncertainty.

**Conditional legacy-name migration:** while all controllers are stopped, inspect
both old/new names. Either `gn-v4-32-spot-1` or `gn-v4-32-spot-2` request/node in
any state blocks management of both renamed 3–4 before any poll mutation.
The watchdog never deletes/adopts legacy capacity or chooses which identity to
retain. If retaining either legacy resource, leave the service disabled or use
two `--disable ID --output NEW` config steps to disable both 3 and 4 before
installing. Otherwise retire each resource through a separately approved operator
workflow, never force-deleting a request with a node. Require both legacy names
absent from both fresh inventories before managing renamed targets. Relabeling
does not clear the guard; retiring pending capacity loses its queue position.

### Separate activation

After handoff, migration if needed, installation and successful fresh preflight,
on TPU1 worker 0:

```bash
sudo systemctl enable --now gfn-spot-watchdog.service
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p SubState -p NRestarts
```

Run the inspection/preflight block above. Verify existing pending/active requests
were not reset and 1–2 remain unchanged, allowing legitimate async transitions.
Enabled units start after reboot; validate that separately in a planned maintenance
window using status and `sudo journalctl -b -u gfn-spot-watchdog.service --no-pager`.

### Rollback

Apply the same handoff checks to the new controller first; if TPU1 is unreachable,
establish it cannot run before starting another controller. Older code lacks the
reverse-name guard: old-name 3–4 management requires both new `spot-3`/`spot-4`
requests/nodes absent. Otherwise keep old management disabled or both targets
unmanaged with a compatible config. Resolve disposition manually.

On TPU1 worker 0 after handoff, restore the chosen stopped backup:

```bash
sudo tar -xzpf "$BACKUP/code.tgz" -C /opt/gfn-spot
sudo tar -xzpf "$BACKUP/state.tgz" -C /var/lib/gfn-spot
sudo install -o root -g root -m 644 "$BACKUP/unit.service" /etc/systemd/system/gfn-spot-watchdog.service
sudo systemd-analyze verify /etc/systemd/system/gfn-spot-watchdog.service
sudo systemctl daemon-reload
```

Review the restored config and run preflight before separate activation. A v4-only
rollback leaves any 5–8 capacity unmanaged. To restore box248 instead, keep TPU1
disabled and restart only the recorded old command/config/manager after these
same handoff gates; do not add a second nohup process.

## Troubleshooting and local validation

For identity/executable errors, check attached identity and fixed-path gcloud;
do not weaken the sandbox or pass user credentials. Inventory errors fail the
whole poll and retry after 60 seconds; inspect API/IAM/scopes. For terminal
failures inspect ownership/runtime/operation errors; an existing node prevents
deletion. For helper mismatches bootstrap matching helpers, including the registry,
on clean idle workers rather than resetting training checkouts.

From the repository root:

```bash
python3 -m unittest discover -s scripts/tests -v
bash -n scripts/tpu/setup_vm.sh
bash -n scripts/tpu.sh
systemd-analyze verify scripts/tpu/gfn-spot-watchdog.service
git diff --check
```

CPU/mock checks cover configuration, reconciliation, isolated authentication,
staged/installed imports and installation without activation. Live IAM, quota,
provisioning, runtime/training readiness and reboot behavior require operator
validation.
