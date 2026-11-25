# Operator installation, update and rollback

These are **operator commands**, not actions performed by this implementation.
The currently running independent TPU1 service is the v4-only deployment. Keep
its requests and trainer checkout intact. This update remains pending until you
perform and confirm it. Never run capture_setup.py, startup.sh, pip or training
setup on TPU1 during this procedure. No command below modifies
`/full-gauss-newton` on TPU1.

## Stage the reviewed source while the old service remains active

On a workstation, from a separate clean checkout of the reviewed commit on
`codex/watchdog-service-account` (eventually the reviewed merged revision):

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

In that TPU1 worker-0 shell, retain these shell variables for subsequent steps.
The archive retains the source layout so the installer finds the shared registry.
Existing Bash, Python 3.10+/zoneinfo/tzdata, systemd and a supported gcloud
installation are prerequisites. No packages are installed here.

```bash
umask 077
sha256sum /tmp/gfn-watchdog-update.tgz  # compare with workstation
STAGE=$(mktemp -d /tmp/gfn-watchdog-update.XXXXXXXX)
mkdir "$STAGE/code"
tar -xzf /tmp/gfn-watchdog-update.tgz -C "$STAGE/code"
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" \
  --config /var/lib/gfn-spot/queue.json --output "$STAGE/queue-v2.json"
# Expect exactly: Managed targets: 3, 4. This command makes no cloud calls.
sudo cat "$STAGE/queue-v2.json"
NEXT_CONFIG="$STAGE/queue-v2.json"
```

Legacy runtime/account values and startup path are copied without inference.
Migration does not change startup bytes or cloud resources, and 5–8 remain
disabled. The entire new config is validated locally, including declared quota
pools. Read/list IAM and scopes for the current location have been operator
validated; review additional locations separately before enabling them.

## Stop and verify every old process before installation

First follow [the process recording and exit procedure](TROUBLESHOOTING.md#record-processes-and-verify-exit)
in this same shell. It records MainPID, all descendants and their identities
**before** stopping or pausing the parent. Inspect and disable any watchdog
relaunch outside this unit too; a timer, cron entry, supervisor or box248
controller must not overlap. For the service, after recording descendants:

```bash
sudo systemctl disable --now gfn-spot-watchdog.service
sudo systemctl is-active gfn-spot-watchdog.service   # inactive; expected exit 3
sudo systemctl is-enabled gfn-spot-watchdog.service  # disabled; expected exit 1
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p SubState -p ControlGroup
```

Require MainPID=0, inactive/disabled state, an empty service cgroup, **and verified
exit of all recorded descendants**, including reparented gcloud/Python commands.
Parent exit alone is insufficient. Do not use broad process kills. A timeout,
unreadable process tree or ambiguous process identity blocks the update.

If a create/delete was in flight, follow [operation resolution](TROUBLESHOOTING.md#resolve-in-flight-mutations).
Fresh queued-resource/node inventory and operation status must resolve its
outcome before activation. Do not let a new controller determine an old
mutation's outcome by reconciling. The local file lock does not coordinate hosts.

## Back up the stopped installed copy, then install

After process exit, preserve old code/unit/config/payloads, not gcloud token
storage. The archives include only the explicit files below. Keep a new backup
per update and retain its path. They contain no downloaded service-account keys;
prepared payloads must remain secret-free.

```bash
BACKUP="$HOME/gfn-watchdog-backups/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP"
sudo tar -czpf "$BACKUP/code.tgz" -C /opt/gfn-spot .
sudo cp -p /etc/systemd/system/gfn-spot-watchdog.service "$BACKUP/unit.service"
sudo bash -c 'cd /var/lib/gfn-spot && tar -czpf "$1/state.tgz" queue.json startup*.sh' _ "$BACKUP"
sudo chmod 600 "$BACKUP/code.tgz" "$BACKUP/unit.service" "$BACKUP/state.tgz"
sudo journalctl -u gfn-spot-watchdog.service -n 200 --no-pager \
  | tee "$BACKUP/journal-before.txt"
sudo /usr/bin/python3 -I "$STAGE/code/tpu/install_watchdog.py" --config "$NEXT_CONFIG"
sudo systemctl is-active gfn-spot-watchdog.service   # inactive
sudo systemctl is-enabled gfn-spot-watchdog.service  # disabled
sudo stat -c '%U:%G %a %n' /opt/gfn-spot /opt/gfn-spot/*.py /var/lib/gfn-spot
sudo bash -c 'stat -c "%U:%G %a %n" /var/lib/gfn-spot/queue.json /var/lib/gfn-spot/startup*.sh'
```

The installer refuses an active/enabled unit and leaves it stopped/disabled.
It copies every runtime module (`spot_queue`, `watchdog_service`, `queue_config`,
`tpu_targets`) into `/opt/gfn-spot`; it never updates a training checkout.
Code is root:root 755/644, state is gfn-spot:gfn-spot 700, config and copied
payloads are root:gfn-spot 640. Startup syntax and service-user readability are
checked without executing startup. The legacy payload becomes
`/var/lib/gfn-spot/startup-v4.sh`, with exact bytes retained.

For the initial legacy migration, compare old/new payload hashes:

```bash
sudo bash -c 'tar -xzOf "$1/state.tgz" startup.sh | sha256sum' _ "$BACKUP"
sudo sha256sum /var/lib/gfn-spot/startup-v4.sh
```

On later v2 updates, compare `startup-v4.sh` from the archive instead. Do not
proceed if runtime/account, ownership labels, resource names or payload bytes for
3–4 unexpectedly differ. `--startup-script SOURCE` is supported only as a legacy
source-path override. For off-host v2 configs, `--payload-dir DIR` reads payloads
by their unique source basenames; ambiguous basenames or missing enabled
payloads fail validation. Keep generated configs/payloads outside Git.

## Read-only preflight, review, then separate activation

Run preflight after installation, before activation; the same isolated identity,
gcloud discovery and environment are used by the service:

```bash
set -o pipefail
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py \
  --preflight | tee "$HOME/gfn-preflight-handoff.log" || { echo 'STOP: preflight failed'; exit 1; }
```

For the first update, expect managed targets **3, 4** and only us-central2-b.
Compare fresh inventory with the stopped controller's final logs/snapshot;
preserve existing pending requests and targets 1–2. Resolve all previously
in-flight mutations before proceeding; repeat preflight after resolution.
List access does not prove create/delete/actAs permission. Confirm the attached
identity, permissions/scopes, resource settings and regional IP headroom using
[the readiness checklist](TROUBLESHOOTING.md#generation-readiness-and-permissions).

Only after every gate above has passed, run this **separate activation block**:

```bash
sudo systemctl enable --now gfn-spot-watchdog.service
sudo systemctl status --no-pager gfn-spot-watchdog.service
sudo systemctl show gfn-spot-watchdog.service -p MainPID -p SubState -p NRestarts
sudo journalctl -u gfn-spot-watchdog.service -n 100 --no-pager
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py \
  --preflight | tee "$HOME/gfn-preflight-after.log"
```

The unit retains its sandbox, unprivileged user, journal logs, 60-second failure
restart and enabled reboot startup. Verify unchanged existing request/node
states. Legitimate asynchronous transitions may occur, but pending/active states
must not be reset by this update. Reboot survival requires a separately approved
maintenance window; after that reboot inspect `systemctl status` and
`sudo journalctl -b -u gfn-spot-watchdog.service --no-pager`. Do not reboot a
training host merely to test this implementation.

## Stage 2: validate and enable target 5 only

First keep 3–4 running through the v2 migration and verify them. Separately
complete all generation-readiness checks for v6e in us-east1-d. The current v4
runtime/closure is **not** evidence of v6e compatibility. On worker 0, stage the
operator-reviewed secret-free payload outside `/full-gauss-newton`, then supply
the verified values (no values are invented here):

```bash
: "${V6E_RUNTIME:?set verified runtime}" "${V6E_RESOURCE_SA:?set reviewed resource identity}"
: "${V6E_STARTUP:?absolute path to separately prepared v6e startup payload}"
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" \
  --config /var/lib/gfn-spot/queue.json --output "$STAGE/queue-enable5.json" \
  --enable 5 --preparation v6e-us --accelerator-type v6e-16 \
  --runtime-version "$V6E_RUNTIME" --service-account "$V6E_RESOURCE_SA" \
  --startup-script "$V6E_STARTUP"
NEXT_CONFIG="$STAGE/queue-enable5.json"
```

This writes a local candidate only. Expect managed targets **3, 4, 5**; 6–8 stay
disabled. Repeat the complete record/stop/exit/operation-resolution/backup/install
sequence above with this `NEXT_CONFIG`, and a new backup. Preflight now must
successfully list both inventories in us-central2-b **and** us-east1-d. Only then
use the separate activation block. Activation authorizes capacity requests for
5 when absent; it does not launch experiments. Validate every allocated worker
before any manual training command.

## Stage 3: validate 6, 7 and 8 separately

After 5 is validated, review one target/location at a time. For each, create a
new config with `queue_config.py --enable ID --preparation NAME` and all four
explicit preparation fields as above. For 6 use a separately reviewed
`v6e-europe` preparation (`v6e-16`, europe-west4-a); for 7 use `v5e-us`
(`v5e-16`, us-central1-a); for 8 use `v5e-europe` (`v5e-16`, europe-west4-b).
Verify supported runtime and headroom in each location even when sharing a
generation. Repeat safe update, backup, preflight and explicit activation for
each stage. Do not enable all 6–8 merely because quota is declared.

For each separately reviewed stage, set these variables for **one** target
only, supply that target's reviewed runtime/account/payload, and generate its
candidate (do not run this as a loop):

```bash
# Stage 6: TARGET_ID=6; PREP_NAME=v6e-europe; ACCELERATOR=v6e-16
# Later stage 7: TARGET_ID=7; PREP_NAME=v5e-us; ACCELERATOR=v5e-16
# Later stage 8: TARGET_ID=8; PREP_NAME=v5e-europe; ACCELERATOR=v5e-16
: "${TARGET_ID:?}" "${PREP_NAME:?}" "${ACCELERATOR:?}"
: "${RUNTIME:?reviewed per location}" "${RESOURCE_SA:?}" "${STARTUP:?absolute prepared path}"
NEXT_CONFIG="$STAGE/queue-enable${TARGET_ID}.json"
sudo /usr/bin/python3 "$STAGE/code/tpu/queue_config.py" \
  --config /var/lib/gfn-spot/queue.json --output "$NEXT_CONFIG" \
  --enable "$TARGET_ID" --preparation "$PREP_NAME" --accelerator-type "$ACCELERATOR" \
  --runtime-version "$RUNTIME" --service-account "$RESOURCE_SA" --startup-script "$STARTUP"
```

Then repeat the full stopped-service update and separate activation using this
candidate. Existing enabled targets and their preparations remain in the input.

To stop future management while leaving resources untouched, generate a
candidate with `--disable ID --output NEW`, then perform the same safe update.
Disabling does not delete a request/node or stop charges. Manual numeric control
remains available. No automatic experiment resume is provided.

## Rollback to the saved controller

Record the **new** controller and descendants before stopping, disable/stop it,
and verify all exits and server-side mutation outcomes as above. If TPU1 is
unreachable, establish that it cannot run before starting another controller.
No simultaneous controllers are permitted, including box248. Then restore the
saved files from the selected backup; do not delete any cloud resources:

```bash
sudo systemctl disable --now gfn-spot-watchdog.service
# STOP HERE until recorded parent/descendant exits and mutation outcomes are verified.
sudo tar -xzpf "$BACKUP/code.tgz" -C /opt/gfn-spot
sudo tar -xzpf "$BACKUP/state.tgz" -C /var/lib/gfn-spot
sudo install -o root -g root -m 644 "$BACKUP/unit.service" /etc/systemd/system/gfn-spot-watchdog.service
sudo systemd-analyze verify /etc/systemd/system/gfn-spot-watchdog.service
sudo systemctl daemon-reload
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py --preflight
```

Review the restored config and fresh inventories before separately enabling and
starting it with the activation block. Extra new runtime modules/payload files
can remain unused; no recursive cleanup is necessary. Rolling back to the
v4-only config leaves any already-created 5–8 resources untouched and unmanaged;
inspect them manually. If restoring box248 instead, keep the TPU1 unit disabled
and restore only the recorded old command/config/relaunch mechanism after
verified TPU1 subprocess exit and resolved mutations. Do not launch nohup in
addition to a restored manager.
