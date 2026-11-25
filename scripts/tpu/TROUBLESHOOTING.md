# Readiness, handoff and troubleshooting

All commands here are for the operator. They were not run against cloud resources
by the implementation. Use [the numeric target table](README.md) for exact
locations/names. Never use watchdog reconciliation to diagnose an uncertain
mutation, runtime or training failure.

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
inventories for each affected registry location. The installed wrapper's
`--preflight` reads all managed locations without reconciling; when updating
while installed code is stopped, these read-only operator commands can also
inspect the affected location directly:

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

## Generation readiness and permissions

Before enabling **each** target/location, record these operator checks:

1. Confirm the accelerator/runtime combination is supported in that zone using
   the current Cloud TPU runtime catalog. Supply an explicitly reviewed runtime;
   no v5e/v6e runtime version is asserted by this repository.
2. Review the declared chip pool **and live use elsewhere in the project**.
   Confirm regional external IP quota and headroom for all allocated workers,
   networking, and other applicable quotas. The reported us-central2 IP quota of
   16 proves nothing about us-east1/europe-west4/us-central1 headroom.
3. Check the controller's metadata identity, OAuth scopes, effective IAM
   allow/deny policies/conditions and organization constraints in every location.
   Successful lists only prove read access. Separately verify the actual API
   operations: queuedResources.create uses `tpu.nodes.create`,
   queuedResources.delete uses `tpu.nodes.delete`; also review
   `iam.serviceAccounts.actAs` on the preparation's captured resource account.
   Use read-only policy evaluation/Policy Troubleshooter, not test mutations or
   broad IAM grants. The service never changes IAM or impersonates another user.
4. Check the **resource account** can read the configured data, warmup checkpoint
   and `wandb-api-key` secret, and write intended output/queue buckets. These are
   separate from controller inventory/create/delete access. Preserve existing
   account/runtime settings for 3–4.
5. Review a **separate** exact dependency closure and startup payload for v6e/v5e.
   Do not relabel v4 pins as validated new-generation dependencies. Existing
   `prepare_startup.py` accepts reviewed package==version pins, a branch/full SHA,
   Python minor version and secret name; it does not discover compatibility.
   Do not upgrade packages automatically or run the generated setup on TPU1.
6. After approved capacity allocation, confirm setup completion on **every**
   discovered worker, exact package versions, model imports, data/secret access
   and a small separately operator-run CPU/readiness check. ACTIVE capacity alone
   is not startup or training readiness.
7. On an idle new target only, run a coordinated distributed topology check on
   all discovered workers. Record process indices/counts, global/local device
   counts, and a suitable mesh. Neither 16 chips nor a `-16` suffix establishes
   worker count or the trainer's mesh. Supply the reviewed mesh and other training
   flags explicitly; presets/experiments are never automatically rewritten.

To inspect startup on an operator-allocated new target:

```bash
# Set to that target's exact registry name/project/zone.
: "${NAME:?}" "${PROJECT:?}" "${ZONE:?}"
gcloud compute tpus tpu-vm describe "$NAME" --project="$PROJECT" --zone="$ZONE" --format=json
gcloud compute tpus tpu-vm ssh "ga2740@$NAME" --project="$PROJECT" --zone="$ZONE" --worker=all \
  --command='cat /tmp/gfn_setup_status; sudo tail -n 30 /var/log/gfn-setup.log'
```

Expect DONE on every worker. FAILED setup is inspected in place; the watchdog
does not replace an ACTIVE request because startup failed. A fresh controller
`status` checks each discovered worker's helper/environment. Before experiments,
review model imports and a small readiness check under that target's reviewed
environment; actual model/training compatibility is not a local mock-test result.
Distributed initialization must run simultaneously across the discovered workers
and be shut down cleanly; never run it alongside active training. Exact-state
resume is not provided; preempted runs require an operator checkpoint decision.

Historical reference only: the 2026-11-23 v4 capture reported
tpu-ubuntu2204-base, Python 3.10.12, JAX/jaxlib 0.6.2, libtpu 0.0.17,
Flax 0.10.7, Optax 0.2.8. The capture script reads a v4 reference and remains
v4-specific. It must not be used as evidence of v5e/v6e compatibility. Original
bootstrap clones a pinned new-worker checkout and refuses an existing checkout;
do not rerun it on a training host. Captured/generated files remain outside Git.

## Identify existing nodes without recreation

These are optional **operator-run metadata updates**, not part of watchdog
reconciliation. Run from the reviewed controller repository root. Select one
existing node by its numeric ID; start with 1 or 2, and use 3 or 4 only after a
node is allocated. A pending queued request without a node cannot be labeled
this way; leave that request untouched. No node names, queued requests or startup
payloads are changed by these commands.

Resolve the exact project/zone/name from the shared registry and inspect it first:

```bash
TARGET_ID=1  # choose one existing allocated target; repeat deliberately for others
read -r PROJECT ZONE NAME < <(
  python3 -c 'import sys; sys.path.insert(0, "scripts"); from tpu_targets import TARGETS; t = TARGETS[sys.argv[1]]; print(t.project, t.zone, t.name)' "$TARGET_ID"
)
printf 'Target %s: %s (%s/%s)\n' "$TARGET_ID" "$NAME" "$PROJECT" "$ZONE"
gcloud compute tpus tpu-vm describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
  --format='json(name,state,labels,description)'
```

After verifying the identity, add just the identifying label in place:

```bash
gcloud compute tpus tpu-vm update "$NAME" --project="$PROJECT" --zone="$ZONE" \
  --update-labels="gfn-target=$TARGET_ID"
```

`--update-labels` preserves other label keys, including any existing ownership
label. Do not use `--clear-labels`, remove `managed-by`, or add a watchdog ownership
label to fixed targets 1–2. The watchdog does not require `gfn-target` on legacy
resources, so these updates are optional. New Spot creates include the label
automatically; existing queued-resource templates remain unchanged.

Optionally append an identifying description while preserving the previous
text. Inspect the proposed value before applying this separate metadata update:

```bash
OLD_DESCRIPTION=$(gcloud compute tpus tpu-vm describe "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --format='value(description)') || exit 1
NEW_DESCRIPTION="${OLD_DESCRIPTION}${OLD_DESCRIPTION:+ | }GFN controller target $TARGET_ID"
printf 'Proposed description: %s\n' "$NEW_DESCRIPTION"
# Run only after reviewing the proposed text:
gcloud compute tpus tpu-vm update "$NAME" --project="$PROJECT" --zone="$ZONE" \
  --description="$NEW_DESCRIPTION"
gcloud compute tpus tpu-vm describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
  --format='json(name,state,labels,description)'
```

The [TPU VM update command](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/update)
supports these label/description fields directly. Use the operator's existing
update permission; do not change IAM or delete/recreate a node to add identifying
metadata. No such updates were run by this implementation.

## Common failures

- **Discovery fails:** check allocation, project/zone overrides and API/auth access.
  Do not infer absent capacity from a failed describe/list.
- **Service cannot find gcloud:** ordinary discovery uses a fixed system PATH;
  direct `/snap/google-cloud-cli/current/bin/gcloud` is the fallback. Preflight
  reports the selected executable. Do not weaken NoNewPrivileges/ProtectSystem/
  ProtectHome or pass operator credential environment settings to repair it.
- **One managed location fails:** the entire watchdog poll exits before
  reconciliation; systemd retries after 60 seconds. Disable management through a
  reviewed stopped-service update if you deliberately want to exclude that
  location. Do not reinterpret the error as missing nodes.
- **Repeated terminal requests:** inspect ownership, setup/runtime and operation
  errors. Deletion is refused with an existing node and never uses force.
- **Registry/helper mismatch:** manually bootstrap reviewed identical helpers,
  including `tpu_targets.py`, onto clean idle workers. Do not reset/stash active
  training checkouts to force compatibility.

API references for operator review:
- https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/create
- https://docs.cloud.google.com/tpu/docs/reference/rest/v2/projects.locations.queuedResources/delete
- https://docs.cloud.google.com/iam/docs/service-accounts-actas
