# TPU capacity and numeric control

The authoritative registry is [tpu_targets.py](../tpu_targets.py). Project is
`fast-second-order` for every target. Names 1–2 and label
`managed-by=gfn-spot-queue` are unchanged. A configured target is a declared slot,
not a claim that capacity is allocated or ready for training.

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

All Spot slots now use numeric-ID suffixes. Fixed targets 1–2 are unchanged.
Legacy `gn-v4-32-spot-1` and `gn-v4-32-spot-2` requests/nodes are never renamed or
deleted automatically. **Either legacy name in either inventory blocks both
renamed targets 3–4**, regardless of state or labels. The whole poll stops before
any reconciliation when a selected managed target 3 or 4 has this conflict.
Read-only service preflight reports it too. Follow the
[stopped-service naming migration](DEPLOYMENT.md#resolve-the-spot-resource-name-migration-with-the-service-stopped)
before activation; pending or deleting legacy requests still block it.
New Spot nodes carry both `managed-by=gfn-spot-queue` and
`gfn-target=ID`. The identifying label is informational and is not required for
ownership of existing resources; the original ownership checks remain in force.
Optional [in-place identifying labels/descriptions](TROUBLESHOOTING.md#identify-existing-nodes-without-recreation)
can make existing allocated nodes easier to find without recreating them.

The logical v5e-16 targets use the documented Cloud TPU API accelerator token
`v5litepod-16` for creation and exact ownership checks; previews show both names.
Preparation `accelerator_type` and quota accounting use the logical `v5e-16`
name. See [Google's v5e examples](https://docs.cloud.google.com/tpu/docs/v5e-training).
No runtime version is inferred from those examples.

Each slot counts **16 chips**, including v4-32. Declared 1–2 use the supplied
32-chip on-demand v4 pool and 3–4 use the 32-chip Spot v4 pool in us-central2-b.
Each of the other zones has its own supplied 64-chip Spot pool for the generation
shown. Registry validation counts every declared slot, including disabled ones,
against its project/zone/accelerator/allocation pool. This checks declared targets
only: unknown project allocations, current quota use, IP quota and actual capacity
are not queried or inferred. It does not automatically allocate unused quota.

## Operational status

Operator report, 2026-11-24: the **previous v4-only installed copy** is enabled and
running on TPU1 (`gn-v4-32`), worker 0, independently under `/opt/gfn-spot`. Its
isolated attached identity and direct Snap executable worked under the systemd
sandbox; it deleted failed requests 3–4, recreated them and observed both in
`WAITING_FOR_RESOURCES` under the legacy names `gn-v4-32-spot-1` and
`gn-v4-32-spot-2`. The us-central2 external IP quota was increased to 16.
Allocated-worker startup, training readiness and reboot survival remain untested.

**The registry/version-2 update in this branch is not deployed.** Repository
updates do not alter the installed copy or its existing requests. Preserve the
working controller until the explicit [update and rollback](DEPLOYMENT.md).
New-generation runtime, dependency, topology and live API compatibility checks
remain pending. No service or target is enabled by repository changes.

## Routine commands (operator-run)

All numeric targets support the same commands, regardless of watchdog enablement.
Use a clean controller checkout with the same helpers/branch/SHA on every worker
for training; see [the training workflow](../../docs/tpu_workflow.md). Worker
counts come from `networkEndpoints`, never a generation-specific constant.

```bash
bash scripts/tpu.sh 5 status
bash scripts/tpu.sh 5 ssh 0
bash scripts/tpu.sh 5 logs 0
bash scripts/tpu.sh 5 run gn_adam --dry-run "${GN_ARGS[@]}"
bash scripts/tpu.sh 5 run gn_adam "${GN_ARGS[@]}"
bash scripts/tpu.sh 5 stop
bash scripts/tpu.sh 5 pull
bash scripts/tpu.sh 5 switch main
```

Preview shows the registry's accelerator, allocation, project, zone and name.
Every controller command starts with `Target ID: NAME`; worker reports also
include the numeric ID. Watchdog state/action lines start with
`Target ID [NAME]:`, for example
`Target 3 [gn-v4-32-spot-3]: WAITING_FOR_RESOURCES`.
No mesh, flags or packages are chosen for a generation. Supply reviewed topology
settings explicitly through `GN_ARGS`; presets are starter settings. A missing or
unreadable allocation produces a describe/access error, not zero workers. A
disabled target remains available for inspection and manual control if allocated.
Reservations on every worker, matching revisions/helpers, launch rollback and
managed-process ownership checks remain mandatory. The registry is now part of
the helper fingerprint: manually bootstrap that same helper revision on clean,
idle workers before using `pull`/`switch` across helper revisions.

For manual control only, `GFN_TPU_NAME`, `GFN_PROJECT` and `GFN_ZONE` override the
selected registry identity; unset them for normal numeric routing. Type/allocation
in preview remain the registry's declaration, not a live validation of an override.
`GFN_SSH_USER`, `GFN_REMOTE_REPO`, `GFN_REMOTE_ENV`, `GFN_STATE_DIR` retain their
existing meanings. Set overrides before starting the controller and use the same
ones for status/logs/stop. The watchdog ignores these controller overrides.

On TPU1 worker 0:

```bash
sudo systemctl status --no-pager gfn-spot-watchdog.service
sudo journalctl -u gfn-spot-watchdog.service -n 100 --no-pager
sudo journalctl -u gfn-spot-watchdog.service -f
# Before stopping, record descendants as described in DEPLOYMENT.md.
sudo systemctl disable --now gfn-spot-watchdog.service
```

## Watchdog configuration and protections

Version 2 configuration associates managed Spot targets with explicit prepared
settings. Legacy three-key configurations normalize to this form, enabling only
3–4 and preserving their captured runtime, resource service account and payload
path. The installer copies the exact payload bytes and rewrites its local path.
This schematic example uses operator-supplied values, not validated runtimes:

```json
{
  "version": 2,
  "preparations": {
    "v4": {
      "accelerator_type": "v4-32",
      "runtime_version": "<existing captured runtime>",
      "service_account": "<existing captured resource account>",
      "startup_script": "/var/lib/gfn-spot/startup-v4.sh"
    }
  },
  "targets": {
    "1": {"managed": false}, "2": {"managed": false},
    "3": {"managed": true, "preparation": "v4"},
    "4": {"managed": true, "preparation": "v4"},
    "5": {"managed": false}, "6": {"managed": false},
    "7": {"managed": false}, "8": {"managed": false}
  }
}
```

Use `queue_config.py --config EXISTING --output NEW` to validate/migrate locally;
it performs no cloud operations and refuses to overwrite an output file. Generated
configs and captured payloads belong outside Git. Omitted target settings use
registry defaults; explicitly disable any target you intend to leave unmanaged.
An enabled target requires an existing payload and matching preparation
accelerator. Disabled targets require no preparation and cause no inventory or
reconciliation of their own. Disabling never cancels their existing resources.
Defined but unused preparations are structurally validated; their files may be
absent until enabling. All enabled preparations and the entire config are
validated even when `spot_queue.py --targets` selects a subset. That option only
narrows management; it cannot enable disabled targets or manage 1–2.

The watchdog polls every **300 seconds**. It inventories queued resources and
nodes once per managed project/zone, completing **all** required inventories
before any reconciliation. Any API/auth/timeout/JSON failure stops the entire
process without reconciling that poll, including otherwise healthy locations.
Systemd retains its 60-second failure restart. Errors never mean absent capacity.
Pending, active, deleting, and unknown states are untouched. Ownership requires
the exact registry name, location, accelerator, Spot allocation and label.
Only owned FAILED/SUSPENDED requests with no node can be deleted, never with
force. Only a later successful inventory showing absence allows recreation.
Unrelated resources and fixed targets 1–2 are never traversed for deletion.

The service preserves the unprivileged `gfn-spot` user, root-owned independent
code, writable state, journald logs and systemd restrictions. Each invocation
uses fresh gcloud storage and an allowlisted environment, excluding inherited
credential files/tokens/impersonation/user caches. Metadata identity must equal
`598447672993-compute@developer.gserviceaccount.com`. Fixed system PATH discovery
prefers ordinary gcloud installations, then uses
`/snap/google-cloud-cli/current/bin/gcloud` directly. Preflight reports that exact
executable and all managed-location inventories. List success does not prove
create/delete permission. No interactive login or service-account keys are used.

The local file lock cannot coordinate different hosts or users. Use one controller;
record and verify all subprocess exits before handoff. Monitoring depends on TPU1
remaining available; reboot survival still needs operator validation.

## Deployment, preparation and validation

- [Installation/update, separate activation, staged enablement and rollback](DEPLOYMENT.md).
- [Generation readiness, IAM review and process/operation troubleshooting](TROUBLESHOOTING.md).

```bash
python3 -m unittest discover -s scripts/tests -v
bash -n scripts/tpu/setup_vm.sh
bash -n scripts/tpu.sh
systemd-analyze verify scripts/tpu/gfn-spot-watchdog.service
git diff --check
```

Local tests are CPU-only with mocked/fake cloud commands. They do not prove
runtime/generation compatibility, IAM, quotas, live provisioning, startup, training
readiness or reboot behavior. No training, optimizer, experiment or dependency
versions are changed by this implementation.

Local implementation validation (2026-11-24): **101 tests passed in 9.631s** in
an Ubuntu WSL temporary Linux snapshot of Git's stored files plus these changes,
avoiding the Windows checkout's existing CRLF shell files. Both script syntax
checks and `systemd-analyze verify` exited 0; all 30 Bash blocks in the operator
and workflow docs passed `bash -n`; Git diff whitespace checks passed. Installer
system commands/cloud calls were mocked; staged and installed module imports
were actually checked with `python -I`. No service was installed/started locally,
and no live cloud validation, IAM changes, SSH, capacity requests or training
were performed. Operator reports above describe the pre-existing deployment.
