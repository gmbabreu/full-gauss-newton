# TPU capacity and numeric control

The shared [registry](../tpu_targets.py) routes numeric IDs in project
`fast-second-order`. A slot does not imply allocated capacity or training readiness.

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

New Spot nodes carry `managed-by=gfn-spot-queue` and informational `gfn-target=ID`.
Legacy ownership does not require the identifying label. Targets 1–2 are never
watchdog managed. Numeric 3–4 address the new names above; either legacy
`gn-v4-32-spot-1`/`gn-v4-32-spot-2` request or node blocks management of both
renamed targets. See [migration and deployment](DEPLOYMENT.md).

The installed v4-only service on TPU1 worker 0 uses the legacy names. This
branch's registry/configuration update is **pending operator deployment**;
updating Git does not update `/opt/gfn-spot` or cloud resources. Targets 5–8
remain disabled. Monitoring depends on TPU1 remaining available.

## Configuration

Legacy three-key configs normalize to version 2 with only 3–4 managed. Captured
runtime, resource service account and exact startup bytes are preserved; the
installer rewrites startup paths to persistent files in `/var/lib/gfn-spot`.
Example (supply the existing captured values):

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

`queue_config.py --config EXISTING --output NEW` validates/migrates locally and
refuses to overwrite the output. Keep generated files outside Git. Omitted
target settings use registry defaults; explicitly disable unwanted management.
Enabled targets require an existing prepared payload and matching accelerator.
Unused preparations are structurally validated but need no existing payload.
The whole configuration is validated even with `spot_queue.py --targets ID`;
that option only narrows management. Disabling leaves resources untouched.

The watchdog polls every **300 seconds**. All managed-location inventories must
succeed before reconciliation; API/auth/timeout/JSON errors stop the entire poll.
Pending, active, deleting and unknown states remain untouched. Ownership requires
the exact name, location, accelerator, Spot allocation and ownership label.
Only owned FAILED/SUSPENDED requests without nodes may be deleted, without force;
recreation waits for a later successful inventory showing absence.

Each target represents 16 chips. Supplied quotas are documentation, not enforced
grants or live headroom:

| Zone | Generation/allocation | Supplied chips |
| --- | --- | --- |
| us-central2-b | v4 on-demand | 32 |
| us-central2-b | v4 Spot | 32 |
| us-east1-d | v6e Spot | 64 |
| europe-west4-a | v6e Spot | 64 |
| us-central1-a | v5e Spot | 64 |
| europe-west4-b | v5e Spot | 64 |

Check **live quota, project-wide usage and regional IP headroom before activation**.
Logical `v5e-16` preparations use API accelerator `v5litepod-16`; no runtime or
dependency compatibility is inferred for a generation.

## Routine commands

Use one target ID for status, SSH, logs, run, stop, pull and switch:

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

Commands/logs display numeric IDs and names. Worker counts come from
`networkEndpoints`. Disabled targets remain available for manual inspection/control.
Supply reviewed topology through `GN_ARGS`; see the [training workflow](../../docs/tpu_workflow.md)
for matching helpers, environment and coordinated launches. Manual controller
overrides `GFN_TPU_NAME`, `GFN_PROJECT`, `GFN_ZONE`, `GFN_SSH_USER`,
`GFN_REMOTE_REPO`, `GFN_REMOTE_ENV`, `GFN_STATE_DIR` retain their meanings;
the watchdog ignores them. Unset identity overrides for normal numeric routing.

On TPU1 worker 0:

```bash
sudo systemctl status --no-pager gfn-spot-watchdog.service
sudo journalctl -u gfn-spot-watchdog.service -n 100 --no-pager
sudo journalctl -u gfn-spot-watchdog.service -f
sudo -u gfn-spot /usr/bin/python3 -I -u /opt/gfn-spot/watchdog_service.py --preflight
```

Preflight verifies the attached identity, reports its selected executable and
managed IDs/locations, and lists concise names/states/failure reasons without
reconciling. It does not verify mutation permission. Installation, stopping,
handoff, activation and rollback follow the single [operator procedure](DEPLOYMENT.md).
