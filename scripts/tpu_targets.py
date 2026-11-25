"""Authoritative TPU identities and declared quota pools (no cloud access)."""
from dataclasses import dataclass

PROJECT = 'fast-second-order'
PROJECT_NUMBER = '598447672993'


@dataclass(frozen=True)
class Target:
    name: str
    zone: str
    accelerator: str
    allocation: str
    managed: bool = False
    project: str = PROJECT
    project_number: str = PROJECT_NUMBER
    chips: int = 16

    @property
    def api_accelerator(self):
        # Cloud TPU's documented v5e accelerator token; logical pool stays v5e.
        return 'v5litepod-16' if self.accelerator == 'v5e-16' else self.accelerator


TARGETS = {
    '1': Target('gn-v4-32', 'us-central2-b', 'v4-32', 'on-demand'),
    '2': Target('gn-v4-32-2', 'us-central2-b', 'v4-32', 'on-demand'),
    '3': Target('gn-v4-32-spot-1', 'us-central2-b', 'v4-32', 'spot', True),
    '4': Target('gn-v4-32-spot-2', 'us-central2-b', 'v4-32', 'spot', True),
    '5': Target('gn-v6e-16-spot-5', 'us-east1-d', 'v6e-16', 'spot'),
    '6': Target('gn-v6e-16-spot-6', 'europe-west4-a', 'v6e-16', 'spot'),
    '7': Target('gn-v5e-16-spot-7', 'us-central1-a', 'v5e-16', 'spot'),
    '8': Target('gn-v5e-16-spot-8', 'europe-west4-b', 'v5e-16', 'spot'),
}
# Project, zone, accelerator, allocation -> supplied TRC chips, not device counts.
QUOTA_POOLS = {
    (PROJECT, 'us-central2-b', 'v4-32', 'on-demand'): 32,
    (PROJECT, 'us-central2-b', 'v4-32', 'spot'): 32,
    (PROJECT, 'us-east1-d', 'v6e-16', 'spot'): 64,
    (PROJECT, 'europe-west4-a', 'v6e-16', 'spot'): 64,
    (PROJECT, 'us-central1-a', 'v5e-16', 'spot'): 64,
    (PROJECT, 'europe-west4-b', 'v5e-16', 'spot'): 64,
}


def validate_registry(targets=None):
    """Count every declared slot, including disabled targets; not live usage."""
    targets = TARGETS if targets is None else targets
    usage, identities = {}, set()
    for key, target in targets.items():
        identity = (target.project, target.zone, target.name)
        if identity in identities:
            raise ValueError('Duplicate target identity: ' + key)
        identities.add(identity)
        if target.chips != 16 or target.accelerator not in {'v4-32', 'v5e-16', 'v6e-16'}:
            raise ValueError('Unreviewed chip accounting: ' + key)
        if target.managed and (key in {'1', '2'} or target.allocation != 'spot'):
            raise ValueError('Fixed/on-demand targets cannot be watchdog managed')
        pool = (target.project, target.zone, target.accelerator, target.allocation)
        usage[pool] = usage.get(pool, 0) + target.chips
        if pool not in QUOTA_POOLS or usage[pool] > QUOTA_POOLS[pool]:
            raise ValueError('Declared targets exceed supplied quota pool: ' + repr(pool))
    return usage


validate_registry()
