"""Authoritative TPU identities (no cloud access or quota enforcement)."""
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

    @property
    def api_accelerator(self):
        # Cloud TPU's documented v5e accelerator token; logical pool stays v5e.
        return 'v5litepod-16' if self.accelerator == 'v5e-16' else self.accelerator


TARGETS = {
    '1': Target('gn-v4-32', 'us-central2-b', 'v4-32', 'on-demand'),
    '2': Target('gn-v4-32-2', 'us-central2-b', 'v4-32', 'on-demand'),
    '3': Target('gn-v4-32-spot-3', 'us-central2-b', 'v4-32', 'spot', True),
    '4': Target('gn-v4-32-spot-4', 'us-central2-b', 'v4-32', 'spot', True),
    '5': Target('gn-v6e-16-spot-5', 'us-east1-d', 'v6e-16', 'spot'),
    '6': Target('gn-v6e-16-spot-6', 'europe-west4-a', 'v6e-16', 'spot'),
    '7': Target('gn-v5e-16-spot-7', 'us-central1-a', 'v5e-16', 'spot'),
    '8': Target('gn-v5e-16-spot-8', 'europe-west4-b', 'v5e-16', 'spot'),
}
# Historical identities are conflict sentinels, never additional managed targets.
LEGACY_SPOT_NAMES = {'3': 'gn-v4-32-spot-1', '4': 'gn-v4-32-spot-2'}


def validate_registry(targets=None):
    """Validate identities and supported accelerators; operators check live quota."""
    targets = TARGETS if targets is None else targets
    identities = set()
    for key, target in targets.items():
        identity = (target.project, target.zone, target.name)
        if identity in identities:
            raise ValueError('Duplicate target identity: ' + key)
        identities.add(identity)
        if target.accelerator not in {'v4-32', 'v5e-16', 'v6e-16'}:
            raise ValueError('Unsupported accelerator: ' + key)
        if target.managed and (key in {'1', '2'} or target.allocation != 'spot'):
            raise ValueError('Fixed/on-demand targets cannot be watchdog managed')


validate_registry()
