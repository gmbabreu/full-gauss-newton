"""Versioned watchdog configuration and legacy migration. No cloud operations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tpu_targets import TARGETS, validate_registry

LEGACY_KEYS = {'runtime_version', 'service_account', 'startup_script'}


def normalize(data, *, check_files=True):
    validate_registry()
    if not isinstance(data, dict):
        raise ValueError('Queue configuration must be an object')
    data = deepcopy(data)
    if set(data) == LEGACY_KEYS:
        data = {'version': 2, 'preparations': {'v4': dict(data, accelerator_type='v4-32')},
                'targets': {key: {'managed': target.managed,
                                  **({'preparation': 'v4'} if target.managed else {})}
                            for key, target in TARGETS.items()}}
    if set(data) != {'version', 'preparations', 'targets'} or type(data['version']) is not int or data['version'] != 2:
        raise ValueError('Expected version 2 configuration with preparations and targets')
    preparations, overrides = data['preparations'], data['targets']
    if not isinstance(preparations, dict) or not isinstance(overrides, dict):
        raise ValueError('Preparations and targets must be objects')
    if set(overrides) - set(TARGETS):
        raise ValueError('Unregistered watchdog target')
    for name, prep in preparations.items():
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name):
            raise ValueError('Preparation name must use lowercase letters, numbers, underscores or dashes')
        if not isinstance(prep, dict) or set(prep) != LEGACY_KEYS | {'accelerator_type'}:
            raise ValueError('Preparation requires accelerator_type, runtime_version, service_account, startup_script')
        if any(not isinstance(value, str) or not value.strip() for value in prep.values()):
            raise ValueError('Preparation fields must be nonempty strings')
        if (prep['accelerator_type'] not in {'v4-32', 'v5e-16', 'v6e-16'} or
                '@' not in prep['service_account'] or not Path(prep['startup_script']).is_absolute()):
            raise ValueError('Invalid accelerator, resource service account or absolute startup path')
    normalized = {}
    for key, target in TARGETS.items():
        settings = overrides.get(key, {'managed': target.managed,
                                       **({'preparation': 'v4'} if target.managed else {})})
        if (not isinstance(settings, dict) or set(settings) - {'managed', 'preparation'} or
                type(settings.get('managed')) is not bool):
            raise ValueError(key + ': target requires a boolean managed setting')
        prep_name = settings.get('preparation')
        if prep_name is not None and not isinstance(prep_name, str):
            raise ValueError(key + ': preparation must be a name')
        if settings['managed']:
            if key in {'1', '2'} or target.allocation != 'spot':
                raise ValueError('Targets 1–2 and on-demand capacity can never be watchdog managed')
            if prep_name not in preparations:
                raise ValueError(key + ': enabling requires explicit prepared settings')
            prep = preparations[prep_name]
            if prep['accelerator_type'] != target.accelerator:
                raise ValueError(key + ': preparation accelerator does not match target')
            if check_files and not Path(prep['startup_script']).is_file():
                raise ValueError(key + ': startup-script must be a prepared local file')
        normalized[key] = settings
    data['targets'] = normalized
    return data


def load(path, *, check_files=True):
    return normalize(json.loads(Path(path).expanduser().read_text()), check_files=check_files)


def enabled(config, selected=None):
    keys = TARGETS if selected is None else dict.fromkeys(selected)
    return [key for key in keys if config['targets'][key]['managed']]


def preparation(config, key):
    return config['preparations'][config['targets'][key]['preparation']]


def locations(config, selected=None):
    return sorted({(TARGETS[key].project, TARGETS[key].zone) for key in enabled(config, selected)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', help='Write normalized/migrated configuration to a NEW file')
    change = parser.add_mutually_exclusive_group()
    change.add_argument('--enable', choices=sorted(TARGETS))
    change.add_argument('--disable', choices=sorted(TARGETS))
    parser.add_argument('--preparation', help='Explicit preparation name for --enable')
    for field in ('accelerator_type', 'runtime_version', 'service_account', 'startup_script'):
        parser.add_argument('--' + field.replace('_', '-'))
    args = parser.parse_args()
    config = load(args.config)
    fields = {field: getattr(args, field) for field in LEGACY_KEYS | {'accelerator_type'}}
    if (args.enable or args.disable) and not args.output:
        parser.error('Changes require --output to a new file; the input is never modified')
    if (args.preparation or any(fields.values())) and not args.enable:
        parser.error('Preparation settings require --enable')
    if args.enable:
        if not args.preparation:
            parser.error('--enable requires an explicit --preparation')
        if any(fields.values()):
            if not all(fields.values()):
                parser.error('Supply all four preparation fields together')
            config['preparations'][args.preparation] = fields
        config['targets'][args.enable] = dict(managed=True, preparation=args.preparation)
    if args.disable:
        config['targets'][args.disable]['managed'] = False
    config = normalize(config)
    if args.output:
        with Path(args.output).open('x') as output:
            output.write(json.dumps(config, indent=2) + '\n')
    print('Managed targets: ' + ', '.join(enabled(config)))
    print('Declared quota accounting passed; unknown project allocations are not included.')


if __name__ == '__main__':
    main()
