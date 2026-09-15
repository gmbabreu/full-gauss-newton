"""Durable two-slot CG bundles. The marker is the sole commit record."""
import hashlib
import json
import os
import pickle
import re
import tempfile
import uuid
from contextlib import contextmanager


def paths(state_path):
    match = re.fullmatch(r'cg_state_([01])', os.path.basename(state_path))
    if not match:
        raise ValueError('cg_resume_state must name cg_state_0 or cg_state_1')
    slot = match[1]
    directory = os.path.dirname(state_path)
    return dict(state=state_path, **{name: os.path.join(directory,
        f'cg_{name}_{slot}.{"json" if name == "complete" else "pkl"}')
        for name in ('metadata', 'dataset', 'complete')})


def blob(path):
    from google.cloud import storage
    bucket, key = path[5:].split('/', 1)
    return storage.Client().bucket(bucket).blob(key)


@contextmanager
def reader(path):
    if path.startswith('gs://'):
        with blob(path).open('rb') as stream:
            yield stream
    else:
        with open(path, 'rb') as stream:
            yield stream


def digest(path):
    h = hashlib.sha256()
    with reader(path) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def durable_write(path, data):
    if path.startswith('gs://'):
        blob(path).upload_from_string(data)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read_bundle(state_path):
    p = paths(state_path)
    with reader(p['complete']) as stream:
        complete = json.load(stream)
    if complete.get('version') != 1 or not complete.get('complete'):
        raise ValueError('Incomplete CG checkpoint')
    for name in ('state', 'metadata', 'dataset'):
        if digest(p[name]) != complete['sha256'][name]:
            raise ValueError('CG checkpoint checksum mismatch: ' + name)
    with reader(p['metadata']) as stream:
        metadata = pickle.load(stream)
    with reader(p['dataset']) as stream:
        dataset = pickle.load(stream)
    if any(item.get('snapshot_id') != complete['snapshot_id']
           for item in (metadata, dataset)) or metadata['step'] != complete['step']:
        raise ValueError('CG checkpoint snapshot mismatch')
    progress = metadata['training_progress']
    if (dataset.get('packed_state_version') != 1
            or dataset['training_step'] != metadata['step']
            or progress['phase_completed_updates'] != metadata['step']
            or dataset['metadata']['dataset_total_tokens'] != progress['dataset_total_tokens']):
        raise ValueError('CG packed cursor/progress mismatch')
    return metadata, dataset, complete


def newest(directory):
    valid = []
    for slot in (0, 1):
        path = os.path.join(directory, f'cg_state_{slot}')
        try:
            _, _, marker = read_bundle(path)
            valid.append((marker['generation'], path))
        except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError):
            continue
        except Exception as error:
            # A missing GCS object is equivalent to a missing local slot.
            if getattr(error, 'code', None) != 404:
                raise
    return max(valid)[1] if valid else None


def validate_flags(saved, current):
    # Everything is trajectory-affecting unless explicitly operational/reporting.
    ignored = {'cg_resume_state', 'load_dataset_state', 'wandb_run_id', 'wandb_dir',
        'wandb_project', 'wandb_entity', 'output_dir', 'tmp_dir', 'experiment_id',
        'notes', 'logger', 'log_all_worker', 'save_model_freq', 'save_milestone_freq',
        'checkpointer', 'param_count', 'param_count_nonembed', 'training_progress',
        'log_time_offset_s'}
    changed = sorted(k for k in set(saved) | set(current)
                     if k not in ignored and saved.get(k) != current.get(k))
    if changed:
        raise ValueError('Resume changes trajectory settings: ' + ', '.join(changed))


def save_bundle(directory, previous, metadata, dataset, write_state, *, enable=True):
    """All hosts call write_state (for gathers); only the writer commits files."""
    if not enable:
        write_state('/dev/null')
        return previous
    generation = previous + 1
    p = paths(os.path.join(directory, f'cg_state_{generation % 2}'))
    # This durable invalidation precedes *every* write to the reused slot.
    durable_write(p['complete'], b'{"complete": false}')
    snapshot_id = uuid.uuid4().hex
    metadata = dict(metadata, snapshot_id=snapshot_id)
    dataset = dict(dataset, snapshot_id=snapshot_id, training_step=metadata['step'])
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, 'state')
        write_state(local)
        state_sha = digest(local)
        if p['state'].startswith('gs://'):
            blob(p['state']).upload_from_filename(local)
        else:
            # Stream the large state; fsync before publishing its completion marker.
            import shutil
            with open(local, 'rb') as source, open(p['state'], 'wb') as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
    metadata_bytes = pickle.dumps(metadata)
    dataset_bytes = pickle.dumps(dataset)
    durable_write(p['metadata'], metadata_bytes)
    durable_write(p['dataset'], dataset_bytes)
    marker = dict(version=1, complete=True, generation=generation,
                  snapshot_id=snapshot_id, step=metadata['step'],
                  sha256=dict(state=state_sha,
                      metadata=hashlib.sha256(metadata_bytes).hexdigest(),
                      dataset=hashlib.sha256(dataset_bytes).hexdigest()))
    durable_write(p['complete'], json.dumps(marker).encode())
    return generation
