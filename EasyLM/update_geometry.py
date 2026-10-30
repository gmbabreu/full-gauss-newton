"""CPU utilities for streamed update-geometry bundles (no model or JAX imports)."""
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path

import msgpack
import numpy as np


FORMAT_VERSION = 1
DIRECTION_FILE = 'direction.msgpack'
PRODUCT_FILE = 'g_direction.msgpack'
METADATA_FILE = 'metadata.json'


def fingerprint_leaves(leaves):
    """SHA256 of ordered (path, array) leaves, including shape and native dtype.

    Callers supply a stable path order. Arrays are consumed one leaf at a time;
    memoryview avoids copying a whole model (or a contiguous leaf) into bytes.
    """
    digest = hashlib.sha256()
    structure = []
    for path, value in leaves:
        value = np.asarray(value)
        entry = dict(path=list(path), shape=list(value.shape), dtype=str(value.dtype))
        header = json.dumps(entry, sort_keys=True, separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        # View bytes first: Python's buffer protocol does not understand BF16.
        digest.update(memoryview(np.ascontiguousarray(value).view(np.uint8)).cast('B'))
        structure.append(entry)
    return dict(sha256=digest.hexdigest(), leaves=structure)


def save_parameter_tree(tree, path, gather_fns, *, writer):
    """Reuse checkpoint serialization; every host enters every leaf gather.

    Nonwriters execute precisely the writer's serializer traversal without
    opening any file. Gather functions must replicate global leaves before
    transferring them to CPU, as make_shard_and_gather_fns does.
    """
    from flax.serialization import to_state_dict
    from flax.traverse_util import flatten_dict
    if writer:
        from EasyLM.checkpoint import StreamingCheckpointer
        StreamingCheckpointer.save_train_state_to_file(
            tree, path, gather_fns, float_dtype='fp32')
    else:
        gathers = flatten_dict(to_state_dict(gather_fns))
        for key, value in flatten_dict(to_state_dict(tree)).items():
            gathers[key](value)


def _restore_fp32_array(code, data):
    # Flax ndarray extension, emitted by StreamingCheckpointer.to_bytes.
    # This deliberately supports only the FP32 leaves in this bundle format,
    # keeping the comparison command independent of Flax, JAX, and TPU runtimes.
    if code != 1:
        raise ValueError(f'Expected Flax ndarray extension, got {code}')
    shape, dtype, buffer = msgpack.unpackb(data, raw=True)
    if dtype != b'float32':
        raise ValueError(f'Expected FP32 export, got {dtype!r}')
    return np.frombuffer(buffer, dtype=np.float32).reshape(shape)


def iter_parameter_leaves(path):
    """Read the existing streaming checkpoint format without loading a tree."""
    with open(path, 'rb') as stream:
        for path, payload in msgpack.Unpacker(stream, raw=False, max_buffer_size=0):
            value = msgpack.unpackb(payload, ext_hook=_restore_fp32_array, raw=False)
            if not isinstance(value, np.ndarray):
                raise ValueError('Export leaf must be an FP32 ndarray')
            yield tuple(path), value


def read_metadata(bundle):
    with open(Path(bundle) / METADATA_FILE) as stream:
        return json.load(stream)


def validate_compatible(left, right):
    for metadata in (left, right):
        if metadata.get('format_version') != FORMAT_VERSION:
            raise ValueError('Unsupported or missing update-geometry format version')
        for key in ('base_parameters', 'solve_batch'):
            fingerprint = metadata.get(key, {})
            if not fingerprint.get('sha256') or not fingerprint.get('leaves'):
                raise ValueError(f'Missing {key} fingerprint/structure')
        if not metadata.get('parameters') or not metadata.get('gn_operator'):
            raise ValueError('Missing parameter structure or GN operator settings')
    for key in ('base_parameters', 'solve_batch', 'parameters', 'gn_operator'):
        if left[key] != right[key]:
            raise ValueError(f'Incompatible exports: {key} mismatch')


def _ratio(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return None
    return numerator / denominator


def compare_bundles(left_bundle, right_bundle):
    """Two passes over leaves, with float64 CPU dots and bounded live storage."""
    left, right = read_metadata(left_bundle), read_metadata(right_bundle)
    validate_compatible(left, right)
    expected = left['parameters']

    def leaves():
        streams = [iter_parameter_leaves(Path(bundle) / filename)
                   for bundle in (left_bundle, right_bundle)
                   for filename in (DIRECTION_FILE, PRODUCT_FILE)]
        count = 0
        for row in zip_longest(*streams):
            if any(item is None for item in row) or count >= len(expected):
                raise ValueError('Export parameter leaf count mismatch')
            entry = expected[count]
            for path, value in row:
                if (list(path) != entry['path'] or list(value.shape) != entry['shape']
                        or str(value.dtype) != 'float32' or not np.all(np.isfinite(value))):
                    raise ValueError('Export parameter path, shape, dtype or finite-value mismatch')
            count += 1
            yield row[0][0], [value.astype(np.float64) for _, value in row]
        if count != len(expected):
            raise ValueError('Export parameter leaf count mismatch')

    def dot(a, b):
        return float(np.einsum('i,i->', a.reshape(-1), b.reshape(-1)))

    sums = dict(a_a=0., c_c=0., a_c=0., a_Ga=0., c_Gc=0., a_Gc=0., c_Ga=0.)
    matrices = []
    for path, (a, ga, c, gc) in leaves():
        aa, cc, ac = dot(a, a), dot(c, c), dot(a, c)
        for key, value in dict(a_a=aa, c_c=cc, a_c=ac, a_Ga=dot(a, ga),
                               c_Gc=dot(c, gc), a_Gc=dot(a, gc), c_Ga=dot(c, ga)).items():
            sums[key] += value
        if a.ndim == 2:
            matrices.append(dict(path='/'.join(path), shape=list(a.shape),
                                 euclidean_cosine=_ratio(ac, math.sqrt(aa) * math.sqrt(cc))))
    projection = _ratio(sums['a_c'], sums['a_a'])
    e_e = e_ge = None
    if projection is not None:
        # Explicit residual leaves avoid cancellation in cc - ac**2/aa.
        e_e = e_ge = 0.
        for _, (a, ga, c, gc) in leaves():
            e = c - projection * a
            ge = gc - projection * ga
            e_e += dot(e, e)
            e_ge += dot(e, ge)
    gn_denominator = (math.sqrt(sums['a_Ga']) * math.sqrt(sums['c_Gc'])
                      if sums['a_Ga'] > 0 and sums['c_Gc'] > 0 else 0.)
    discrepancy = abs(sums['a_Gc'] - sums['c_Ga'])
    result = dict(
        euclidean_cosine=_ratio(sums['a_c'], math.sqrt(sums['a_a']) * math.sqrt(sums['c_c'])),
        gn_cosine=_ratio(sums['a_Gc'], gn_denominator),
        cross_symmetry_absolute_discrepancy=discrepancy,
        cross_symmetry_relative_discrepancy=_ratio(
            discrepancy, max(abs(sums['a_Gc']), abs(sums['c_Ga']))),
        rayleigh_a=_ratio(sums['a_Ga'], sums['a_a']),
        rayleigh_c=_ratio(sums['c_Gc'], sums['c_c']),
        projection_coefficient=projection,
        difference_norm=math.sqrt(e_e) if e_e is not None else None,
        difference_e_G_e=e_ge,
        difference_rayleigh=_ratio(e_ge, e_e) if e_e is not None else None,
        undefined_policy='null means undefined: zero norm, nonpositive GN denominator, or nonfinite result',
        **sums)
    return result, matrices
