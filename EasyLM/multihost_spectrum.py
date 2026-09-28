"""One CPU Lanczos controller; all hosts participate in each device product."""
import json

import jax
import numpy as np
from jax.experimental import multihost_utils


def _broadcast_json(value):
    """Broadcast the small, JSON-compatible estimator report (never the basis)."""
    source = jax.process_index() == 0
    payload = json.dumps(value).encode('utf-8') if source else b''
    size = int(multihost_utils.broadcast_one_to_all(
        np.asarray(len(payload), dtype=np.int32)))
    data = (np.frombuffer(payload, dtype=np.uint8).copy() if source else
            np.zeros(size, dtype=np.uint8))
    return json.loads(multihost_utils.broadcast_one_to_all(data).tobytes())


def run_spectrum(estimate, apply_operator, dimension, **options):
    """Run the unchanged estimator on host 0, with collective operator calls.

    Other hosts hold only the current vector, not a Lanczos basis. Host 0
    controls every product and stop decision. CPU estimator failures (including
    memory preflight) release the waiting hosts; device/collective failures
    still rely on JAX distributed failure handling.
    """
    if jax.process_count() == 1:
        return estimate(apply_operator, dimension, **options)

    def command(value):
        return int(multihost_utils.broadcast_one_to_all(
            np.asarray(value, dtype=np.int32)))

    def product(vector):
        vector = multihost_utils.broadcast_one_to_all(vector)
        return apply_operator(vector)

    if jax.process_index() == 0:
        def controlled_product(vector):
            # Validate before telling peers to enter the vector collective.
            vector = np.asarray(vector, dtype=np.float32).reshape(dimension)
            command(1)
            return product(vector)

        try:
            result = estimate(controlled_product, dimension, **options)
        except Exception as error:
            command(2)
            _broadcast_json(f'{type(error).__name__}: {error}')
            raise
        command(0)
        # Normalize JSON's integer dictionary keys on all hosts identically.
        return _broadcast_json(result)

    buffer = np.zeros(dimension, dtype=np.float32)
    while True:
        action = command(0)
        if action == 0:
            return _broadcast_json(None)
        if action == 2:
            raise RuntimeError('Spectrum controller failed: ' + _broadcast_json(None))
        if action != 1:
            raise RuntimeError(f'Unknown spectrum controller command: {action}')
        product(buffer)


def gather_product(product):
    """Fetch global leaves without adding a host axis to local/replicated ones."""
    return jax.tree.map(
        lambda leaf: (multihost_utils.process_allgather(leaf)
                      if not leaf.is_fully_addressable else jax.device_get(leaf)),
        product)
