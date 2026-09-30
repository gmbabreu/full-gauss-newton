"""Controller protocol checks; the transport is simulated, not a multi-host test."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from EasyLM import matrix_spectrum
from EasyLM import multihost_spectrum as transport


@pytest.mark.parametrize('failure',
                         [None, 'controlled', 'before_product', 'after_product'])
def test_controller_owns_estimator_and_releases_peers(failure):
    barrier = threading.Barrier(2, timeout=20)
    local = threading.local()
    shared = {}
    def broadcast(value):
        if local.rank == 0:
            shared['value'] = np.array(value, copy=True)
        barrier.wait()
        result = shared['value'].copy()
        barrier.wait()
        return result
    # Real JAX operator; only the inter-host transport is simulated here.
    rng = np.random.default_rng(12)
    jacobian = rng.normal(size=(160, 128)).astype(np.float32) / np.sqrt(160.)
    exact = np.linalg.eigvalsh(jacobian.astype(np.float64).T @ jacobian)[::-1]
    apply = jax.jit(lambda v: jnp.asarray(jacobian).T @ (jnp.asarray(jacobian) @ v))
    apply(jnp.ones(128)).block_until_ready()
    counts = [0, 0]
    def product(vector):
        counts[local.rank] += 1
        return np.asarray(apply(vector))
    def estimate(apply, dimension, **kwargs):
        assert local.rank == 0
        if failure == 'controlled':
            return matrix_spectrum.unavailable_spectrum_report(
                100, 'time budget', 128)[0], dict(
                    values=[], residuals=[], direct_residuals={},
                    failure_reasons=['time_budget_exhausted'],
                    orthogonality_error=None, restart_count=0,
                    memory_preflight=None)
        if failure:
            if failure == 'after_product':
                apply(np.ones(dimension, np.float32))
            raise MemoryError('test allocation failure')
        return matrix_spectrum.estimate_top_spectrum(apply, dimension, **kwargs)
    def run(rank):
        local.rank = rank
        try:
            return transport.run_spectrum(estimate, product, 128, top_k=100,
                check_every=4, max_basis=128, restart_keep=100, max_products=400,
                residual_tol=1e-3, stability_tol=1e-3)
        except (MemoryError, RuntimeError) as error:
            assert failure not in (None, 'controlled')
            assert 'test allocation failure' in str(error)
            return 'released'
    with patch.object(transport, 'jax', SimpleNamespace(
            process_count=lambda: 2, process_index=lambda: local.rank)), \
         patch.object(transport, 'multihost_utils', SimpleNamespace(
            broadcast_one_to_all=broadcast)), \
         patch.object(matrix_spectrum, 'available_host_memory', return_value=10**15), \
         ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, rank) for rank in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert results[0] == results[1]
    assert counts[0] == counts[1]
    if failure == 'controlled':
        scalars, table = results[0]
        assert not scalars['accepted']
        assert table['failure_reasons'] == ['time_budget_exhausted']
        assert counts == [0, 0]
    elif failure:
        assert results == ['released', 'released']
    else:
        scalars, table = results[0]
        assert scalars['accepted'], table
        assert counts[0] == scalars['gn_products']
        np.testing.assert_allclose(table['values'][:100], exact[:100], rtol=1e-3)
        print('top-100 max relative error:',
              np.max(np.abs(np.asarray(table['values'][:100]) / exact[:100] - 1)))
