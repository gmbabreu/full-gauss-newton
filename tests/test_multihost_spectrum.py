"""Real two-process CPU collectives, dense ground truth, and controller failures."""
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get('RUN_MULTIHOST_TESTS') != '1',
                    reason='opt-in real multi-process CPU transport test')
def test_two_process_spectrum(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    use_mpi = os.environ.get('SPECTRUM_TEST_MPI') == '1'
    env.update(JAX_PLATFORMS='cpu', JAX_CPU_COLLECTIVES_IMPLEMENTATION=('mpi' if use_mpi else 'gloo'),
               XLA_FLAGS='--xla_force_host_platform_device_count=2',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    root = Path(__file__).resolve().parents[1]
    env['PYTHONPATH'] = str(root)
    processes = []
    logs = []
    try:
        for rank in range(1 if use_mpi else 2):
            path = tmp_path / f'worker-{rank}.log'
            logs.append(path)
            with path.open('w') as log:
                command = [sys.executable, str(Path(__file__).resolve()),
                           '--worker', str(rank), str(port)]
                if use_mpi:
                    command = ['mpiexec', '-n', '2', *command]
                processes.append(subprocess.Popen(
                    command,
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=180)
        output = '\n'.join(path.read_text() for path in logs)
        assert all(p.returncode == 0 for p in processes), output
        assert output.count('MULTIHOST_OK') == 2, output
        print(output)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


def test_four_local_devices(tmp_path):
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=4',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--local', '0', '0'],
        env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'LOCAL_SHARDS_OK' in result.stdout
    print(result.stdout)


def worker(rank, port, distributed=True):
    # Limit CPU thread pools even on large shared CI hosts, before JAX imports.
    if hasattr(os, 'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(cpus[rank * 2:rank * 2 + 2] or cpus[:2]))
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax.experimental import multihost_utils

    if distributed:
        jax.distributed.initialize(coordinator_address=f'127.0.0.1:{port}',
                                   num_processes=2, process_id=rank,
                                   initialization_timeout=60)
    from EasyLM import matrix_spectrum
    from EasyLM.condition_diagnostics import ConditionDiagnostics
    from EasyLM.multihost_spectrum import run_spectrum, gather_product
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ('host', 'local'))
    shards = NamedSharding(mesh, P('host'))
    replicated = NamedSharding(mesh, P())
    def put(value, sharding=shards):
        return jax.make_array_from_callback(value.shape, sharding,
                                            lambda index: value[index])
    rng = np.random.default_rng(42)
    left, _ = np.linalg.qr(rng.normal(size=(160, 128)))
    right, _ = np.linalg.qr(rng.normal(size=(128, 128)))
    eigenvalues = np.linspace(3., 1., 128)
    eigenvalues[0], eigenvalues[-1] = 10., .1  # resolved endpoint controls
    jacobian = ((left * np.sqrt(eigenvalues)) @ right.T).astype(np.float32)
    exact = np.linalg.eigvalsh(jacobian.astype(np.float64).T @ jacobian)[::-1]
    distributed_j = put(jacobian)
    params = {'w': put(np.ones(128, np.float32))}
    original = gather_product(params)['w'].copy()
    @jax.jit
    def matvec(vector):
        return distributed_j.T @ (distributed_j @ vector)
    calls = []
    def apply_g(params, batch, vector, wd):
        calls.append(1)
        return {'w': jax.device_put(matvec(vector['w']), shards)}
    flags = SimpleNamespace(inner_loop_wd=0., spectrum_top_k=100,
        spectrum_check_every=4, spectrum_max_basis=128, spectrum_restart_keep=100,
        spectrum_max_gn_products=400, spectrum_residual_tol=1e-3,
        spectrum_stability_tol=1e-3, spectrum_seed=0,
        spectrum_max_validation_attempts=3, spectrum_max_seconds=3600,
        spectrum_endpoint_maxiter=150, spectrum_endpoint_num_starts=2,
        spectrum_endpoint_agreement_tol=1e-3, spectrum_endpoint_residual_tol=1e-3,
        spectrum_inverse_cg_maxiter=128, spectrum_inverse_cg_tol=1e-5)
    controller = ConditionDiagnostics(config=flags, apply_g=apply_g,
                                      param_shards={'w': put})
    estimator = matrix_spectrum.estimate_top_spectrum
    captured = []
    def checked_estimator(*args, **kwargs):
        assert rank == 0, 'non-controller allocated a Lanczos basis'
        result = estimator(*args, **kwargs)
        captured.append(result)
        return result
    with mesh, patch.object(matrix_spectrum, 'available_host_memory', return_value=10**15), \
            patch.object(matrix_spectrum, 'estimate_top_spectrum', checked_estimator):
        report = controller.run(params, {}, step=0)
        assert report['spectrum/G/accepted'], report
        assert report['spectrum/G/gn_products'] == len(calls)
        for k in (1, 10, 100):
            np.testing.assert_allclose(report[f'spectrum/G/lambda_{k}_est'],
                                       exact[k - 1], rtol=1e-3)
        if rank == 0:
            values = captured[0][1]['values'][:100]
            error = float(np.max(np.abs(np.asarray(values) / exact[:100] - 1)))
            np.testing.assert_allclose(values, exact[:100], rtol=1e-3)
            print('TOP100_MAX_REL_ERROR', error, flush=True)
        # Both hosts receive identical scientific metrics and product counts.
        scientific = {k: v for k, v in report.items() if '/seconds' not in k}
        encoded = np.frombuffer(json.dumps(scientific, sort_keys=True).encode(), np.uint8)
        multihost_utils.assert_equal(encoded)
        np.testing.assert_array_equal(gather_product(params)['w'], original)
        # A second invocation exercises exhausted-budget fallback and A endpoints.
        flags.spectrum_top_k = 2
        flags.spectrum_max_gn_products = 5
        calls.clear()
        report = controller.run(params, {}, step=1, cg_diagonal=params,
                                effective_lambda=.3, safe_adam_lr=.7)
        assert not report['spectrum/G/accepted']
        assert report['spectrum/G/fallback_resolved'], report
        assert report['spectrum/G/lambda_1_from_fallback']
        np.testing.assert_allclose(report['spectrum/G/lambda_1_est'], exact[0], rtol=.003)
        assert report['spectrum/A/resolved'], report
        np.testing.assert_allclose(
            [report['spectrum/A/lambda_min_est'], report['spectrum/A/lambda_max_est']],
            [.3 * exact[-1] + 1., .3 * exact[0] + 1.], rtol=.003)
        # Fetching replicated leaves must not append a host axis.
        gathered = gather_product({'r': put(np.arange(4, dtype=np.float32), replicated)})
        np.testing.assert_array_equal(gathered['r'], np.arange(4))
        # Failures before and after a product release peers; subsequent calls work.
        for after_product in (False, True):
            def fail(apply, dimension, **kwargs):
                if after_product:
                    apply(np.ones(dimension, np.float32))
                raise MemoryError('intentional preflight failure')
            def collective_apply(vector):
                return gather_product({'w': matvec(put(vector))})['w']
            try:
                run_spectrum(fail, collective_apply, 128)
            except (MemoryError, RuntimeError) as error:
                assert 'intentional preflight failure' in str(error)
            else:
                raise AssertionError('controller error was not propagated')
        multihost_utils.sync_global_devices('finished')
    print('MULTIHOST_OK' if distributed else 'LOCAL_SHARDS_OK', rank, flush=True)
    if distributed:
        jax.distributed.shutdown()


if __name__ == '__main__':
    rank = int(os.environ.get('PMI_RANK', sys.argv[2]))
    worker(rank, int(sys.argv[3]), distributed=sys.argv[1] != '--local')
