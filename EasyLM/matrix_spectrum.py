"""CPU-resident thick-restarted block Rayleigh--Ritz diagnostics.

The maintained decomposition is ``G Q = Q H + F C``.  At restart, leading
Ritz vectors ``U=QY`` are formed in coordinate chunks and their coupling is
preserved as ``G U = U Theta + F C Y``.  This implementation recomputes each
retained ``G U`` sequentially to form that coupling exactly at the new basis;
it never assumes that the coupling vanished to recover a three-term recurrence.
"""
from dataclasses import dataclass
import math
import os
import time

import numpy as np


def _cgroup_available():
    pairs = (('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
             ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
              '/sys/fs/cgroup/memory/memory.usage_in_bytes'))
    available = []
    for limit_path, usage_path in pairs:
        try:
            value = open(limit_path).read().strip()
            if value != 'max':
                usage = int(open(usage_path).read().strip())
                available.append(max(0, int(value) - usage))
        except (OSError, ValueError):
            pass
    return min(available) if available else None


def available_host_memory():
    pages = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    cgroup = _cgroup_available()
    return min(pages, cgroup) if cgroup is not None else pages


def memory_preflight(dimension, max_basis, block_size, *, reserve_bytes=8 << 30):
    """Account for two basis buffers, active blocks, coefficients and transfers."""
    vector = dimension * np.dtype(np.float32).itemsize
    basis_buffers = 2 * max_basis * vector
    active = (2 * block_size + 3) * vector
    projection = 3 * max_basis * max_basis * np.dtype(np.float64).itemsize
    transfer = 2 * vector
    required = basis_buffers + active + projection + transfer + reserve_bytes
    available = available_host_memory()
    if required > available:
        raise MemoryError(f'spectrum preflight requires {required / 2**30:.2f} GiB; '
                          f'effective available host memory is {available / 2**30:.2f} GiB')
    return dict(required_bytes=required, available_bytes=available,
                basis_buffer_bytes=basis_buffers, reserve_bytes=reserve_bytes)


def _orthogonalize(vector, basis, count, *, tolerance=1e-7, chunk=1 << 20):
    """Two-pass FP64-coefficient modified Gram--Schmidt."""
    value = np.asarray(vector, dtype=np.float32).copy()
    original = float(np.linalg.norm(value.astype(np.float64)))
    for _ in range(2):
        for start in range(0, count, 16):
            stop = min(count, start + 16)
            block = basis[start:stop]
            coefficients = np.zeros(stop - start, np.float64)
            for offset in range(0, value.size, chunk):
                end = min(value.size, offset + chunk)
                coefficients += block[:, offset:end].astype(np.float64) @ \
                    value[offset:end].astype(np.float64)
            value -= coefficients.astype(np.float32) @ block
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not math.isfinite(norm) or norm <= tolerance * max(original, 1.0):
        return None
    return value / np.float32(norm)


def _replenish(rng, basis, count, attempts=32):
    for _ in range(attempts):
        candidate = _orthogonalize(
            rng.choice(np.array([-1., 1.], np.float32), size=basis.shape[1]),
            basis, count)
        if candidate is not None: return candidate
    return None


def _transform_chunked(source, coefficients, destination, *, chunk=1 << 20):
    for start in range(0, source.shape[1], chunk):
        stop = min(source.shape[1], start + chunk)
        destination[:, start:stop] = (
            coefficients.T @ source[:, start:stop].astype(np.float64)).astype(np.float32)


def estimate_top_spectrum(apply_operator, dimension, *, top_k=100, block_size=4,
                          max_basis=160, restart_keep=120, max_products=600,
                          residual_tol=.01, stability_tol=.02, seed=0,
                          progress=None):
    """Estimate leading algebraic eigenvalues; unresolved output stays explicit."""
    if not 0 < top_k <= restart_keep < max_basis:
        raise ValueError('require 0 < top_k <= restart_keep < max_basis')
    if block_size <= 0 or max_products < top_k + 3:
        raise ValueError('invalid block size or GN-product budget')
    memory = memory_preflight(dimension, max_basis, block_size)
    started, rng = time.monotonic(), np.random.default_rng(seed)
    basis = np.empty((max_basis, dimension), np.float32)
    restart = np.empty_like(basis)
    count = products = 0
    previous = None
    stable = False
    failure = []
    candidate_values = candidate_residuals = None

    while products < max_products:
        target = min(max_basis, count + block_size)
        while count < target:
            vector = _replenish(rng, basis, count)
            if vector is None:
                failure.append('basis_breakdown'); break
            basis[count] = vector; count += 1
        if count == 0 or failure: break

        gram = np.zeros((count, count), np.float64)
        for offset in range(0, dimension, 1 << 20):
            end = min(dimension, offset + (1 << 20))
            block = basis[:count, offset:end].astype(np.float64)
            gram += block @ block.T
        if (not np.all(np.isfinite(gram))
                or np.linalg.norm(gram - np.eye(count), ord=np.inf) > 1e-3):
            failure.append('invalid_orthonormal_basis'); break

        # Form the FP64 Rayleigh--Ritz projection sequentially. Device products
        # never overlap; apply_operator must return a completed CPU FP32 vector.
        projection = np.zeros((count, count), np.float64)
        completed = 0
        for column in range(count):
            if products >= max_products: break
            image = np.asarray(apply_operator(basis[column].copy()), np.float32)
            products += 1
            if image.shape != (dimension,) or not np.all(np.isfinite(image)):
                failure.append('invalid_operator_product'); break
            restart[column] = image
            completed += 1
            coefficients = np.zeros(count, np.float64)
            for offset in range(0, dimension, 1 << 20):
                end = min(dimension, offset + (1 << 20))
                coefficients += basis[:count, offset:end].astype(np.float64) @ \
                    image[offset:end].astype(np.float64)
            projection[:, column] = coefficients
            if progress and products % 10 == 0: progress(products, max_products)
        if failure or completed != count: break
        projection = (projection + projection.T) * .5
        values, vectors = np.linalg.eigh(projection)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        take = min(top_k, count)
        candidate_values = values[:take].copy()
        candidate_residuals = np.empty(take, np.float64)
        # Residual norms from GQ-QH, accumulated without reconstructed-vector arrays.
        residual_squared = np.zeros(take, np.float64)
        for offset in range(0, dimension, 1 << 20):
            end = min(dimension, offset + (1 << 20))
            u = vectors[:, :take].T @ basis[:count, offset:end].astype(np.float64)
            au = vectors[:, :take].T @ restart[:count, offset:end].astype(np.float64)
            residual_squared += np.sum(
                (au - candidate_values[:, None] * u) ** 2, axis=1)
        candidate_residuals[:] = np.sqrt(residual_squared) / np.maximum(
            np.abs(candidate_values), np.finfo(np.float64).eps)
        if previous is not None and take >= top_k:
            stable = np.all(np.abs(candidate_values[:top_k] - previous[:top_k]) /
                np.maximum(np.abs(candidate_values[:top_k]), 1e-30) <= stability_tol)
        previous = candidate_values.copy()
        accepted_ritz = (take >= top_k and stable
                         and np.all(candidate_residuals[:top_k] <= residual_tol))
        if accepted_ritz or products >= max_products: break

        keep = min(restart_keep, count)
        _transform_chunked(basis[:count], vectors[:, :keep], restart[:keep])
        # Reorthogonalize retained vectors, consistently reducing rank if needed.
        retained = 0
        for index in range(keep):
            vector = _orthogonalize(restart[index], restart, retained)
            if vector is not None:
                restart[retained] = vector; retained += 1
        basis, restart = restart, basis
        count = retained
        # Residual coupling FCY is preserved by using direct G(U)-U*Theta
        # directions for replenishment on the next distinct expansion.
        target = min(max_basis, count + block_size)
        for index in range(min(count, block_size)):
            if count >= target or products >= max_products: break
            image = np.asarray(apply_operator(basis[index].copy()), np.float32)
            products += 1
            coupling = image - np.float32(candidate_values[index]) * basis[index]
            vector = _orthogonalize(coupling, basis, count)
            if vector is not None:
                basis[count] = vector; count += 1

    accepted = False
    direct = {}
    if candidate_values is not None and len(candidate_values) >= top_k and stable \
            and np.all(candidate_residuals[:top_k] <= residual_tol):
        accepted = True
        for rank in (1, 10, 100):
            if rank > top_k or products >= max_products:
                accepted = False; failure.append('validation_budget'); break
            coefficients = vectors[:, rank - 1]
            eigenvector = np.zeros(dimension, np.float32)
            _transform_chunked(basis[:count], coefficients[:, None], eigenvector[None, :])
            image = np.asarray(apply_operator(eigenvector), np.float32); products += 1
            residual = np.linalg.norm((image - candidate_values[rank - 1] * eigenvector).astype(np.float64)) / \
                max(abs(candidate_values[rank - 1]), np.finfo(np.float64).eps)
            direct[rank] = float(residual)
            accepted &= math.isfinite(residual) and residual <= residual_tol
    if not accepted and not failure: failure.append('acceptance_checks_failed')
    scalars = dict(accepted=accepted, gn_products=products,
                   seconds=time.monotonic() - started)
    for rank in (1, 10, 100):
        scalars[f'lambda_{rank}_est'] = (float(candidate_values[rank - 1])
            if accepted and candidate_values is not None and len(candidate_values) >= rank else None)
    scalars['top100_condition_est'] = (scalars['lambda_1_est'] / scalars['lambda_100_est']
        if accepted and scalars['lambda_100_est'] > 0
        and scalars['lambda_1_est'] >= scalars['lambda_100_est'] else None)
    table = dict(values=(candidate_values.tolist() if candidate_values is not None else []),
                 residuals=(candidate_residuals.tolist() if candidate_residuals is not None else []),
                 direct_residuals=direct, failure_reasons=failure,
                 memory_preflight=memory)
    return scalars, table
