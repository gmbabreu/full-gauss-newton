"""CPU-resident thick-restarted block Rayleigh--Ritz diagnostics.

The maintained decomposition is ``G Q = Q H + F C``.  At restart, leading
Ritz vectors ``U=QY`` are formed in coordinate chunks and their coupling is
preserved as ``G U = U Theta + F C Y``.  The retained ``GQ`` products are
transformed with the retained basis; they are not recomputed at restart.
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
    coordinate_chunk = min(dimension, 1 << 20)
    # Peak: FP64 views/copies for Q, GQ, Ritz U/AU and restart multiplication.
    chunk_workspace = (2 * max_basis + 2 * block_size + 2) * coordinate_chunk * 8
    projection = 4 * max_basis * max_basis * np.dtype(np.float64).itemsize
    transfer = 2 * vector
    required = (basis_buffers + active + projection + transfer
                + chunk_workspace + reserve_bytes)
    available = available_host_memory()
    if required > available:
        raise MemoryError(f'spectrum preflight requires {required / 2**30:.2f} GiB; '
                          f'effective available host memory is {available / 2**30:.2f} GiB')
    return dict(required_bytes=required, available_bytes=available,
                basis_buffer_bytes=basis_buffers,
                fp64_chunk_workspace_bytes=chunk_workspace,
                reserve_bytes=reserve_bytes)


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


def _transform_in_place(buffer, count, coefficients, *, chunk=1 << 20):
    """Apply coefficients.T to rows using only a bounded coordinate temporary."""
    keep = coefficients.shape[1]
    for start in range(0, buffer.shape[1], chunk):
        stop = min(buffer.shape[1], start + chunk)
        transformed = coefficients.T @ buffer[:count, start:stop].astype(np.float64)
        buffer[:keep, start:stop] = transformed.astype(np.float32)


def _residual_directions(q, gq, coefficients, values, *, chunk=1 << 20):
    """Form an ordered block of Ritz residuals with one basis scan per chunk."""
    directions = np.empty((len(values), q.shape[1]), np.float32)
    for start in range(0, q.shape[1], chunk):
        stop = min(q.shape[1], start + chunk)
        image = coefficients.T @ gq[:, start:stop].astype(np.float64)
        ritz = coefficients.T @ q[:, start:stop].astype(np.float64)
        image -= values[:, None] * ritz
        directions[:, start:stop] = image
    return directions


def _small_projection(q, gq, count, *, chunk=1 << 20):
    projection = np.zeros((count, count), np.float64)
    for start in range(0, q.shape[1], chunk):
        stop = min(q.shape[1], start + chunk)
        projection += q[:count, start:stop].astype(np.float64) @ \
            gq[:count, start:stop].astype(np.float64).T
    return .5 * (projection + projection.T)


def _gram(q, count, *, chunk=1 << 20):
    result = np.zeros((count, count), np.float64)
    for start in range(0, q.shape[1], chunk):
        stop = min(q.shape[1], start + chunk)
        block = q[:count, start:stop].astype(np.float64)
        result += block @ block.T
    return result


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
    q = np.empty((max_basis, dimension), np.float32)
    gq = np.empty_like(q)
    validation_ranks = tuple(rank for rank in (1, 10, 100) if rank <= top_k)
    expansion_budget = max_products - len(validation_ranks)
    count = products = 0
    previous = None
    stable = False
    failure = []
    candidate_values = candidate_residuals = vectors = None
    orthogonality_error = math.inf
    restart_count = 0
    timings = {name: 0. for name in (
        'operator', 'orthogonalization', 'gram', 'projection', 'eigensolve',
        'ritz_residuals', 'expansion', 'restart', 'validation')}

    def add_vector(vector):
        nonlocal count, products
        if products >= expansion_budget or count >= max_basis:
            return False
        timer = time.monotonic()
        vector = _orthogonalize(vector, q, count)
        if vector is None:
            vector = _replenish(rng, q, count)
        timings['orthogonalization'] += time.monotonic() - timer
        if vector is None:
            failure.append('basis_breakdown'); return False
        timer = time.monotonic()
        image = np.asarray(apply_operator(vector.copy()), np.float32)
        timings['operator'] += time.monotonic() - timer
        products += 1
        if image.shape != (dimension,) or not np.all(np.isfinite(image)):
            failure.append('invalid_operator_product'); return False
        q[count], gq[count] = vector, image
        count += 1
        if progress and products % 10 == 0: progress(products, max_products)
        return True

    for _ in range(min(block_size, expansion_budget)):
        if not add_vector(rng.choice(np.array([-1., 1.], np.float32), dimension)): break

    while count and not failure:
        timer = time.monotonic()
        gram = _gram(q, count)
        orthogonality_error = float(np.linalg.norm(gram - np.eye(count), ord=np.inf))
        timings['gram'] += time.monotonic() - timer
        if not np.all(np.isfinite(gram)) or orthogonality_error > 1e-3:
            failure.append('invalid_orthonormal_basis'); break
        timer = time.monotonic()
        projection = _small_projection(q, gq, count)
        timings['projection'] += time.monotonic() - timer
        timer = time.monotonic()
        values, vectors = np.linalg.eigh(projection)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        timings['eigensolve'] += time.monotonic() - timer
        take = min(top_k, count)
        candidate_values = values[:take].copy()
        timer = time.monotonic()
        residual_squared = np.zeros(take, np.float64)
        for offset in range(0, dimension, 1 << 20):
            end = min(dimension, offset + (1 << 20))
            u = vectors[:, :take].T @ q[:count, offset:end].astype(np.float64)
            au = vectors[:, :take].T @ gq[:count, offset:end].astype(np.float64)
            residual_squared += np.sum(
                (au - candidate_values[:, None] * u) ** 2, axis=1)
        candidate_residuals = np.sqrt(residual_squared) / np.maximum(
            np.abs(candidate_values), np.finfo(np.float64).eps)
        timings['ritz_residuals'] += time.monotonic() - timer
        stable = False
        if previous is not None and len(previous) >= top_k and take >= top_k:
            relative_change = (
                np.abs(candidate_values[:top_k] - previous[:top_k])
                / np.maximum(np.abs(candidate_values[:top_k]), 1e-30)
            )
            stable = bool(np.all(relative_change <= stability_tol))
        previous = candidate_values.copy()
        positive_ordered = (take >= top_k
            and np.all(np.isfinite(candidate_values[:top_k]))
            and np.all(candidate_values[:top_k] > 0)
            and np.all(candidate_values[:top_k - 1] >= candidate_values[1:top_k]))
        accepted_ritz = (positive_ordered and stable
                         and np.all(candidate_residuals[:top_k] <= residual_tol))
        if accepted_ritz: break
        if products >= expansion_budget:
            failure.append('product_budget_exhausted'); break

        # Target the least-converged wanted Ritz vectors, wherever they occur in
        # the requested spectrum (including the rank-top_k frontier).
        timer = time.monotonic()
        wanted = np.argsort(candidate_residuals)[
            -min(block_size, take):
        ][::-1]
        residual_directions = _residual_directions(
            q[:count], gq[:count], vectors[:, wanted], candidate_values[wanted])
        timings['expansion'] += time.monotonic() - timer

        if count + len(residual_directions) > max_basis:
            timer = time.monotonic()
            restart_count += 1
            keep = min(restart_keep, count)
            coefficients = vectors[:, :keep].copy()
            _transform_in_place(q, count, coefficients)
            _transform_in_place(gq, count, coefficients)
            count = keep
            # Q and GQ receive exactly the same Ritz transformation; because Y
            # is orthogonal, no independent reorthogonalization is necessary.
            restart_gram = _gram(q, count)
            timings['restart'] += time.monotonic() - timer
            if np.linalg.norm(restart_gram - np.eye(count), ord=np.inf) > 1e-3:
                failure.append('restart_orthogonality'); break

        added = 0
        for direction in residual_directions:
            if add_vector(direction): added += 1
            if products >= expansion_budget: break
        while added < block_size and products < expansion_budget and count < max_basis:
            if not add_vector(rng.choice(np.array([-1., 1.], np.float32), dimension)): break
            added += 1
        if added == 0:
            failure.append('incomplete_expansion'); break

    accepted = False
    direct = {}
    if (not failure and candidate_values is not None and len(candidate_values) >= top_k
            and stable and np.all(candidate_residuals[:top_k] <= residual_tol)):
        accepted = True
        # Coefficients and Q are still the exact pair used for the accepted Ritz solve.
        for rank in validation_ranks:
            timer = time.monotonic()
            coefficients = vectors[:, rank - 1]
            eigenvector = np.zeros(dimension, np.float32)
            _transform_chunked(q[:count], coefficients[:, None], eigenvector[None, :])
            timings['validation'] += time.monotonic() - timer
            timer = time.monotonic()
            image = np.asarray(apply_operator(eigenvector), np.float32); products += 1
            timings['operator'] += time.monotonic() - timer
            timer = time.monotonic()
            residual = np.linalg.norm((image - candidate_values[rank - 1] * eigenvector).astype(np.float64)) / \
                max(abs(candidate_values[rank - 1]), np.finfo(np.float64).eps)
            timings['validation'] += time.monotonic() - timer
            direct[rank] = float(residual)
            accepted &= math.isfinite(residual) and residual <= residual_tol
    if not accepted and not failure: failure.append('acceptance_checks_failed')
    scalars = dict(accepted=accepted, gn_products=products,
                   seconds=time.monotonic() - started,
                   restart_count=restart_count, basis_size=count,
                   orthogonality_error=orthogonality_error,
                   max_relative_ritz_residual=(
                       float(np.max(candidate_residuals[:top_k]))
                       if candidate_residuals is not None else None),
                   memory_required_gib=memory['required_bytes'] / 2**30)
    scalars.update({f'seconds_{name}': value for name, value in timings.items()})
    for rank in (1, 10, 100):
        scalars[f'lambda_{rank}_est'] = (float(candidate_values[rank - 1])
            if accepted and candidate_values is not None and len(candidate_values) >= rank else None)
    lambda_1 = scalars["lambda_1_est"]
    lambda_10 = scalars["lambda_10_est"]
    lambda_100 = scalars["lambda_100_est"]
    scalars["top10_condition_est"] = (
        lambda_1 / lambda_10
        if (accepted and lambda_1 is not None and lambda_10 is not None
            and 0 < lambda_10 <= lambda_1) else None)
    scalars["top100_condition_est"] = (
        lambda_1 / lambda_100
        if (
            accepted
            and lambda_1 is not None
            and lambda_100 is not None
            and 0 < lambda_100 <= lambda_1
        )
        else None
    )
    table = dict(values=(candidate_values.tolist() if candidate_values is not None else []),
                 residuals=(candidate_residuals.tolist() if candidate_residuals is not None else []),
                 direct_residuals=direct, failure_reasons=failure,
                 orthogonality_error=orthogonality_error,
                 restart_count=restart_count,
                 memory_preflight=memory)
    return scalars, table
