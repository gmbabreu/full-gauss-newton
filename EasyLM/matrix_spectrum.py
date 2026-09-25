"""CPU-resident thick-restarted Lanczos with adaptive reorthogonalization.

Only Q is stored. The small symmetric recurrence matrix becomes an arrowhead
at thick restart; the retained coupling is beta * Y[-1, :keep]. Residuals from
this recurrence screen convergence, and fresh products validate every Ritz
value published as a scalar metric.
"""
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
    """Account for one Lanczos basis, validation blocks and bounded temporaries."""
    vector = dimension * np.dtype(np.float32).itemsize
    basis_buffers = max_basis * vector
    active = (2 * block_size + 3) * vector
    coordinate_chunk = min(dimension, 1 << 20)
    # Peak: FP64 basis conversion plus retained restart output coexist.
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


def _gram(q, count, *, chunk=1 << 20):
    result = np.zeros((count, count), np.float64)
    for start in range(0, q.shape[1], chunk):
        stop = min(q.shape[1], start + chunk)
        block = q[:count, start:stop].astype(np.float64)
        result += block @ block.T
    return result


def _norm(vector):
    # FP64 accumulation without a parameter-sized FP64 allocation.
    return float(np.sqrt(np.einsum('i,i->', vector, vector, dtype=np.float64)))


def _reorthogonalize(value, basis):
    """One FP32 overlap scan; correct only when orthogonality is at risk.

    Two corrective passes are used when triggered. This is adaptive full
    reorthogonalization, not a claim to implement a PRO error-bound recurrence.
    """
    norm = _norm(value)
    coefficients = basis @ value
    threshold = 8 * np.finfo(np.float32).eps * norm
    if _norm(coefficients) > threshold:
        value -= coefficients @ basis
        value -= (basis @ value) @ basis
    return value


def estimate_top_spectrum(apply_operator, dimension, *, top_k=100, block_size=4,
                          max_basis=160, restart_keep=120, max_products=600,
                          residual_tol=.01, stability_tol=.02, seed=0,
                          progress=None):
    """Leading algebraic eigenvalues of a fixed symmetric PSD operator.

    block_size is the small projected-eigensolve cadence. max_products includes
    fresh validation of all published ranks. Failure never publishes scalars.
    """
    if not 0 < top_k <= restart_keep < max_basis or dimension < top_k:
        raise ValueError('require 0 < top_k <= restart_keep < max_basis and dimension >= top_k')
    if block_size <= 0 or max_products < top_k + 3:
        raise ValueError('invalid block size or GN-product budget')
    if not (0 < residual_tol < 1 and 0 <= stability_tol < 1):
        raise ValueError('invalid spectrum tolerances')
    max_basis = min(max_basis, dimension)
    memory = memory_preflight(dimension, max_basis, block_size)
    started, rng = time.monotonic(), np.random.default_rng(seed)
    q = np.empty((max_basis, dimension), np.float32)
    h = np.zeros((max_basis, max_basis), np.float64)
    validation_ranks = tuple(sorted({1, top_k, *range(10, top_k + 1, 10)}))
    expansion_budget = max_products - len(validation_ranks)
    count = products = restart_count = 0
    previous = None
    stable = accepted = False
    failure, direct = [], {}
    candidate_values = candidate_residuals = vectors = None
    orthogonality_error = math.inf
    timings = {name: 0. for name in (
        'operator', 'orthogonalization', 'gram', 'projection', 'eigensolve',
        'ritz_residuals', 'expansion', 'restart', 'validation')}

    def random_direction():
        for _ in range(8):
            value = rng.standard_normal(dimension, dtype=np.float32)
            value = _reorthogonalize(value, q[:count])
            norm = _norm(value)
            if norm > 1e-6:
                return value / np.float32(norm)
        return None

    def apply(value):
        nonlocal products
        timer = time.monotonic()
        image = np.asarray(apply_operator(value.copy()), np.float32)
        timings['operator'] += time.monotonic() - timer
        products += 1
        if progress and products % 10 == 0:
            progress(products, max_products)
        if image.shape != (dimension,) or not np.all(np.isfinite(image)):
            raise ValueError('invalid_operator_product')
        return image.copy()

    next_vector = random_direction()
    while products < expansion_budget:
        q[count] = next_vector
        try:
            w = apply(next_vector)
        except ValueError as error:
            if str(error) != 'invalid_operator_product': raise
            failure.append(str(error)); break
        timer = time.monotonic()
        image_norm = _norm(w)
        # Three-term recurrence except for the first step after thick restart.
        coupling = h[:count, count]
        nonzero = np.flatnonzero(coupling)
        if len(nonzero) == 1:
            index = nonzero[0]
            w -= np.float32(coupling[index]) * q[index]
        elif len(nonzero) > 1:
            w -= coupling.astype(np.float32) @ q[:count]
        alpha = float(np.einsum('i,i->', next_vector, w, dtype=np.float64))
        h[count, count] = alpha
        w -= np.float32(alpha) * next_vector
        timings['expansion'] += time.monotonic() - timer
        timer = time.monotonic()
        w = _reorthogonalize(w, q[:count + 1])
        beta = _norm(w)
        timings['orthogonalization'] += time.monotonic() - timer
        count += 1
        breakdown = beta <= 32 * np.finfo(np.float32).eps * max(image_norm, 1e-30)
        check = (count >= top_k and (products % block_size == 0 or breakdown
                 or count == max_basis or products == expansion_budget))
        if check:
            timer = time.monotonic()
            values, vectors = np.linalg.eigh(h[:count, :count])
            values, vectors = values[::-1], vectors[:, ::-1]
            timings['eigensolve'] += time.monotonic() - timer
            candidate_values = values[:top_k].copy()
            timer = time.monotonic()
            candidate_residuals = beta * np.abs(vectors[-1, :top_k]) / np.maximum(
                np.abs(candidate_values), np.finfo(np.float64).eps)
            stable = previous is not None and bool(np.all(
                np.abs(candidate_values - previous) / np.maximum(
                    np.abs(candidate_values), 1e-30) <= stability_tol))
            # A complete orthonormal basis has no unexplored subspace.
            stable = stable or count == dimension
            previous = candidate_values.copy()
            timings['ritz_residuals'] += time.monotonic() - timer
            if (stable and np.all(candidate_values > 0)
                    and np.all(np.isfinite(candidate_values))
                    and np.all(candidate_residuals <= residual_tol)):
                accepted = True
                break
        if products == expansion_budget:
            break
        if count == dimension:
            failure.append('acceptance_checks_failed'); break
        if breakdown:
            timer = time.monotonic()
            next_vector = random_direction()
            timings['orthogonalization'] += time.monotonic() - timer
            beta = 0.
            if next_vector is None:
                failure.append('basis_breakdown'); break
        else:
            next_vector = w / np.float32(beta)
        if count == max_basis:
            timer = time.monotonic()
            keep = min(restart_keep, count - 1)
            _transform_in_place(q, count, vectors[:, :keep])
            h.fill(0)
            h[np.arange(keep), np.arange(keep)] = values[:keep]
            h[:keep, keep] = h[keep, :keep] = beta * vectors[-1, :keep]
            count = keep
            restart_count += 1
            timings['restart'] += time.monotonic() - timer
        else:
            h[count - 1, count] = h[count, count - 1] = beta

    if accepted:
        timer = time.monotonic()
        orthogonality_error = float(np.linalg.norm(
            _gram(q, count) - np.eye(count), ord=np.inf))
        timings['gram'] += time.monotonic() - timer
        if not math.isfinite(orthogonality_error) or orthogonality_error > 1e-3:
            accepted = False
            failure.append('invalid_orthonormal_basis')
        else:
            # Batched reconstruction of only the scalar ranks we publish.
            for start in range(0, len(validation_ranks), block_size):
                timer = time.monotonic()
                ranks = validation_ranks[start:start + block_size]
                indices = np.asarray(ranks, dtype=np.int64) - 1
                u = np.empty((len(ranks), dimension), np.float32)
                _transform_chunked(q[:count], vectors[:, indices], u)
                timings['validation'] += time.monotonic() - timer
                for rank, index, vector in zip(ranks, indices, u):
                    try:
                        image = apply(vector)
                    except ValueError as error:
                        if str(error) != 'invalid_operator_product': raise
                        failure.append(str(error)); accepted = False; break
                    timer = time.monotonic()
                    norm = _norm(vector)
                    residual = _norm(image - np.float32(candidate_values[index]) * vector) / max(
                        abs(candidate_values[index]) * norm, np.finfo(np.float64).eps)
                    direct[rank] = residual
                    accepted &= math.isfinite(residual) and residual <= residual_tol
                    timings['validation'] += time.monotonic() - timer
                if failure: break
            if not accepted and not failure:
                failure.append('direct_residual_failed')
    if not accepted and not failure:
        failure.append('product_budget_exhausted')
    scalars = dict(accepted=bool(accepted), gn_products=products,
                   seconds=time.monotonic() - started,
                   restart_count=restart_count, basis_size=count,
                   orthogonality_error=orthogonality_error,
                   max_relative_ritz_residual=(float(np.max(candidate_residuals))
                       if candidate_residuals is not None else None),
                   memory_required_gib=memory['required_bytes'] / 2**30)
    scalars.update({f'seconds_{name}': value for name, value in timings.items()})
    # Keep legacy absent-rank fields for dashboard compatibility.
    for rank in sorted({1, 10, 100, top_k, *range(10, top_k + 1, 10)}):
        scalars[f'lambda_{rank}_est'] = (float(candidate_values[rank - 1])
            if accepted and rank <= top_k else None)
    for rank in (10, 100):
        endpoint = scalars[f'lambda_{rank}_est']
        scalars[f'top{rank}_condition_est'] = (
            scalars['lambda_1_est'] / endpoint if endpoint is not None else None)
    table = dict(values=candidate_values.tolist() if candidate_values is not None else [],
                 residuals=candidate_residuals.tolist() if candidate_residuals is not None else [],
                 direct_residuals=direct, failure_reasons=failure,
                 orthogonality_error=orthogonality_error,
                 restart_count=restart_count, memory_preflight=memory)
    return scalars, table
