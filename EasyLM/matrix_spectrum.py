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


def _memory_report(dimension, max_basis, check_every, reserve_bytes, available):
    """Account for one Lanczos basis, validation blocks and bounded temporaries."""
    vector = dimension * np.dtype(np.float32).itemsize
    basis_buffers = max_basis * vector
    active = (2 * check_every + 3) * vector
    coordinate_chunk = min(dimension, 1 << 20)
    # Peak: FP64 basis conversion plus retained restart output coexist.
    chunk_workspace = (2 * max_basis + 2 * check_every + 2) * coordinate_chunk * 8
    projection = 4 * max_basis * max_basis * np.dtype(np.float64).itemsize
    gram_cache = max_basis * max_basis * np.dtype(np.float64).itemsize
    transfer = 2 * vector
    required = (basis_buffers + active + projection + gram_cache + transfer
                + chunk_workspace + reserve_bytes)
    return dict(required_bytes=required, available_bytes=available,
                basis_buffer_bytes=basis_buffers,
                gram_cache_bytes=gram_cache,
                fp64_chunk_workspace_bytes=chunk_workspace,
                reserve_bytes=reserve_bytes)


def memory_preflight(dimension, max_basis, check_every, *, reserve_bytes=8 << 30,
                     available_bytes=None):
    """Check that one Lanczos basis and its bounded temporaries fit in RAM."""
    available = (available_host_memory() if available_bytes is None
                 else available_bytes)
    report = _memory_report(
        dimension, max_basis, check_every, reserve_bytes, available)
    required = report['required_bytes']
    if required > available:
        raise MemoryError(f'spectrum preflight requires {required / 2**30:.2f} GiB; '
                          f'effective available host memory is {available / 2**30:.2f} GiB')
    return report


def fit_basis_to_memory(dimension, max_basis, min_basis, check_every,
                        *, reserve_bytes=8 << 30):
    """Use the largest requested basis capacity that fits current host RAM."""
    available = available_host_memory()
    requested = max_basis
    if _memory_report(dimension, max_basis, check_every, reserve_bytes,
                      available)['required_bytes'] > available:
        low, high = min_basis, max_basis
        while low < high:
            middle = (low + high + 1) // 2
            required = _memory_report(
                dimension, middle, check_every, reserve_bytes,
                available)['required_bytes']
            if required <= available:
                low = middle
            else:
                high = middle - 1
        max_basis = low
    report = memory_preflight(
        dimension, max_basis, check_every, reserve_bytes=reserve_bytes,
        available_bytes=available)
    report.update(requested_max_basis=requested,
                  effective_max_basis=max_basis,
                  memory_limited=max_basis < requested)
    return max_basis, report


def unavailable_spectrum_report(top_k, reason, configured_max_basis):
    """Return the normal failure schema when even the minimum basis cannot fit."""
    scalars = dict(accepted=False, gn_products=0, seconds=0., restart_count=0,
                   basis_size=0, basis_capacity=0,
                   configured_max_basis=configured_max_basis,
                   memory_limited=True, validation_attempts=0,
                   orthogonality_error=None, max_direct_residual=None,
                   worst_direct_residual_rank=None,
                   max_relative_ritz_residual=None, memory_required_gib=None)
    for name in ('operator', 'orthogonalization', 'gram', 'projection',
                 'eigensolve', 'ritz_residuals', 'expansion', 'restart',
                 'validation'):
        scalars[f'seconds_{name}'] = 0.
    for rank in sorted({1, 10, 100, top_k, *range(10, top_k + 1, 10)}):
        scalars[f'lambda_{rank}_est'] = None
    scalars['top10_condition_est'] = None
    scalars['top100_condition_est'] = None
    table = dict(values=[], residuals=[], direct_residuals={},
                 failure_reasons=['insufficient_host_memory'],
                 orthogonality_error=None, restart_count=0,
                 memory_preflight=None, diagnostic_error=reason)
    return scalars, table


def _transform_chunked(source, coefficients, destination, *, chunk=1 << 20,
                       check_deadline=None, heartbeat=None, phase='transform'):
    for start in range(0, source.shape[1], chunk):
        if check_deadline:
            check_deadline()
        stop = min(source.shape[1], start + chunk)
        destination[:, start:stop] = (
            coefficients.T @ source[:, start:stop].astype(np.float64)).astype(np.float32)
        if check_deadline:
            check_deadline()
        if heartbeat:
            heartbeat(phase, stop, source.shape[1])


def _transform_in_place(buffer, count, coefficients, *, chunk=1 << 20,
                        check_deadline=None, heartbeat=None):
    """Apply coefficients.T to rows using only a bounded coordinate temporary."""
    keep = coefficients.shape[1]
    for start in range(0, buffer.shape[1], chunk):
        if check_deadline:
            check_deadline()
        stop = min(buffer.shape[1], start + chunk)
        transformed = coefficients.T @ buffer[:count, start:stop].astype(np.float64)
        buffer[:keep, start:stop] = transformed.astype(np.float32)
        if check_deadline:
            check_deadline()
        if heartbeat:
            heartbeat('restart transform', stop, buffer.shape[1])


def _incremental_gram(q, count, cache=None, cached_count=0, *, chunk=1 << 20,
                      check_deadline=None, heartbeat=None):
    """Extend an FP64 Gram cache without recomputing its old-old block."""
    if cache is None:
        cache = np.zeros((q.shape[0], q.shape[0]), np.float64)
        cached_count = 0
    if not 0 <= cached_count <= count or cache.shape[0] < count:
        raise ValueError('invalid Gram cache state')
    if cached_count == count:
        return cache, count
    for start in range(0, q.shape[1], chunk):
        if check_deadline:
            check_deadline()
        stop = min(q.shape[1], start + chunk)
        new = q[cached_count:count, start:stop].astype(np.float64)
        cache[cached_count:count, cached_count:count] += new @ new.T
        if cached_count:
            old = q[:cached_count, start:stop].astype(np.float64)
            cross = old @ new.T
            cache[:cached_count, cached_count:count] += cross
            cache[cached_count:count, :cached_count] += cross.T
        if check_deadline:
            check_deadline()
        if heartbeat:
            heartbeat('Gram', stop, q.shape[1])
    return cache, count


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


class _TimeBudgetExceeded(Exception):
    pass


def estimate_top_spectrum(apply_operator, dimension, *, top_k=100, check_every=4,
                          max_basis=160, restart_keep=120, max_products=600,
                          residual_tol=.01, stability_tol=.02, seed=0,
                          max_validation_attempts=3, max_seconds=3600,
                          progress=None, clock=time.monotonic,
                          heartbeat_seconds=30):
    """Leading algebraic eigenvalues of a fixed symmetric PSD operator.

    check_every is the small projected-eigensolve cadence. max_products includes
    fresh validation of all published ranks. A failed direct validation resumes
    expansion with exponential retry spacing. Failure never publishes scalars.
    max_seconds is a cooperative, soft host-controller deadline; zero disables it.
    """
    if not 0 < top_k <= restart_keep < max_basis or dimension < top_k:
        raise ValueError('require 0 < top_k <= restart_keep < max_basis and dimension >= top_k')
    if check_every <= 0 or max_products < top_k + 3:
        raise ValueError('invalid check cadence or GN-product budget')
    if max_validation_attempts <= 0 or int(max_validation_attempts) != max_validation_attempts:
        raise ValueError('max_validation_attempts must be a positive integer')
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('max_seconds must be finite and nonnegative')
    if not (0 < residual_tol < 1 and 0 <= stability_tol < 1):
        raise ValueError('invalid spectrum tolerances')
    started = clock()
    deadline = started + max_seconds if max_seconds else None
    max_basis = min(max_basis, dimension)
    max_basis, memory = fit_basis_to_memory(
        dimension, max_basis, restart_keep + 1, check_every)
    rng = np.random.default_rng(seed)
    q = np.empty((max_basis, dimension), np.float32)
    h = np.zeros((max_basis, max_basis), np.float64)
    gram_cache = np.zeros((max_basis, max_basis), np.float64)
    gram_cached_count = 0
    validation_ranks = tuple(sorted({1, top_k, *range(10, top_k + 1, 10)}))
    count = products = expansion_products = restart_count = 0
    previous = None
    stable = accepted = False
    failure, direct = [], {}
    validation_attempts = 0
    next_validation_expansion = top_k
    candidate_values = candidate_residuals = vectors = None
    orthogonality_error = math.inf
    last_heartbeat = started
    timings = {name: 0. for name in (
        'operator', 'orthogonalization', 'gram', 'projection', 'eigensolve',
        'ritz_residuals', 'expansion', 'restart', 'validation')}

    def elapsed(now=None):
        return (clock() if now is None else now) - started

    def check_deadline():
        if deadline is not None and clock() >= deadline:
            raise _TimeBudgetExceeded

    def heartbeat(phase, done, total):
        nonlocal last_heartbeat
        now = clock()
        if heartbeat_seconds >= 0 and now - last_heartbeat >= heartbeat_seconds:
            print(f'[spectrum] {phase}: {done}/{total}; products={products}; '
                  f'basis={count}/{max_basis}; elapsed={elapsed(now):.1f}s',
                  flush=True)
            last_heartbeat = now

    def progress_line(event, *, attempt=None, recurrence=None, direct_value=None,
                      worst_rank=None):
        fields = [f'[spectrum] {event}',
                  f'attempt={validation_attempts + 1 if attempt is None else attempt}',
                  f'products={products}', f'basis={count}/{max_basis}',
                  f'elapsed={elapsed():.1f}s',
                  f'recurrence_residual={recurrence if recurrence is not None else "n/a"}',
                  f'direct_residual={direct_value if direct_value is not None else "n/a"}']
        if worst_rank is not None:
            fields.append(f'worst_rank={worst_rank}')
        print(' '.join(fields), flush=True)

    def random_direction():
        for _ in range(8):
            check_deadline()
            value = rng.standard_normal(dimension, dtype=np.float32)
            value = _reorthogonalize(value, q[:count])
            norm = _norm(value)
            if norm > 1e-6:
                return value / np.float32(norm)
        return None

    def apply(value):
        nonlocal products
        if products >= max_products:
            raise RuntimeError('spectrum product budget exceeded')
        check_deadline()
        timer = clock()
        image = np.asarray(apply_operator(value.copy()), np.float32)
        timings['operator'] += clock() - timer
        products += 1
        if progress and products % 10 == 0:
            progress(products, max_products)
        if image.shape != (dimension,) or not np.all(np.isfinite(image)):
            raise ValueError('invalid_operator_product')
        check_deadline()
        return image.copy()

    def validate_candidate():
        """Freshly check published Ritz pairs; return False to keep expanding."""
        nonlocal orthogonality_error, direct, validation_attempts
        nonlocal gram_cache, gram_cached_count
        validation_attempts += 1
        recurrence = (float(np.max(candidate_residuals))
                      if candidate_residuals is not None else None)
        progress_line('validation start', attempt=validation_attempts,
                      recurrence=recurrence)
        check_deadline()
        timer = clock()
        try:
            gram_cache, new_cached_count = _incremental_gram(
                q, count, gram_cache, gram_cached_count,
                check_deadline=check_deadline, heartbeat=heartbeat)
        finally:
            timings['gram'] += clock() - timer
        orthogonality_error = float(np.linalg.norm(
            gram_cache[:count, :count] - np.eye(count), ord=np.inf))
        gram_cached_count = new_cached_count
        check_deadline()
        if not math.isfinite(orthogonality_error) or orthogonality_error > 1e-3:
            failure.append('invalid_orthonormal_basis')
            progress_line('validation end', attempt=validation_attempts,
                          recurrence=recurrence)
            return False

        attempt = {}
        valid = True
        for start in range(0, len(validation_ranks), check_every):
            check_deadline()
            timer = clock()
            ranks = validation_ranks[start:start + check_every]
            indices = np.asarray(ranks, dtype=np.int64) - 1
            u = np.empty((len(ranks), dimension), np.float32)
            try:
                _transform_chunked(q[:count], vectors[:, indices], u,
                                   check_deadline=check_deadline,
                                   heartbeat=heartbeat,
                                   phase='validation transform')
            finally:
                timings['validation'] += clock() - timer
            for rank, index, vector in zip(ranks, indices, u):
                try:
                    image = apply(vector)
                except ValueError as error:
                    if str(error) != 'invalid_operator_product':
                        raise
                    failure.append(str(error))
                    return False
                timer = clock()
                norm = _norm(vector)
                residual = _norm(
                    image - np.float32(candidate_values[index]) * vector
                ) / max(abs(candidate_values[index]) * norm,
                        np.finfo(np.float64).eps)
                attempt[rank] = residual
                valid &= math.isfinite(residual) and residual <= residual_tol
                timings['validation'] += clock() - timer
                check_deadline()
        # Publish only a complete attempt; deadline exits leave the prior result.
        direct = attempt
        worst_rank = max(attempt, key=attempt.get) if attempt else None
        progress_line('validation end', attempt=validation_attempts,
                      recurrence=recurrence,
                      direct_value=attempt.get(worst_rank) if worst_rank else None,
                      worst_rank=worst_rank)
        return bool(valid)

    try:
        next_vector = random_direction()
        while products < max_products:
            check_deadline()
            if products + 1 > max_products:
                break
            q[count] = next_vector
            try:
                w = apply(next_vector)
                expansion_products += 1
            except ValueError as error:
                if str(error) != 'invalid_operator_product':
                    raise
                failure.append(str(error)); break
            timer = clock()
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
            timings['expansion'] += clock() - timer
            check_deadline()
            timer = clock()
            w = _reorthogonalize(w, q[:count + 1])
            beta = _norm(w)
            timings['orthogonalization'] += clock() - timer
            check_deadline()
            count += 1
            breakdown = beta <= 32 * np.finfo(np.float32).eps * max(image_norm, 1e-30)
            validation_fits = products + len(validation_ranks) <= max_products
            final_validation_point = (count == max_basis or count == dimension
                                      or products + len(validation_ranks) == max_products)
            check = (count >= top_k and (
                     expansion_products % check_every == 0 or breakdown
                     or final_validation_point))
            if check:
                timer = clock()
                values, vectors = np.linalg.eigh(h[:count, :count])
                values, vectors = values[::-1], vectors[:, ::-1]
                timings['eigensolve'] += clock() - timer
                check_deadline()
                candidate_values = values[:top_k].copy()
                timer = clock()
                candidate_residuals = beta * np.abs(vectors[-1, :top_k]) / np.maximum(
                    np.abs(candidate_values), np.finfo(np.float64).eps)
                stable = previous is not None and bool(np.all(
                    np.abs(candidate_values - previous) / np.maximum(
                        np.abs(candidate_values), 1e-30) <= stability_tol))
                stable = stable or count == dimension
                previous = candidate_values.copy()
                timings['ritz_residuals'] += clock() - timer
                check_deadline()
                recurrence_passed = (stable and np.all(candidate_values > 0)
                        and np.all(np.isfinite(candidate_values))
                        and np.all(candidate_residuals <= residual_tol))
                validation_due = (expansion_products >= next_validation_expansion
                                  or final_validation_point)
                if recurrence_passed and validation_due and validation_fits:
                    accepted = validate_candidate()
                    if accepted or failure:
                        break
                    if validation_attempts >= max_validation_attempts:
                        failure.append('validation_attempt_budget_exhausted')
                        break
                    # Failed complete validations back off by 8, 16, ... cadences.
                    next_validation_expansion = (expansion_products
                        + (2 ** (validation_attempts + 2)) * check_every)
            if products + len(validation_ranks) >= max_products:
                break
            if count == dimension:
                failure.append('direct_residual_failed' if validation_attempts
                               else 'acceptance_checks_failed')
                break
            if breakdown:
                timer = clock()
                next_vector = random_direction()
                timings['orthogonalization'] += clock() - timer
                beta = 0.
                if next_vector is None:
                    failure.append('basis_breakdown'); break
            else:
                next_vector = w / np.float32(beta)
            if count == max_basis:
                recurrence = (float(np.max(candidate_residuals))
                              if candidate_residuals is not None else None)
                progress_line('basis restart', attempt=validation_attempts,
                              recurrence=recurrence,
                              direct_value=max(direct.values()) if direct else None,
                              worst_rank=max(direct, key=direct.get) if direct else None)
                timer = clock()
                keep = min(restart_keep, count - 1)
                try:
                    _transform_in_place(q, count, vectors[:, :keep],
                                        check_deadline=check_deadline,
                                        heartbeat=heartbeat)
                finally:
                    timings['restart'] += clock() - timer
                # Stored rows were rounded to FP32; rebuild from those rows later.
                gram_cache.fill(0.)
                gram_cached_count = 0
                h.fill(0)
                h[np.arange(keep), np.arange(keep)] = values[:keep]
                h[:keep, keep] = h[keep, :keep] = beta * vectors[-1, :keep]
                count = keep
                restart_count += 1
                check_deadline()
            else:
                h[count - 1, count] = h[count, count - 1] = beta
    except _TimeBudgetExceeded:
        failure.append('time_budget_exhausted')

    if not accepted and not failure:
        failure.append('direct_residual_failed' if validation_attempts
                       else 'product_budget_exhausted')
    worst_rank = max(direct, key=direct.get) if direct else None
    scalars = dict(accepted=bool(accepted), gn_products=products,
                   seconds=elapsed(),
                   restart_count=restart_count, basis_size=count,
                   basis_capacity=max_basis,
                   configured_max_basis=memory['requested_max_basis'],
                   memory_limited=memory['memory_limited'],
                   validation_attempts=validation_attempts,
                   orthogonality_error=orthogonality_error,
                   max_direct_residual=(float(max(direct.values()))
                       if direct else None),
                   worst_direct_residual_rank=worst_rank,
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
