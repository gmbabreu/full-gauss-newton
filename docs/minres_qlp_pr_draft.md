Title: Add undamped MINRES-QLP for opt-in plain Gauss-Newton solves

Adds a matrix-free JAX MINRES-QLP solver with zero initialization, QLP from the
first iteration, constant vector storage, explicit rank truncation, and verified
residual/stationarity exits. `optimizer_type=cg`, `cg_solver=minres_qlp`,
`cg_plain_gn=True`, `cg_preconditioner=none` solves Gd=-g using the existing
gradient, GN operator, microbatching and line search. Active damping and
direction modifications are rejected. The earlier local MINRES finite/descent
and actual-candidate acceptance guards are reused, including fixed-step checks.

Legacy optimizer defaults and checkpoint slots remain intact. Missing solver
fields normalize to legacy defaults; explicit solver/rank-policy changes reject
resume. Existing `gn_quad` metrics are retained, with QLP status, rank events,
iterations, true residual/stationarity, and counted verification matvecs added.

CPU validation covers FP32/FP64, JIT/pytrees, 42 dense acceptance controls and
the supplied pinned prototype, plus masked-softmax GN against independent
JTWJ with unequal microbatches. Acceptance-control solution errors are at most
1.30e-6 (FP32) and 4.25e-15 (FP64). Severe 1e10 conditioning remains a documented
limitation: small residuals do not imply accurate pseudoinverse solutions.
Full measurements, reproduction commands and licensing are in
`docs/minres_qlp.md`. The recurrence port uses the official BSD-licensed Fortran
v27 source; the Python reference is not redistributed.

Focused CPU tests pass; the pre-existing local ignored quadratic-logging test
failures were reproduced against the original main source. No TPU training or
distributed execution was performed. Local preparation only; no push or merge.
