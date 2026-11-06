# Status — 2026-11-06

Adam-GN damping experiment, based on current main. No training launched.

- `gn_damping_mu` (default 0) adds mu*d to iterative GN gradients and
  mu/2*||d||^2 to the inner objective; d = candidate - outer parameters.
- Existing `inner_loop_wd` does not damp the GN gradient: it is constant
  with respect to the logits differentiated by this route.
- CPU test of the actual trainer kernel against explicit J/H: maximum
  float32 gradient error 2.98e-8 at mu=0, .03, .3. Adam converges to the
  dense damped solve with maximum displacement error 2.98e-7 and relative
  residual 2.10e-7. Independent float64 check: error 6.94e-18.

Next: paired Adam-GN runs with mu=0 and the Jacobi-PCG mu, identical
checkpoint, batches, seed, budgets, LR, reset behavior and line search.
Set optimizer_wd=0 in both for an exact quadratic-system comparison.
Use single_batch_inner=True; confirm dropout is disabled for a fixed operator.
Compare solve/line-search/eval improvements, step size, direction norm and
residuals. More inner iterations distinguish incomplete solving from damping.
Open: TPU behavior and whether damping explains the observed PCG loss gap.
