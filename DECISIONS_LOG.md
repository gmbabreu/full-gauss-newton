# Validated decisions

## 2026-11-06 — Matched damping control

- Minimize g*d + d*G*d/2 + mu*||d||^2/2 using Adam to target the same
  system (G+mu I)d=-g used by Jacobi-PCG. Verified against an explicit
  Jacobian/logit Hessian and a dense solve in tests/test_gn_damping.py.
- Keep gn_damping_mu separate from cg_damping_mu, optimizer_wd and
  inner_loop_wd. Default zero preserves the existing computation path.
- Existing relative_residual / inner_relative_residual now measure the
  damped gradient when enabled, at the iterate BEFORE each Adam update.
  gn_quad/final_relative_residual remains the raw undamped final residual;
  all gn_quad predictions remain data-only. These are different diagnostics.
- Damping improves conditioning for PSD G, but changes the target direction.
  Better conditioning alone does not guarantee better nonlinear or eval loss.
- Finite-iteration Adam and PCG may differ even on the identical system;
  different results do not alone establish a PCG bug.
