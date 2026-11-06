# Adam-GN damping command additions

Use the existing Adam-GN launch command on branch `adam-gn-damping`.
Keep its checkpoint, data, LR, total_steps, inner_loop_iter and line search.
Use distinct experiment IDs / output directories to avoid automatic resume.

Common control flags (replace existing occurrences):
```bash
--optimizer_type=adamw --gauss_newton=True --single_batch_inner=True \
--optimizer_wd=0 --inner_loop_wd=0 --cg_damping_mu=0
```

Run A: `--gn_damping_mu=0`

Run B: `--gn_damping_mu=0.03` (replace .03 with the actual PCG mu).

Optional one-update diagnostic: `--stop_after_first_outer_step=True`.
Leave total_steps unchanged so any schedule retains its original horizon.
For the corresponding PCG comparison use gn_damping_mu=0,
optimizer_type=cg, cg_preconditioner=gn_jacobi, cg_damping_mu=.03,
cg_interpolation_lambda=1 and optimizer.adamw_optimizer.b1=0.
Disable lambda schedules as required by the existing Jacobi validation.

For the initial Adam ablation preserve reset_start, clipping and iteration
budget in both arms; test longer solves separately. Isotropic damping acts on
all parameter coordinates without a per-leaf or per-token normalization.
