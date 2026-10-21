# Logger
EasyLM uses the [MLXU](https://github.com/young-geng/mlxu) library for logging.
Specifically, EasyLM uses the `mlxu.WandBLogger` module for logging. The `WandBLogger`
module is a wrapper of the [Weights & Biases](https://wandb.ai/site) library. The
following options are available for configuring the `WandBLogger` module:
* `online`: Whether to log online. If `False`, the logger will not upload to W&B service.
* `prefix`: The prefix of the W&B project name.
* `project`: The W&B project name.
* `output_dir`: The output directory for checkpointing, can be a local directory or a
  Google Cloud Storage directory.
* `wandb_dir`: The output directory for W&B logs. Must be a local directory.
* `random_delay`: Whether to add a random delay to the logging process.
* `experiment_id`: The experiment ID. If not specified, a random ID will be generated.
* `anonymous`: Whether to log anonymously.
* `notes`: The notes for the experiment.
* `entity`: The W&B entity name.
* `prefix_to_id`: Whether to add a prefix to the experiment ID.

For more information about the logger configuration, please refer to the
[MLXU](https://github.com/young-geng/mlxu) library.

## GN inner residual plots

For nonadaptive SGD, Adam, or Muon GN solves, enable `--log_inner_steps=True`.
The trainer logs numeric `inner_relative_residual` scalars directly to W&B at
the initial iterate, every tenth inner iteration, and the final iteration.
In a regular line panel, use Y=`inner_relative_residual`, X=`inner_step`.
Each experiment remains a separate W&B run, so selecting multiple experiments
overlays separate curves. The trainer currently initializes a separate run on
each JAX worker; each of those runs receives its own inner scalar history.

The x-axis is `outer_step * inner_loop_iter + i` and continues across outer
solves. Residuals describe the pre-update iterate: with 1000 inner iterations,
the logged indices in the first solve are 0, 9, 19, ..., 999. A new quadratic
problem starts at each outer step, so jumps at solve boundaries are expected.
The post-1000th-update residual is not separately evaluated. Console output
and the deferred `inner_diagnostics` table still contain every inner iteration.
Outer training metrics retain their `total_tokens` axis.

This change takes effect in a newly started Python process. Updating a checkout
does not update an already-running trainer, and missing historical scalars are
not backfilled. For older runs, retained console logs or table artifact versions
may be used for recovery; the latest table alone contains only the latest solve.
