# Project reference

## Training progress schema 2

Adam and GN trainers expose reporting-only `log_step_offset`, `log_token_offset`,
and `log_initial_eval` flags. Explicit offsets identify a comparison prefix for
a parameter-only optimizer branch; they never alter optimizer counters, learning
rate schedules, dataset skipping, or training horizons. Use `-1` to inherit
supported full-resume reporting state, or zero for a new zero-origin experiment.

Every emitted update row uses cumulative completed updates for `step`,
`global_step`, and the explicit W&B history step. `total_tokens` and
`cumulative_tokens` include only distinct fresh training positions used by a
solve or line-search selection. `phase_skipped_tokens` records fetched positions
that were discarded, and `dataset_total_tokens` reports the training-stream
cursor. Validation tokens and repeated inner/line-search evaluation do not add
to training-token totals.

`log_initial_eval=True` emits a pre-update validation row at local step `-1`
without advancing the live training RNG. Runs replayed from older checkpoints
must use a fresh W&B run ID to avoid backward history.
