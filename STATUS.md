# Status

Resumable raw Hugging Face Adam checkpoints are implemented on CPU-facing repository code. Muon-GN parameter-only branches can explicitly retain a packed Adam data-stream snapshot while choosing a new batch size. The focused test suite is present, but could not be collected in this checkout because JAX, NumPy, datasets, MLXU, and google-cloud-storage are not installed.

No TPU training, GCS transfer, multi-host execution, or live training process was run.

Progress schema 3 reporting is integrated into Adam and GN logging and checkpoint metadata. `total_tokens` and `cumulative_tokens` count only the parent solve-token prefix plus distinct solve batches. Line-search tokens and skipped fetches remain separate diagnostics; the absolute dataset cursor still includes every fetched batch. Schema-2 component metadata remains readable for existing Adam resumes. Focused host-side accounting tests are included; full trainer tests remain limited by unavailable runtime dependencies in this checkout.

Trainer integration now copies evaluation RNGs before donated calls, retains Adam's local evaluation cadence, validates parameter-only parent bundles, distinguishes line-search baseline loss, and reports terminal summaries at the current boundary.

Adam and GN now report synchronized `update_time_s`, cumulative process-local `train_time_s`, and cumulative process-local `eval_time_s`, including initial and terminal validation where enabled.

Interrupted GN outer work is retained in final training time, and final timing/progress summaries are exported even when terminal evaluation is disabled.

Both trainers expose and record the reporting-only `log_time_offset_s` annotation.

The ordinary single-batch Muon path on `main` already excludes line-search tokens; schema 3 aligns the resumable branch with that convention. Existing schema-2 Adam-parent Muon history can be plotted as `total_tokens - phase_linesearch_tokens`, without rerunning training. Do not apply that subtraction to schema 3 or older solve-only runs. Tokens are a solve-data budget, not total data access or computation; actual line-search time stays included in timing.
