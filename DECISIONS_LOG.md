# Decisions Log

- Exact continuation is limited to raw Hugging Face AdamW with FP32 parameters/checkpoints, optimizer-state saving, gradient accumulation 1, and no weight averaging.
- Completion markers are written after model, metadata, and packed-dataset state and are tied together by snapshot UUID and completed optimizer step.
- Packed dataset snapshots retain the unconsumed token/mask remainder and replay the seeded document stream to the consumed-document count.
- Existing legacy dataset snapshots remain loadable outside the exact-resume validation route.
- Batch-size changes are accepted only through an explicit dataset-loader option, used by raw Hugging Face Muon-GN branches initialized from parameter-only checkpoints; strict full-state continuation remains the default.
- Progress schema 3 defines `total_tokens` and `cumulative_tokens` as the explicit parent solve-token prefix plus `phase_solve_tokens`. Line-search and skipped-fetch counters remain separate and still advance the physical dataset cursor. Reporting offsets never affect optimizer or schedule state. Schema 2 counted solve plus line-search tokens; its separate saved counters remain readable, preserving explicit prefixes and Adam resume coordinates. A new GN branch must use a solve-only parent prefix, not an arbitrary parent's physical dataset cursor.
- Muon-GN full-state continuation remains explicitly unsupported; parameter-only branches with packed dataset state validate metadata, completion, and dataset snapshot identity before claiming a cumulative prefix.
- Timing uses synchronized process-local wall time: training includes fetch/compile/solve/search/update work, validation is accumulated separately, and no elapsed-time prefix is inferred or checkpointed.
- Partially completed GN work caused by dataset exhaustion contributes to process training time without advancing the latest completed-update duration or progress counters.
- `log_time_offset_s` is run-config reporting metadata only; measured process-local timing continues to start from zero.
- The ordinary single-batch Muon logger on `main` already counts one solve batch per outer update, excluding line search; no change to that path is required. Schema 3 restores the same plotting convention on the resumable branch. This change does not remove line-search computation or alter fetch order, optimizer state, timing, or checkpoint policy.
- Existing schema-2 Muon runs branched from these Adam checkpoints can be corrected for plotting with `total_tokens - phase_linesearch_tokens`. This is a version-specific display correction; do not subtract from schema-3 totals. Use a new W&B run for subsequent schema-3 launches so an old run does not acquire a discontinuous token axis.
