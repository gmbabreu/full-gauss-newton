# Decisions Log

- Exact continuation is limited to raw Hugging Face AdamW with FP32 parameters/checkpoints, optimizer-state saving, gradient accumulation 1, and no weight averaging.
- Completion markers are written after model, metadata, and packed-dataset state and are tied together by snapshot UUID and completed optimizer step.
- Packed dataset snapshots retain the unconsumed token/mask remainder and replay the seeded document stream to the consumed-document count.
- Existing legacy dataset snapshots remain loadable outside the exact-resume validation route.
- Batch-size changes are accepted only through an explicit dataset-loader option, used by raw Hugging Face Muon-GN branches initialized from parameter-only checkpoints; strict full-state continuation remains the default.
