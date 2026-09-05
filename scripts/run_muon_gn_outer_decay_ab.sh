#!/usr/bin/env bash
# Controlled A/B launch template. Review paths and resource launcher before use.
set -euo pipefail

: "${START_CHECKPOINT:?set START_CHECKPOINT to the identical params-only or full-state checkpoint}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
COMMON=(
  --load_checkpoint="$START_CHECKPOINT"
  --output_dir="$OUTPUT_ROOT"
  --optimizer_type=muon --gauss_newton=True --reset_start=False
  --adaptive_inner_loop=False --inner_loop_iter=100
  --inner_loop_lr=0.001 --optimizer_wd=0.0 --inner_loop_wd=0.0
  --lr_sched=cosine --linesearch=True --single_batch_inner=False
  --weight_average=False
  --total_steps=1000 --log_outer_update_stats=True
)

# Submit these through the repository's normal TPU launcher; this script does not
# provision hardware. Run one command per identically configured worker/job.
python -m EasyLM.models.llama.llama_train_gn "${COMMON[@]}" \
  --experiment_id=muon-gn-outer-decay-A --outer_weight_decay=0.0
python -m EasyLM.models.llama.llama_train_gn "${COMMON[@]}" \
  --experiment_id=muon-gn-outer-decay-B --outer_weight_decay=0.0001
