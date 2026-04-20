#!/usr/bin/env bash
set -euo pipefail

# Reproduce Figure-1-style curve with full Gauss-Newton on a Cloud TPU VM.
# This script runs for 4,000 GN outer-loop steps (not early-stopping at target loss).
#
# Usage (on TPU VM host):
#   export WANDB_API_KEY=...
#   export OUTPUT_DIR=gs://<your-bucket>/figure1-gn
#   export TRAIN_PRETOKENIZED_DIR=gs://<your-bucket>/c4-tokenized
#   export EVAL_PRETOKENIZED_DIR=gs://<your-bucket>/c4-tokenized-val
#   bash templates/gn-tpu-vm-figure1-4000.sh

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export JAX_TRACEBACK_FILTERING=off

# Avoid inheriting incompatible TPU client flags from host shells.
unset LIBTPU_INIT_ARGS || true
unset TPU_LIBRARY_PATH || true

# Optional TPU throughput tuning flags.
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

: "${OUTPUT_DIR:?Set OUTPUT_DIR to a local path or gs:// bucket prefix}"
: "${TRAIN_PRETOKENIZED_DIR:?Set TRAIN_PRETOKENIZED_DIR}"
: "${EVAL_PRETOKENIZED_DIR:?Set EVAL_PRETOKENIZED_DIR}"

python sweep_launcher.py \
  --program='EasyLM.models.llama.llama_train_gn' \
  --mesh_dim='1,1,1' \
  --dtype='bf16' \
  --param_dtype='bf16' \
  --total_steps=4000 \
  --log_freq=1 \
  --eval_freq=10 \
  --eval_steps=100 \
  --inner_loop_iter=122 \
  --gradient_accumulation_steps=1 \
  --save_model_freq=100 \
  --save_milestone_freq=0 \
  --load_llama_config='' \
  --update_llama_config='' \
  --llama.base_model='45M' \
  --llama.initializer_range=1.0 \
  --optimizer_type='muon' \
  --lr_sched='constant_with_inner_cosine' \
  --inner_loop_lr=0.01 \
  --global_warmup=0 \
  --inner_loop_warmup=0 \
  --inner_loop_wd=0 \
  --parameter_wd=0 \
  --optimizer_wd=0.001 \
  --inner_b1=0.9 \
  --inner_b2=0.999 \
  --inner_clip_gradient=1 \
  --weight_average=True \
  --weight_average_decay=0.999 \
  --linesearch=True \
  --ls_range=5 \
  --gauss_newton=True \
  --target_loss=0.0 \
  --tokenizer='google-t5/t5-base' \
  --train_dataset.type='huggingface' \
  --train_dataset.text_processor.fields='text' \
  --train_dataset.text_processor.add_bos_token=False \
  --train_dataset.huggingface_dataset.pretokenized_dataset_dir="${TRAIN_PRETOKENIZED_DIR}" \
  --train_dataset.huggingface_dataset.tokens_count_at_start=32768000 \
  --train_dataset.huggingface_dataset.path='allenai/c4' \
  --train_dataset.huggingface_dataset.name='en' \
  --train_dataset.huggingface_dataset.streaming=True \
  --train_dataset.huggingface_dataset.split='train' \
  --train_dataset_batch_size=32 \
  --eval_dataset.text_processor.fields='text' \
  --eval_dataset.text_processor.add_bos_token=False \
  --eval_dataset.huggingface_dataset.pretokenized_dataset_dir="${EVAL_PRETOKENIZED_DIR}" \
  --eval_dataset.huggingface_dataset.path='allenai/c4' \
  --eval_dataset.huggingface_dataset.name='en' \
  --eval_dataset.huggingface_dataset.split='validation' \
  --eval_dataset.huggingface_dataset.batch_size=128 \
  --checkpointer.save_optimizer_state=True \
  --wandb_project='full-gn-figure1-repro' \
  --output_dir="${OUTPUT_DIR}" \
  --notes='Figure1 reproduction: Full GN 45M, 4000 steps'
