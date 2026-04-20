#!/usr/bin/env bash
set -euo pipefail

# Reproduce Figure-1-style curve with AdamW on a Cloud TPU VM.
# This script runs for 4,000 optimizer steps (not early-stopping at target loss).
#
# Usage (on TPU VM host):
#   export WANDB_API_KEY=...
#   export OUTPUT_DIR=gs://<your-bucket>/figure1-adamw
#   export TRAIN_PRETOKENIZED_DIR=gs://<your-bucket>/c4-tokenized
#   export EVAL_PRETOKENIZED_DIR=gs://<your-bucket>/c4-tokenized-val
#   bash templates/adam-tpu-vm-figure1-4000.sh

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export JAX_TRACEBACK_FILTERING=off

# Optional TPU throughput tuning flags.
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

: "${OUTPUT_DIR:?Set OUTPUT_DIR to a local path or gs:// bucket prefix}"
: "${TRAIN_PRETOKENIZED_DIR:?Set TRAIN_PRETOKENIZED_DIR}"
: "${EVAL_PRETOKENIZED_DIR:?Set EVAL_PRETOKENIZED_DIR}"

python sweep_launcher.py \
  --program='EasyLM.models.llama.llama_train' \
  --mesh_dim='1,1,1' \
  --dtype='bf16' \
  --param_dtype='bf16' \
  --total_steps=488000 \
  --log_freq=122 \
  --eval_freq=122 \
  --eval_steps=100 \
  --save_model_freq=1220 \
  --save_milestone_freq=0 \
  --load_llama_config='' \
  --update_llama_config='' \
  --llama.base_model='45M' \
  --llama.initializer_range=1.0 \
  --optimizer.type='adamw' \
  --optimizer.accumulate_gradient_steps=122 \
  --optimizer.adamw_optimizer.lr_sched=constant_with_warmup \
  --optimizer.adamw_optimizer.weight_decay=0 \
  --optimizer.adamw_optimizer.lr=0.01 \
  --optimizer.adamw_optimizer.init_lr=0.0 \
  --optimizer.adamw_optimizer.end_lr=0.0 \
  --optimizer.adamw_optimizer.lr_warmup_steps=0 \
  --optimizer.adamw_optimizer.lr_decay_steps=4000 \
  --optimizer.adamw_optimizer.b1=0.9 \
  --optimizer.adamw_optimizer.b2=0.95 \
  --weight_average=True \
  --weight_average_decay=0.9 \
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
  --notes='Figure1 reproduction: AdamW 45M, 4000 optimizer steps'
