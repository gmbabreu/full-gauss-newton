#!/bin/bash
# Train 45M LLaMA with AdamW on GCP TPU for 4000 gradient steps (~33 optimizer steps).
#
# Step accounting (matches Figure 1 paper axis):
#   total_steps=4000 micro-batch steps, accumulate_gradient_steps=122
#   → 4000 / 122 ≈ 33 optimizer steps  ≈ 4026 effective gradient steps
#
# Usage:
#   WANDB_API_KEY=<your_key> WANDB_ENTITY=<your_entity> \
#   OUTPUT_DIR=gs://<bucket>/adamw bash scripts/run_adamw_tpu.sh

set -e

# ── Required user config ──────────────────────────────────────────────────────
WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_ENTITY="${WANDB_ENTITY:-ga2740-new-york-university}"
WANDB_PROJECT="${WANDB_PROJECT:-Gauss Newton}"
OUTPUT_DIR="${OUTPUT_DIR:-}"          # e.g. gs://your-bucket/experiments/adamw
# ─────────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${PWD}:$PYTHONPATH"
export JAX_TRACEBACK_FILTERING=off
export WANDB_API_KEY

# TPU XLA flags for improved throughput (uncomment if on TPU)
python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --dtype='fp32' \
    --total_steps=4000 \
    --log_freq=1 \
    --eval_freq=10 \
    --eval_steps=100 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_checkpoint='' \
    --load_dataset_state='' \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='45M' \
    --llama.initializer_range=1.0 \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=122 \
    --optimizer.adamw_optimizer.lr_sched='constant_with_warmup' \
    --optimizer.adamw_optimizer.lr=0.01 \
    --optimizer.adamw_optimizer.init_lr=0.0 \
    --optimizer.adamw_optimizer.end_lr=0.0 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 \
    --optimizer.adamw_optimizer.lr_decay_steps=227 \
    --optimizer.adamw_optimizer.b1=0.9 \
    --optimizer.adamw_optimizer.b2=0.95 \
    --optimizer.adamw_optimizer.weight_decay=0 \
    --weight_average=True \
    --weight_average_decay=0.9 \
    --tokenizer='google-t5/t5-base' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_bos_token=False \
    --train_dataset.huggingface_dataset.pretokenized_dataset_dir='' \
    --train_dataset.huggingface_dataset.tokens_count_at_start=0 \
    --train_dataset.huggingface_dataset.path='allenai/c4' \
    --train_dataset.huggingface_dataset.name='en' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset_batch_size=8 \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.text_processor.add_bos_token=False \
    --eval_dataset.huggingface_dataset.pretokenized_dataset_dir='' \
    --eval_dataset.huggingface_dataset.path='allenai/c4' \
    --eval_dataset.huggingface_dataset.name='en' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --eval_dataset.huggingface_dataset.batch_size=128 \
    --eval_dataset.huggingface_dataset.streaming=True \
    --checkpointer.save_optimizer_state=False \
    --wandb_project="${WANDB_PROJECT}" \
    --wandb_entity="${WANDB_ENTITY}" \
    --wandb_dir='/tmp' \
    --output_dir="${OUTPUT_DIR}" \
    --notes='AdamW 45M 4000 steps TPU'
