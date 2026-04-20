#!/bin/bash
# Train 45M LLaMA with Muon on GCP TPU for 4000 gradient steps (~33 optimizer steps).
#
# Step accounting (matches Figure 1 paper axis):
#   total_steps=4000 micro-batch steps, accumulate_gradient_steps=122
#   → 4000 / 122 ≈ 33 optimizer steps  ≈ 4026 effective gradient steps
#
# Usage:
#   WANDB_API_KEY=<your_key> WANDB_ENTITY=<your_entity> \
#   OUTPUT_DIR=gs://<bucket>/muon bash scripts/run_muon_tpu.sh

set -e

# ── Required user config ──────────────────────────────────────────────────────
WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_ENTITY="${WANDB_ENTITY:-ga2740-new-york-university}"
WANDB_PROJECT="${WANDB_PROJECT:-Gauss Newton}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
# ─────────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${PWD}:$PYTHONPATH"
export JAX_TRACEBACK_FILTERING=off
export WANDB_API_KEY

export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 \
  --xla_tpu_spmd_threshold_for_allgather_cse=10000 \
  --xla_enable_async_all_gather=true \
  --xla_tpu_enable_latency_hiding_scheduler=true \
  TPU_MEGACORE=MEGACORE_DENSE'

python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --dtype='fp32' \
    --total_steps=27694 \
    --log_freq=122 \
    --eval_freq=122 \
    --eval_steps=100 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_checkpoint='' \
    --load_dataset_state='' \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='45M' \
    --llama.initializer_range=1.0 \
    --optimizer.type='muon' \
    --optimizer.accumulate_gradient_steps=122 \
    --optimizer.muon_optimizer.lr_sched='constant_with_warmup' \
    --optimizer.muon_optimizer.lr=0.01 \
    --optimizer.muon_optimizer.init_lr=0.0 \
    --optimizer.muon_optimizer.end_lr=0.0 \
    --optimizer.muon_optimizer.lr_warmup_steps=0 \
    --optimizer.muon_optimizer.lr_decay_steps=227 \
    --optimizer.muon_optimizer.b1=0.9 \
    --optimizer.muon_optimizer.b2=0.95 \
    --optimizer.muon_optimizer.beta=0.95 \
    --optimizer.muon_optimizer.weight_decay=0 \
    --optimizer.muon_optimizer.clip_gradient=1.0 \
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
    --train_dataset_batch_size=32 \
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
    --notes='Muon 45M 4000 steps TPU'
