#!/bin/bash
# Train 45M LLaMA with Full Gauss-Newton on GCP TPU.
#
# Step accounting (matches Figure 1 paper axis):
#   total_steps=33 outer (Taylor) steps × inner_loop_iter=122 = 4026 inner steps
#   This aligns with the ~4000 gradient steps run by AdamW/Muon scripts.
#   log_freq=1 → one WandB entry per outer step.
#
# Usage:
#   WANDB_API_KEY=<your_key> WANDB_ENTITY=<your_entity> \
#   OUTPUT_DIR=gs://<bucket>/gn bash scripts/run_gn_tpu.sh

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

python -m EasyLM.models.llama.llama_train_gn \
    --mesh_dim='1,-1,1' \
    --dtype='fp32' \
    --total_steps=32 \
    --log_freq=1 \
    --eval_freq=3 \
    --eval_steps=100 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_checkpoint='' \
    --load_dataset_state='' \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='45M' \
    --llama.initializer_range=1.0 \
    --inner_loop_iter=122 \
    --gradient_accumulation_steps=1 \
    --gauss_newton=True \
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
    --linesearch=True \
    --ls_range=5 \
    --reset_start=False \
    --weight_average=False \
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
    --notes='GN 45M 33 outer steps TPU'
