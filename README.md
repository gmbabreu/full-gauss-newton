# The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton

This repository accompanies the paper **“[The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](https://arxiv.org/abs/2510.09378)” (Abreu et al., 2025)**. It builds off of the [EasyLM](https://github.com/young-geng/EasyLM) framework to support full and layer-wise **Gauss–Newton (GN)** preconditioning, as well as a prox-linear variant, to study the performance limits of second-order optimization in transformer-based language models.

---

## Repository Structure

We build directly on top of the EasyLM codebase. Files marked with `(*)` were modified from the EasyLM repo, and those with `(+)` were added for this project.

```text
EasyLM/
├── data.py (*)                         # Modified to handle option for pretokenized dataset
├── gcs_utils.py (+)                   # Utilities for checkpoint/data storage
├── jax_utils.py (*)                   # Additional JAX/training utilities
├── layerwise_utils.py (+)             # Utilities for layer-wise GN computations
├── models/llama/
│   ├── llama_model.py (*)             # Added configs for 45M and 150M models
│   ├── llama_train.py (*)             # Baseline training: AdamW, Muon, SOAP
│   ├── llama_train_gn.py (+)          # Full GN and GN-prox-linear methods
│   ├── llama_train_gn_layerwise.py (+)# Layer-wise GN and GN-prox-linear
├── optimizers.py (*)                  # Modified to include additional baselines
├── pretokenize.py (+)                 # Data preprocessing and tokenization
templates/
├── adam-template.sbatch (+)           # SLURM template for baseline optimizers
├── gn-template.sbatch (+)             # SLURM template for GN runs
sweep.py (+)                           # Main experiment launcher
sweep_launcher.py (+)                  # Sweep launcher for hyperparameter tuning
```

---

## Training Scripts

| Script | Function |
|---|---|
| `llama_train.py` | Runs baseline optimizers (AdamW, Muon, SOAP). |
| `llama_train_gn.py` | Runs full Gauss–Newton (GN) and GN-prox-linear methods. |
| `llama_train_gn_layerwise.py` | Runs layer-wise GN and layer-wise GN-prox-linear methods. |

---
### llama_train_gn Structure
```text
                 ┌─────────────────────┐
                 │     params0, batch   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    build_gn_model   │
                 │                     │
                 │ b = Jᵀg             │
                 │ G(v) = JᵀHJv        │
                 │ q(Δ)                │
                 └──────────┬──────────┘
                            │
                         GNModel
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          ┌─────────────┐       ┌─────────────┐
          │    Muon     │       │     CG      │
          │ inner solve │       │ inner solve │
          └──────┬──────┘       └──────┬──────┘
                 │                     │
                 └──────────┬──────────┘
                            ▼
                      candidate_params
                            │
                            ▼
                         line search
```
---
## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/gmbabreu/full-gauss-newton.git
cd full-gauss-newton

pip install "setuptools<68"
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install \
    transformers==4.40.0 \
    datasets==3.0.0 \
    fsspec==2024.6.1 \
    gcsfs==2024.6.1 \
    flax optax tensorflow mlxu tqdm wandb einops
```

For GPU-only setups, install the appropriate CUDA-enabled JAX version instead of the TPU build.

Set the repository root in your Python path:

```bash
export PYTHONPATH="${PWD}:$PYTHONPATH"
```

---

## Dataset Preparation

Experiments in the paper use the **C4 dataset**.

The codebase supports either:

- Streaming datasets directly from HuggingFace
- Pretokenized datasets generated with `pretokenize.py`

Example streaming dataset configuration:

```bash
--train_dataset.type='huggingface' \
--train_dataset.huggingface_dataset.path='allenai/c4' \
--train_dataset.huggingface_dataset.name='en' \
--train_dataset.huggingface_dataset.streaming=True
```

---

## Running Experiments

The repository supports both direct training commands and hyperparameter sweeps through `sweep.py` / `sweep_launcher.py`.

### Baseline Optimizers (AdamW / Muon / SOAP)

```bash
python sweep.py \
    --program='EasyLM.models.llama.llama_train' \
    --optimizer.type='adamw'
```

### Full Gauss–Newton (GN)

```bash
python sweep.py \
    --program='EasyLM.models.llama.llama_train_gn' \
    --gauss_newton=True
```

### Layer-wise Gauss–Newton

```bash
python sweep.py \
    --program='EasyLM.models.llama.llama_train_gn_layerwise'
```

Example SLURM templates are provided in `templates/` for multi-node or cluster execution.

---

# Example Experiments

## 150M AdamW Warmup Run

This warmup checkpoint is used as the initialization for subsequent AdamW and GN experiments.

Replace the placeholder values below with your own configuration:

- `<WANDB_PROJECT>`
- `<WANDB_ENTITY>`
- `<OUTPUT_DIR>`

```bash
export SLURM_ARRAY_JOB_ID=1
export SLURM_ARRAY_TASK_ID=1

python sweep_launcher.py \
    --program='EasyLM.models.llama.llama_train' \
    --mesh_dim='1,1,8' \
    --dtype='fp32' \
    --total_steps=4578 \
    --log_freq=100 \
    --eval_freq=500 \
    --eval_steps=50 \
    --save_model_freq=4578 \
    --save_milestone_freq=0 \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='150M' \
    --llama.initializer_range=1.0 \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.lr_sched=constant_with_warmup \
    --optimizer.adamw_optimizer.lr=0.003 \
    --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.init_lr=0.0 \
    --optimizer.adamw_optimizer.end_lr=0.0 \
    --optimizer.adamw_optimizer.lr_warmup_steps=457 \
    --optimizer.adamw_optimizer.lr_decay_steps=4578 \
    --optimizer.adamw_optimizer.b1=0.9 \
    --optimizer.adamw_optimizer.b2=0.95 \
    --tokenizer='google-t5/t5-base' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_bos_token=False \
    --train_dataset.huggingface_dataset.path='allenai/c4' \
    --train_dataset.huggingface_dataset.name='en' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset_batch_size=32 \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.text_processor.add_bos_token=False \
    --eval_dataset.huggingface_dataset.path='allenai/c4' \
    --eval_dataset.huggingface_dataset.name='en' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --eval_dataset.huggingface_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=True \
    --wandb_project='<WANDB_PROJECT>' \
    --wandb_entity='<WANDB_ENTITY>' \
    --wandb_dir='/tmp' \
    --output_dir='<OUTPUT_DIR>/warmup_150m' \
    --notes='AdamW warmup 150M 5pct chinchilla'
```

---

## 150M Full Gauss–Newton Run

This reproduces the main GN experiment from Figure 1 of the paper.

Before running:

1. Complete the warmup run above
2. Replace:
   - `<WANDB_PROJECT>`
   - `<WANDB_ENTITY>`
   - `<OUTPUT_DIR>`
   - `<WARMUP_CHECKPOINT_PATH>`

Example checkpoint path:

```text
trainstate_params::<OUTPUT_DIR>/warmup_150m/1_0/streaming_train_state
```

```bash
export SLURM_ARRAY_JOB_ID=2
export SLURM_ARRAY_TASK_ID=1

python sweep_launcher.py \
    --program='EasyLM.models.llama.llama_train_gn' \
    --mesh_dim='1,1,8' \
    --dtype='fp32' \
    --total_steps=500000 \
    --log_freq=1 \
    --eval_freq=1 \
    --eval_steps=50 \
    --inner_loop_iter=1831 \
    --gradient_accumulation_steps=1 \
    --save_model_freq=3 \
    --save_milestone_freq=0 \
    --load_checkpoint='<WARMUP_CHECKPOINT_PATH>' \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='150M' \
    --llama.initializer_range=1.0 \
    --load_dataset_state='' \
    --optimizer_type='muon' \
    --lr_sched='global_cosine' \
    --inner_loop_lr=0.003 \
    --global_warmup=0 \
    --inner_loop_warmup=0 \
    --inner_loop_wd=0.01 \
    --parameter_wd=0 \
    --optimizer_wd=0.001 \
    --inner_b1=0.9 \
    --inner_b2=0.999 \
    --inner_clip_gradient=1 \
    --weight_average=False \
    --linesearch=True \
    --ls_range=10 \
    --log_inner_steps=False \
    --tokenizer='google-t5/t5-base' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_bos_token=False \
    --train_dataset.huggingface_dataset.path='allenai/c4' \
    --train_dataset.huggingface_dataset.name='en' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset_batch_size=128 \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.text_processor.add_bos_token=False \
    --eval_dataset.huggingface_dataset.path='allenai/c4' \
    --eval_dataset.huggingface_dataset.name='en' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --eval_dataset.huggingface_dataset.batch_size=128 \
    --checkpointer.save_optimizer_state=True \
    --wandb_project='<WANDB_PROJECT>' \
    --wandb_entity='<WANDB_ENTITY>' \
    --wandb_dir='/tmp' \
    --output_dir='<OUTPUT_DIR>/gn_150m' \
    --notes='GN 150M bsz=240M lr=0.003 wd=0.01 fig1' \
    --gauss_newton=True \
    --reset_start=False
```

---

## Reproducing Paper Results

Experiments are conducted on:

- **45M- and 150M-parameter LLaMA models**
- **C4 dataset**
- AdamW, Muon, SOAP, Full GN, and Layer-wise GN optimizers

For exact hyperparameters and setup details, see Appendix G of the paper.

---

## Logging and Checkpointing

The repository supports:

- Weights & Biases logging
- Local or remote checkpoint directories
- Automatic checkpoint resume
- Evaluation during training

Typical logging flags:

```bash
--wandb_project='<WANDB_PROJECT>' \
--wandb_entity='<WANDB_ENTITY>' \
--output_dir='<OUTPUT_DIR>'
```

---

## Known Issues

| Issue | Fix |
|---|---|
| `FlaxLogitsWarper` import error | Use `transformers==4.40.0` |
| Dataset streaming issues | Use `datasets==3.0.0` |
| `canonicalize_version` TypeError | Use `setuptools<68` |
| GN instability / negative loss | Reduce `inner_loop_lr` |
| Disk filling with `.msgpack` files | Periodically clear `/tmp/*.msgpack` |

---

## Acknowledgements

We gratefully acknowledge the creators of the [EasyLM](https://github.com/young-geng/EasyLM) project for developing the original framework on which this work builds.

---

## Citation

If you use this codebase or build upon our work, please cite:

```bibtex
@misc{abreu2025potentialsecondorderoptimizationllms,
      title={The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton},
      author={Natalie Abreu and Nikhil Vyas and Sham Kakade and Depen Morwani},
      year={2025},
      eprint={2510.09378},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.09378},
}
```

---

## License

This project follows the same license as the original EasyLM repository (Apache 2.0). See `LICENSE` for details.
