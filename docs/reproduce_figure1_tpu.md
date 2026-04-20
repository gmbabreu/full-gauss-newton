# Reproducing Figure 1 Curves on Google Cloud TPU (AdamW vs Full GN)

This runbook targets the **`fast-second-order`** GCP project and reproduces
training-step vs validation-loss curves for:

- AdamW (`EasyLM.models.llama.llama_train`)
- Full Gauss-Newton (`EasyLM.models.llama.llama_train_gn` with `--gauss_newton=True`)

The provided templates run until **4000 steps** (instead of early stopping at
validation loss 3.25).

## Fastest way (from your local machine)

After setting environment variables once, you can drive the full flow with one helper script:

```bash
export PROJECT=fast-second-order
export ZONE=europe-west4-a
export TPU_NAME=fig1-v6e-8
export ACCELERATOR_TYPE=v6e-8
export REPO_URL=https://github.com/<you>/full-gauss-newton.git
export REPO_BRANCH=main
export WANDB_API_KEY=<YOUR_WANDB_KEY>
export OUTPUT_DIR=gs://<YOUR_BUCKET>/figure1-runs
export TRAIN_PRETOKENIZED_DIR=gs://<YOUR_BUCKET>/c4-tokenized
export EVAL_PRETOKENIZED_DIR=gs://<YOUR_BUCKET>/c4-tokenized-val

bash scripts/gcloud_tpu_figure1.sh create
bash scripts/gcloud_tpu_figure1.sh setup
bash scripts/gcloud_tpu_figure1.sh configure
bash scripts/gcloud_tpu_figure1.sh run-both
```

This will create the TPU VM, install dependencies, write `.env.fig1`, and run AdamW then full GN.

## 1) Pick TPU type / zone from available free quota

From your available quota, suggested first choice is:

- **v6e spot** in `europe-west4-a` (64 chips available)

Alternative zones/chips listed in your allocation can be used with identical
commands by changing `ZONE`/`ACCELERATOR_TYPE`.

## 2) Create TPU VM(s)

> You can run AdamW and GN sequentially on one TPU VM, or in parallel on two TPU VMs.

Example for a single-host TPU VM (8 chips):

```bash
gcloud config set project fast-second-order

ZONE=europe-west4-a
TPU_NAME=fig1-v6e-8
ACCELERATOR_TYPE=v6e-8
RUNTIME=tpu-ubuntu2204-base

gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${RUNTIME} \
  --spot
```

## 3) Prepare code + environment on TPU VM

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command "
  set -euo pipefail
  git clone <YOUR_FORK_OR_REPO_URL> full-gauss-newton
  cd full-gauss-newton
  export PYTHONPATH=\"$PWD${PYTHONPATH:+:$PYTHONPATH}\"
  bash scripts/setup_tpu_vm.sh
"
```

## 4) Set runtime variables

Use a GCS bucket for outputs/checkpoints and pretokenized C4 paths.

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command "
  cd full-gauss-newton
  cat > .env.fig1 <<'ENVVARS'
export WANDB_API_KEY=<YOUR_WANDB_KEY>
export OUTPUT_DIR=gs://<YOUR_BUCKET>/figure1-runs
export TRAIN_PRETOKENIZED_DIR=gs://<YOUR_BUCKET>/c4-tokenized
export EVAL_PRETOKENIZED_DIR=gs://<YOUR_BUCKET>/c4-tokenized-val
ENVVARS
"
```

## 5) Run AdamW (4000 optimizer steps)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command "
  set -euo pipefail
  cd full-gauss-newton
  source .env.fig1
  bash templates/adam-tpu-vm-figure1-4000.sh
"
```

## 6) Run Full GN (4000 GN steps)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command "
  set -euo pipefail
  cd full-gauss-newton
  source .env.fig1
  bash templates/gn-tpu-vm-figure1-4000.sh
"
```

## 7) Export/plot step-vs-val-loss curves

Use your preferred tracker (W&B recommended by this repo). A minimal local
matplotlib script after exporting CSVs from W&B:

```python
import pandas as pd
import matplotlib.pyplot as plt

adam = pd.read_csv('adamw_history.csv')
gn = pd.read_csv('gn_history.csv')

plt.plot(adam['step'], adam['eval_loss'], label='AdamW')
plt.plot(gn['step'], gn['eval_loss'], label='Full Gauss-Newton')
plt.xlabel('Training step')
plt.ylabel('Validation loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figure1_adamw_vs_gn_4000steps.png', dpi=200)
```

## Notes on step accounting

- In baseline `llama_train.py`, `total_steps` counts gradient-accumulation
  steps. With `optimizer.accumulate_gradient_steps=122`, `total_steps=488000`
  corresponds to **4000 optimizer steps**.
- In `llama_train_gn.py`, `total_steps=4000` directly controls GN outer steps.
- Both scripts set `target_loss=0.0`, which disables early stop at a target
  validation loss.

## Cleanup

```bash
gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE} --quiet
```


## Troubleshooting

- `NOT_FOUND ... nodes/<TPU_NAME>`: the TPU VM was not created yet (or wrong zone/name).
  Run `bash scripts/gcloud_tpu_figure1.sh create` first, then `setup`.
- `PYTHONPATH: unbound variable`: use safe export form
  `export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"` (already used by helper script).
- `ensurepip is not available` / `python3-venv` missing: `scripts/setup_tpu_vm.sh` now tries
  `apt-get update` + `apt-get install python3-venv` and falls back to `pip --user` if venv cannot be created.
- `apt ... 404 Not Found`: package index is stale; run `sudo apt-get update` before install.
- `ImportError: cannot import name 'Jaxpr' from jax.core` from `neural_tangents`: this flow no longer requires `neural_tangents`; rerun `bash scripts/setup_tpu_vm.sh` to reinstall pinned TPU JAX (`jax[tpu]==0.4.28`) and updated deps.
- If you accidentally pasted secrets (e.g., W&B API key) into terminal logs, rotate them immediately.
